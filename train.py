import argparse
import yaml
import os
import re

import numpy as np
import torch
import torch.optim as optim
from scipy.io.wavfile import read
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import itertools as it
import matplotlib.pyplot as plt

from infowavegan import WaveGANGenerator, WaveGANDiscriminator, WaveGANQNetwork
from utils import get_continuation_fname


class AudioDataSet:
    def __init__(self, datadir, slice_len, norm_coef = 1, rand_pad_data = False):
        print("Loading data")
        dir = os.listdir(datadir)
        x = np.zeros((len(dir), 1, slice_len))
        i = 0
        for file in tqdm(dir):
            audio = read(os.path.join(datadir, file))[1]
            if audio.shape[0] < slice_len:
                dist_to_pad = slice_len - audio.shape[0]
                if(rand_pad_data):
                    front_pad = np.random.randint(0, dist_to_pad+1)
                    back_pad = dist_to_pad - front_pad
                else:
                    front_pad = 0
                    back_pad = dist_to_pad
                audio = np.pad(audio, (front_pad, back_pad))
            audio = audio[:slice_len]

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32767
            elif audio.dtype == np.float32:
                pass
            else:
                raise NotImplementedError('Scipy cannot process atypical WAV files.')
            audio /= np.max(np.abs(audio)) * norm_coef 
            x[i, 0, :] = audio
            i += 1

        self.len = len(x)
        self.audio = torch.from_numpy(np.array(x, dtype=np.float32))

    def __getitem__(self, index):
        return self.audio[index]

    def __len__(self):
        return self.len


def gradient_penalty(G, D, real, fake, epsilon):
    x_hat = epsilon * real + (1 - epsilon) * fake
    scores = D(x_hat)
    grad = torch.autograd.grad(
        outputs=scores,
        inputs=x_hat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True
    )[0]
    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1)  # norm along each batch
    penalty = ((grad_norm - 1) ** 2).unsqueeze(1)
    return penalty

def synthesize(model, x, spk_embed):
    '''
    Given batch of EMA data and EMA model, synthesizes speech output
    Args:
        x: (batch, num_feats, art_len)

    Return:
        signal: (batch, 1, audio_len)
    '''
    batch_size = x.shape[0]
    spk_embed = np.repeat(spk_embed[np.newaxis, :], batch_size, axis = 0)
    out = model.decode_with_grad(x, spk_embed)

    #shape from (batch, audio_len) to (batch, 1, audio_len)
    torch.unsqueeze(out, 1)
    return out

if __name__ == "__main__":
    # Training Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datadir',
        type=str,
        required=True,
        help='Training Directory'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        required=True,
        help='Log/Results Directory'
    )
    parser.add_argument(
        '--spk_embed_path',
        type=str,
        required=True,
        help='Path to the speaker embedding .npy file'
    )
    parser.add_argument(
        '--num_categ',
        type=int,
        default=0,
        help='Q-net categories'
    )
    parser.add_argument(
        '--norm_coef',
        type=float,
        default=1,
        help='Data normalization coefficient'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=5000,
        help='Epochs'
    )
    parser.add_argument(
        '--slice_len',
        type=int,
        default=20480,
        help='Length of training data'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--cont',
        type=str,
        default="",
        help='''continue: default from the last saved iteration. '''
             '''Provide the epoch number if you wish to resume from a specific point'''
             '''Or set "last" to continue from last available'''
    )

    parser.add_argument(
        '--save_int',
        type=int,
        default=50,
        help='Save interval in epochs'
    )

    parser.add_argument(
        '--num_channels',
        type=int,
        default=14,
        help='Size of articulatory generator output'
    )

    parser.add_argument(
        '--log_audio',
        action='store_true',
        help='Save audio and articulator plots'
    )
    parser.add_argument(
        '--rand_pad_data',
        action='store_true',
        help='Randomly pad the training data'
    )
    parser.add_argument(
        '--norm_in_training',
        action='store_true',
        help='Normalize EMA outputs during the training loop'
    )
    parser.add_argument(
        '--do_not_update_G_with_Q',
        action = 'store_true',
        help='If enabled, does not update G using the Q optimizer'
    )
    parser.add_argument(
        '--kernel_len',
        type=int,
        default=7,
        help='Sets the generator kernel length, must be odd'
    )
    parser.add_argument(
        '--Q_phaseshuffle',
        type=int,
        default=2,
        help='Phase shuffle of the Q network, in radians'
    )

    # Q-net Arguments
    Q_group = parser.add_mutually_exclusive_group()
    Q_group.add_argument(
        '--ciw',
        action='store_true',
        help='Trains a ciwgan'
    )
    Q_group.add_argument(
        '--fiw',
        action='store_true',
        help='Trains a fiwgan'
    )
    args = parser.parse_args()
    train_Q = args.ciw or args.fiw

    assert args.kernel_len % 2 == 1, f"generator kernel length must be odd, got: {args.kernel_len}"

    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from sparc import load_model

    with open(args.spk_embed_path, 'rb') as f:
        spk_embed = np.load(f)

    datadir = args.datadir
    logdir = args.logdir
    SLICE_LEN = args.slice_len
    NUM_CATEG = args.num_categ
    NUM_EPOCHS = args.num_epochs
    WAVEGAN_DISC_NUPDATES = 5
    BATCH_SIZE = args.batch_size
    LAMBDA = 10
    LEARNING_RATE = 1e-4
    BETA1 = 0.5
    BETA2 = 0.9

    CONT = args.cont
    SAVE_INT = args.save_int

    # Load data
    dataset = AudioDataSet(datadir, SLICE_LEN, norm_coef = args.norm_coef, rand_pad_data = args.rand_pad_data)
    dataloader = DataLoader(
        dataset,
        BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    def make_new():
        padding_len = (int)((args.kernel_len - 1)/2)
        G = WaveGANGenerator(nch=args.num_channels, kernel_len=args.kernel_len, padding_len=padding_len, use_batchnorm=False).to(device).train()
        EMA = load_model("en", device = device)

        D = WaveGANDiscriminator(slice_len=SLICE_LEN).to(device).train()

        # Optimizers
        optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

        Q, optimizer_Q, criterion_Q = (None, None, None)
        if train_Q:
            Q = WaveGANQNetwork(slice_len=SLICE_LEN, num_categ=NUM_CATEG, phaseshuffle_rad=args.Q_phaseshuffle).to(device).train()
        if(not args.do_not_update_G_with_Q):
            Q_opt_params = it.chain(G.parameters(), Q.parameters())
        else:
            Q_opt_params = Q.parameters()
        if args.fiw:
            optimizer_Q = optim.RMSprop(Q_opt_params, lr=LEARNING_RATE)
            criterion_Q = torch.nn.BCEWithLogitsLoss()
        elif args.ciw:
            optimizer_Q = optim.RMSprop(Q_opt_params, lr=LEARNING_RATE)
            criterion_Q = lambda inpt, target: torch.nn.CrossEntropyLoss()(inpt, target.max(dim=1)[1])

        return G, D, EMA, optimizer_G, optimizer_D, Q, optimizer_Q, criterion_Q


    # Load models
    G, D, EMA, optimizer_G, optimizer_D, Q, optimizer_Q, criterion_Q = make_new()
    start_epoch = 0
    start_step = 0

    if CONT.lower() != "":
        try:
            print("Loading model from existing checkpoints...")
            fname, start_epoch = get_continuation_fname(CONT, logdir)

            G.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_G.pt")))
            D.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_D.pt")))
            if train_Q:
                Q.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Q.pt")))

            optimizer_G.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Gopt.pt")))
            optimizer_D.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Dopt.pt")))

            if train_Q:
                optimizer_Q.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Qopt.pt")))

            start_step = int(re.search(r'_step(\d+).*', fname).group(1))
            print(f"Successfully loaded model. Continuing training from epoch {start_epoch},"
                  f" step {start_step}")

        # Don't care why it failed
        except Exception as e:
            print("Could not load from existing checkpoint, initializing new model...")
            print(e)
    else:
        print("Starting a new training")

    # Set Up Writer
    writer = SummaryWriter(logdir)
    step = start_step

    for epoch in range(start_epoch + 1, NUM_EPOCHS):

        print("Epoch {} of {}".format(epoch, NUM_EPOCHS))
        print("-----------------------------------------")
        pbar = tqdm(dataloader)
        real = dataset[:BATCH_SIZE].to(device)

        for i, real in enumerate(pbar):
            # D Update
            optimizer_D.zero_grad()
            real = real.to(device)
            epsilon = torch.rand(BATCH_SIZE, 1, 1).repeat(1, 1, SLICE_LEN).to(device)
            _z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_CATEG).uniform_(-1, 1).to(device)
            if train_Q:
                if args.fiw:
                    c = torch.FloatTensor(BATCH_SIZE, NUM_CATEG).bernoulli_().to(device)
                else:
                    c = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                                                    num_classes=NUM_CATEG).to(device)
                z = torch.cat((c, _z), dim=1)
            else:
                z = _z

            fake = synthesize(EMA, G(z), spk_embed)
            if(args.norm_in_training):
                fake = torch.nn.functional.normalize(fake, dim = -1)
            penalty = gradient_penalty(G, D, real, fake, epsilon)

            D_loss = torch.mean(D(fake) - D(real) + LAMBDA * penalty)
            writer.add_scalar('Loss/Discriminator', D_loss.detach().item(), step)
            D_loss.backward()
            optimizer_D.step()

            if i % WAVEGAN_DISC_NUPDATES == 0:
                optimizer_G.zero_grad()
                EMA.zero_grad()
                if train_Q:
                    optimizer_Q.zero_grad()
                _z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_CATEG).uniform_(-1, 1).to(device)

                if train_Q:
                    if args.fiw:
                        c = torch.FloatTensor(BATCH_SIZE, NUM_CATEG).bernoulli_().to(device)
                    else:
                        c = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                                                        num_classes=NUM_CATEG).to(device)

                    z = torch.cat((c, _z), dim=1)
                else:
                    z = _z
                
                articul_out = G(z)
                G_z = synthesize(EMA, articul_out, spk_embed)

                # G Loss
                G_loss = torch.mean(-D(G_z))
                G_loss.backward(retain_graph=True)
                writer.add_scalar('Loss/Generator', G_loss.detach().item(), step)

                # Q Loss
                if train_Q:
                    Q_loss = criterion_Q(Q(G_z), c)
                    Q_loss.backward()
                    writer.add_scalar('Loss/Q_Network', Q_loss.detach().item(), step)
                    optimizer_Q.step()

                # Update
                optimizer_G.step()
            step += 1

        if args.log_audio:
            for i in range(1):
                audio = G_z[i,0,:]
                writer.add_audio(f'Audio/sample{i}', audio, step, sample_rate=16000)
            
            articul_np = articul_out.cpu().detach().numpy()
            for i in range(args.num_channels):
                articul = articul_np[0,i,:]
                fig, ax = plt.subplots()
                ax.plot(range(len(articul)), articul)
                writer.add_figure(f"Articul/articul{i}", fig, step)

        if not epoch % SAVE_INT:
            torch.save(G.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_G.pt'))
            torch.save(D.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_D.pt'))
            if train_Q:
                torch.save(Q.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q.pt'))

            torch.save(optimizer_G.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Gopt.pt'))
            torch.save(optimizer_D.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Dopt.pt'))
            if train_Q:
                torch.save(optimizer_Q.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Qopt.pt'))


