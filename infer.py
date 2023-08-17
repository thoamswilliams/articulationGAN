import torch
from articulatory.utils import load_model
from infowavegan import WaveGANGenerator
import matplotlib.pyplot as plt
import yaml
from train import synthesize
import os
from torch.utils.tensorboard import SummaryWriter
from scipy.io.wavfile import write

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

synthesis_checkpoint_path = "/workspace/synthesis/wu_weights/best_mel_ckpt.pkl"
synthesis_config_path = "/workspace/synthesis/wu_weights/config.yml"
ckpt_path = "/workspace/gan_training_12ch_7k/epoch480_step49440"
logdir = "/workspace/12ch_7k_sample_outputs"

with open(synthesis_config_path) as f:
    synthesis_config = yaml.load(f, Loader=yaml.Loader)

EMA = load_model(synthesis_checkpoint_path, synthesis_config)
EMA.remove_weight_norm()
EMA = EMA.eval().to(device)

num_ch = 12
BATCH_SIZE = 50
NUM_CATEG = 9

G = WaveGANGenerator(nch=num_ch, kernel_len=7, padding_len=3).to(device).eval()
G.load_state_dict(torch.load(f=ckpt_path + "_G.pt"))

_z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_CATEG).uniform_(-1, 1).to(device)
c = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                                num_classes=NUM_CATEG).to(device)

z = torch.cat((c, _z), dim=1)
articul_out = G(z)
G_z = synthesize(EMA, articul_out.permute(0, 2, 1), synthesis_config).cpu().detach().numpy()


# writer = SummaryWriter(logdir)
# for i in range(BATCH_SIZE):
#     audio = G_z[i,0,:]
#     writer.add_audio(f'Audio/sample{i}', audio, 0, sample_rate=16000)

articul_np = articul_out.cpu().detach().numpy()
# for i in range(num_ch):
#     articul = articul_np[0,i,:]
#     fig, ax = plt.subplots()
#     ax.plot(range(len(articul)), articul)
#     writer.add_figure(f"Articul/articul{i}", fig, 0)

c_cpu = c.cpu().detach().numpy()
for i in range(BATCH_SIZE):
    audio = G_z[i,0,:]
    filename = logdir + "/12ch7k_sample" + str(i) + "_" + str(c_cpu[i])
    write(filename + ".wav", 16000, audio)

    articul = articul_np[i,:,:]
    fig, ax = plt.subplots(3,4, figsize = (12, 9))
    for i in range(num_ch):
        ax[(int)(i/4),i%4].plot(range(len(articul[i,:])), articul[i,:])
    plt.savefig(filename + ".png")
    plt.close()
    