import torch
from articulatory.utils import load_model
from infowavegan import WaveGANGenerator
import matplotlib.pyplot as plt
import yaml
from train import synthesize
import os
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

synthesis_checkpoint_path = "/global/scratch/users/thomaslu/articulationGAN/wu_weights/best_mel_ckpt.pkl"
synthesis_config_path = "/global/scratch/users/thomaslu/articulationGAN/wu_weights/config.yml"
ckpt_path = ""
logdir = ""

with open(synthesis_config_path) as f:
    synthesis_config = yaml.load(f, Loader=yaml.Loader)

EMA = load_model(synthesis_checkpoint_path, synthesis_config)
EMA.remove_weight_norm()
EMA = EMA.eval().to(device)

num_ch = 12
BATCH_SIZE = 5
NUM_CATEG = 9

G = WaveGANGenerator(nch=num_ch, kernel_len=3, padding_len=1, use_batchnorm=False).to(device).eval()
G.load_state_dict(torch.load(f=ckpt_path + "_G.pt"))

_z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_CATEG).uniform_(-1, 1).to(device)
c = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                                num_classes=NUM_CATEG).to(device)

z = torch.cat((c, _z), dim=1)
articul_out = G(z)
G_z = synthesize(EMA, articul_out.permute(0, 2, 1), synthesis_config)


print(G_z.shape)
writer = SummaryWriter(logdir)
for i in range(BATCH_SIZE):
    audio = G_z[i,0,:]
    writer.add_audio(f'Audio/sample{i}', audio, 0, sample_rate=16000)

articul_np = articul_out.cpu().detach().numpy()
for i in range(num_ch):
    articul = articul_np[0,i,:]
    fig, ax = plt.subplots()
    ax.plot(range(len(articul)), articul)
    writer.add_figure(f"Articul/articul{i}", fig, 0)