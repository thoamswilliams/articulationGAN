import torch
import torch.nn.functional as F
import numpy as np

class UpConv(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        filters,
        kernel_size=25,
        stride=4,
        padding=11,
        relu=True,
        use_batchnorm=False,
        output_padding=1
    ):
        self.relu = relu
        super(UpConv, self).__init__()
        self.conv = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=(kernel_size),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.batch_norm = torch.nn.BatchNorm1d(filters) if use_batchnorm else torch.nn.Identity()

    def forward(self, x):
        output = self.conv(x)
        output = self.batch_norm(output)
        output = F.relu(output) if self.relu else 3*torch.tanh(output) #multiply tanh by 3 for EMA
        return(output)

class DownConv(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        filters,
        kernel_size=25,
        stride=4,
        use_batchnorm=False,
        alpha=0.2,
        phaseshuffle_rad=0,
        padding = 11
    ):
        super(DownConv, self).__init__()
        self.alpha = alpha
        self.phaseshuffle_rad = phaseshuffle_rad

        self.conv = torch.nn.Conv1d(
            in_channels,
            filters,
            kernel_size,
            stride,
            padding=padding
        )
        self.batchnorm = torch.nn.BatchNorm1d(filters) if use_batchnorm else torch.nn.Identity()

    def forward(self, x):
        output = self.conv(x)
        output = self.batchnorm(output)
        output = F.leaky_relu(output, self.alpha)
        output = self.phaseshuffle(output, self.phaseshuffle_rad)
        return output

    def phaseshuffle(self, x, rad):
        phase = np.random.randint(-rad, rad+1)
        pad_l = np.max(phase, 0)
        pad_r = np.max(-phase, 0)
        shuffle = torch.nn.ReflectionPad1d((pad_l, pad_r))
        x = shuffle(x)
        return(x)

class WaveGANGenerator(torch.nn.Module):
    def __init__(
        self,
        nch=1,
        kernel_len=25,
        dim=64,
        use_batchnorm=False,
        latent_dim=100,
        padding_len=12,
        lrelu_alpha = 0.2
    ):
        # assert slice_len in [16384]
        super(WaveGANGenerator, self).__init__()
        # dim_mul = 16 if slice_len == 16384 else 32
        dim_mul = 16
        self.dim = dim
        self.dim_mul = dim_mul

        # [100] -> [16, 1024]
        self.z_project = torch.nn.Linear(latent_dim, 4 * 4 * dim * dim_mul)
        self.z_batchnorm = torch.nn.BatchNorm1d(dim*dim_mul) if use_batchnorm else torch.nn.Identity()
        self.avg_pool = torch.nn.AvgPool1d(kernel_size=5, stride = 4, padding = 2)
        dim_mul //= 2

        # Leaky ReLU after the dense layer
        self.lrelu_alpha = lrelu_alpha

        # [16, 1024] -> [32, 512]
        self.upconv0 = UpConv(
            dim * dim_mul * 2,
            dim * dim_mul,
            kernel_len,
            stride=2,
            padding=padding_len,
            use_batchnorm=use_batchnorm
        )

        # [32, 512] -> [64, 512]
        self.upconv1 = UpConv(
            dim * dim_mul,
            dim * dim_mul,
            kernel_len,
            stride=2,
            padding=padding_len,
            use_batchnorm=use_batchnorm
        )
        dim_mul //= 2

        # [64, 512] -> [128, 256]
        self.upconv2 = UpConv(
            dim * dim_mul * 2,
            dim * dim_mul,
            kernel_len,
            stride=2,
            padding=padding_len,
            use_batchnorm=use_batchnorm
        )

        # [128, 256] -> [128, 256]
        self.upconv3 = UpConv(
            dim * dim_mul,
            dim * dim_mul,
            kernel_len,
            stride=1,
            padding=padding_len,
            output_padding=0,
            use_batchnorm=use_batchnorm

        )

        # [128, 256] -> [256, nch]
        self.upconv4 = UpConv(
            dim * dim_mul,
            nch,
            kernel_len,
            stride=2,
            padding=padding_len,
            relu=True,
            use_batchnorm=use_batchnorm
        )

        #[256, nch] -> [64, nch]
        self.downconv = DownConv(
            in_channels = nch, 
            filters = nch, 
            kernel_size = kernel_len, 
            stride = 4, 
            use_batchnorm = use_batchnorm,
            padding = padding_len,
            alpha = 1, 
            phaseshuffle_rad = 0)
        

    def forward(self, z):
        # Project and reshape
        output = self.z_project(z)
        output = self.z_batchnorm(output.view(-1, self.dim * self.dim_mul, 16))
        output = F.leaky_relu(output, self.lrelu_alpha)

        # Conv layers
        output = self.upconv0(output)
        output = self.upconv1(output)
        output = self.upconv2(output)
        output = self.upconv3(output)
        output = self.upconv4(output)
        # output = self.downconv(output)
        output = self.avg_pool(output)
        #activation: empirically ema channels in (-4, 4), loudness in (0 ,2), pitch in (80, 255)
        ema, loudness, pitch = torch.split(output, [12, 1, 1], dim = 1)
        
        ema_max, _ = torch.max(torch.abs(ema), dim=1, keepdim=True)
        ema = 4*(ema / ema_max)
        # ema = 4*F.normalize(ema, dim = 2)

        loudness = F.relu(loudness)
        pitch = F.relu(pitch)
        comb_out = torch.cat((ema, loudness, pitch), dim = 1)
        
        return comb_out

class WaveGANDiscriminator(torch.nn.Module):
    def __init__(
        self,
        kernel_len=25,
        dim=64,
        stride=4,
        use_batchnorm=False,
        slice_len=20480,
        lrelu_alpha = 0.2,
        phaseshuffle_rad=0
    ):
        super(WaveGANDiscriminator, self).__init__()
        self.dim=dim
        self.slice_len = slice_len

        # Conv Layers
        self.downconv_0 = DownConv(
            in_channels = 1, 
            filters = dim, 
            kernel_size = kernel_len, 
            stride = stride, 
            use_batchnorm = use_batchnorm, 
            alpha = lrelu_alpha, 
            phaseshuffle_rad = phaseshuffle_rad)
        
        self.downconv_1 = DownConv(
            in_channels = dim, 
            filters = dim*2, 
            kernel_size = kernel_len, 
            stride = stride, 
            use_batchnorm = use_batchnorm, 
            alpha = lrelu_alpha, 
            phaseshuffle_rad = phaseshuffle_rad)
        
        self.downconv_2 = DownConv(
            in_channels = dim*2, 
            filters = dim*4, 
            kernel_size = kernel_len, 
            stride = stride, 
            use_batchnorm = use_batchnorm, 
            alpha = lrelu_alpha, 
            phaseshuffle_rad = phaseshuffle_rad)
        
        self.downconv_3 = DownConv(
            in_channels = dim*4, 
            filters = dim*8, 
            kernel_size = kernel_len,
            stride = stride, 
            use_batchnorm = use_batchnorm, 
            alpha = lrelu_alpha, 
            phaseshuffle_rad = phaseshuffle_rad)
        
        self.downconv_4 = DownConv(
            in_channels = dim*8, 
            filters=dim*16, 
            kernel_size = kernel_len, 
            stride = stride, 
            use_batchnorm = use_batchnorm, 
            alpha = lrelu_alpha, 
            phaseshuffle_rad = phaseshuffle_rad)

        # Logit
        self.fc_out = torch.nn.Linear(self.slice_len, 1)

    def forward(self, x):
        output = self.downconv_0(x)
        output = self.downconv_1(output)
        output = self.downconv_2(output)
        output = self.downconv_3(output)
        output = self.downconv_4(output)
        output = self.fc_out(output.view(-1, self.slice_len))
        return output

class WaveGANQNetwork(WaveGANDiscriminator):
    def __init__(
        self,
        num_categ,
        slice_len = 20480,
        kernel_len=25,
        dim=64,
        stride=4,
        use_batchnorm=False,
        phaseshuffle_rad=0,
    ):
        super(WaveGANQNetwork, self).__init__(
                                        slice_len = slice_len,
                                        kernel_len=kernel_len,
                                        dim=dim,
                                        stride=stride,
                                        use_batchnorm=use_batchnorm,
                                        phaseshuffle_rad=phaseshuffle_rad
                                    )
        self.fc_out = torch.nn.Linear(self.slice_len, num_categ)


# z = torch.Tensor(np.random.uniform(-1, 1, (25, 100)))
# G = WaveGANGenerator(nch=30, stride=2)
# D = WaveGANDiscriminator(phaseshuffle_rad=20)
# Q = WaveGANQNetwork(latent_dim=10, phaseshuffle_rad=20)
# G_z = G(z)
# print(G_z.shape)
# D_G_z = D(G_z)
# Q_G_z = Q(G_z)
