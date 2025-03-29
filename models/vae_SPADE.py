import torch
import torch.nn as nn
import torch.nn.functional as F

l1_loss = torch.nn.L1Loss()

# ----------------------------------------
# SPADE Layer
# ----------------------------------------
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)

        self.conv = nn.Sequential(
            nn.Conv2d(label_nc, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
    
    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.shape[2:], mode='nearest')  # Rescale segmap
        features = self.conv(segmap)
        gamma = self.mlp_gamma(features)
        beta = self.mlp_beta(features)

        return normalized * (1 + gamma) + beta

# ----------------------------------------
# SPADE Residual Block
# ----------------------------------------
class SPADEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, label_nc):
        super().__init__()

        # Skip connection learned
        self.skip_connection = (in_channels != out_channels)
        if self.skip_connection:
            self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.spade_s = SPADE(in_channels, label_nc)

        fmiddle = min(in_channels, out_channels)

        # Normal flow layers
        self.spade1 = SPADE(in_channels, label_nc)
        self.conv1 = nn.Conv2d(in_channels, fmiddle, kernel_size=3, padding=1)
        
        self.spade2 = SPADE(fmiddle, label_nc)
        self.conv2 = nn.Conv2d(fmiddle, out_channels, kernel_size=3, padding=1)
        
        self.lrelu = nn.LeakyReLU(2e-1)

    def shortcut(self, x, seg):
        if self.skip_connection:
            x_s = self.conv_s(self.spade_s(x, seg))
        else:
            x_s = x
        return x_s

    def forward(self, x, segmap):
        # Skip connection
        x_s = self.shortcut(x, segmap)

        # Normal layers
        x = self.conv1(self.lrelu(self.spade1(x, segmap)))
        x = self.conv2(self.lrelu(self.spade2(x, segmap)))

        # Concatenate
        out = x + x_s

        return out

# ----------------------------------------
# Encoder
# ----------------------------------------
class Encoder(nn.Module):
    def __init__(self, z_dim=256, chs=(1, 64, 128, 256), spatial_size=[64, 64]):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(chs[i], chs[i + 1], 3, padding=1),
                nn.BatchNorm2d(chs[i + 1]),
                nn.LeakyReLU(),
                nn.MaxPool2d(2)
            ) for i in range(len(chs) - 1)]
        )
        _h, _w = spatial_size[0] // (2 ** len(self.enc_blocks)), spatial_size[1] // (2 ** len(self.enc_blocks))
        self.out = nn.Sequential(nn.Flatten(1), nn.Linear(chs[-1] * _h * _w, 2 * z_dim))
    
    def forward(self, x):
        for block in self.enc_blocks:
            x = block(x)
        x = self.out(x)
        return torch.chunk(x, 2, dim=1)

# ----------------------------------------
# Generator with SPADE
# ----------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=256, chs=(256, 128, 64, 32), h=8, w=8, label_nc=1):
        super().__init__()
        self.proj_z = nn.Linear(z_dim, chs[0] * h * w)
        self.reshape = lambda x: x.view(-1, chs[0], h, w)
        
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(chs[i], chs[i], 2, 2) for i in range(len(chs) - 1)
        ])
        
        self.dec_blocks = nn.ModuleList([
            SPADEBlock(chs[i], chs[i + 1], label_nc) for i in range(len(chs) - 1)
        ])
        
        self.head = nn.Sequential(
            nn.Conv2d(chs[-1], 1, 1),
            nn.Tanh()
        )
    
    def forward(self, z, segmap):
        x = self.proj_z(z)
        x = self.reshape(x)
        for i in range(len(self.dec_blocks)):
            x = self.upconvs[i](x)
            x = self.dec_blocks[i](x, segmap)
        return self.head(x)

# ----------------------------------------
# Full VAE Model
# ----------------------------------------
class VAE(nn.Module):
    def __init__(self, z_dim=256, enc_chs=(1, 64, 128, 256), dec_chs=(256, 128, 64, 32), label_nc=1):
        super().__init__()
        self.encoder = Encoder(z_dim, enc_chs)
        self.generator = Generator(z_dim, dec_chs, label_nc=label_nc)
    
    def forward(self, x, segmap):
        mu, logvar = self.encoder(x)
        latent_z = sample_z(mu, logvar)
        output = self.generator(latent_z, segmap)
        return output, mu, logvar

# ----------------------------------------
# Helper Functions
# ----------------------------------------
def sample_z(mu, logvar):
    eps = torch.randn_like(mu)
    return (logvar / 2).exp() * eps + mu

def kld_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def vae_loss(inputs, recons, mu, logvar, beta=1):
    return l1_loss(inputs, recons) + beta * kld_loss(mu, logvar)
def get_noise(n_samples, z_dim, device="cpu"):
    return torch.randn(n_samples, z_dim, device=device)