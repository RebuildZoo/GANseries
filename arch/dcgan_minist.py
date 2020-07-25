import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_num):
        # latent_num = 16
        super(Generator, self).__init__()

        ngf = 16
        # (16, 1, 1) -> (16*8, 4,4) -> (16*4, 8, 8) -> (16*2, 14, 14) -> (1, 28, 28)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_num, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2,  1, 2, 2, 0, bias=False),
            nn.Sigmoid()
            # nn.Tanh()
        )
    
    def forward(self, pz_Tsor):
        BS = pz_Tsor.shape[0]
        pz_Tsor = pz_Tsor.view(BS, -1 , 1, 1)
        qreon_Tsor = self.dec_conv(pz_Tsor)
        return qreon_Tsor

def test_generator():
    gm_gnr = Generator(latent_num = 16)
    z_Tsor = torch.rand(5, 16)
    reon_Tsor = gm_gnr(z_Tsor)

    print("Generator output:", reon_Tsor.shape) # [5, 1, 28, 28]

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        ndf = 16

        # (16, 1, 1) -> (16*8, 4,4) -> (16*4, 8, 8) -> (16*2, 14, 14) -> (1, 28, 28)
        # (1, 28, 28) -> (16*2, 14, 14) -> (16*4, 8, 8) -> (16*8, 4,4) -> (1)
        self.body = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, ndf*2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf * 4, 2, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )
    def forward(self, px_Tsor):
        # Spectral Normalization from https://arxiv.org/abs/1802.05957
        BS = px_Tsor.shape[0]
        qjudge_Tsor = self.body(px_Tsor)
        qjudge_Tsor = qjudge_Tsor.view(BS, -1)
        return qjudge_Tsor

def test_discriminator():
    gm_dcm = Discriminator()
    x_Tsor = torch.rand(5, 1, 28, 28)
    j_Tsor = gm_dcm(x_Tsor)

    print("Discriminator output:", j_Tsor.shape) # [5, 1, 28, 28]

if __name__ == "__main__":
    test_discriminator()
    # test_generator()