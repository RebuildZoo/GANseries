import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias = False):
        super(BasicDeconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                padding, bias = bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Generator(nn.Module):
    def __init__(self, latent_num):
        # latent_num = 16
        super(Generator, self).__init__()

        ngf = 16
        # (16, 1, 1) -> (16*8, 4,4) -> (16*4, 8, 8) -> (16*2, 14, 14) -> (1, 28, 28)
        self.G_deconv = nn.Sequential(
            BasicDeconv2d(latent_num, ngf * 8, 4, 1, 0), 
            BasicDeconv2d(   ngf * 8, ngf * 4, 4, 2, 1), 
            BasicDeconv2d(   ngf * 4, ngf * 2, 2, 2, 1), 
            nn.ConvTranspose2d(ngf * 2,  1, 2, 2, 0, bias=False),
            nn.Sigmoid()
            # nn.Tanh()
        )
    
    def forward(self, pz_Tsor):
        BS = pz_Tsor.shape[0]
        pz_Tsor = pz_Tsor.view(BS, -1 , 1, 1)
        qreon_Tsor = self.G_deconv(pz_Tsor)
        return qreon_Tsor

def test_generator():
    gm_gnr = Generator(latent_num = 16)
    z_Tsor = torch.rand(5, 16)
    reon_Tsor = gm_gnr(z_Tsor)

    print("Generator output:", reon_Tsor.shape) # [5, 1, 28, 28]

class Conv2d_wSN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias = False, bn_flag = False):
        super(Conv2d_wSN, self).__init__()
        # Spectral Normalization from https://arxiv.org/abs/1802.05957
        # SN has no para. 
        self.sn_conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding, bias = bias)
            )
        #)
        self.bn_flag = bn_flag
        if bn_flag:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.sn_conv(x)
        if self.bn_flag:
            x = self.bn(x)
        return F.leaky_relu(x, 0.2,inplace=True)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        ndf = 16

        # (16, 1, 1) -> (16*8, 4,4) -> (16*4, 8, 8) -> (16*2, 14, 14) -> (1, 28, 28)
        # (1, 28, 28) -> (16*2, 14, 14) -> (16*4, 8, 8) -> (16*8, 4,4) -> (1)
        self.D_conv = nn.Sequential(
            Conv2d_wSN(1, ndf*2, 4, 2, 1),
            Conv2d_wSN(ndf * 2, ndf * 4, 2, 2, 1, bn_flag = True),  
            Conv2d_wSN(ndf * 4, ndf * 8, 4, 2, 1, bn_flag = True), 
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), 
            
        )
        self.D_head = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )
    def forward(self, px_Tsor):
        BS = px_Tsor.shape[0]
        px_Tsor = self.D_conv(px_Tsor)
        qjudge_Tsor = self.D_head(px_Tsor)
        
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