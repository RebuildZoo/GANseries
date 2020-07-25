import torch
import torch.nn as nn
import torch.nn.functional as F


'''
reference 
paper <Unsupervised representation learning with deep convolutional generative adversarial networks.> 2015; ->(ICLR 2016)
code & para https://github.com/pytorch/examples/blob/234bcff4a2d8480f156799e6b9baae06f7ddc96a/dcgan/main.py#L113

'''

class Generator(nn.Module):
    def __init__(self, latent_num, ngpu=1):
        # latent_num = 16
        super(Generator, self).__init__()
        self.ngpu = ngpu # try to use parrallel

        # (16, 1, 1) -> (16, 3, 3) -> (8, 7, 7) -> (3, 14, 14) -> (1, 28, 28) 
        # as same as convVAE

        # fractional-strided convolutions ; 
        # batchnorm
        # no FC
        # ReLU activation;  output uses Tanh
        ngf = 64
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_num, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, pz_Tsor):
        BS = pz_Tsor.shape[0]
        pz_Tsor = pz_Tsor.view(BS, -1 , 1, 1)

        # switch if parallel
        if pz_Tsor.is_cuda and self.ngpu > 1:
            qreon_Tsor = nn.parallel.data_parallel(self.dec_conv, pz_Tsor, range(self.ngpu))
            # qreon_Tsor = torch.sigmoid(qreon_Tsor)
        else:
            qreon_Tsor = self.dec_conv(pz_Tsor)
        return qreon_Tsor



def test_generator():
    gm_gnr = Generator(latent_num = 16)

    z_Tsor = torch.rand(5, 16)

    reon_Tsor = gm_gnr(z_Tsor)

    print("Generator output:", reon_Tsor.shape) # [5, 1, 28, 28]



class Discriminator(nn.Module):
    def __init__(self, ngpu = 1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        # strided convolutions ;
        # batchnorm; 
        ndf = 64
        # LeakyReLU for all layers
        # (1, 28, 28) -> (6, 14, 14) -> (32, 7, 7) -> (64, 4, 4) -> (64, 2, 2) -> 
        self.body = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, px_Tsor):
        # input : fake or real img
        BS = px_Tsor.shape[0]
        if px_Tsor.is_cuda and self.ngpu > 1:
            qjudge_Tsor = nn.parallel.data_parallel(self.body, px_Tsor, range(self.ngpu))
        else:
            qjudge_Tsor = self.body(px_Tsor)
        
        qjudge_Tsor = qjudge_Tsor.view(BS, -1)
        return qjudge_Tsor


def test_discriminator():
    x_Tsor = torch.rand(5, 1, 64, 64)

    gm_dcm = Discriminator(1)

    j_Tsor = gm_dcm(x_Tsor)

    print("Discriminator output:", j_Tsor.shape) # [5, 1]
    # print(gm_dcm)



if __name__ == "__main__":

    test_discriminator()
    #test_generator()
    

