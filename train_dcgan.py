import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np
import time 
import random
import os 
import sys
ROOT = os.getcwd()
sys.path.append(ROOT)

import arch.dcgan_minist as dc_m
import loaders.ministLoader as mnstld 
import custom_utils.config as ut_cfg 
import custom_utils.initializer as ut_init
import custom_utils.logger as ut_log 

class train_config(ut_cfg.config):
    def __init__(self):
        super(train_config, self).__init__(saving_id = "dcgan_mnist_BCE",
            pBs = 64, pWn = 2, p_force_cpu = False)

        self.total_epoch = 100
        self.save_epoch_begin = 0
        self.save_epoch_interval = 20
        self.validate_epoch_interval = 3

        self.netin_size = 28
        self.latent_num = 16

        self.method_init ="norm" # "xavier", "kaiming", "preTrain", "norm"
        # self.preTrain_model_path = pretrain_path

        self.dataset_root_path = os.path.join(ROOT, "datasets") # load the data
        self.log_root_path = self.check_path_valid(os.path.join(ROOT, "outputs", "dcgan")) # save logs and checkpoints. 
        self.checkpoints_root_path = self.check_path_valid(os.path.join(self.log_root_path, self.saving_id + "_checkpoints"))
        
        self.opt_baseLr_D = 5e-4
        self.opt_baseLr_G = 2e-4
        self.opt_beta1 = 0.5
        self.opt_weightdecay = 3e-6

        self.rand_seed = 2673 # random.randint(1, 10000)
        print("Random Seed: ", self.rand_seed)
        random.seed(self.rand_seed)
        torch.manual_seed(self.rand_seed)
        self.fixed_noise = self.noise_generate_func(self.ld_batchsize, self.latent_num)
    
    def noise_generate_func(self, *size):
        # torch.randn N(0, 1) ; torch.rand U(0, 1)
        # U(-1 ,1)
        return torch.rand(*size).to(self.device)* 2 - 1 # to (-1, 1)
    
    def create_dataset(self, istrain):
        if istrain:
            imgUbyte_absfilename = os.path.join(self.dataset_root_path, r"MNIST\train-images-idx3-ubyte.gz")
            labelUbyte_absfilename = os.path.join(self.dataset_root_path, r"MNIST\train-labels-idx1-ubyte.gz")
        else:
            imgUbyte_absfilename = os.path.join(self.dataset_root_path, r"MNIST\t10k-images-idx3-ubyte.gz")
            labelUbyte_absfilename = os.path.join(self.dataset_root_path, r"MNIST\t10k-labels-idx1-ubyte.gz")
        
        basic_transform = transforms.Compose([
                # transforms.ToPILImage(), 
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), 
                transforms.ToTensor(), # (0, 255) uint8 HWC-> (0, 1.0) float32 CHW
                # transforms.RandomApply([ReverseColor()]), # AddGaussianNoise()
                # transforms.RandomApply([AddGaussianNoise()]),
                ])
        q_dataset = mnstld.minist_Loader(imgUbyte_absfilename, labelUbyte_absfilename, basic_transform)

        return q_dataset 
    
    def validate(self, pnetD, pnetG, p_log, p_epoch):
        pnetD.eval()
        pnetG.eval()
        # use the fixed noise to test the GAN performance
        w_layout = 8
        imgF_Tsor_bacth_i = pnetG(self.fixed_noise)
        # imgF_Tsor_bacth_i = imgF_Tsor_bacth_i/2 + 0.5
        view_x_Tsor = torchvision.utils.make_grid(tensor = imgF_Tsor_bacth_i, nrow= w_layout)

        p_log.board_imgs_singlefig("Generator Outputs", view_x_Tsor, p_epoch)

        gm_log.board_net_weightdist("D weights Dist", pnetD , p_epoch)
        gm_log.board_net_weightdist("G weights Dist", pnetG , p_epoch)
        
if __name__ == "__main__":
    
    gm_real_label = 1
    gm_real_confidence = 0.9
    gm_fake_label = 0

    gm_cfg = train_config()
    gm_log = ut_log.logger(gm_cfg.log_root_path, gm_cfg.saving_id)
    gm_trainloader = torch.utils.data.DataLoader(
        dataset = gm_cfg.create_dataset(istrain = True), 
        batch_size= gm_cfg.ld_batchsize,
        shuffle= True,
        num_workers= gm_cfg.ld_workers
    )

    gm_netG = dc_m.Generator(gm_cfg.latent_num)
    gm_netD = dc_m.Discriminator()

    gm_cfg.init_net(gm_netG, gm_cfg.method_init, istrain=True)
    gm_cfg.init_net(gm_netD, gm_cfg.method_init, istrain=True)

    # optimizer & scheduler
    gm_optimizerG = optim.Adam(
        params = gm_netG.parameters(),
        lr = gm_cfg.opt_baseLr_G,
        betas= (gm_cfg.opt_beta1, 0.99),
        # weight_decay = gm_cfg.opt_weightdecay
    )

    gm_optimizerD = optim.Adam(
        params = gm_netD.parameters(),
        lr = gm_cfg.opt_baseLr_D,
        betas= (gm_cfg.opt_beta1, 0.99),
        # weight_decay = gm_cfg.opt_weightdecay
    )

    gm_schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = gm_optimizerG,
        mode='min',
        factor=0.8, patience=5, verbose=True, 
        threshold=0.0001, threshold_mode='rel', 
        cooldown=0, min_lr=0, eps=1e-08
    )

    gm_schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = gm_optimizerD,
        mode='min',
        factor=0.8, patience=5, verbose=True, 
        threshold=0.0001, threshold_mode='rel', 
        cooldown=0, min_lr=0, eps=1e-08
    )

    gm_criterion = nn.BCELoss()

    lossD_an_epoch_Lst = []
    lossG_an_epoch_Lst = []

    try:
        gm_log.summarize_netarch(gm_netD)
        gm_log.summarize_netarch(gm_netG)
        gm_log.summarize_config(gm_cfg)
        print("Train_Begin".center(40, "*"))

        for epoch_i in range(gm_cfg.total_epoch):
            start=time.time()
            # single epoch
            gm_netD.train()
            gm_netG.train()

            for iter_idx, (img_Tsor_bacth_i, _ ) in enumerate(gm_trainloader):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                # k(update) = 2; 1 turn for real, another for fake
                # min D(imgR) ~ 1 && D(G(z)) ~ 0
                ###########################
                BS = img_Tsor_bacth_i.shape[0]
                gm_optimizerD.zero_grad()

                # train D with real
                imgR_Tsor_bacth_i = img_Tsor_bacth_i.to(gm_cfg.device)
                gt_Tsor_bacth_i = torch.full((BS, 1), gm_real_label*gm_real_confidence, device = gm_cfg.device)

                predR_Tsor_bacth_i = gm_netD(imgR_Tsor_bacth_i)

                lossD_R = gm_criterion(predR_Tsor_bacth_i, gt_Tsor_bacth_i)

                lossD_R.backward()

                # train D with fake
                noise_Tsor_bacth_i = gm_cfg.noise_generate_func(BS, gm_cfg.latent_num)
                imgF_Tsor_bacth_i = gm_netG(noise_Tsor_bacth_i)
                gt_Tsor_bacth_i.fill_(gm_fake_label) # reuse the space

                predF_Tsor_bacth_i = gm_netD(imgF_Tsor_bacth_i.detach())

                lossD_F = gm_criterion(predF_Tsor_bacth_i, gt_Tsor_bacth_i)

                lossD_F.backward()

                lossD = lossD_R + lossD_F
                # for the gt_Tsor_bacth_i has been modified in-place,
                # cannot use lossD to backward; 
                
                lossD_an_epoch_Lst.append(lossD.item())

                gm_optimizerD.step() ### upgrade the D.para()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                # k(update) = 1
                # min  D(G(z)) ~ 1
                ###########################
                gm_optimizerG.zero_grad()
                gt_Tsor_bacth_i.fill_(gm_real_label) # reuse the space

                # current fake generated by G: imgF_Tsor_bacth_i

                # use the updated D to judge the imgF
                judge_Tsor_bacth_i = gm_netD(imgF_Tsor_bacth_i) # D has updated, diff from line252.

                # the request for G:
                # produce D'judge as close as '1'.

                lossG = gm_criterion(judge_Tsor_bacth_i, gt_Tsor_bacth_i)

                lossG.backward()                
                lossG_an_epoch_Lst.append(lossG.item())

                gm_optimizerG.step()
            # end an epoch
            delta_t = (time.time()- start)/60
            avgD_loss = sum(lossD_an_epoch_Lst)/len(lossD_an_epoch_Lst)
            avgG_loss = sum(lossG_an_epoch_Lst)/len(lossG_an_epoch_Lst)

            gm_schedulerD.step(avgD_loss)
            gm_schedulerG.step(avgG_loss)

            gm_log.board_scalars_singlechart("dcmnist_loss", 
                {"d_loss": avgD_loss, 
                "g_loss": avgG_loss, 
                },epoch_i
                )
            
            gm_log.log_scalars_singleline([
                    ["epoch", epoch_i], 
                    ["time_cost(min)", delta_t], 
                    ["d_loss", avgD_loss], 
                    ["g_loss", avgG_loss], 
                ])
            
            if epoch_i % gm_cfg.validate_epoch_interval == 0:
                gm_cfg.validate(gm_netD, gm_netG, gm_log, epoch_i)

            lossD_an_epoch_Lst.clear()
            lossG_an_epoch_Lst.clear()

            if (epoch_i >gm_cfg.save_epoch_begin and epoch_i %gm_cfg.save_epoch_interval == 1):
                # save weight at checkpoint = save_epoch_begin + t * save_epoch_interval
                torch.save(obj = gm_netD.state_dict(), 
                    f = gm_cfg.name_save_model("processing", gm_netD, epoch_i))
                torch.save(obj = gm_netG.state_dict(), 
                    f = gm_cfg.name_save_model("processing", gm_netG, epoch_i))
            
        # end the train process
        torch.save(obj = gm_netD.state_dict(),  f = gm_cfg.name_save_model("ending", gm_netD, gm_cfg.total_epoch))
        torch.save(obj = gm_netG.state_dict(),  f = gm_cfg.name_save_model("ending", gm_netG, gm_cfg.total_epoch))



    except KeyboardInterrupt :
        print("get Keyboard error, Save the Inter.pth")
        torch.save(obj = gm_netD.state_dict(), f = gm_cfg.name_save_model("interrupt", gm_netD))
        torch.save(obj = gm_netG.state_dict(), f = gm_cfg.name_save_model("interrupt", gm_netG))



    
