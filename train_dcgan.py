
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

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

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("please pip install tensorboard==2.0.2")

class train_config(ut_cfg.config):
    def __init__(self):
        super(train_config, self).__init__(pBs = 64, pWn = 2, p_force_cpu = False)
        self.path_save_mdroot = self.check_path_valid(os.path.join(ROOT, "outputs"))
        localtime = time.localtime(time.time())
        self.path_save_mdid = "dcmnist" + "%02d%02d"%(localtime.tm_mon, localtime.tm_mday)

        self.save_epoch_begin = 50
        self.save_epoch_interval = 20

        self.log_epoch_txt = open(os.path.join(self.path_save_mdroot, "dcmnist_epoch_loss_log.txt"), 'a+')
        self.writer = SummaryWriter(log_dir=os.path.join(self.path_save_mdroot, "board"))

        self.height_in = 28
        self.width_in = 28
        self.latent_num = 16

        self.method_init ="xavier"  #"preTrain" #"kaming" #"xavier" # "norm"
        self.training_epoch_amount = 150
        

        self.dtroot = os.path.join(ROOT, "datasets")

        self.opt_baseLr_D = 5e-4
        self.opt_baseLr_G = 2e-4
        self.opt_bata1 = 0.5
        self.opt_weightdecay = 3e-6
        
        
        '''
        DCGAN: 
        suggested learning rate of 0.001, to be too high, using 0.0002 instead. 
        the momentum term β1 at the suggested value of 0.9 resulted in training oscillation and instability while reducing it to 0.5 helped stabilize training.
        '''

        # synchronize the rand seed
        self.rand_seed = 2673 # random.randint(1, 10000)
        print("Random Seed: ", self.rand_seed)
        random.seed(self.rand_seed)
        torch.manual_seed(self.rand_seed)
        self.fixed_noise = self.noise_generate_func(self.ld_batchsize, self.latent_num)

    def noise_generate_func(self, *size):
        # torch.randn N(0, 1) ; torch.rand U(0, 1)
        # U(-1 ,1)
        return torch.rand(*size).to(self.device)* 2 - 1 # to (-1, 1)

    def init_net(self, pNet):
        if self.method_init == "xavier":
            ut_init.init_xavier(pNet)
        elif self.method_init == "kaiming":
            ut_init.init_kaiming(pNet)
        elif self.method_init == "norm":
            ut_init.init_norm(pNet)
        elif self.method_init == "preTrain":
            assert self.preTrain_model_path is not None, "weight path ungiven"
            pNet.load_state_dict(torch.load(self.preTrain_model_path))

        pNet.to(self.device).train()

    def create_dataset(self, istrain):
        if istrain:
            imgUbyte_absfilename = r"datasets\MNIST\train-images-idx3-ubyte.gz"
            labelUbyte_absfilename = r"datasets\MNIST\train-labels-idx1-ubyte.gz"
        else:
            imgUbyte_absfilename = r"datasets\MNIST\t10k-images-idx3-ubyte.gz"
            labelUbyte_absfilename = r"datasets\MNIST\t10k-labels-idx1-ubyte.gz"
        
        basic_transform = transforms.Compose([
                # transforms.ToPILImage(), 
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), 
                transforms.ToTensor(), # (0, 255) uint8 HWC-> (0, 1.0) float32 CHW
                # transforms.RandomApply([ReverseColor()]), # AddGaussianNoise()
                # transforms.RandomApply([AddGaussianNoise()]),
                ])
        q_dataset = mnstld.minist_Loader(imgUbyte_absfilename, labelUbyte_absfilename, basic_transform)

        return q_dataset 

    def name_save_model(self, save_mode, epochX = None):
        model_type = save_mode.split("_")[1] # netD / netG
        model_filename = self.path_save_mdid + model_type
        
        if "processing" in save_mode:
            assert epochX is not None, "miss the epoch info" 
            model_filename += "_%03d"%(epochX) + ".pth"
        elif "ending" in save_mode:
            model_filename += "_%03d"%(self.training_epoch_amount) + ".pth"
        elif "interrupt" in save_mode:
            model_filename += "_interrupt"+ ".pth"
        assert os.path.splitext(model_filename)[-1] == ".pth"
        q_abs_path = os.path.join(self.path_save_mdroot, model_filename)
        return q_abs_path

    def log_in_file(self, *print_paras):
        for para_i in print_paras:
            print(para_i, end= "")
            print(para_i, end= "", file = self.log_epoch_txt)
        print("")
        print("", file = self.log_epoch_txt)
    
    def log_in_board(self, chartname,data_Dic, epoch):
        # for key_i, val_i in data_Dic:
        self.writer.add_scalars(chartname, 
            data_Dic, epoch)

    def validate(self, pnetD, pnetG, p_epoch):
        
        # use the fixed noise to test the GAN performance
        w_layout = 8
        imgF_Tsor_bacth_i = pnetG(self.fixed_noise)
        # imgF_Tsor_bacth_i = imgF_Tsor_bacth_i/2 + 0.5
        view_x_Tsor = torchvision.utils.make_grid(tensor = imgF_Tsor_bacth_i, nrow= w_layout)

        self.writer.add_image("Generator Outputs", view_x_Tsor, p_epoch)

        judge_Tsor_batch_i = pnetD(imgF_Tsor_bacth_i)

        judge_Arr = np.zeros(gm_cfg.ld_batchsize)
        for idex, ele_i in enumerate(judge_Tsor_batch_i):
            judge_Arr[idex] = round(ele_i.item(), 3)
        print("[validate] epoch %d  D's judgement:\n"%(p_epoch), np.reshape(judge_Arr, (-1, w_layout)))

        

if __name__ == "__main__":

    gm_cfg = train_config()
    
    # prepare data
    gm_trainloader = torch.utils.data.DataLoader(
        dataset = gm_cfg.create_dataset(istrain = True), 
        batch_size= gm_cfg.ld_batchsize,
        shuffle= True,
        num_workers= gm_cfg.ld_workers
    ) # 1875 * 32

    gm_real_label = 1
    gm_real_confidence = 0.9
    gm_fake_label = 0

    # prepare nets
    gm_netG = dc_m.Generator(gm_cfg.latent_num)
    gm_netD = dc_m.Discriminator()
    
    gm_cfg.init_net(gm_netG)
    gm_cfg.init_net(gm_netD)

    # optimizer & scheduler
    gm_optimizerG = optim.Adam(
        params = gm_netG.parameters(),
        lr = gm_cfg.opt_baseLr_G,
        betas= (gm_cfg.opt_bata1, 0.99),
        # weight_decay = gm_cfg.opt_weightdecay
    )

    gm_optimizerD = optim.Adam(
        params = gm_netD.parameters(),
        lr = gm_cfg.opt_baseLr_D,
        betas= (gm_cfg.opt_bata1, 0.99),
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
        print("Train_Begin".center(40, "*"))
        print("Generator:", end="")
        gm_cfg.check_arch_para(gm_netG)
        print("Discriminator:", end="")
        gm_cfg.check_arch_para(gm_netD)
        gm_cfg.log_in_file("net_id = ", gm_cfg.path_save_mdid, ", batchsize = ", gm_cfg.ld_batchsize, ", workers = ", gm_cfg.ld_workers)
        gm_cfg.log_in_file("criterion_use: ",gm_criterion, ", init: ", gm_cfg.method_init)
        for epoch_i in range(gm_cfg.training_epoch_amount):
            start=time.time()
            # single epoch
            for iter_idx, (img_Tsor_bacth_i, _ ) in enumerate(gm_trainloader):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                # k(update) = 2; 1 turn for real, another for fake
                # min D(imgR) ~ 1 && D(G(z)) ~ 0
                ###########################
                BS = img_Tsor_bacth_i.shape[0]
                gm_netD.zero_grad()

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
                
                lossD_an_epoch_Lst.append(lossD.item())

                gm_optimizerD.step() ### upgrade the D.para()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                # k(update) = 1
                # min  D(G(z)) ~ 1
                ###########################
                gm_netG.zero_grad()

                gt_Tsor_bacth_i.fill_(gm_real_label) # reuse the space

                # current fake generated by G: imgF_Tsor_bacth_i

                # use the updated D to judge the imgF
                judge_Tsor_bacth_i = gm_netD(imgF_Tsor_bacth_i)

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

            gm_cfg.log_in_board( "dcmnist loss", 
                {"d_loss": avgD_loss, 
                "g_loss": avgG_loss, 
                },  epoch_i
            )

            gm_cfg.log_in_file("epoch = %03d, time_cost(min)= %2.2f, dcGAN_d_loss = %2.5f, dcGAN_g_loss = %2.5f"
                %(epoch_i, delta_t, avgD_loss, avgG_loss)
            )
            
            # validate the accuracy
            gm_cfg.validate(gm_netD, gm_netG, epoch_i)

            lossD_an_epoch_Lst.clear()
            lossG_an_epoch_Lst.clear()

            if (epoch_i >gm_cfg.save_epoch_begin and epoch_i %gm_cfg.save_epoch_interval == 1):
                # save weight at regular interval
                torch.save(obj = gm_netD.state_dict(), 
                    f = gm_cfg.name_save_model("processing_netD", epoch_i))
                torch.save(obj = gm_netG.state_dict(), 
                    f = gm_cfg.name_save_model("processing_netG", epoch_i))
                
            gm_cfg.log_epoch_txt.flush()
        
        # end the train process(training_epoch_amount times to reuse the data)
        torch.save(obj = gm_netD.state_dict(),  f = gm_cfg.name_save_model("ending_netD"))
        torch.save(obj = gm_netG.state_dict(),  f = gm_cfg.name_save_model("ending_netG"))
        gm_cfg.log_epoch_txt.close()
        gm_cfg.writer.close()
    
    except KeyboardInterrupt:
        print("Save the Inter.pth".center(60, "*"))
        torch.save(obj = gm_netD.state_dict(), f = gm_cfg.name_save_model("interrupt_netD"))
        torch.save(obj = gm_netG.state_dict(), f = gm_cfg.name_save_model("interrupt_netG"))
























    
