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

import arch.infogan_mnist as info_m
import loaders.ministLoader as mnstld 
import custom_utils.config as ut_cfg 
import custom_utils.initializer as ut_init

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("please pip install tensorboard==2.0.2")
'''
tensorboard --logdir outputs --port 8890
'''

class NormalNLLLoss(nn.Module):
    """
    https://github.com/Natsu6767/InfoGAN-PyTorch/blob/master/utils.py
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """
    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super(NormalNLLLoss, self).__init__()
        self.reduction = reduction
    def __str__(self):
        return "customed NormalNLLLoss"
    def forward(self, x, mu, var):
        logli = -0.5 * torch.log(2 * np.pi* var + 1e-6) - (x - mu)**2 / (2*var +  1e-6)
        if self.reduction == "mean":
            nll = - logli.sum(1).mean()
        return nll
    


class train_config(ut_cfg.config):
    def __init__(self):
        super(train_config, self).__init__(pBs = 64, pWn = 2, p_force_cpu = False)
        self.path_save_mdroot = self.check_path_valid(os.path.join(ROOT, "outputs", "infogan"))
        localtime = time.localtime(time.time())
        self.path_save_mdid = "infomnist_z10_MSE" + "%02d%02d"%(localtime.tm_mon, localtime.tm_mday)

        self.save_epoch_begin = 20
        self.save_epoch_interval = 10

        self.log_epoch_txt = open(os.path.join(self.path_save_mdroot, "infomnist_z10_MSE_epoch_loss_log.txt"), 'a+')
        self.writer = SummaryWriter(log_dir=os.path.join(self.path_save_mdroot, "board"))

        self.height_in = 28
        self.width_in = 28
    
        self.latent_dim = 10 # z dim
        self.class_num = 10 # class: one-hot discrete
        self.code_dim = 2  # continuous
        self.test_edge = 2.0 # [-test_edge, test_edge]
        self.method_init ="norm"  #"preTrain" #"kaming" #"xavier" # "norm"
        self.training_epoch_amount = 100
        
        self.dtroot = os.path.join(ROOT, "datasets")

        self.opt_baseLr_D = 5e-4
        self.opt_baseLr_G = 1e-3
        self.opt_bata1 = 0.5
        self.opt_weightdecay = 3e-6

        # synchronize the rand seed
        self.rand_seed = 2673 # random.randint(1, 10000)
        print("Random Seed: ", self.rand_seed)
        random.seed(self.rand_seed)
        torch.manual_seed(self.rand_seed)
        
        # design the vali instance 
        fixed_z = self.noise_generate_func(self.class_num**2, self.latent_dim) # (100, 62)
        fixed_discrete, _ = self.label_generate_func("fixed", self.class_num**2)
        fixed_continuous_zero = torch.zeros(self.class_num**2, 1)
        fixed_continuous_lsp = (torch.stack([torch.linspace(-self.test_edge, self.test_edge, self.class_num) for _ in range(self.class_num)]).transpose(0,1)).reshape(-1,1)
        fixed_continuous1 = torch.cat([fixed_continuous_lsp,fixed_continuous_zero],dim=-1).to(self.device)
        fixed_continuous2 = torch.cat([fixed_continuous_zero, fixed_continuous_lsp],dim=-1).to(self.device)
        
        self.combined_noise1 = torch.cat([fixed_z, fixed_discrete, fixed_continuous1],dim = -1)
        self.combined_noise2 = torch.cat([fixed_z, fixed_discrete, fixed_continuous2],dim = -1)


    def noise_generate_func(self, *size):
        # torch.randn N(0, 1) ; torch.rand U(0, 1)
        return torch.rand(*size).to(self.device)* 2 - 1 # to U(-1, 1)
    
    def label_generate_func(self, mode:str, batchsize:int):
        figure_Tsor = None
        if mode == "fixed":
            figure_Tsor = torch.cat([torch.arange(self.class_num) for _ in range(self.class_num)]).view(-1,1) # (10*10, 1)
        elif mode == "random":
            figure_Tsor = torch.randint(self.class_num, (batchsize, 1)) # (Bs, 1)
        assert figure_Tsor is not None, "invalid mode"
        
        onehot_Tensor = torch.zeros(figure_Tsor.shape[0], self.class_num).scatter_(dim= 1, index = figure_Tsor, value = 1.0).to(self.device) # (BS, 10)

        return onehot_Tensor, figure_Tsor.view(-1).to(self.device)

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
        w_layout = self.class_num
        imgF_Tsor_bacth1 = pnetG(self.combined_noise1)
        imgF_Tsor_bacth2 = pnetG(self.combined_noise2)
        # imgF_batch_i = imgF_batch_i/2 + 0.5
        view_x_Tsor1 = torchvision.utils.make_grid(tensor = imgF_Tsor_bacth1, nrow= w_layout)
        view_x_Tsor2 = torchvision.utils.make_grid(tensor = imgF_Tsor_bacth2, nrow= w_layout)

        self.writer.add_image("infomnist_z10_MSE_ctndim0", view_x_Tsor1, p_epoch)
        self.writer.add_image("infomnist_z10_MSE_ctndim1", view_x_Tsor2, p_epoch)

        # judge_Tsor_batch_i = pnetD(imgF_batch_i)

        # judge_Arr = np.zeros(gm_cfg.ld_batchsize)
        # for idex, ele_i in enumerate(judge_Tsor_batch_i):
        #     judge_Arr[idex] = round(ele_i.item(), 3)
        # print("[validate] epoch %d  D's judgement:\n"%(p_epoch), np.reshape(judge_Arr, (-1, w_layout)))

    
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
    
    gm_lambda_con = 0.1 # # Loss weights

    # prepare nets
    gm_netG = info_m.Generator(gm_cfg.latent_dim, gm_cfg.class_num, gm_cfg.code_dim)
    gm_netD = info_m.Discriminator_wQ(gm_cfg.class_num, gm_cfg.code_dim)
    
    gm_cfg.init_net(gm_netG)
    gm_cfg.init_net(gm_netD)

    # Optimizers
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
    gm_optimizerINFO = optim.Adam(
        params = [{'params':gm_netG.parameters()}, {'params':gm_netD.parameters()}],
        lr = gm_cfg.opt_baseLr_G * 0.5,
        betas= (gm_cfg.opt_bata1, 0.99),
        # weight_decay = gm_cfg.opt_weightdecay
    )

    gm_schedulerG = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = gm_optimizerG,
        mode='min',
        factor=0.8, patience=5, verbose=True, 
        threshold=0.0001, threshold_mode='rel', 
        cooldown=0, min_lr=0, eps=1e-08
    )

    gm_schedulerD = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = gm_optimizerD,
        mode='min',
        factor=0.8, patience=5, verbose=True, 
        threshold=0.0001, threshold_mode='rel', 
        cooldown=0, min_lr=0, eps=1e-08
    )

    gm_schedulerINFO = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = gm_optimizerINFO,
        mode='min',
        factor=0.8, patience=5, verbose=True, 
        threshold=0.0001, threshold_mode='rel', 
        cooldown=0, min_lr=0, eps=1e-08
    )

    # Loss functions
    adversarial_criterion = nn.BCELoss()
    discrete_criterion = nn.CrossEntropyLoss()
    continuous_criterion = nn.MSELoss() # NormalNLLLoss() #
    nn.NLLLoss()
    gm_criterion = [adversarial_criterion, discrete_criterion, continuous_criterion]
    lossD_an_epoch_Lst = []
    lossG_an_epoch_Lst = []
    lossINFO_an_epoch_Lst = []

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
            for iter_idx, (img_batch_i, label_batch_i) in enumerate(gm_trainloader):
                BS = img_batch_i.shape[0]
                imgR_batch_i = img_batch_i.to(gm_cfg.device)
                labelR_batch_i = label_batch_i.to(gm_cfg.device)
                ############################
                # (1) Update G network: maximize log(D(G(z)))
                # k(update) = 1
                # min  D(G(z)) ~ 1
                ###########################
                gm_optimizerG.zero_grad()

                noise_batch_i = gm_cfg.noise_generate_func(BS, gm_cfg.latent_dim) # (Bs, 62)
                discrete_batch_i, labelF_batch_i = gm_cfg.label_generate_func("random", BS) # (Bs,10)
                continuous_batch_i = gm_cfg.noise_generate_func(BS, gm_cfg.code_dim) #(Bs, 2)
                combined_batch_i = torch.cat([noise_batch_i, discrete_batch_i, continuous_batch_i],dim = -1)
                
                imgF_batch_i = gm_netG(combined_batch_i)

                gt_batch_i = torch.full((BS, 1), gm_real_label, device = gm_cfg.device)
                judge_batch_i, _, _, _ = gm_netD(imgF_batch_i)

                lossG = adversarial_criterion(judge_batch_i, gt_batch_i)

                lossG.backward() 
                lossG_an_epoch_Lst.append(lossG.item())

                gm_optimizerG.step()

                ############################
                # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                # k(update) = 2; 1 turn for real, another for fake
                # min D(imgR) ~ 1 && D(G(z)) ~ 0
                ###########################
                gm_optimizerD.zero_grad()
                # train D with real
                gt_batch_i.fill_(gm_real_label*gm_real_confidence) # reuse the space
                predR_batch_i, _, _, _ = gm_netD(imgR_batch_i)
                lossD_R = adversarial_criterion(predR_batch_i, gt_batch_i)
                lossD_R.backward()
                # train D with fake
                gt_batch_i.fill_(gm_fake_label) # reuse the space
                predF_batch_i, _, _, _ = gm_netD(imgF_batch_i.detach())
                lossD_F = adversarial_criterion(predF_batch_i, gt_batch_i)
                lossD_F.backward()

                lossD = lossD_R + lossD_F
                lossD_an_epoch_Lst.append(lossD.item())

                gm_optimizerD.step()

                ############################
                # (3) Information Loss
                ###########################
                gm_optimizerINFO.zero_grad()

                # labelF_batch_i
                imgF_batch_i = gm_netG(combined_batch_i) # use the updated G; diff from line 257

                _, predDsctF_batch_i, predCtnFmu_batch_i, predCtnFexpvar_batch_i = gm_netD(imgF_batch_i) # use the updated D; diff from line 260/ 282

                _, predDsctR_batch_i, _, _ = gm_netD(imgR_batch_i) ### add novelly...

                lossINFO =  discrete_criterion(predDsctF_batch_i, labelF_batch_i) + \
                    discrete_criterion(predDsctR_batch_i, labelR_batch_i) + \
                    2 * gm_lambda_con * continuous_criterion(continuous_batch_i, predCtnFmu_batch_i)
                

                lossINFO.backward()

                lossINFO_an_epoch_Lst.append(lossINFO.item())

                gm_optimizerINFO.step()
            
            # end an epoch
            delta_t = (time.time()- start)/60
            avgD_loss = sum(lossD_an_epoch_Lst)/len(lossD_an_epoch_Lst)
            avgG_loss = sum(lossG_an_epoch_Lst)/len(lossG_an_epoch_Lst)
            avgINFO_loss = sum(lossINFO_an_epoch_Lst)/len(lossINFO_an_epoch_Lst)

            gm_schedulerD.step(avgD_loss)
            gm_schedulerG.step(avgG_loss)
            gm_schedulerINFO.step(avgINFO_loss)

            gm_cfg.log_in_board( "infomnist_z10_MSE loss", 
                {"d_loss": avgD_loss, 
                "g_loss": avgG_loss, 
                "info_loss": avgINFO_loss, 
                },  epoch_i
            )

            gm_cfg.log_in_file("epoch = %03d, time_cost(min)= %2.2f, d_loss = %2.5f, g_loss = %2.5f,info_loss = %2.5f"
                %(epoch_i, delta_t, avgD_loss, avgG_loss, avgINFO_loss)
            )

            # validate the accuracy
            gm_cfg.validate(gm_netD, gm_netG, epoch_i)

            lossD_an_epoch_Lst.clear()
            lossG_an_epoch_Lst.clear()
            lossINFO_an_epoch_Lst.clear()

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






