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
import custom_utils.logger as ut_log 

class NormalNLLLoss(nn.Module):
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    let Var = \sigma^2, 
    - f(x)=\frac{1}{ \sqrt{2 \pi \cdot Var}} e^{-\frac{(x-\mu)^2}{2 \cdot Var}}
    - -ln(f) = 0.5 * log(2 \pi \cdot Var) + \frac{(x-\mu)^2}{2 \cdot Var}
    """
    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super(NormalNLLLoss, self).__init__()
        self.reduction = reduction
    def __str__(self):
        return "customed NormalNLLLoss"
    def forward(self, x, mu, var):
        # shape: (BS, cont_dim)
        logli = 0.5 * torch.log(2 * np.pi* var + 1e-6) + (x - mu)**2 / (2*var +  1e-6)
        if self.reduction == "mean":
            nll = logli.sum(1).mean()
        elif self.reduction == "sum":
            nll = logli.sum()
        return nll
    
class train_config(ut_cfg.config):
    def __init__(self):
        super(train_config, self).__init__(saving_id = "infogan_mnist_spv_BCEwMSE",
                pBs = 64, pWn = 2, p_force_cpu = False)
        
        self.total_epoch = 100
        self.save_epoch_begin = 30
        self.save_epoch_interval = 10
        self.validate_epoch_interval = 3

        self.netin_size = 28
        self.latent_dim = 10 # z dim
        self.class_num = 10 # class: one-hot discrete
        self.code_dim = 2  # continuous
        self.test_edge = 2.0 # [-test_edge, ]
        
        self.method_init ="norm" # "xavier", "kaiming", "preTrain", "norm"
        # self.preTrain_model_path = pretrain_path

        self.dataset_root_path = os.path.join(ROOT, "datasets") # load the data
        self.log_root_path = self.check_path_valid(os.path.join(ROOT, "outputs", "infogan")) # save logs and checkpoints. 
        self.checkpoints_root_path = self.check_path_valid(os.path.join(self.log_root_path, self.saving_id + "checkpoints"))

        self.opt_baseLr_D = 5e-4
        self.opt_baseLr_G = 2e-4
        self.opt_baseLr_INFO = 0.5 * self.opt_baseLr_G
        self.opt_beta1 = 0.5
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
        
        x = torch.linspace(-self.test_edge, self.test_edge, self.class_num)
        fixed_continuous_lsp = torch.meshgrid(x, x)[0].reshape(-1,1) # [(-2)x10, ..., (2)x10]
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
            x = torch.arange(self.class_num)
            figure_Tsor = torch.meshgrid(x, x)[-1].reshape(-1,1) # [(0~9)x10]
        elif mode == "random":
            figure_Tsor = torch.randint(self.class_num, (batchsize, 1)) # (Bs, 1)
        assert figure_Tsor is not None, "invalid mode"
        
        onehot_Tensor = torch.zeros(figure_Tsor.shape[0], self.class_num).scatter_(dim= 1, index = figure_Tsor, value = 1.0).to(self.device) # (BS, 10)

        return onehot_Tensor, figure_Tsor.view(-1).to(self.device)
    
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
        w_layout = self.class_num
        imgF_Tsor_bacth1 = pnetG(self.combined_noise1)
        imgF_Tsor_bacth2 = pnetG(self.combined_noise2)
        # imgF_batch_i = imgF_batch_i/2 + 0.5
        view_x_Tsor1 = torchvision.utils.make_grid(tensor = imgF_Tsor_bacth1, nrow= w_layout)
        view_x_Tsor2 = torchvision.utils.make_grid(tensor = imgF_Tsor_bacth2, nrow= w_layout)

        p_log.board_imgs_singlefig(self.saving_id + "_ctndim0", view_x_Tsor1, p_epoch)
        p_log.board_imgs_singlefig(self.saving_id + "_ctndim1", view_x_Tsor2, p_epoch)

        gm_log.board_net_weightdist(self.saving_id +"d_weights", pnetD , p_epoch)
        gm_log.board_net_weightdist(self.saving_id +"g_weights", pnetG , p_epoch)
    

if __name__ == "__main__":
    
    gm_real_label = 1
    gm_real_confidence = 0.9
    gm_fake_label = 0
    
    gm_lambda_con = 0.1 # Loss weights

    gm_cfg = train_config()
    gm_log = ut_log.logger(gm_cfg.log_root_path, gm_cfg.saving_id)
    # prepare data
    gm_trainloader = torch.utils.data.DataLoader(
        dataset = gm_cfg.create_dataset(istrain = True), 
        batch_size= gm_cfg.ld_batchsize,
        shuffle= True,
        num_workers= gm_cfg.ld_workers
    ) # 1875 * 32

    # prepare nets
    gm_netG = info_m.Generator(gm_cfg.latent_dim, gm_cfg.class_num, gm_cfg.code_dim)
    gm_netD = info_m.Discriminator_wQ(gm_cfg.class_num, gm_cfg.code_dim)
    
    gm_cfg.init_net(gm_netG, gm_cfg.method_init, istrain=True)
    gm_cfg.init_net(gm_netD, gm_cfg.method_init, istrain=True)
    
    # Optimizers
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
    gm_optimizerINFO = optim.Adam(
        params = [{'params':gm_netG.parameters()}, {'params':gm_netD.parameters()}],
        lr = gm_cfg.opt_baseLr_INFO,
        betas= (gm_cfg.opt_beta1, 0.99),
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
    gm_criterion = [adversarial_criterion, discrete_criterion, continuous_criterion]
    lossD_an_epoch_Lst = []
    lossG_an_epoch_Lst = []
    lossINFO_an_epoch_Lst = []

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
                judge_batch_i, _, _ = gm_netD(imgF_batch_i)

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
                predR_batch_i, _, _ = gm_netD(imgR_batch_i)
                lossD_R = adversarial_criterion(predR_batch_i, gt_batch_i)
                lossD_R.backward()
                # train D with fake
                gt_batch_i.fill_(gm_fake_label) # reuse the space
                predF_batch_i, _, _ = gm_netD(imgF_batch_i.detach())
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

                _, predDsctF_batch_i, predCtnFmu_batch_i = gm_netD(imgF_batch_i) # use the updated D; diff from line 260/ 282

                _, predDsctR_batch_i, _ = gm_netD(imgR_batch_i) ### add novelly...

                # encourage ctn[0] to represent the thickness; 
                roiF_batch_i = imgF_batch_i[..., 9:19] # encourage x \in (7, 21)
                aux_thickness_batch_i = roiF_batch_i.reshape(BS, -1).sum(-1) / (roiF_batch_i.nelement()/BS)
                aux_thickness_batch_i = aux_thickness_batch_i.detach() # (BS)
                
                
                # encourage ctn[1] to represent the rot;
                # aux_rot_batch_i = torch.zeros(BS).to(gm_cfg.device)
                # diag_line_amout = 0
                # fliplr_imgF_batch_i = imgF_batch_i[..., [i for i in range(-1, -(imgF_batch_i.shape[-1]+1), -1)]]
                # # fliplr_imgF_batch_i = torch.fliplr(imgF_batch_i) for torch1.6
                # for off_j in range(-1, 2, 1):
                #     aux_rot_batch_i += torch.diagonal(fliplr_imgF_batch_i, offset = off_j,  dim1 = -2, dim2 = -1).mean(dim= -1).squeeze(dim = -1)
                #     diag_line_amout += 1
                # aux_rot_batch_i = aux_rot_batch_i.detach() / diag_line_amout # get avg
                
                
                continuous_batch_i[:,0] = (continuous_batch_i[:,0] + 1) * (aux_thickness_batch_i + 0.52)  - 1
                # continuous_batch_i[:,1] = (continuous_batch_i[:,1] + 1) * (aux_rot_batch_i + 0.55)/ (aux_thickness_batch_i + 0.55) - 1

                lossINFO =  discrete_criterion(predDsctF_batch_i, labelF_batch_i) + \
                    discrete_criterion(predDsctR_batch_i, labelR_batch_i) + \
                    2 * gm_lambda_con * continuous_criterion(continuous_batch_i, predCtnFmu_batch_i)
                # # discrete_criterion(predDsctR_batch_i, labelR_batch_i) + \
                #
                
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

            gm_log.board_scalars_singlechart(gm_cfg.saving_id + "_loss", 
                {"d_loss": avgD_loss, 
                "g_loss": avgG_loss, 
                "info_loss": avgINFO_loss, 
                },  epoch_i
                )
            
            gm_log.log_scalars_singleline([
                    ["epoch", epoch_i], 
                    ["time_cost(min)", delta_t], 
                    ["d_loss", avgD_loss], 
                    ["g_loss", avgG_loss], 
                    ["info_loss",  avgINFO_loss]
                ])
            
            if epoch_i % gm_cfg.validate_epoch_interval == 0:
                gm_cfg.validate(gm_netD, gm_netG, gm_log, epoch_i)
            
            lossD_an_epoch_Lst.clear()
            lossG_an_epoch_Lst.clear()
            lossINFO_an_epoch_Lst.clear()

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


