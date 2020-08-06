import time 
import os 
import sys 
sys.path.append(os.getcwd())
import arch.infogan_mnist as info_m
try:
    from terminaltables import AsciiTable
except ImportError: 
    print("please pip install terminaltables")

import torch 
import torch.nn as nn 
# from torchsummary import summary
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("for torch1.4, please pip install tensorboard==2.0.2")
'''
tensorboard --logdir outputs --port 8890
'''

class logger(object):
    def __init__(self, log_dir : str, tag : str):
        os.makedirs(log_dir, exist_ok= True)
        localtime = time.localtime(time.time()); date = f"{localtime.tm_mon}{localtime.tm_mday}"
        logfile_path = os.path.join(log_dir, tag + date + "logfile.txt")
        self.txt_writer = open(logfile_path, 'a+')
        self.board_writer = SummaryWriter(log_dir=os.path.join(log_dir, "board"))

    def log_the_str(self, *print_paras):
        for para_i in print_paras:
            print(para_i, end= "")
            print(para_i, end= "", file = self.txt_writer)
        print("")
        print("", file = self.txt_writer)
    
    def log_the_table(self, title, info_table: list):

        table = AsciiTable(info_table, title).table
        self.log_the_str(table)

    def summarize_config(self, pConfig):
        info_table = [['item', 'detail']]
        info_table.append(["training id", pConfig.path_save_mdid])
        info_table.append(["# epochs", pConfig.training_epoch_amount])
        info_table.append(["# checkpoint begin", pConfig.save_epoch_begin])
        info_table.append(["batch size", pConfig.ld_batchsize])
        info_table.append(["#workers ", pConfig.ld_ld_workers])
        info_table.append(["init mode", pConfig.method_init])
        info_table.append(["base Lr", pConfig.opt_baseLr])
        info_table.append(["device", pConfig.device])
        

    def summarize_netarch(self, pNet):
        # input_size : (C, H, W)
        # in terminal:
        # summary(pNet, input_size)
        # in file
        info_table = [['item', 'detail']]
        para_num = 0
        for para_i in pNet.parameters():
            # para_i is a torch.nn.parameter.Parameter, grad is True by default. 
            # print(para_i.shape, para_i.requires_grad)
            para_num += para_i.numel()
        info_table.append(["#parameters", para_num])

        conv_num = 0; fc_num = 0
        for m in pNet.modules():
            if isinstance(m, nn.modules.conv._ConvNd):
                # nn.convNd
                conv_num += 1
            elif isinstance(m, nn.Linear):
                fc_num += 1
        info_table.append(["#conv_layers", conv_num])
        info_table.append(["#fc_layers", fc_num])

        self.log_the_table(pNet._get_name() , info_table)
    
    def fetch_netweights(self, pNet):
        para_Lst = []
        for para_i in pNet.parameters():
            para_Lst.append(para_i.reshape(-1))
        return torch.cat(para_Lst)

    def board_net_weightdist(self, pNet, step_i):
        '''
        view the distribution of the net weights in tfboard. 
        '''
        weights = self.fetch_netweights(pNet)
        self.writer.add_histogram("G weights Dist", weights, step_i)

    def board_scalars_singlechart(self, chart_tag, data_dic, step_i):
        self.board_writer.add_scalars(chart_tag, data_dic, step_i)
    
    def board_scalars_multicharts(self, charts_tag, data_dic, step_i):
        # key : str; item : val
        for key, val in data_dic.item():
            self.board_writer.add_scalar(chart_tag + "_"+ key, val, step_i)
        
    def board_imgs_singlefig(self, fig_tag, img_Tsor, step_i):
        '''
        img_Tsor : (C, H, W)
        img_Tsor = torchvision.utils.make_grid(tensor = imgF_Tsor_bacth_i, nrow= w_layout)
        '''
        self.board_writer.add_image(fig_tag, img_Tsor, step_i)

    def board_geos_singlefig(self, fig_tag, vert_Tsor, color_Tsor, faces_Tsor, step_i):
        '''
        vert_Tsor, color_Tsor, faces_Tsor (BS, N, 3)
        '''
        self.board_writer.add_mesh(fig_tag, vert_Tsor, color_Tsor, faces_Tsor, global_step = step_i)


if __name__ == "__main__":
    
    gm_logger = logger(r"E:\ZimengZhao_Program\RebuidZoo\GANseries\custom_utils", "test")
    gm_net = info_m.Generator(10 , 10, 2)
    gm_logger.summarize_netarch(gm_net)

