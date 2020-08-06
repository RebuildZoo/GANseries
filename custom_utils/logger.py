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
    
    def summary_net(self, pNet):
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
        info_table.append(["#parameters", str(para_num)])

        conv_num = 0; fc_num = 0
        for m in pNet.modules():
            if isinstance(m, nn.modules.conv._ConvNd):
                # nn.convNd
                conv_num += 1
            elif isinstance(m, nn.Linear):
                fc_num += 1
        info_table.append(["#conv_layers", str(conv_num)])
        info_table.append(["#fc_layers", str(fc_num)])

        self.log_the_str(AsciiTable(info_table).table)



if __name__ == "__main__":
    
    gm_logger = logger(r"E:\ZimengZhao_Program\RebuidZoo\GANseries\custom_utils", "test")
    gm_net = info_m.Generator(10 , 10, 2)
    gm_logger.summary_net(gm_net)

