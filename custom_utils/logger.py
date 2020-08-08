import time 
import os 
import sys 
sys.path.append(os.getcwd())
import arch.infogan_mnist as info_m
try:
    from terminaltables import AsciiTable
except ImportError: 
    print("pip install terminaltables...")
    os.system("pip install terminaltables")
    from terminaltables import AsciiTable

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
        localtime = time.localtime(time.time()); date = "%02d%02d"%(localtime.tm_mon, localtime.tm_mday)
        logfile_path = os.path.join(log_dir, tag + "_" + date + "_" +"log.txt")
        self.txt_writer = open(logfile_path, 'a+')
        self.board_writer = SummaryWriter(log_dir=os.path.join(log_dir, "board"))

    def __del__(self):
        self.txt_writer.close()
        self.board_writer.close()

    def log_the_str(self, *print_paras):
        for para_i in print_paras:
            print(para_i, end= "")
            print(para_i, end= "", file = self.txt_writer)
        print("")
        print("", file = self.txt_writer)
        self.txt_writer.flush()

    def log_the_table(self, title, info_table: list):
        table = AsciiTable(info_table, title).table
        self.log_the_str(table)


    def log_scalars_singleline(self, info_table: list):
        line_str = ""
        for idx_i, info_pair in enumerate(info_table):
            info_val = info_pair[1]
            if isinstance(info_val, float): 
                info_val = str(round(info_val, 5))
            elif isinstance(info_val, float):
                info_val = "%03d"%(info_val)
            info_str = "= ".join([str(info_pair[0]), str(info_val)])
            if idx_i != len(info_table): 
                info_str += ", "
            line_str += info_str
        self.log_the_str(line_str)
        

    def summarize_config(self, pConfig):
        info_table = [['item', 'detail']]
        info_table.append(["config id", pConfig.saving_id])
        info_table.append(["# epochs", pConfig.training_epoch_amount])
        info_table.append(["# checkpoint begin", pConfig.save_epoch_begin])
        info_table.append(["batch size", pConfig.ld_batchsize])
        info_table.append(["#workers ", pConfig.ld_workers])
        info_table.append(["init mode", pConfig.method_init])
        info_table.append(["device", pConfig.device])
        
        self.log_the_table("config" , info_table)

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

        self.log_the_table("Net-" + pNet._get_name() , info_table)
    
    def fetch_netweights(self, pNet):
        para_Lst = []
        for para_i in pNet.parameters():
            para_Lst.append(para_i.reshape(-1))
        return torch.cat(para_Lst)

    def board_net_weightdist(self, chart_tag, pNet, step_i):
        '''
        view the distribution of the net weights in tfboard. 
        '''
        weights = self.fetch_netweights(pNet)
        self.board_writer.add_histogram(chart_tag, weights, step_i)

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

    for i in range(10):
        gm_logger.log_scalars_singleline([
            ["epoch", i], 
            ["time_cost(min)", 0.2], 
            ["avg_loss", torch.rand(1).item()], 
        ])
