import os
import time
import torch
try : 
    import custom_utils.initializer as ut_init
except: 
    import initializer as ut_init

class config(object):
    def __init__(self, saving_id: str, pBs : int, pWn : int, p_force_cpu : bool):
        # device & its name
        self.saving_id = saving_id
        self.device = self.select_device(p_force_cpu)
        
        # path 
        self.checkpoints_root_path = None

        # loader
        self.ld_batchsize = pBs #8
        self.ld_workers = pWn #2
        

    def select_device(self, force_cpu = False):
        '''
        auto-select the device, equals to: 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if cuda_flag: try enable apex & cudnn.benchmark 
        '''
        cuda_flag = False if force_cpu else torch.cuda.is_available()
        device = torch.device('cuda:0' if cuda_flag else 'cpu')

        if not cuda_flag:
            print("[divice info] Using CPU")
        else:
            try:
                from apex import amp
                apex_str = 'with Apex '
            except:
                apex_str = ''
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True 
            c = 1024 ** 2  # bytes to MB
            ng = torch.cuda.device_count()
            x = [torch.cuda.get_device_properties(i) for i in range(ng)]
            cuda_str = 'Using CUDA ' + apex_str
            for i in range(0, ng):
                if i == 1:
                    # torch.cuda.set_device(0)  # OPTIONAL: Set GPU ID
                    cuda_str = ' ' * len(cuda_str)
                print("[divice info]%sdevice%g _Properties(name='%s', total_memory=%dMB)" %
                      (cuda_str, i, x[i].name, x[i].total_memory / c))

        print('')  # skip a line
        return device

    def check_path_valid(self, path) -> str:
        os.makedirs(path, exist_ok = True)
        return(path)
    
    def name_save_model(self, save_mode, pNet, epochX = None) -> str:
        model_filename = self.saving_id + "-" + pNet._get_name()
        
        if "processing" in save_mode or "ending" in save_mode:
            assert epochX is not None, "miss the epoch info" 
            model_filename += "_%03d"%(epochX) + ".pth"
        elif "interrupt" in save_mode:
            model_filename += "_interrupt"+ ".pth"
        assert os.path.splitext(model_filename)[-1] == ".pth"
        q_abs_path = os.path.join(self.checkpoints_root_path, model_filename)
        return q_abs_path

    def init_net(self, pNet, init_mode, istrain, pretrain_path = None) -> None:
        if init_mode == "xavier":
            ut_init.init_xavier(pNet)
        elif init_mode == "kaiming":
            ut_init.init_kaiming(pNet)
        elif init_mode == "norm":
            ut_init.init_norm(pNet)
        elif init_mode == "preTrain":
            assert pretrain_path is not None, "weight path ungiven"
            if ".weight" in self.path_weight_file:
                pNet.load_darknet_weights(self.path_weight_file)
            elif ".pth" in self.path_weight_file:
                pNet.load_state_dict(torch.load(self.path_weight_file))

        pNet.to(self.device)
        if istrain : pNet.train()
        else : pNet.eval()

if __name__ == '__main__':
    train_para = config()
    print(train_para.device)
