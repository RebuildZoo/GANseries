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
        self.device_index = 0 # default
        self.device, self.device_info = self.select_device(p_force_cpu)
        
        # path 
        self.checkpoints_root_path = None

        # loader
        self.ld_batchsize = pBs #8
        self.ld_workers = pWn #2
        

    def select_device(self, force_cpu = False):
        cuda_flag = False if force_cpu else torch.cuda.is_available()
        device = torch.device('cuda:%d'%(self.device_index) if cuda_flag else 'cpu')

        device_str = None
        if not cuda_flag:
            import platform 
            device_str = "cpu" + "(name: %s)"%(platform.processor())
            

        else: 
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True 

            c = 1024 ** 3  # bytes to MB

            ng = torch.cuda.device_count()
            device_prop = torch.cuda.get_device_properties(self.device_index)
            device_str = "cuda" + f"(index: {self.device_index}/{ng}, name: {device_prop.name}, mem: {float(device_prop.total_memory/c)} G)" 

        return device, device_str

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
            if ".weight" in pretrain_path:
                pNet.load_darknet_weights(pretrain_path)
            elif ".pth" in pretrain_path:
                try : 
                    pNet.load_state_dict(torch.load(pretrain_path))
                except RuntimeError: 
                    pre_state_Dic= torch.load(pretrain_path)
                    temp_state_Dic = OrderedDict()
                    for key_self, key_out in zip(pNet.state_dict(), pre_state_Dic.keys()):
                        temp_state_Dic[key_self] = pre_state_Dic[key_out]
                    pNet.load_state_dict(temp_state_Dic)


        pNet.to(self.device)
        if istrain : pNet.train()
        else : pNet.eval()
    
    def to_cpu(self, tensor):
        if tensor.requires_grad: 
            tensor = tensor.detach()
        if tensor.is_cuda : 
            tensor = tensor.cpu()
        return tensor

    def to_device(self, tensor):
        return tensor.to(self.device)

if __name__ == '__main__':
    train_para = config()
    print(train_para.device)
