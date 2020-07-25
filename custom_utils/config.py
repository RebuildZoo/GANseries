import os
import time
import torch


class config(object):
    def __init__(self, pBs, pWn, p_force_cpu):
        # device & its name
        self.device = self.select_device(p_force_cpu)
        
        # loader
        self.ld_batchsize = pBs #8
        self.ld_workers = pWn #2

        self.method_criterion = None

        # None for inherint to rewrite


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
    
    def check_arch_para(self, pNet):
        para_amount = 0
        for para_i in pNet.parameters():
            # para_i is a torch.nn.parameter.Parameter, grad 默认 True
            # print(para_i.shape, para_i.requires_grad)
            para_amount+= para_i.numel()
        print("[net info] para_amount=%d"%para_amount)

    def check_path_valid(self, path)->str:
        assert os.path.isdir(path) or os.path.isfile(path), "invalid path in " + path
        return(path)
    


if __name__ == '__main__':
    train_para = train_hyper_para_Keeper()
    print(train_para.device)
