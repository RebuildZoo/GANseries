import os 
import sys 
import gzip 
import numpy as np 
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt 
def view_tensor(p_img_Tsor):
    # p_img_Tsor = p_img_Tsor / 2 + 0.5     # unnormalize
    img_Arr = p_img_Tsor.numpy()
    plt.imshow(np.transpose(img_Arr, (1, 2, 0)))
    plt.show()

class ReverseColor(object):
    def __call__(self, img_Tsor):
        assert isinstance(img_Tsor, torch.Tensor), "not a valid data-type"
        assert torch.max(img_Tsor) < 1.1 , "not a valid image tensor"
        return 1 - img_Tsor

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.15):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        tensor += torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor, min=0, max=1)

class minist_Loader(torch.utils.data.Dataset):
    def __init__(self, pImgUbyte_absfilename, pLabelUbyte_absfilename, pTansfom = None):
        '''
        load the oringinal MNIST dataset from ubyte file

        pImgUbyte_dir = datasets\MNIST\train-images-idx3-ubyte
        pLabelUbyte_dir = datasets\MNIST\train-labels-idx1-ubyte
        '''
        assert os.path.isfile(pImgUbyte_absfilename), "invalid image ubyte file: " + pImgUbyte_absfilename
        assert os.path.isfile(pLabelUbyte_absfilename), "invalid label ubyte file: " + pLabelUbyte_absfilename

        if not pTansfom:
            self.transform = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), 
                transforms.ToTensor(), # (0, 255) uint8 HWC-> (0, 1.0) float32 CHW
                transforms.RandomApply([ReverseColor()]), # AddGaussianNoise()
                transforms.RandomApply([AddGaussianNoise()]),
                ])
             
        else:
            self.transform = pTansfom
        
        with gzip.open(pImgUbyte_absfilename,'rb') as fin:
            magic_number = int.from_bytes(fin.read(4), 'big')
            self.img_count = int.from_bytes(fin.read(4), 'big')
            row_count = int.from_bytes(fin.read(4), 'big')
            column_count = int.from_bytes(fin.read(4), 'big')
            img_data = fin.read()
            self.imgLst_Arr = np.frombuffer(img_data, dtype=np.uint8) \
                    .reshape((self.img_count, row_count, column_count))
        
        with gzip.open(pLabelUbyte_absfilename,'rb') as fin:
            magic_number = int.from_bytes(fin.read(4), 'big')
            self.label_count = int.from_bytes(fin.read(4), 'big')
            label_data = fin.read()
            self.labelLst_Arr = np.frombuffer(label_data, dtype=np.uint8)
            self.class_num = np.max(self.labelLst_Arr) + 1 # 10
        assert self.img_count == self.label_count, "image and label nums mismatch."

    def __len__(self):
        return self.img_count

    def __getitem__(self, index):

        img_Arr = self.imgLst_Arr[index]
        label_idx = int(self.labelLst_Arr[index])#.reshape(1).astype(np.int64) # index: 0,1, ..., 9; 


        img_Tsor = self.transform(img_Arr)
        # label_Tsor = torch.zeros(self.class_num).scatter_(0, torch.tensor(label_idx), 1.0)
        # torch.tensor(label_Arr) #torch.from_numpy(label_Arr)

        return img_Tsor, label_idx


if __name__ == "__main__":

    imgUbyte_absfilename = r"datasets\MNIST\train-images-idx3-ubyte.gz"
    labelUbyte_absfilename = r"datasets\MNIST\train-labels-idx1-ubyte.gz"


    gm_dataset = minist_Loader(imgUbyte_absfilename, labelUbyte_absfilename)

    trainloader = torch.utils.data.DataLoader(dataset = gm_dataset, batch_size = 64, 
                        shuffle= False, num_workers = 1)
    
    for i_idx, pac_i in enumerate(trainloader):
        img_Tsor_bacth_i, label_Tsor_bacth_i = pac_i
        print(img_Tsor_bacth_i.shape, label_Tsor_bacth_i.shape)
        print(torch.max(img_Tsor_bacth_i[0]), torch.min(img_Tsor_bacth_i[0]))
        view_tensor(torchvision.utils.make_grid(
                        tensor = img_Tsor_bacth_i, 
                        nrow= 8)
            )









