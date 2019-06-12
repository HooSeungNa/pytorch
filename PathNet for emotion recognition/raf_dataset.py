import numpy as np
import os 
import cv2
import torch
from torch.utils.data import Dataset
import scipy.io
import torchvision.transforms as transforms
from skimage import io, transform

#load raf dataset 

class ImageDataset(Dataset):
    def __init__(self, fnames, transform=None):
        
        #label 값을 불러온다
        self.image=np.load(fnames[0])
        self.label=np.load(fnames[1])
        self.transform=transform
    def __len__(self):
        return len(self.image)
#         return 1000
    
    def __getitem__(self, i):
        #load grayscale image
        image=cv2.imread(self.image[i])
        image=cv2.resize(image,(32,32))
        label=self.label[i]
        
        sample = [image,label]
        if self.transform:
            sample = self.transform(sample)
        return sample
# #using face mask
# class Fer2013(Dataset):
#     def __init__(self, fnames, transform=None):
        
#         #label 값을 불러온다
#         self.image=np.load(fnames[0])
#         self.label=np.load(fnames[1])
#         self.transform=transform
#     def __len__(self):
#         return len(self.image)
# #         return 1000
    
#     def __getitem__(self, i):
#         #load grayscale image
#         image=self.image[i]
#         label=self.label[i]
        
#         sample = [image,label]
#         if self.transform:
#             sample = self.transform(sample)
#         return sample
        
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        ia,ib= sample[0],sample[1]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        ia = ia/255
        ia = ia.transpose((2, 0, 1))
        ia = torch.from_numpy(ia)
        
#         #color image
#         ia = ia/255
#         ia = ia.transpose((2, 0, 1))
#         ia = torch.from_numpy(ia)
        
        #label
        ib = torch.tensor(ib)
        
        result=[ia.type(torch.FloatTensor),ib.type(torch.LongTensor)]
        return result