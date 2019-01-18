import numpy as np
import os 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
from skimage import io, transform
from torchvision import transforms, utils

class ImageDataset(Dataset):
    def __init__(self,path,transform=None):
        #label 값을 불러온다
        self.images = np.load(path+"/images.npy")
        self.labels = np.load(path+"/labels.npy")
        self.landmarks = np.load(path+"/landmarks.npy")
        self.hog_features = np.load(path+"/hog_features.npy")
        self.transform=transform
        

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        
        
        images = self.images[i]
        labels = self.labels[i]
        landmarks=self.landmarks[i]
        hog_features=self.hog_features[i]
        
        sample = [images,labels,landmarks,hog_features]
        if self.transform:
            sample = self.transform(sample)
        return sample
        
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        images, labels, landmarks, hog_features= sample[0], sample[1], sample[2], sample[3]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        images = images.reshape((1, images.shape[0], images.shape[1]))

        images = images/255
        images = torch.from_numpy(images)
        labels = torch.from_numpy(np.array(labels))
        landmarks = torch.from_numpy(landmarks)
        hog_features = torch.from_numpy(hog_features)
        
        result=[images.type(torch.FloatTensor),labels.type(torch.LongTensor),landmarks.type(torch.FloatTensor),hog_features.type(torch.FloatTensor)]
        return result
    
class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels, landmarks, hog_features= sample[0], sample[1], sample[2], sample[3]
        
        h, w = image.shape[0],image.shape[1]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        sample=[img,labels,landmarks, hog_features]

        return sample