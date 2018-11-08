import cv2
import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.files = pd.read_csv(csv_file)
        self.transform=transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.files.iloc[idx,0])
        img = cv2.imread(image_name)
        b, g, r = cv2.split(img)   # img파일을 b,g,r로 분리
        img2 = cv2.merge([r,g,b]) # b, r을 바꿔서 Merge
        labels = self.files.iloc[idx,1]
#         labels=labels.astype('int')
        
        
        sample = [img2,labels]
        if self.transform:
            sample = self.transform(sample)
        return sample
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample[0], sample[1]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        image = image/255
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        label = torch.from_numpy(np.array(label))
        result=[image.type(torch.FloatTensor),label.type(torch.LongTensor)]
        return result


