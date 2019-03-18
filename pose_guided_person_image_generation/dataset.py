import h5py
import numpy as np
import torch
import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    '''Returns a customized sample of data'''

    def __init__(self, filename, transform = None):
        '''Initialization'''
        self.file = filename
        self.transform=transform


    def __len__(self):
        '''Denotes the total number of samples'''
        return 10000
#         return len(list(h5py.File(self.file, 'r').keys()))

    def proc_rawimage(self, img, mean, norm):
        '''Processing the raw image. Here mean and norm are scalar values'''
        return (img - mean)/norm

    def __getitem__(self, index):
        file = self.file
        group = 'Data_{}'.format(index)
        with h5py.File(file, 'r') as f_read:
            image_raw_0 = np.asarray(f_read[group].get('image_raw_0'), dtype = np.float32)
            image_raw_1 = np.asarray(f_read[group].get('image_raw_1'), dtype = np.float32)
            pose_r4_0 = np.asarray(f_read[group].get('pose_r8_0'), dtype = np.float32)
            pose_r4_1 = np.asarray(f_read[group].get('pose_r8_1'), dtype = np.float32)
            pose_mask_r4_0 = np.asarray(f_read[group].get('pose_mask_r8_0'), dtype = np.float32)
            pose_mask_r4_1 = np.asarray(f_read[group].get('pose_mask_r8_1'), dtype = np.float32)

        #processing
        proc_raw_0 = self.proc_rawimage(image_raw_0, 127.5, 127.5)
        proc_raw_1 = self.proc_rawimage(image_raw_1, 127.5, 127.5)
        proc_pose_0 = 2 * pose_r4_0 - 1
        proc_pose_1 = 2 * pose_r4_1 - 1

        sample=[proc_raw_0, proc_raw_1, proc_pose_0, proc_pose_1, pose_mask_r4_0, pose_mask_r4_1]
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def hwc_to_chw(arr):
        return arr.transpose([2,0,1])
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        proc_raw_0, proc_raw_1, proc_pose_0, proc_pose_1, pose_mask_r4_0, pose_mask_r4_1 = sample[0], sample[1],sample[2], sample[3],sample[4], sample[5]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        proc_raw_0 = hwc_to_chw(proc_raw_0)
        proc_raw_0 = torch.from_numpy(proc_raw_0)
        
        proc_raw_1 = hwc_to_chw(proc_raw_1)
        proc_raw_1 = torch.from_numpy(proc_raw_1)
        
        proc_pose_0 = hwc_to_chw(proc_pose_0)
        proc_pose_0 = torch.from_numpy(proc_pose_0)
        
        proc_pose_1 = hwc_to_chw(proc_pose_1)
        proc_pose_1 = torch.from_numpy(proc_pose_1)
        
        pose_mask_r4_0 = hwc_to_chw(pose_mask_r4_0)
        pose_mask_r4_0 = torch.from_numpy(pose_mask_r4_0)
        
        pose_mask_r4_1 = hwc_to_chw(pose_mask_r4_1)
        pose_mask_r4_1 = torch.from_numpy(pose_mask_r4_1)
      
        result=[proc_raw_0.type(torch.FloatTensor),proc_raw_1.type(torch.FloatTensor),
               proc_pose_0.type(torch.FloatTensor),proc_pose_1.type(torch.FloatTensor),
               pose_mask_r4_0.type(torch.FloatTensor),pose_mask_r4_1.type(torch.FloatTensor)]
        return result
