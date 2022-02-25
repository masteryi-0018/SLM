import os
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


# RGB
@register('labeled-folder')
class LabeledFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat_1=1, repeat_2=2, cache='none'):
        self.repeat = repeat_1
        self.cache = cache
        
        city_list = sorted(os.listdir(root_path))
        self.files = []
        
        for city_name in city_list:
            img_dir = os.path.join(root_path, city_name, 'BDORTHO')
            img_filenames = sorted(os.listdir(img_dir))

            gt_dir = os.path.join(root_path, city_name, 'UrbanAtlas')
            gt_filenames = sorted(os.listdir(gt_dir))
            
            for i in range(len(img_filenames)):
                img_file = os.path.join(img_dir, img_filenames[i])
                gt_file = os.path.join(gt_dir, gt_filenames[i])

                if cache == 'none':
                    self.files.append((img_file, gt_file))
                
                elif cache == 'in_memory':
                    img = transforms.ToTensor()(cv2.imread(img_file, -1))
                    gt = torch.tensor(cv2.imread(gt_file, -1))
                    self.files.append((img, gt))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.files[idx]

        if self.cache == 'none':
            img = transforms.ToTensor()(cv2.imread(x[0], -1))
            gt = torch.tensor(cv2.imread(x[1], -1))
            return (img, gt)
        
        elif self.cache == 'in_memory':
            return x


@register('unlabeled-folder')
class UnlabeledFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat_1=3, repeat_2=2, cache='none'):
        self.repeat = repeat_2
        self.cache = cache

        city_list = sorted(os.listdir(root_path))
        self.files = []

        for city_name in city_list:
            img_dir = os.path.join(root_path, city_name, 'BDORTHO')
            img_filenames = sorted(os.listdir(img_dir))
            
            for i in range(len(img_filenames)):
                img_file = os.path.join(img_dir, img_filenames[i])

                if cache == 'none':
                    self.files.append(img_file)
                
                elif cache == 'in_memory':
                    img = transforms.ToTensor()(cv2.imread(img_file, -1))
                    self.files.append(img)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.files[idx]

        if self.cache == 'none':
            img = transforms.ToTensor()(cv2.imread(x, -1))
            return img
        
        elif self.cache == 'in_memory':
            return x


# DEM
@register('lbdem-folder')
class LbDemFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat_1=3, repeat_2=2, cache='none'):
        self.repeat = repeat_1
        self.cache = cache
        
        city_list = sorted(os.listdir(root_path))
        self.files = []
        
        for city_name in city_list:
            dem_dir = os.path.join(root_path, city_name, 'RGEALTI')
            dem_filenames = sorted(os.listdir(dem_dir))
            
            for i in range(len(dem_filenames)):
                dem_file = os.path.join(dem_dir, dem_filenames[i])

                if cache == 'none':
                    self.files.append(dem_file)
                
                elif cache == 'in_memory':
                    dem = torch.relu(torch.tensor((cv2.imread(dem_file, -1)))).div_(3022.26)
                    self.files.append(dem)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.files[idx]

        if self.cache == 'none':
            dem = torch.relu(torch.tensor((cv2.imread(x, -1)))).div_(3022.26)
            return dem
        
        elif self.cache == 'in_memory':
            return x


@register('ulbdem-folder')
class UlbDemFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat_1=3, repeat_2=2, cache='none'):
        self.repeat = repeat_1
        self.cache = cache
        
        city_list = sorted(os.listdir(root_path))
        self.files = []
        
        for city_name in city_list:
            dem_dir = os.path.join(root_path, city_name, 'RGEALTI')
            dem_filenames = sorted(os.listdir(dem_dir))
            
            for i in range(len(dem_filenames)):
                dem_file = os.path.join(dem_dir, dem_filenames[i])

                if cache == 'none':
                    self.files.append(dem_file)
                
                elif cache == 'in_memory':
                    dem = torch.relu(torch.tensor((cv2.imread(dem_file, -1)))).div_(3022.26)
                    self.files.append(dem)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.files[idx]

        if self.cache == 'none':
            dem = torch.relu(torch.tensor((cv2.imread(x, -1)))).div_(3022.26)
            return dem
        
        elif self.cache == 'in_memory':
            return x


# HHA
@register('lbhha-folder')
class LbHhaFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat_1=1, repeat_2=2, cache='none'):
        self.repeat = repeat_1
        self.cache = cache
        
        city_list = sorted(os.listdir(root_path))
        self.files = []
        
        for city_name in city_list:
            hha_dir = os.path.join(root_path, city_name, 'HHA')
            hha_filenames = sorted(os.listdir(hha_dir))
            
            for i in range(len(hha_filenames)):
                hha_file = os.path.join(hha_dir, hha_filenames[i])

                if cache == 'none':
                    self.files.append(hha_file)
                
                elif cache == 'in_memory':
                    hha = transforms.ToTensor()(cv2.imread(hha_file, -1))
                    self.files.append(hha)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.files[idx]

        if self.cache == 'none':
            hha = transforms.ToTensor()(cv2.imread(x, -1))
            return hha
        
        elif self.cache == 'in_memory':
            return x


@register('ulbhha-folder')
class UlbHhaFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat_1=1, repeat_2=2, cache='none'):
        self.repeat = repeat_1
        self.cache = cache
        
        city_list = sorted(os.listdir(root_path))
        self.files = []
        
        for city_name in city_list:
            hha_dir = os.path.join(root_path, city_name, 'HHA')
            hha_filenames = sorted(os.listdir(hha_dir))
            
            for i in range(len(hha_filenames)):
                hha_file = os.path.join(hha_dir, hha_filenames[i])

                if cache == 'none':
                    self.files.append(hha_file)
                
                elif cache == 'in_memory':
                    hha = transforms.ToTensor()(cv2.imread(hha_file, -1))
                    self.files.append(hha)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.files[idx]

        if self.cache == 'none':
            hha = transforms.ToTensor()(cv2.imread(x, -1))
            return hha
        
        elif self.cache == 'in_memory':
            return x


# 去0的RGB和DEM，已切块的
import random
import numpy as np
def resize_cv(img, size):
    img = img.numpy().transpose(1,2,0)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return img

@register('nozero-all-folder')
class No0AllFolder(Dataset):

    def __init__(self, root_path, scale=True, repeat=1):
        self.repeat = repeat
        self.inp_size = 512
        self.step = 256
        self.scale = scale
        self.scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
        
        city_list = sorted(os.listdir(root_path))
        self.files = []
        
        for city_name in city_list:
            img_dir = os.path.join(root_path, city_name, 'BDORTHO')
            img_filenames = sorted(os.listdir(img_dir))

            gt_dir = os.path.join(root_path, city_name, 'UrbanAtlas')
            gt_filenames = sorted(os.listdir(gt_dir))
            
            dem_dir = os.path.join(root_path, city_name, 'RGEALTI')
            dem_filenames = sorted(os.listdir(dem_dir))
            
            for i in range(len(img_filenames)):
                
                for j in range(self.repeat):
                    img_file = os.path.join(img_dir, img_filenames[i])
                    gt_file = os.path.join(gt_dir, gt_filenames[i])
                    dem_file = os.path.join(dem_dir, dem_filenames[i])
                    
                    img = transforms.ToTensor()(cv2.imread(img_file, -1))
                    gt = torch.tensor(cv2.imread(gt_file, -1))
                    dem = (torch.tensor((cv2.imread(dem_file, -1)))-(-79.18)).div_(3020.26+79.18)
                    
                    if self.scale:
                        scale = random.choice(self.scales)
                        w_lr = int(self.inp_size * scale)
                        step = int(self.step * scale)
                    else:
                        w_lr = self.inp_size
                        step = self.step
                    
                    pointlist = [step * i for i in range((img.shape[-2]-w_lr) // step + 1)]
                    pointlist.append(min(img.shape[-2], img.shape[-1])-w_lr)
                    x0 = random.choice(pointlist)
                    y0 = random.choice(pointlist)
                    
                    dem = dem.unsqueeze(0)
                    shape = img.shape[-2:]
                    dem_big = torch.tensor(resize_cv(dem, shape))
                    crop_img = img[:, x0: x0 + w_lr, y0: y0 + w_lr]
                    crop_gt = gt[x0: x0 + w_lr, y0: y0 + w_lr]
                    crop_dem = dem_big[x0: x0 + w_lr, y0: y0 + w_lr]
                    
                    if self.scale:
                        if crop_img.shape[-2] != self.inp_size:
                            shape = (self.inp_size, self.inp_size)
                            crop_gt = crop_gt.unsqueeze(0)
                            crop_dem = crop_dem.unsqueeze(0)
                            crop_img = torch.tensor(resize_cv(crop_img, shape)).permute(2,0,1)
                            crop_gt = torch.tensor(resize_cv(crop_gt, shape))
                            crop_dem = torch.tensor(resize_cv(crop_dem, shape))
                    # print(crop_img.shape, crop_gt.shape, crop_dem.shape)
                    crop_gt = crop_gt.numpy()
                    if not np.all(crop_gt == 0):
                        crop_gt = torch.tensor(crop_gt)
                        self.files.append((crop_img, crop_gt, crop_dem))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        return self.files[idx]


# 不裁剪
@register('nozero-all-folder-nocut')
class No0AllFolderNocut(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat_1=1, repeat_2=2, cache='none'):
        self.repeat = repeat_1
        self.cache = cache
        
        city_list = sorted(os.listdir(root_path))
        self.files = []
        
        for city_name in city_list:
            img_dir = os.path.join(root_path, city_name, 'BDORTHO')
            img_filenames = sorted(os.listdir(img_dir))

            gt_dir = os.path.join(root_path, city_name, 'UrbanAtlas')
            gt_filenames = sorted(os.listdir(gt_dir))
            
            dem_dir = os.path.join(root_path, city_name, 'RGEALTI')
            dem_filenames = sorted(os.listdir(dem_dir))
            
            for i in range(len(img_filenames)):
            
                img_file = os.path.join(img_dir, img_filenames[i])
                gt_file = os.path.join(gt_dir, gt_filenames[i])
                dem_file = os.path.join(dem_dir, dem_filenames[i])
                
                gt = cv2.imread(gt_file, -1)
                if not np.all(gt == 0):
                    gt = torch.tensor(gt)
                    
                    if self.cache == 'none':
                        self.files.append((img_file, gt_file, dem_file))
                        
                    if self.cache == 'in_memory':
                        img = transforms.ToTensor()(cv2.imread(img_file, -1))
                        dem = (torch.tensor((cv2.imread(dem_file, -1)))+79.18).div_(3020.26+79.18)
                        self.files.append((img, gt, dem))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.files[idx]
        
        if self.cache == 'none':
            img = transforms.ToTensor()(cv2.imread(x[0], -1))
            dem = (torch.tensor((cv2.imread(x[2], -1)))+79.18).div_(3020.26+79.18)
            gt = torch.tensor(cv2.imread(x[1], -1))
            return img, gt, dem
            
        if self.cache == 'in_memory':
            return x


# 全部 RGB
@register('paired-folders')
class PairedFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = LabeledFolder(root_path_1, **kwargs)
        self.dataset_2 = UnlabeledFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx][0], self.dataset_1[idx][1], self.dataset_2[idx]


# 有标签 RGB+DEM
@register('rgbdem-folders')
class RgbDemFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = LabeledFolder(root_path_1, **kwargs)
        self.dataset_2 = LbDemFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx][0], self.dataset_1[idx][1], self.dataset_2[idx]


# 有标签 RGB+HHA
@register('rgbhha-folders')
class RgbHhaFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = LabeledFolder(root_path_1, **kwargs)
        self.dataset_2 = LbHhaFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx][0], self.dataset_1[idx][1], self.dataset_2[idx]



if __name__ == '__main__':
    # 读入内存约50-60G
    
    # 只读取有标签RGB
    # labeled_dataset = LabeledFolder(root_path=r'F:\DFC22\labeled_train', cache='none')
    # print(len(labeled_dataset), labeled_dataset[0][0].shape, labeled_dataset[0][1].shape)
    
    # 只读取无标签
    # unlabeled_dataset = UnlabeledFolder(root_path=r'F:\DFC22\unlabeled_train', cache='none')
    # print(len(unlabeled_dataset), unlabeled_dataset[0].shape)
    
    # 一起读取，需要分配repeat使其长度相同
    # dataset = PairedFolders(root_path_1=r'F:\DFC22\labeled_train', root_path_2=r'F:\DFC22\unlabeled_train')
    # print(len(dataset), dataset[0][0].shape, dataset[0][1].shape, dataset[0][2].shape)
    
    # 有标签RGB+DEM
    # rgbdem_dataset = RgbDemFolders(root_path_1=r'F:\DFC22\labeled_train', root_path_2=r'F:\DFC22\labeled_train')
    # print(len(rgbdem_dataset), rgbdem_dataset[0][0].shape, rgbdem_dataset[0][1].shape, rgbdem_dataset[0][2].shape)
    
    rgbdem_dataset_ori = No0AllFolderNocut(root_path=r'F:\DFC22\labeled_train')
    rgbdem_dataset = rgbdem_dataset_ori[2]
    print(len(rgbdem_dataset_ori), rgbdem_dataset[0].shape, rgbdem_dataset[1].shape, rgbdem_dataset[2].shape)