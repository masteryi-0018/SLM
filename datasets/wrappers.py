import random
import cv2
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import torch

from datasets import register


@register('crop-aug')
class CropAug(Dataset):

    def __init__(self, dataset, repeat_1=16, inp_size=None, flip=False,
                 scale=True, class_12=False, color=True):
        self.dataset = dataset
        self.repeat = repeat_1
        self.inp_size = inp_size
        self.flip = flip
        self.step = 256
        self.scale = scale
        self.scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
        self.class_12 = class_12
        self.color = color

    def __len__(self):
        return len(self.dataset) * self.repeat

    def __getitem__(self, idx):
        data = self.dataset[idx % len(self.dataset)]
        img, gt = data[0], data[1]

        if self.inp_size is None:
            self.inp_size = 512
        
        if self.scale:
            scale = random.choice(self.scales)
            w_lr = int(self.inp_size * scale)
            step = int(self.step * scale)
        else:
            w_lr = self.inp_size
            step = self.step
        
        # 随机选取一个点进行裁剪
        pointlist = [step * i for i in range((img.shape[-2]-w_lr) // step + 1)]
        pointlist.append(min(img.shape[-2], img.shape[-1])-w_lr)
        # print(pointlist)
        x0 = random.choice(pointlist)
        y0 = random.choice(pointlist)
        # x0 = random.randint(0, img.shape[-2] - w_lr) // self.step * self.step
        # y0 = random.randint(0, img.shape[-1] - w_lr) // self.step * self.step
        # print(x0, y0)
        crop_img = img[:, x0: x0 + w_lr, y0: y0 + w_lr]
        crop_gt = gt[x0: x0 + w_lr, y0: y0 + w_lr]

        if self.flip:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                
                return x
            
            crop_img = augment(crop_img)
            crop_gt = augment(crop_gt)
        
        if self.color:
            def color(x):
                return transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(transforms.ToPILImage()(x))
            
            crop_img = transforms.ToTensor()(color(crop_img))
        
        if self.scale:
            if crop_img.shape[-2] != self.inp_size:
                shape = (self.inp_size, self.inp_size)
                crop_gt = crop_gt.unsqueeze(0)
                crop_img = torch.tensor(resize_cv(crop_img, shape)).permute(2,0,1)
                crop_gt = torch.tensor(resize_cv(crop_gt, shape))
        
        gt = crop_gt
        if self.class_12:
            gt[gt == 1] = 0
            gt[gt == 2] = 1
            gt[gt == 3] = 2
            gt[gt == 4] = 3
            gt[gt == 5] = 4
            gt[gt == 6] = 5
            gt[gt == 7] = 6
            gt[gt == 8] = 7
            gt[gt == 9] = 8
    
            gt[gt == 10] = 7
            gt[gt == 11] = 8
            gt[gt == 12] = 9
            gt[gt == 13] = 10
            gt[gt == 14] = 11
    
            gt[gt == 15] = 11
        else:
            gt[gt == 15] = 0
        crop_gt = gt
        
        return {
            'data': crop_img,
            'label': crop_gt,
        }


# resize, transforms.InterpolationMode.BICUBIC
def resize_fn(img, size):
    return transforms.ToTensor()(transforms.Resize(size, transforms.InterpolationMode.BICUBIC)
                                 (transforms.ToPILImage()(img)))

def resize_cv(img, size):
    img = img.numpy().transpose(1,2,0)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return img


@register('crop-aug-dem')
class CropAugDem(Dataset):

    def __init__(self, dataset, repeat_1=16, inp_size=None, flip=True, scale=True, class_12=False):
        self.dataset = dataset
        self.repeat = repeat_1
        self.inp_size = inp_size
        self.flip = flip
        self.step = 256
        self.scale = scale
        self.scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
        self.class_12 = class_12

    def __len__(self):
        return len(self.dataset) * self.repeat

    def __getitem__(self, idx):
        img, gt, dem = self.dataset[idx % len(self.dataset)]
        
        dem = dem.unsqueeze(0)
        shape = img.shape[-2:]
        dem_big = torch.tensor(resize_cv(dem, shape))
        
        if self.inp_size is None:
            self.inp_size = 512
        
        if self.scale:
            scale = random.choice(self.scales)
            w_lr = int(self.inp_size * scale)
            step = int(self.step * scale)
        else:
            w_lr = self.inp_size
            step = self.step
        
        # 随机选取一个点进行裁剪
        pointlist = [step * i for i in range((img.shape[-2]-w_lr) // step + 1)]
        pointlist.append(min(img.shape[-2], img.shape[-1])-w_lr)
        # print(pointlist)
        x0 = random.choice(pointlist)
        y0 = random.choice(pointlist)
        
        crop_img = img[:, x0: x0 + w_lr, y0: y0 + w_lr]
        crop_gt = gt[x0: x0 + w_lr, y0: y0 + w_lr]
        crop_dem = dem_big[x0: x0 + w_lr, y0: y0 + w_lr]

        if self.flip:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_img = augment(crop_img)
            crop_gt = augment(crop_gt)
            crop_dem = augment(crop_dem)
        
        if self.scale:
            if crop_img.shape[-2] != self.inp_size:
                shape = (self.inp_size, self.inp_size)
                crop_gt = crop_gt.unsqueeze(0)
                crop_dem = crop_dem.unsqueeze(0)
                crop_img = torch.tensor(resize_cv(crop_img, shape)).permute(2,0,1)
                crop_gt = torch.tensor(resize_cv(crop_gt, shape))
                crop_dem = torch.tensor(resize_cv(crop_dem, shape))
        
        crop_gt[crop_gt == 15] = 0
        
        return {
            'data': crop_img,
            'label': crop_gt,
            'dem': crop_dem,
        }


@register('crop-aug-hha')
class CropAugHha(Dataset):

    def __init__(self, dataset, repeat_1=16, inp_size=None, flip=True, scale=True, class_12=False):
        self.dataset = dataset
        self.repeat = repeat_1
        self.inp_size = inp_size
        self.flip = flip
        self.step = 256
        self.scale = scale
        self.scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
        self.class_12 = class_12

    def __len__(self):
        return len(self.dataset) * self.repeat

    def __getitem__(self, idx):
        img, gt, dem = self.dataset[idx % len(self.dataset)]
        
        # dem = dem.unsqueeze(0)
        shape = img.shape[-2:]
        dem_big = torch.tensor(resize_cv(dem, shape)).permute(2,0,1)
        
        if self.inp_size is None:
            self.inp_size = 512
        
        if self.scale:
            scale = random.choice(self.scales)
            w_lr = int(self.inp_size * scale)
            step = int(self.step * scale)
        else:
            w_lr = self.inp_size
            step = self.step
        
        # 随机选取一个点进行裁剪
        pointlist = [step * i for i in range((img.shape[-2]-w_lr) // step + 1)]
        pointlist.append(min(img.shape[-2], img.shape[-1])-w_lr)
        # print(pointlist)
        x0 = random.choice(pointlist)
        y0 = random.choice(pointlist)
        
        crop_img = img[:, x0: x0 + w_lr, y0: y0 + w_lr]
        crop_gt = gt[x0: x0 + w_lr, y0: y0 + w_lr]
        crop_dem = dem_big[:, x0: x0 + w_lr, y0: y0 + w_lr]

        if self.flip:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_img = augment(crop_img)
            crop_gt = augment(crop_gt)
            crop_dem = augment(crop_dem)
        
        if self.scale:
            if crop_img.shape[-2] != self.inp_size:
                shape = (self.inp_size, self.inp_size)
                crop_gt = crop_gt.unsqueeze(0)
                crop_img = torch.tensor(resize_cv(crop_img, shape)).permute(2,0,1)
                crop_gt = torch.tensor(resize_cv(crop_gt, shape))
                crop_dem = torch.tensor(resize_cv(crop_dem, shape)).permute(2,0,1)
        
        crop_gt[crop_gt == 15] = 0
        
        return {
            'data': crop_img,
            'label': crop_gt,
            'dem': crop_dem,
        }


@register('crop-aug-paired')
class CropAugMask(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, gt, ulb = self.dataset[idx]

        if self.inp_size is None:
            self.inp_size = 512
        
        w_lr = self.inp_size
        x0 = random.randint(0, img.shape[-2] - w_lr - 2)
        y0 = random.randint(0, img.shape[-1] - w_lr - 2)
        crop_img = img[:, x0: x0 + w_lr, y0: y0 + w_lr]
        crop_gt = gt[x0: x0 + w_lr, y0: y0 + w_lr]
        crop_ulb = ulb[:, x0: x0 + w_lr, y0: y0 + w_lr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_img = augment(crop_img)
            crop_gt = augment(crop_gt)
            crop_ulb = augment(crop_ulb)
        
        return {
            'data': crop_img,
            'label': crop_gt,
            'ulb': crop_ulb,
        }


# 单独的去0版本
@register('crop-nozero')
class CropNo0(Dataset):

    def __init__(self, dataset, flip=False):
        self.dataset = dataset
        self.flip = flip

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        crop_img, crop_gt, crop_dem = self.dataset[idx]

        if self.flip:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                
                return x
            
            crop_img = augment(crop_img)
            crop_gt = augment(crop_gt)
            crop_dem = augment(crop_dem)

        crop_gt[crop_gt == 15] = 0
        return {
            'data': crop_img,
            'label': crop_gt,
            'dem': crop_dem
        }



if __name__ == '__main__':
    import image_folder
    
    # 只有RGB
    dataset_ori = image_folder.LabeledFolder(root_path=r'F:\DFC22\labeled_train')
    dataset = CropAug(dataset_ori)
    batch = dataset[1]
    img, gt = batch['data'], batch['label']
    print(len(dataset), img.shape, gt.shape)
    '''
    img = transforms.ToPILImage()(img)
    img.save("img.tif")
    gt = transforms.ToPILImage()(gt)
    gt.save("gt.tif")
    '''
    
    # RGB+DEM
    dataset_rgbdem = image_folder.RgbDemFolders(root_path_1=r'F:\DFC22\labeled_train', root_path_2=r'F:\DFC22\labeled_train')
    dataset_rd = CropAugDem(dataset_rgbdem)
    print(len(dataset_rd), dataset_rd[1]['data'].shape, dataset_rd[1]['label'].shape, dataset_rd[1]['dem'].shape)
    
    # 寻找dem数据的最大值
    '''
    maxi = 0
    for i in range(len(dataset_rd)):
        if dataset_rd[i]['dem'].max() > maxi:
            maxi = dataset_rd[i]['dem'].max()
            print(dataset_rd[i]['dem'].max())
    '''
    
    # RGB+HHA
    dataset_rgbhha = image_folder.RgbHhaFolders(root_path_1=r'F:\DFC22\labeled_train', root_path_2=r'F:\DFC22\labeled_train')
    dataset_rh = CropAugHha(dataset_rgbhha)
    print(len(dataset_rh), dataset_rh[1]['data'].shape, dataset_rh[1]['label'].shape, dataset_rh[1]['dem'].shape)
    
    # 去0版本
    dataset_rgbhha = image_folder.No0AllFolderNocut(root_path=r'F:\DFC22\labeled_train')
    dataset_rh = CropAugDem(dataset_rgbhha)
    print(len(dataset_rh), dataset_rh[1]['data'].shape, dataset_rh[1]['label'].shape, dataset_rh[1]['dem'].shape)
    