# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:57:29 2022

@author: masteryi
"""


import os
import cv2
import numpy as np

from torchvision import transforms
import torch

import models


# size=512切块
def generat_val_512(dir_path1, dir_path2, model_path):
    path1 = dir_path1
    path2 = dir_path2
    img_name_list = sorted(os.listdir(path1))
    size = 512
    class12 = False
    aux = False
    
    model_spec = torch.load(model_path)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    model.eval()
    with torch.no_grad():
    
        for i in range(len(img_name_list)):
            print(path1 + '/' + img_name_list[i], flush=True)
            img = cv2.imread(path1 + '/' + img_name_list[i], -1)
            shape = img.shape[:2]
            out_img = np.zeros(shape, dtype=np.uint8)
            
            for r in range(4):
                for c in range(4):
                    if c < 3 and r < 3:
                        img_c = img[size*r:size*(r+1), size*c:size*(c+1)]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        if aux:
                            _, pred = model(img_c)
                        else:
                            pred = model(img_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[size*r:size*(r+1), size*c:size*(c+1)] = np.int8(premax[0].cpu())
                        
                    elif r < 3:
                        img_c = img[size*r:size*(r+1), -size:]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        if aux:
                            _, pred = model(img_c)
                        else:
                            pred = model(img_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[size*r:size*(r+1), -size:] = np.int8(premax[0].cpu())
                        
                    elif c < 3:
                        img_c = img[-size:, size*c:size*(c+1)]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        if aux:
                            _, pred = model(img_c)
                        else:
                            pred = model(img_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[-size:, size*c:size*(c+1)] = np.int8(premax[0].cpu())
                        
                    else:
                        img_c = img[-size:, -size:]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        if aux:
                            _, pred = model(img_c)
                        else:
                            pred = model(img_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[-size:, -size:] = np.int8(premax[0].cpu())
            
            if class12:
                gt = out_img
                
                gt[gt == 11] = 14
                gt[gt == 10] = 13
                gt[gt == 9] = 12
                gt[gt == 8] = 11
                gt[gt == 7] = 10
                gt[gt == 6] = 7
                gt[gt == 5] = 6
                gt[gt == 4] = 5
                gt[gt == 3] = 4
                gt[gt == 2] = 3
                gt[gt == 1] = 2
                gt[gt == 0] = 1
                
                out_img = gt
            
            idx = img_name_list[i].find('.')
            bdortho = img_name_list[i][:idx]
            out_img_name = path2 + '/' + bdortho + '_prediction.tif'
            cv2.imwrite(out_img_name, out_img)


# size=512切块，加dem数据
def generat_val_512_dem(dir_path1, dir_path2, dir_path3, model_path):
    path1 = dir_path1
    path2 = dir_path2
    path3 = dir_path3
    img_name_list = sorted(os.listdir(path1))
    dem_name_list = sorted(os.listdir(path3))
    
    size = 512
    class12 = False
    
    model_spec = torch.load(model_path)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    model.eval()
    with torch.no_grad():
    
        for i in range(len(img_name_list)):
            print(path1 + '/' + img_name_list[i], flush=True)
            img = cv2.imread(path1 + '/' + img_name_list[i], -1)
            dem = cv2.imread(path3 + '/' + dem_name_list[i], -1)
            shape = h, w = img.shape[:2] # [height, width, channel]
            dem = cv2.resize(dem, (w, h))
            
            out_img = np.zeros(shape, dtype=np.uint8)
            
            for r in range(4):
                for c in range(4):
                    if c < 3 and r < 3:
                        img_c = img[size*r:size*(r+1), size*c:size*(c+1)]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        dem_c = dem[size*r:size*(r+1), size*c:size*(c+1)]
                        dem_c = torch.relu(torch.tensor(dem_c)).div_(3022.26).unsqueeze(0).unsqueeze(0).cuda()
                        pred = model(img_c, dem_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[size*r:size*(r+1), size*c:size*(c+1)] = np.int8(premax[0].cpu())
                        
                    elif r < 3:
                        img_c = img[size*r:size*(r+1), -size:]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        dem_c = dem[size*r:size*(r+1), -size:]
                        dem_c = torch.relu(torch.tensor(dem_c)).div_(3022.26).unsqueeze(0).unsqueeze(0).cuda()
                        pred = model(img_c, dem_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[size*r:size*(r+1), -size:] = np.int8(premax[0].cpu())
                        
                    elif c < 3:
                        img_c = img[-size:, size*c:size*(c+1)]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        dem_c = dem[-size:, size*c:size*(c+1)]
                        dem_c = torch.relu(torch.tensor(dem_c)).div_(3022.26).unsqueeze(0).unsqueeze(0).cuda()
                        pred = model(img_c, dem_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[-size:, size*c:size*(c+1)] = np.int8(premax[0].cpu())
                        
                    else:
                        img_c = img[-size:, -size:]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        dem_c = dem[-size:, -size:]
                        dem_c = torch.relu(torch.tensor(dem_c)).div_(3022.26).unsqueeze(0).unsqueeze(0).cuda()
                        pred = model(img_c, dem_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[-size:, -size:] = np.int8(premax[0].cpu())
            
            if class12:
                gt = out_img
                
                gt[gt == 11] = 14
                gt[gt == 10] = 13
                gt[gt == 9] = 12
                gt[gt == 8] = 11
                gt[gt == 7] = 10
                gt[gt == 6] = 7
                gt[gt == 5] = 6
                gt[gt == 4] = 5
                gt[gt == 3] = 4
                gt[gt == 2] = 3
                gt[gt == 1] = 2
                gt[gt == 0] = 1
                
                out_img = gt
            
            idx = img_name_list[i].find('.')
            bdortho = img_name_list[i][:idx]
            out_img_name = path2 + '/' + bdortho + '_prediction.tif'
            cv2.imwrite(out_img_name, out_img)

def generat_val_512_hha(dir_path1, dir_path2, dir_path3, model_path):
    path1 = dir_path1
    path2 = dir_path2
    path3 = dir_path3
    img_name_list = sorted(os.listdir(path1))
    dem_name_list = sorted(os.listdir(path3))
    
    size = 512
    class12 = False
    
    model_spec = torch.load(model_path)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    model.eval()
    with torch.no_grad():
    
        for i in range(len(img_name_list)):
            print(path1 + '/' + img_name_list[i], flush=True)
            img = cv2.imread(path1 + '/' + img_name_list[i], -1)
            dem = cv2.imread(path3 + '/' + dem_name_list[i], -1)
            shape = h, w = img.shape[:2] # [height, width, channel]
            dem = cv2.resize(dem, (w, h))
            
            out_img = np.zeros(shape, dtype=np.uint8)
            
            for r in range(4):
                for c in range(4):
                    if c < 3 and r < 3:
                        img_c = img[size*r:size*(r+1), size*c:size*(c+1)]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        dem_c = dem[size*r:size*(r+1), size*c:size*(c+1)]
                        dem_c = transforms.ToTensor()(dem_c).unsqueeze(0).cuda()
                        pred = model(img_c, dem_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[size*r:size*(r+1), size*c:size*(c+1)] = np.int8(premax[0].cpu())
                        
                    elif r < 3:
                        img_c = img[size*r:size*(r+1), -size:]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        dem_c = dem[size*r:size*(r+1), -size:]
                        dem_c = transforms.ToTensor()(dem_c).unsqueeze(0).cuda()
                        pred = model(img_c, dem_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[size*r:size*(r+1), -size:] = np.int8(premax[0].cpu())
                        
                    elif c < 3:
                        img_c = img[-size:, size*c:size*(c+1)]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        dem_c = dem[-size:, size*c:size*(c+1)]
                        dem_c = transforms.ToTensor()(dem_c).unsqueeze(0).cuda()
                        pred = model(img_c, dem_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[-size:, size*c:size*(c+1)] = np.int8(premax[0].cpu())
                        
                    else:
                        img_c = img[-size:, -size:]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        dem_c = dem[-size:, -size:]
                        dem_c = transforms.ToTensor()(dem_c).unsqueeze(0).cuda()
                        pred = model(img_c, dem_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[-size:, -size:] = np.int8(premax[0].cpu())
            
            if class12:
                gt = out_img
                
                gt[gt == 11] = 14
                gt[gt == 10] = 13
                gt[gt == 9] = 12
                gt[gt == 8] = 11
                gt[gt == 7] = 10
                gt[gt == 6] = 7
                gt[gt == 5] = 6
                gt[gt == 4] = 5
                gt[gt == 3] = 4
                gt[gt == 2] = 3
                gt[gt == 1] = 2
                gt[gt == 0] = 1
                
                out_img = gt
            
            idx = img_name_list[i].find('.')
            bdortho = img_name_list[i][:idx]
            out_img_name = path2 + '/' + bdortho + '_prediction.tif'
            cv2.imwrite(out_img_name, out_img)


# size=1024切块
def generat_val_1024(dir_path1, dir_path2, model_path):
    path1 = dir_path1
    path2 = dir_path2
    img_name_list = sorted(os.listdir(path1))
    size = 1024
    
    model_spec = torch.load(model_path)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    model.eval()
    with torch.no_grad():
    
        for i in range(len(img_name_list)):
            print(path1 + '/' + img_name_list[i], flush=True)
            img = cv2.imread(path1 + '/' + img_name_list[i], -1)
            shape = img.shape[:2]
            out_img = np.zeros(shape, dtype=np.uint8)
            
            for r in range(2):
                for c in range(2):
                    if c < 1 and r < 1:
                        img_c = img[size*r:size*(r+1), size*c:size*(c+1)]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        pred = model(img_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[size*r:size*(r+1), size*c:size*(c+1)] = np.int8(premax[0].cpu())
                        
                    elif r < 1:
                        img_c = img[size*r:size*(r+1), -size:]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        pred = model(img_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[size*r:size*(r+1), -size:] = np.int8(premax[0].cpu())
                        
                    elif c < 1:
                        img_c = img[-size:, size*c:size*(c+1)]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        pred = model(img_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[-size:, size*c:size*(c+1)] = np.int8(premax[0].cpu())
                        
                    else:
                        img_c = img[-size:, -size:]
                        img_c = transforms.ToTensor()(img_c).unsqueeze(0).cuda()
                        pred = model(img_c)
                        _, premax = torch.max(pred, dim=1, keepdim=False)
                        out_img[-size:, -size:] = np.int8(premax[0].cpu())
                
            idx = img_name_list[i].find('.')
            bdortho = img_name_list[i][:idx]
            out_img_name = path2 + '/' + bdortho + '_prediction.tif'
            cv2.imwrite(out_img_name, out_img)


# 不切块
def generat_val(dir_path1, dir_path2, model_path):
    path1 = dir_path1
    path2 = dir_path2
    img_name_list = sorted(os.listdir(path1))
    
    model_spec = torch.load(model_path)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    model.eval()
    with torch.no_grad():
    
        for i in range(len(img_name_list)):
            print(path1 + '/' + img_name_list[i], flush=True)
            img = cv2.imread(path1 + '/' + img_name_list[i], -1)
            
            img = transforms.ToTensor()(img).unsqueeze(0).cuda()
            pred = model(img)
            _, premax = torch.max(pred, dim=1, keepdim=False)
            out_img = np.int8(premax[0].cpu())
            
            idx = img_name_list[i].find('.')
            bdortho = img_name_list[i][:idx]
            out_img_name = path2 + '/' + bdortho + '_prediction.tif'
            cv2.imwrite(out_img_name, out_img)


def gen(model_path):
    val_path = 'F:/DFC22/val'
    sub_val_list = sorted(os.listdir(val_path))
    out_path = '../DFC22_val'
    os.makedirs(out_path, exist_ok=True)
    
    for sub_name in sub_val_list:
        sub_path = val_path + '/' + sub_name + '/' + 'BDORTHO'
        sub_out_path = out_path + '/' + sub_name
        os.makedirs(sub_out_path, exist_ok=True)
        generat_val_512(sub_path, sub_out_path, model_path)


# dem
def gen_dem(model_path):
    val_path = 'F:/DFC22/val'
    sub_val_list = sorted(os.listdir(val_path))
    out_path = '../DFC22_val'
    os.makedirs(out_path, exist_ok=True)
    
    for sub_name in sub_val_list:
        sub_path = val_path + '/' + sub_name + '/' + 'BDORTHO'
        dem_path = val_path + '/' + sub_name + '/' + 'RGEALTI'
        sub_out_path = out_path + '/' + sub_name
        os.makedirs(sub_out_path, exist_ok=True)
        generat_val_512_dem(sub_path, sub_out_path, dem_path, model_path)


# hha
def gen_hha(model_path):
    val_path = 'F:/DFC22/val'
    sub_val_list = sorted(os.listdir(val_path))
    out_path = '../DFC22_val'
    os.makedirs(out_path, exist_ok=True)
    
    for sub_name in sub_val_list:
        sub_path = val_path + '/' + sub_name + '/' + 'BDORTHO'
        hha_path = val_path + '/' + sub_name + '/' + 'HHA'
        sub_out_path = out_path + '/' + sub_name
        os.makedirs(sub_out_path, exist_ok=True)
        generat_val_512_hha(sub_path, sub_out_path, hha_path, model_path)



if __name__ == '__main__':
    
    model_path = 'save/_train_baseline_focal/epoch-100.pth'
    gen(model_path)