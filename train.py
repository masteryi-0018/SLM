import argparse
import os
import yaml
from tqdm import tqdm
import numpy as np
# from apex import amp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
from valid import eval_miou
from utils import utils


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    
    valid_size = 1
    train_size = len(dataset) - valid_size
    
    trainset, validset = random_split(dataset=dataset, lengths=[train_size, valid_size])
    
    trainset = datasets.make(spec['wrapper'], args={'dataset': trainset})
    validset = datasets.make(spec['wrapper'], args={'dataset': validset})
    
    log('{} trainset: size={}, validset: size={}'.format(tag, len(trainset), len(validset)))
    for k, v in trainset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))
    
    train_loader = DataLoader(trainset, batch_size=spec['batch_size'],
        shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(validset, batch_size=1,
        shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
    return train_loader, val_loader


def make_data_loaders():
    return make_data_loader(config.get('sup_dataset'))


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def cutmix_data(img, dem, gt):
    """
    输入数据均为tensor:BCHW
    """
    batch_size = img.size()[0]
    lam = np.random.beta(1, 1) 
    index = torch.randperm(batch_size).cuda() # 随机生成batch内顺序
    bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
    img[:, :, bbx1:bbx2, bby1:bby2] = img[index, :, bbx1:bbx2, bby1:bby2]
    if dem is not None:
        dem[:, :, bbx1:bbx2, bby1:bby2] = dem[index, :, bbx1:bbx2, bby1:bby2]
    gt[:, bbx1:bbx2, bby1:bby2] = gt[index, bbx1:bbx2, bby1:bby2]
    return img, dem, gt

 
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
 
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
 
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train(train_loader, model, optimizer):
    model.train()
    loss_fn = utils.make_loss_fn(config['loss_fn'])
    train_loss = utils.Averager()

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['data']
        gt = batch['label'].long()
        
        if config.get('dem_input'):
            dem = batch['dem']
            if config.get('cutmix'):
                inp, dem, gt = cutmix_data(inp, dem, gt) # 加入cutmix增强
            if config.get('aux_loss'):
                aux, pred = model(inp, dem)
                aux_loss = loss_fn(aux, gt)
                loss = loss_fn(pred, gt)
                loss += 0.4 * aux_loss
            else:
                pred = model(inp, dem)
                loss = loss_fn(pred, gt)
        else:
            dem = None
            if config.get('cutmix'):
                inp, dem, gt = cutmix_data(inp, dem, gt)
            if config.get('aux_loss'):
                aux, pred = model(inp)
                aux_loss = loss_fn(aux, gt)
                loss = loss_fn(pred, gt)
                loss += 0.4 * aux_loss
            else:
                pred = model(inp)
                loss = loss_fn(pred, gt)
        
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        # amp
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
            # scaled_loss.backward()
        optimizer.step()

        pred = None; loss = None

    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    train_loader, val_loader = make_data_loaders()

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    # amp
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    num_class = config['model']['args']['num_classes']
    max_val_v = -1e18
    
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()

        sv_file = {
            'model': model_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_miou(config, val_loader, model_, num_class)

            log_info.append('val: miou={:.4f}'.format(val_res))
            writer.add_scalars('miou', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_baseline.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)