import argparse
import os
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import datasets as datasets
import models
from utils import utils, metric


def eval_miou(config, loader, model, num_class=15, eval_type=None, verbose=False):
    model.eval()

    if eval_type is None:
        metric_fn = metric.SegmentationMetric(num_class)
    else:
        raise NotImplementedError

    val_miou = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()
        
        inp = batch['data']
        with torch.no_grad():
            if config.get('dem_input'):
                dem = batch['dem']
                if config.get('aux_loss'):
                    _, pred = model(inp, dem)
                else:
                    pred = model(inp, dem)
            else:
                if config.get('aux_loss'):
                    _, pred = model(inp)
                else:
                    pred = model(inp)
        
        _, premax = torch.max(pred, dim=1, keepdim=True)
        
        premax = premax.cpu()
        gt = batch['label'].unsqueeze(1).cpu()
        
        metric_fn.addBatch(premax, gt)
        res = metric_fn.meanIntersectionOverUnion()
        
        val_miou.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('psnr val {:.4f}'.format(val_miou.item()))

    return val_miou.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_baseline.yaml')
    parser.add_argument('--model', default='save/_train_baseline/epoch_last.pth')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['sup_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=1,
        num_workers=0, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_miou(loader, model, verbose=True)
    print('result miou: {:.4f}'.format(res))
    