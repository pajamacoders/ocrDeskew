import cv2
import os
import torch
import logging
import argparse
import json
from modules.dataset.ocrDataset import OCRDataset

from modules.utils import build_transformer, MLLogger
from modules.dataset import build_dataloader

logger = logging.getLogger('deskew')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

def visualizer(path, img, ):
    img = data['img'].data.numpy().copy().squeeze()
    degree = -data['degree'].item()
    h,w = img.squeeze().shape
    m = cv2.getRotationMatrix2D((w/2, h/2), degree, 1)
    dst = cv2.warpAffine(img, m, (w,h))
    name = os.path.basename(data['imgpath'][-1]).replace('.jpg', '_rev.jpg')
    cv2.imwrite(f'vis/{name}', dst)

def valid(model, loader, fn_loss):
    pass

def train(model, loader, fn_loss, optimizer):
    for i, data in enumerate(loader):
        for k,v in data.items():
            if isinstance(v, torch.Tensor):
                data[k]=v.cuda()
        logit = model(data['img'])
        loss = fn_loss(logit, data['degree'])
        loss.backward()
        
        if (i+1)%10==0:
            break

def parse_args():
    parser = argparse.ArgumentParser(description="handle arguments")
    parser.add_argument("--config", help="path to the configuration file", type=str, default='config/renet_ocr.json')
    parser.add_argument("--run_name", help="run name for mlflow tracking", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = json.load(f)
        cfg['config_file']=args.config
        if args.run_name:
            cfg['mllogger_cfg']['run_name']=args.run_name
    
    tr = build_transformer(cfg['transform_cfg'])
    dataset = OCRDataset(**cfg['dataset_cfg']['train'])
    dataset.set_transformer(tr)
    dataset.compute_average()
    print(dataset)
    # train_loader, valid_loader = build_dataloader(**cfg['dataset_cfg'])

    # logger.info('create model')
    # model = torch.hub.load('pytorch/vision:v0.10.0', cfg['model_cfg']['type'])

    # logger.info('create loss function')
    # fn_loss = torch.nn.MSELoss()

    # logger.info('create optimizer')
    # opt=torch.optim.Adam(model.parameters(), **cfg['optimizer_cfg']['args'])
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,**cfg['lr_scheduler_cfg']['args'])

    # max_epoch = cfg['train_cfg']['max_epoch']
    # valid_ecpoh = cfg['train_cfg']['validation_every_n_epoch']
    # logger.info(f'max_epoch :{max_epoch}')
    # logger.info('set mlflow tracking')
    # mltracker = MLLogger(cfg, logger)
    # for step in range(max_epoch):
    #     train(model, train_loader, fn_loss, opt)
    #     if (step+1)%valid_ecpoh==0:
    #         valid(model, valid_loader, fn_loss)