import cv2
import os
import torch
import logging
import argparse
import json
import numpy as np
from tqdm import tqdm
from modules.utils import build_transformer, MLLogger
from modules.dataset import build_dataloader, OCRDataset
from modules.model import build_model

logger = logging.getLogger('deskew')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
minloss = 999
def visualizer(data, logit, mllogger):
    
    for ind in range(len(data['img'])):
        mean, std = data['mean'][ind], data['std'][ind]
        img = data['img'][ind].squeeze()
        img = img*std+mean
        img = img.data.cpu().numpy().copy().astype(np.uint8)
        rotation=-logit[ind].item()

        h,w = img.shape
        m = cv2.getRotationMatrix2D((w/2, h/2), rotation, 1)
        dst = cv2.warpAffine(img, m, (w,h))
        name = os.path.basename(data['imgpath'][ind]).replace('.jpg', '_rev.jpg')
        resimg=cv2.hconcat([img, dst])
        mllogger.log_image(resimg, name=name)

def test(model, loader, fn_loss, mllogger):
    global minloss
    model.eval()
    total_loss = 0
    num_samples=0

    for i, data in tqdm(enumerate(loader)):
        data['img']=data['img'].cuda(non_blocking=True).float()
        data['degree']=data['degree'].cuda(non_blocking=True).float()

        with torch.no_grad():
            logit = model(data['img'])
        logit=logit.squeeze()
        loss = fn_loss(logit, data['degree'])
        total_loss+=loss.detach()*logit.shape[0]
        num_samples+=logit.shape[0]
        visualizer(data, logit, mllogger)

    avgloss =  (total_loss/num_samples).item()
    logger.info(f'test-0 epoch:{avgloss:.4f}')

    mllogger.log_metric('test_loss', avgloss, 0)

   
    #mllogger.log_image(img, name=f'{step}_sample.jpg')

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
    train_loader, valid_loader = build_dataloader(**cfg['dataset_cfg'], augment_fn=tr)
 
    logger.info('create model')
    model = build_model(**cfg['model_cfg'])
    model.cuda()
    logger.info('create loss function')
    fn_loss = torch.nn.MSELoss()

    logger.info('create optimizer')
    max_epoch = cfg['train_cfg']['max_epoch']
    valid_ecpoh = cfg['train_cfg']['validation_every_n_epoch']
    logger.info(f'max_epoch :{max_epoch}')
    logger.info('set mlflow tracking')
    mltracker = MLLogger(cfg, logger)
    test(model, valid_loader, fn_loss,  mltracker)
       
        