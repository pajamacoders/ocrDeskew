import cv2
import os
import torch
import logging
import argparse
import json
import numpy as np
from tqdm import tqdm
from modules.utils import build_transformer, MLLogger, build_lr_scheduler
from modules.dataset import build_dataloader, OCRDataset
from modules.model import build_model

logger = logging.getLogger('deskew')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
minloss = 999
def visualizer(data, logit):
    ind = np.random.randint(0, data['img'].shape[0])

    mean, std = data['mean'][ind], data['std'][ind]
    img = data['img'][ind].squeeze()
    img = img*std+mean
    img = img.data.cpu().numpy().copy().astype(np.uint8)
    if len(logit)>1:
        rotation, cls = -logit[0][ind].item(), logit[1][ind].sigmoid().item()
    else:
        rotation=-logit[ind].item()
        cls = 0

    h,w = img.shape
    m = cv2.getRotationMatrix2D((w/2, h/2), rotation, 1)
    dst = cv2.warpAffine(img, m, (w,h))
    #name = os.path.basename(data['imgpath'][-1]).replace('.jpg', '_rev.jpg')
    resimg=cv2.hconcat([img, dst])
    return resimg, cls
    



def valid(model, loader, fn_reg_loss,fn_cls_loss, mllogger, step):
    global minloss
    model.eval()
    avg_total_loss = 0
    avg_reg_loss = 0
    avg_cls_loss = 0
    num_samples=0

    for i, data in tqdm(enumerate(loader)):
        data['img']=data['img'].cuda(non_blocking=True).float()
        data['degree']=data['degree'].cuda(non_blocking=True).float()
        if 'cls' in data.keys():
            data['cls'] = data['cls'].cuda(non_blocking=True).float()

        with torch.no_grad():
            reg_logit, cls_logit = model(data['img'])
        reg_logit = reg_logit.squeeze()
        cls_logit = cls_logit.squeeze()
        loss = fn_reg_loss(reg_logit, data['degree'])
        cls_loss = fn_cls_loss(cls_logit, data['cls'])
        total_loss = 0.1*loss +cls_loss

        avg_total_loss+=total_loss.detach()*reg_logit.shape[0]
        avg_reg_loss += loss.detach()*reg_logit.shape[0]
        avg_cls_loss += cls_loss.detach()*reg_logit.shape[0]
        num_samples+=reg_logit.shape[0]

    avgloss =  (avg_total_loss/num_samples).item()
    stat_reg_loss =  (avg_reg_loss/num_samples).item()
    stat_cls_loss =  (avg_cls_loss/num_samples).item()
    mllogger.log_metric('valid_loss',avgloss, step)
    mllogger.log_metric('reg_loss',stat_reg_loss, step)
    mllogger.log_metric('cls_loss',stat_cls_loss, step)
    logger.info(f'valid-{step} epoch: total_loss:{avgloss:.4f}, reg_loss:{stat_reg_loss:.4f}, cls_loss:{stat_cls_loss:.4f}')

    if minloss > avgloss:
        mllogger.log_state_dict(step, model, isbest=True)

    img, cls = visualizer(data, [reg_logit,cls_logit] )
    mllogger.log_image(img, name=f'cls_{cls:.2f}_{step}_sample.jpg')
    
    

def train(model, loader, fn_reg_loss, fn_cls_loss, optimizer, mllogger, step):
    model.train()
    avg_total_loss = 0
    avg_reg_loss = 0
    avg_cls_loss = 0
    num_samples=0
    total_loss = 0
    for i, data in tqdm(enumerate(loader)):
        optimizer.zero_grad()
        data['img'] = data['img'].cuda(non_blocking=True).float()
        data['degree'] = data['degree'].cuda(non_blocking=True).float()
        if 'cls' in data.keys():
            data['cls'] = data['cls'].cuda(non_blocking=True).float()
                
        reg_logit, cls_logit = model(data['img'])
        reg_logit = reg_logit.squeeze()
        cls_logit = cls_logit.squeeze()
        loss = fn_reg_loss(reg_logit, data['degree'])
        cls_loss = fn_cls_loss(cls_logit, data['cls'])

        total_loss = 0.1*loss +cls_loss

        avg_total_loss+=total_loss.detach()*reg_logit.shape[0]
        avg_reg_loss += loss.detach()*reg_logit.shape[0]
        avg_cls_loss += cls_loss.detach()*reg_logit.shape[0]
        num_samples+=reg_logit.shape[0]
        total_loss.backward()
        optimizer.step()

    avgloss =  (avg_total_loss/num_samples).item()
    stat_reg_loss =  (avg_reg_loss/num_samples).item()
    stat_cls_loss =  (avg_cls_loss/num_samples).item()
    mllogger.log_metric('train_loss',avgloss, step)
    mllogger.log_metric('reg_loss',stat_reg_loss, step)
    mllogger.log_metric('cls_loss',stat_cls_loss, step)
    logger.info(f'train-{step} epoch: total_loss:{avgloss:.4f}, reg_loss:{stat_reg_loss:.4f}, cls_loss:{stat_cls_loss:.4f}')
    mllogger.log_state_dict(step, model, isbest=False)

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
    fn_reg_loss = torch.nn.MSELoss()
    fn_cls_loss = torch.nn.BCEWithLogitsLoss()

    logger.info('create optimizer')
    opt=torch.optim.Adam(model.parameters(), **cfg['optimizer_cfg']['args'])
    lr_scheduler = build_lr_scheduler(**cfg['lr_scheduler_cfg'], opt=opt)#torch.optim.lr_scheduler.CosineAnnealingLR(opt,**cfg['lr_scheduler_cfg']['args'])

    max_epoch = cfg['train_cfg']['max_epoch']
    valid_ecpoh = cfg['train_cfg']['validation_every_n_epoch']
    logger.info(f'max_epoch :{max_epoch}')
    logger.info('set mlflow tracking')
    mltracker = MLLogger(cfg, logger)
    for step in range(max_epoch):
        train(model, train_loader, fn_reg_loss, fn_cls_loss, opt, mltracker, step)
        if (step+1)%valid_ecpoh==0:
            valid(model, valid_loader, fn_reg_loss, fn_cls_loss,  mltracker, step)
        lr_scheduler.step()
        mltracker.log_metric(key='learning_rate', value=opt.param_groups[0]['lr'], step=step)
    
        