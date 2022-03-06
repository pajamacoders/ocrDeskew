from functools import partial
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
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger('deskew')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
minloss = 999

def flip_correction_visualizer(data, logit):
    ind = np.random.randint(0, data['img'].shape[0])
    mean, std = data['mean'][ind], data['std'][ind]
    img = data['img'][ind].squeeze()
    img = img*std+mean
    img = img.data.cpu().numpy().copy().astype(np.uint8)
    gt_flip = data['flip'][ind].data.cpu().numpy()
    prob = torch.sigmoid(logit[ind])
    pred_flip = (prob>0.5).int()

    rotation = 180 if pred_flip else 0
    h,w = img.shape
    m = cv2.getRotationMatrix2D((w/2, h/2), rotation, 1)
    dst = cv2.warpAffine(img, m, (w,h))
    resimg = cv2.hconcat([img, dst])
    return resimg, prob, gt_flip



def visualizer(data, logit, info):
    ind = np.random.randint(0, data['img'].shape[0])

    mean, std = data['mean'][ind], data['std'][ind]
    img = data['img'][ind].squeeze()
    img = img*std+mean
    img = img.data.cpu().numpy().copy().astype(np.uint8)
    rot_id = torch.argmax(torch.softmax(logit,-1),-1)[ind]
    rad_range = np.deg2rad(info['degree'])
    gt_deg = data['degree'][ind].item()
    rotation = rot_id*rad_range*2/info['buckets']-rad_range
    if isinstance(rotation, torch.Tensor):
        rotation = np.rad2deg(rotation.item())
    h,w = img.shape
    m = cv2.getRotationMatrix2D((w/2, h/2), -rotation, 1)
    dst = cv2.warpAffine(img, m, (w,h))
    #name = os.path.basename(data['imgpath'][-1]).replace('.jpg', '_rev.jpg')
    resimg=cv2.hconcat([img, dst])
    return resimg, rotation, gt_deg
    
def binary_prediction_outputs(cls_logit):
    return (torch.sigmoid(cls_logit)>0.5).int().tolist()

def multiclass_prediction_outputs(cls_logit):
    return torch.argmax(torch.softmax(cls_logit, -1), -1).tolist()


def valid(model, loader, fn_cls_loss, key_target, mllogger, step, vis_func, prediction_parser=None):
    global minloss
    model.eval()
    avg_total_loss = 0
    avg_cls_loss = 0
    num_samples=0
    labels = []
    preds = []
    for i, data in tqdm(enumerate(loader)):
        data['img']=data['img'].cuda(non_blocking=True).float()
        if key_target in data.keys():
            labels+=data[key_target].tolist()
            data[key_target]= data[key_target].cuda(non_blocking=True).float()

        with torch.no_grad():
            cls_logit = model(data['img'])
        cls_logit = cls_logit.squeeze()
    
        cls_loss = fn_cls_loss(cls_logit, data[key_target])
        total_loss = cls_loss
        avg_total_loss+=total_loss.detach()*cls_logit.shape[0]
        avg_cls_loss += cls_loss.detach()*cls_logit.shape[0]
        num_samples+=cls_logit.shape[0]
        if prediction_parser:
            preds+=prediction_parser(cls_logit)

    avgloss =  (avg_total_loss/num_samples).item()
    stat_cls_loss =  (avg_cls_loss/num_samples).item()
    
    mllogger.log_metric('valid_loss',avgloss, step)
    mllogger.log_metric('valid_cls_loss',stat_cls_loss, step)
    log_msg = f'valid-{step} epoch: cls_loss:{stat_cls_loss:.4f}'
    if prediction_parser:
        precision, recall, f1_score, support = precision_recall_fscore_support(labels, preds, average='macro')
        mllogger.log_metric('valid_precision',precision, step),
        mllogger.log_metric('valid_recall',recall, step)
        mllogger.log_metric('valid_f1_score',f1_score, step)
        log_msg+=f' precision:{precision:.4f}, recall:{recall:.4f}, f1_score:{f1_score:.4f},support:{support}.'

    if minloss > avgloss:
        mllogger.log_state_dict(step, model, isbest=True)

    img, pred_rot, gt_deg = vis_func(data, cls_logit)
    mllogger.log_image(img, name=f'{step}_sample_cls_{pred_rot:.2f}_gt_deg_{gt_deg:.2f}.jpg')
    
def train(model, loader, fn_cls_loss, key_target, optimizer, mllogger, step, prediction_parser=None):
    model.train()
    avg_total_loss = 0
    avg_cls_loss = 0
    num_samples=0
    total_loss = 0
    labels = []
    preds = []
    for i, data in tqdm(enumerate(loader)):
        optimizer.zero_grad()
        data['img'] = data['img'].cuda(non_blocking=True).float()
        data['img']=data['img'].cuda(non_blocking=True).float()
        if key_target in data.keys():
            labels+=data[key_target].tolist()
            data[key_target]= data[key_target].cuda(non_blocking=True).float()
                
        cls_logit = model(data['img'])
        cls_logit = cls_logit.squeeze()
        cls_loss = fn_cls_loss(cls_logit, data[key_target])
        total_loss = cls_loss
        avg_total_loss+=total_loss.detach()*cls_logit.shape[0]
        avg_cls_loss += cls_loss.detach()*cls_logit.shape[0]
        num_samples+=cls_logit.shape[0]
        total_loss.backward()
        optimizer.step()
        if prediction_parser:
            preds+=prediction_parser(cls_logit)

    avgloss =  (avg_total_loss/num_samples).item()
    stat_cls_loss =  (avg_cls_loss/num_samples).item()
   
    mllogger.log_metric('train_total_loss',avgloss, step)
    mllogger.log_metric('train_cls_loss',stat_cls_loss, step)
    log_msg = f'train-{step} epoch: cls_loss:{stat_cls_loss:.6f}'
    if prediction_parser:
        precision, recall, f1_score, support = precision_recall_fscore_support(labels, preds, average='macro')
        mllogger.log_metric('train_precision',precision, step)
        mllogger.log_metric('train_recall',recall, step)
        mllogger.log_metric('train_f1_score',f1_score, step)
    log_msg+=f', precision:{precision:.6f}, recall:{recall:.6f}, f1_score:{f1_score:.6f},support:{support}.'
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
    fn_cls_loss = torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss()

    logger.info('create optimizer')
    opt=torch.optim.Adam(model.parameters(), **cfg['optimizer_cfg']['args'])
    lr_scheduler = build_lr_scheduler(**cfg['lr_scheduler_cfg'], opt=opt)#torch.optim.lr_scheduler.CosineAnnealingLR(opt,**cfg['lr_scheduler_cfg']['args'])

    max_epoch = cfg['train_cfg']['max_epoch']
    valid_ecpoh = cfg['train_cfg']['validation_every_n_epoch']
    logger.info(f'max_epoch :{max_epoch}')
    logger.info('set mlflow tracking')
    mltracker = MLLogger(cfg, logger)
    vis_func = flip_correction_visualizer
    prediction_parser = binary_prediction_outputs
    key_metric = 'flip'
    for step in range(max_epoch):
        train(model, train_loader, fn_cls_loss, key_metric, opt, mltracker, step, prediction_parser)
        if (step+1)%valid_ecpoh==0:
            valid(model, valid_loader, fn_cls_loss, key_metric,  mltracker, step, vis_func, prediction_parser)
        lr_scheduler.step()
        mltracker.log_metric(key='learning_rate', value=opt.param_groups[0]['lr'], step=step)
    
        