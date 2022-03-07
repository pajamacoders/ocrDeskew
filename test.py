import cv2
import os
import torch
import logging
import argparse
import json
import numpy as np
from tqdm import tqdm
from modules.utils import build_transformer, MLLogger
from modules.dataset import build_dataloader
from modules.model import build_model
from sklearn.metrics import precision_recall_fscore_support
from utils import parse_rotation_prediction_outputs, parse_orientation_prediction_outputs, visualize_rotation_corrected_image, visualize_orientation_prediction_outputs
from functools import partial

logger = logging.getLogger('deskew')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
minloss = 999


def test(model, loader, fn_cls_loss, key_target, mllogger, vis_func, prediction_parser=None):
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
            if isinstance(fn_cls_loss, torch.nn.CrossEntropyLoss):
                data[key_target]=data[key_target].long()

        with torch.no_grad():
            cls_logit = model(data['img'])
        if not cls_logit.shape == data[key_target].shape:
            cls_logit=cls_logit.reshape(data[key_target].shape)
        cls_loss = fn_cls_loss(cls_logit, data[key_target])
        total_loss = cls_loss
        avg_total_loss+=total_loss.detach()*cls_logit.shape[0]
        avg_cls_loss += cls_loss.detach()*cls_logit.shape[0]
        num_samples+=cls_logit.shape[0]
        preds+=prediction_parser(cls_logit).tolist()
        vis_func(data, cls_logit, mllogger)

    avgloss =  (avg_total_loss/num_samples).item()
    stat_cls_loss =  (avg_cls_loss/num_samples).item()
    precision, recall, f1_score, support = precision_recall_fscore_support(labels, preds, average='macro')
    mllogger.log_metric('test_loss',avgloss, 0)
    mllogger.log_metric('test_cls_loss',stat_cls_loss, 0)
    mllogger.log_metric('test_precision',precision, 0)
    mllogger.log_metric('test_recall',recall, 0)
    mllogger.log_metric('test_f1_score',f1_score, 0)
    logger.info(f'test-{0} epoch: cls_loss:{stat_cls_loss:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1_score:{f1_score:.4f},support:{support}.')


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

    max_epoch = cfg['train_cfg']['max_epoch']
    valid_ecpoh = cfg['train_cfg']['validation_every_n_epoch']
    logger.info(f'max_epoch :{max_epoch}')
    logger.info('set mlflow tracking')
    mltracker = MLLogger(cfg, logger)
    if cfg['task']=='deskew':
        vis_func = partial(visualize_rotation_corrected_image,info=cfg['transform_cfg']['RandomRotation'])
        prediction_parser = parse_rotation_prediction_outputs
        key_metric = 'rot_id'
        fn_cls_loss = torch.nn.CrossEntropyLoss()
    else:
        vis_func = visualize_orientation_prediction_outputs
        prediction_parser = parse_orientation_prediction_outputs
        key_metric = 'flip'
        fn_cls_loss = torch.nn.BCEWithLogitsLoss()

    test(model, valid_loader, fn_cls_loss, key_metric, mltracker, vis_func, prediction_parser)
       
        