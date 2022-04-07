from functools import partial
import torch
import logging
import argparse
import json
from tqdm import tqdm
from modules.utils import build_transformer, MLLogger, build_lr_scheduler
from modules.dataset import build_dataloader
from modules.model import build_model
from modules.loss import FocalLoss
from sklearn.metrics import precision_recall_fscore_support
from utils import parse_rotation_prediction_outputs, parse_orientation_prediction_outputs, visualize_rotation_corrected_image, visualize_orientation_prediction_outputs
from functools import partial
logger = logging.getLogger('deskew')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
minloss = 999

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
        data['gray']=data['gray'].cuda(non_blocking=True).float()
        if key_target in data.keys():
            labels+=data[key_target].tolist()
            data[key_target]= data[key_target].cuda(non_blocking=True).float()
            if isinstance(fn_cls_loss, (FocalLoss,torch.nn.CrossEntropyLoss)):
                data[key_target]=data[key_target].long()

        with torch.no_grad():
            cls_logit = model(data['gray'])

        cls_loss = fn_cls_loss(cls_logit, data[key_target])
        total_loss = cls_loss
        avg_total_loss+=total_loss.detach()*cls_logit.shape[0]
        avg_cls_loss += cls_loss.detach()*cls_logit.shape[0]
        num_samples+=cls_logit.shape[0]
        if prediction_parser:
            preds+=prediction_parser(cls_logit).tolist()

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
    logger.info(log_msg)

    if minloss > avgloss:
        mllogger.log_state_dict(step, model, isbest=True)

    vis_func(data, cls_logit, mllogger, step=step)
    
    
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
        data['gray']=data['gray'].cuda(non_blocking=True).float()
        if key_target in data.keys():
            labels+=data[key_target].tolist()
            data[key_target]= data[key_target].cuda(non_blocking=True).float()
            if isinstance(fn_cls_loss, (FocalLoss,torch.nn.CrossEntropyLoss)):
                data[key_target]=data[key_target].long()
                
        cls_logit = model(data['gray'])

        cls_loss = fn_cls_loss(cls_logit, data[key_target])
        total_loss = cls_loss
        avg_total_loss+=total_loss.detach()*cls_logit.shape[0]
        avg_cls_loss += cls_loss.detach()*cls_logit.shape[0]
        num_samples+=cls_logit.shape[0]
        total_loss.backward()
        optimizer.step()
        if prediction_parser:
            preds+=prediction_parser(cls_logit).tolist()

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
    logger.info(log_msg)
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
    

    logger.info('create optimizer')
    opt=torch.optim.Adam(model.parameters(), **cfg['optimizer_cfg']['args'])
    lr_scheduler = build_lr_scheduler(**cfg['lr_scheduler_cfg'], opt=opt)#torch.optim.lr_scheduler.CosineAnnealingLR(opt,**cfg['lr_scheduler_cfg']['args'])

    max_epoch = cfg['train_cfg']['max_epoch']
    valid_ecpoh = cfg['train_cfg']['validation_every_n_epoch']
    logger.info(f'max_epoch :{max_epoch}')
    logger.info('set mlflow tracking')
    mltracker = MLLogger(cfg, logger)

    if cfg['task']=='deskew':
        vis_func = partial(visualize_rotation_corrected_image,info=cfg['transform_cfg']['RandomRotation'])
        prediction_parser = parse_rotation_prediction_outputs
        key_metric = 'rot_id'
        fn_cls_loss = FocalLoss(None,2.0) #torch.nn.CrossEntropyLoss()
    else:
        vis_func = visualize_orientation_prediction_outputs
        prediction_parser = parse_orientation_prediction_outputs
        key_metric = 'flip'
        fn_cls_loss = torch.nn.BCEWithLogitsLoss()      
    
    for step in range(max_epoch):
        train(model, train_loader, fn_cls_loss, key_metric, opt, mltracker, step, prediction_parser)
        if (step+1)%valid_ecpoh==0:
            valid(model, valid_loader, fn_cls_loss, key_metric,  mltracker, step, vis_func, prediction_parser)
        lr_scheduler.step()
        mltracker.log_metric(key='learning_rate', value=opt.param_groups[0]['lr'], step=step)

    
        