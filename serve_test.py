import cv2
import os
import torch
import logging
import argparse
import json
import glob
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as ttf
from modules.utils import build_transformer, MLLogger
from modules.model import build_model
from utils import cvt2HeatmapImg
import torchvision
logger = logging.getLogger('deskew')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
minloss = 999
blur = torchvision.transforms.GaussianBlur(7, sigma=(3.0, 3.0))
def visualizer_for_single_input(data, logit, logger, deskew_deg, prob):
    mean, std = data['mean'], data['std']
    img = data['img'].squeeze()
    if len(img.shape)==3:
        img = img.permute(1,2,0)
    img = img.cpu()*std+mean
    img = img.data.numpy().copy().astype(np.uint8)
    if len(img.shape)==3:
            h,w,c = img.shape
    else:
        h,w = img.shape
    m = cv2.getRotationMatrix2D((w/2, h/2), deskew_deg, 1)
    dst = cv2.warpAffine(img, m, (w,h))
    if 'text_score' in data.keys():
        mask = data['text_score'].squeeze().cpu().numpy().copy()
        mask = cvt2HeatmapImg(mask)
        mask=cv2.resize(mask,(w,h), interpolation=cv2.INTER_LINEAR)
        resimg=cv2.hconcat([img, dst, mask])
    else:
        resimg=cv2.hconcat([img, dst])
    # resimg = cv2.hconcat([img, dst])
    name = os.path.basename(data['imgpath'])
    gt_deg = data['degree']
    save_name = f'pred_{prob.item():.2f}_{rotation:.2f}_gt_{gt_deg:.2f}_{name}'
    if logger:
        logger.log_image(resimg, name=save_name)

def visualizer(data, logit,):
    mean, std = data['mean'], data['std']
    img = data['img'].squeeze()
    img = img*std+mean
    img = img.data.cpu().numpy().copy().astype(np.uint8)
    rot_id = torch.argmax(torch.softmax(logit,-1),-1)
    rad_range = np.deg2rad(89) #89 와 356 은 모델 training 시에 사용된 constant
    rotation = rot_id*rad_range*2/356-rad_range
    if isinstance(rotation, torch.Tensor):
        rotation = np.rad2deg(rotation.item())

    h,w = img.shape
    m = cv2.getRotationMatrix2D((w/2, h/2), -rotation, 1)
    dst = cv2.warpAffine(img, m, (w,h))
    resimg=cv2.hconcat([img, dst])
    cv2.imwrite('account1_30.png',resimg)


def parse_args():
    parser = argparse.ArgumentParser(description="handle arguments")
    parser.add_argument("--deskew_config", help="path to the configuration file", type=str, default='config/renet_ocr.json')
    parser.add_argument("--run_name", help="run name for mlflow tracking", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.deskew_config, 'r') as f:
        cfg = json.load(f)
        cfg['config_file'] = args.deskew_config
        if args.run_name:
            cfg['mllogger_cfg']['run_name'] = args.run_name

    tr = build_transformer(cfg['transform_cfg'])
    logger.info('create model')
    deskew_model = build_model(**cfg['model_cfg'])
    deskew_model.cuda()
    deskew_model.eval() # 이게 없으면 "expected more than 1 value per channel when training, got input size" 에러 발생
    craft = None
    if 'feature_ext_model_cfg' in cfg.keys():
        logger.info('create text detection model')
        craft = build_model(**cfg['feature_ext_model_cfg'])
        craft = craft.cuda()
        craft.eval()
    mltracker = MLLogger(cfg, logger)
    root = '/train_data/valid/*' #'./test_file/*'
    imgs = glob.glob(root)
    i=0
    for impath in tqdm(imgs):
        img = cv2.imread(impath)
        if len(img.shape) == 2: 
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = 255-img
        inp={'img':255-img, 'imgpath': impath}
        
        data = tr(inp) # resize and change shape
        data['img'] = torch.from_numpy(data['img']).float()
        data['gray'] = torch.from_numpy(data['gray']).float()
        if len(data['img'].shape)<4:
            data['img'] = data['img'].unsqueeze(0)
        with torch.no_grad():
            in_img = data['img'].cuda()#torch.nn.functional.interpolate(data['img'].cuda(), size=(768,768), mode='bilinear')
            if craft:
                y,feature = craft(in_img)
                mask = torch.nn.functional.interpolate(y[:,:,:,0].unsqueeze(1),size=(512,512), mode='bilinear')
                mask=blur(mask)
                data['text_score']=mask
                if len(data['gray'])!=4:
                    data['gray'] = data['gray'].unsqueeze(0)
                in_img = torch.concat([data['gray'].cuda(), mask],dim=1)
                
                # score_text = y[0,:,:,0].cpu().data.numpy()
                # ret_score_text = cvt2HeatmapImg(score_text)
                # cv2.imwrite(f'./samples/{i}.jpg',ret_score_text )
                # i+=1
            out = deskew_model(in_img)
        prob = torch.softmax(out,-1)
        rot_id = torch.argmax(prob,-1)
        rad_range = np.deg2rad(90) 
        rotation = rot_id*rad_range*2/360-rad_range
        if isinstance(rotation, torch.Tensor):
            rotation = np.rad2deg(rotation.item())
        visualizer_for_single_input(data, out, mltracker, -rotation, prob[0][rot_id])



       
        