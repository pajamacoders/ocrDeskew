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

logger = logging.getLogger('deskew')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
minloss = 999

def visualizer_for_single_input(data, logit, logger, deskew_deg):
    mean, std = data['mean'], data['std']
    img = data['img'].squeeze()
    img = img*std+mean
    img = img.data.cpu().numpy().copy().astype(np.uint8)
    h,w = img.shape
    m = cv2.getRotationMatrix2D((w/2, h/2), deskew_deg, 1)
    dst = cv2.warpAffine(img, m, (w,h))
    resimg = cv2.hconcat([img, dst])
    name = os.path.basename(data['imgpath'])
    save_name = f'pred_{rotation:.2f}_{name}'
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
    mltracker = MLLogger(cfg, logger)
    imgs = glob.glob('./test_file/*')
    for impath in tqdm(imgs):
        img = cv2.imread(impath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = 255-img
        inp={'img':img, 'imgpath': impath}
        
        data = tr(inp) # resize and change shape
        data['img'] = torch.from_numpy(data['img']).float()
        if len(data['img'].shape)<4:
            data['img'] = data['img'].unsqueeze(0)

        out = deskew_model(data['img'].cuda())

        rot_id = torch.argmax(torch.softmax(out,-1),-1)
        rad_range = np.deg2rad(89) 
        rotation = rot_id*rad_range*2/356-rad_range
        if isinstance(rotation, torch.Tensor):
            rotation = np.rad2deg(rotation.item())
        visualizer_for_single_input(data, out, mltracker, -rotation)



       
        