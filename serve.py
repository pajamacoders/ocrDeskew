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

def correct_direction_and_visualize(data, logit, deskew_deg):
    degrees = [-90,0,90,180]
    mean, std = data['mean'], data['std']
    img = data['img'].squeeze()
    img = img*std+mean
    img = img.data.cpu().numpy().copy().astype(np.uint8)
    
    prob = torch.softmax(logit, -1)
    cls = torch.argmax(prob, -1)
    prob = prob.data.cpu().numpy().copy()

    rotation = degrees[cls]
    h,w = img.shape
    m = cv2.getRotationMatrix2D((w/2, h/2), -rotation+deskew_deg, 1)
    dst = cv2.warpAffine(img, m, (w,h))
    resimg = cv2.hconcat([img, dst])
    name = os.path.basename(data['imgpath'])
    if 'direction' in data.keys():
        gt_direction = degrees[data['direction']]
        save_name = f'pred_{prob[0,cls.item()]:.2f}_{rotation:.2f}_gt_{gt_direction:.2f}_{name}'
    else:
        save_name = f'pred_{prob[0,cls.item()]:.2f}_{rotation:.2f}_{name}'
    cv2.imwrite('./samples/'+save_name, resimg)


def visualizer_for_single_input(data, logit, logger, deskew_deg):
    mean, std = data['mean'], data['std']
    img = data['img'].squeeze()
    img = img*std+mean
    img = img.data.cpu().numpy().copy().astype(np.uint8)
    gt_flip = data['flip']
    prob = torch.sigmoid(logit[0])
    pred_flip = (prob>0.5).int()

    rotation = 180 if pred_flip else 0
    h,w = img.shape
    m = cv2.getRotationMatrix2D((w/2, h/2), rotation+deskew_deg, 1)
    dst = cv2.warpAffine(img, m, (w,h))
    resimg = cv2.hconcat([img, dst])
    name = os.path.basename(data['imgpath'])
    save_name = f'pred_{rotation:.2f}_gt_{gt_flip:.2f}_{name}'
    if logger:
        logger.log_image(resimg, name=save_name)

def parse_args():
    parser = argparse.ArgumentParser(description="handle arguments")
    parser.add_argument("--deskew_config", help="path to the configuration file", type=str, default='config/renet_ocr.json')
    parser.add_argument("--orientation_config", help="path to the configuration file", type=str, default='config/renet_ocr.json')
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

    with open(args.orientation_config, 'r') as f:
        orientation_cfg = json.load(f)
        if args.run_name:
            orientation_cfg['mllogger_cfg']['run_name'] = args.run_name
    
    tr = build_transformer(cfg['transform_cfg'])
    logger.info('create model')
    deskew_model = build_model(**cfg['model_cfg'])
    orientation_model = build_model(**orientation_cfg['model_cfg'])
    deskew_model.cuda()
    deskew_model.eval() # 이게 없으면 "expected more than 1 value per channel when training, got input size" 에러 발생
    orientation_model.cuda()
    orientation_model.eval()
    data_root = './test_file/*'
    imgs = glob.glob(data_root)
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
        rad_range = np.deg2rad(cfg['transform_cfg']['RandomRotation']['degree']) 
        rotation = rot_id*rad_range*2/cfg['transform_cfg']['RandomRotation']['buckets']-rad_range
        if isinstance(rotation, torch.Tensor):
            rotation = np.rad2deg(rotation.item())
        deskew_img = ttf.rotate(data['img'], angle=-rotation)
        out = orientation_model(deskew_img.cuda())
        correct_direction_and_visualize(data, out, -rotation)

        



       
        