import os
import argparse
import json
from tqdm import tqdm
import numpy as np
import cv2
import torch
from modules.utils import build_transformer
from modules.dataset.ocrDataset import OCRDataset
from torch.utils.data import DataLoader
from modules.dataset import FontDataSet


def parse_args():
    parser = argparse.ArgumentParser(description="handle arguments")
    parser.add_argument("--config", help="path to the configuration file", type=str, default='config/renet_ocr.json')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = json.load(f)
        cfg['config_file']=args.config
    
    tr = build_transformer(cfg['transform_cfg'])
    train_dataset = OCRDataset(**cfg['dataset_cfg']['valid'], transformer=tr)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)
    output_dir = './pipeline_test_result'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if 0:
        i=0
        for data in tqdm(train_loader):
            #data = train_dataset.__getitem__(np.random.randint(len(train_dataset)))
            print(data['img'].shape)
            cv2.imwrite(f'{output_dir}/{i}.jpg',data['gray'].astype(np.uint8))
            i+=0
    else:
         for i in tqdm(range(len(train_dataset))):
            data = train_dataset.__getitem__(np.random.randint(len(train_dataset)))
            color = data['img']
            # gray = data['gray']
            # deg = data['degree']
            # h,w,c = data['img'].shape
            # matrix = cv2.getRotationMatrix2D((w/2, h/2), -deg, 1)
            # border = np.random.randint(128,256)
            # dst = cv2.warpAffine(color, matrix, (w, h),borderValue = (0,0,0))
            # merge = cv2.hconcat([color,dst])
            cv2.imwrite(f'{output_dir}/{i}.jpg',color.astype(np.uint8))
 
            print(f"color:{color.shape}, gray:{color.shape}")
           
    
