import argparse
import json
from tqdm import tqdm
import numpy as np
import cv2
import torch
from modules.utils import build_transformer
from modules.dataset.ocrDataset3ch import OCRDataset3ch
from torch.utils.data import DataLoader
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
    train_dataset = OCRDataset3ch(**cfg['dataset_cfg']['valid'], transformer=tr)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)
    for data in tqdm(train_loader):
        #data = train_dataset.__getitem__(np.random.randint(len(train_dataset)))
        print(data['img'].shape)
        #cv2.imwrite(f'./samples/{i}.jpg',data['img'].astype(np.uint8))
        
