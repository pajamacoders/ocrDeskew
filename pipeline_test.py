import argparse
import json
from tqdm import tqdm
import numpy as np
import cv2
from modules.utils import build_transformer
from modules.dataset.ocrDataset3ch import OCRDataset3ch

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
    train_dataset = OCRDataset3ch(**cfg['dataset_cfg']['train'], transformer=tr)
    for i in tqdm(range(len(train_dataset))):
        data = train_dataset.__getitem__(np.random.randint(len(train_dataset)))
        print(data['img'].shape)
        cv2.imwrite(f'./samples/{i}.jpg',data['img'].astype(np.uint8))
        
