from modules.utils import build_transformer
from modules.dataset import build_dataloader
import cv2
import os
import torch
import logging
logger = logging.getLogger('deskew')
logger.setLevel(logging.DEBUG)

def visualizer(path, img, ):
    img = data['img'].data.numpy().copy().squeeze()
    degree = -data['degree'].item()
    h,w = img.squeeze().shape
    m = cv2.getRotationMatrix2D((w/2, h/2), degree, 1)
    dst = cv2.warpAffine(img, m, (w,h))
    name = os.path.basename(data['imgpath'][-1]).replace('.jpg', '_rev.jpg')
    cv2.imwrite(f'vis/{name}', dst)

def train(model, loader, fn_loss, optimizer):
    for i, data in enumerate(loader):
        for k,v in data.items():
            if isinstance(v, torch.Tensor):
                data[k]=v.cuda()
        logit = model(data['img'])
        loss = fn_loss(logit, data['degree'])
        loss.backward()
        
        if (i+1)%10==0:
            break
if __name__ == "__main__":
    
    root = '/data/**/*.jpg'
    tr = build_transformer()
    train_loader, valid_loader = build_dataloader(**{'type':'OCRDataset', 
    'train':{'dataroot':'/data/**/*.jpg', 'transformer': tr},
    'valid':{'dataroot':'/data/**/*.jpg', 'transformer':tr}
    })
    max_epoch = 300
    logger.info('create model')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
    logger.info('create loss function')
    fn_loss = torch.nn.MSELoss()
    logger.info('create optimizer')
    opt=torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epoch, eta_min=1e-6, last_epoch=max_epoch)

    logger.info('')

    for i, data in enumerate(train_loader):
      
        if (i+1)%10==0:
            break
    