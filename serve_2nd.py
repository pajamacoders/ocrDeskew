import cv2
import os
import torch
import logging
import argparse
import json
import glob
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tqdm import tqdm
from modules.utils import build_transformer, MLLogger
from modules.model import build_model
from utils import cvt2HeatmapImg
from utils import craft_utils
from utils import imgproc
from utils import file_utils
from collections import Counter
import random
from sklearn.metrics import classification_report
import pickle

logger = logging.getLogger('deskew')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
minloss = 999

def visualizer_for_single_input(org_img, deskew, corrected_img, logger):
    resimg=cv2.hconcat([org_img, deskew, corrected_img])
    h,w,c = resimg.shape
    resimg = cv2.resize(resimg, (w//2, h//2), interpolation=cv2.INTER_AREA)
    name = os.path.basename(data['imgpath'])
    save_name = f'{name}'
    logger.log_image(resimg, name=save_name)

def getDegreeFromDeskewOuput(cls):
    cls = cls.squeeze()
    rad_range = np.deg2rad(90)
    degree = cls*rad_range*2/360-rad_range
    if isinstance(degree, torch.Tensor):
        degree = np.rad2deg(degree.item())
    return degree
    
def rotateImage(img, deg, border=255):
    if len(img.shape)==3:
        h,w,c = img.shape
    else:
        h,w = img.shape
    m = cv2.getRotationMatrix2D((w/2, h/2), deg, 1)
    dst = cv2.warpAffine(img, m, (w,h),borderValue = border)
    return dst


def getCharacterPatches(net, image, text_threshold, link_threshold, low_text):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    # org, image = image
    # if len(image.shape)<3:
    #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))   # [h, w, c] to [b, c, h, w]
    if torch.cuda.is_available():
        x = x.cuda()
    # forward pass
    with torch.no_grad():
        y, feature = net(x)
    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    # Post-processing
    boxes = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    patches = file_utils.getCharacterPatchFromImage(image[:,:,::-1], boxes, max_num_patch=256)
    return patches

def preprocesssin(img, canvas_size, mag_ratio):
    h,w,c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    if max(h,w)/canvas_size >= 2:
        inv_gray = cv2.dilate(255-gray, k)
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio_gray(255-inv_gray, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    else:
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio_gray(gray, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    _, bin_img = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY)
    im2, contour, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, \
                                                cv2.CHAIN_APPROX_NONE)
    white_img = 255-np.zeros(img_resized.shape, dtype=np.uint8)
    mask = np.zeros(img_resized.shape, dtype=np.uint8)
    mask = cv2.fillPoly(mask, pts=contour, color=1, lineType=cv2.LINE_8)

    white_img[mask!=0] = img_resized[mask!=0]
  
    return img, white_img

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
        cfg['config_file'] = args.config
        if args.run_name:
            cfg['mllogger_cfg']['run_name'] = args.run_name

    tr = build_transformer(cfg['transform_cfg'])
    logger.info('create model')
    # load deskew model
    deskew_model = build_model(**cfg['deskew_model_cfg'])

    # 이게 없으면 "expected more than 1 value per channel when training, got input size" 에러 발생
    # load character extraciont model
    logger.info('create text detection model')
    craft = build_model(**cfg['craft_model_cfg'])

    # load detection correction model
    logger.info('create direction correction model')
    stdNet = build_model(**cfg['direction_model_cfg'])

    if torch.cuda.is_available():
        deskew_model = deskew_model.cuda()
        craft = craft.cuda()
        stdNet = stdNet.cuda()  
    deskew_model.eval()
    # craft = torch.nn.DataParallel(craft)
    # cudnn.benchmark = False
    craft.eval()
    stdNet.eval()
    # init logger
    logger.info('init logger')
    mltracker = MLLogger(cfg, logger)
    root = '/train_data/valid/*' 
    imgs = glob.glob(root)
    i=0
    degrees = [0,90,180,270]
    gt_list=[]
    pred_list=[]
    gt_degree = []
    abs_errors=[]
    for iter in range(2):
        for impath in tqdm(imgs):
            img = cv2.imread(impath)
            
            gt_list.append(0)
            org_img, rm_bg_img  = preprocesssin(img,1280, 1.5)
            
            if len(rm_bg_img.shape) >2: 
                rm_bg_img = cv2.cvtColor(rm_bg_img, cv2.COLOR_BGR2GRAY)
            inp={'img':255-rm_bg_img, 'imgpath': impath}
            
            data = tr(inp) # resize and change shape
            gt_degree.append(data['degree'])
            org_img = rotateImage(org_img, data['degree'], (255,255,255) if np.random.rand()<0.5 else (0,0,0))
           
            data['img'] = torch.from_numpy(data['img']).float()
            if len(data['img'].shape)<4:
                data['img'] = data['img'].unsqueeze(0)
            #deskew
            if torch.cuda.is_available():
                data['img'] = data['img'].cuda(non_blocking=True)
                with torch.no_grad():
                    deskew_cls = deskew_model.predict(data['img'])
                deskew_deg = getDegreeFromDeskewOuput(deskew_cls)
                #deg_orgimg2_deskewimg = int(deskew_deg)#int(data['degree']-deskew_deg)
                dsk_img = rotateImage(org_img, -deskew_deg, (255,255,255))
                #get character patches

                patches= getCharacterPatches(craft, dsk_img, 0.8, 0.5, 0.5)
                if patches:
                    patches2tensor = [torch.from_numpy(patch).unsqueeze(0).float()/255.0 for patch in patches]
                    in_img = torch.stack(patches2tensor, dim=0)
                    if torch.cuda.is_available():
                        in_img = in_img.cuda(non_blocking=True)
                    with torch.no_grad():
                        dir_cls = stdNet.predict(in_img)
                    candidates = Counter(dir_cls.tolist())
                    pred = candidates.most_common(1)[0][0]
                    degree = degrees[pred]
                    result_deg = int(abs(data['degree']-deskew_deg-degree))
                    if result_deg in [0, 1,2, 358,359, 360, 361,362]:
                        pred_list.append(0)
                    else:
                        abs_errors.append(result_deg)
                        pred_list.append(1)
                        filename = os.path.basename(impath)
                        try:
                            tmp_degree = data['degree']
                            cv2.imwrite(f'issuefiles/{iter}_deg_{tmp_degree:.2f}_{filename}', img)
                        except OSError:
                            pass
                    corrected_img = rotateImage(dsk_img, -degree, (255,255,255))
                    visualizer_for_single_input(org_img, dsk_img, corrected_img, mltracker)
                else:
                    pred_list.append(0 if deskew_cls==data['rot_id'] else 1)
                    filename = os.path.basename(impath)
                    try:
                        tmp_degree = data['degree']
                        cv2.imwrite(f'issuefiles/{iter}_deg_{tmp_degree:.2f}_{filename}', img)
                    except OSError:
                        pass

    with open("abs_errors.pickle","wb") as fw:
        pickle.dump(abs_errors, fw)
    with open("gt_degree.pickle","wb") as fw:
        pickle.dump(gt_degree, fw)
    with open("pred_list.pickle","wb") as fw:
        pickle.dump(pred_list, fw)
    with open("gt_cls_list.pickle","wb") as fw:
        pickle.dump(gt_list, fw)
    report = classification_report(gt_list, pred_list, target_names=['0', '1'])
    with open('report_deskew_and_text_direction_correction.txt', 'w') as f:
        print(report, file=f)





       
        