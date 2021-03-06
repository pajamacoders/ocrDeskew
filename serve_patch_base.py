import cv2
import os
import torch
import logging
import argparse
import json
import glob
import numpy as np
from tqdm import tqdm
from modules.model import build_model
from utils import craft_utils
from utils import imgproc
from utils import file_utils
from collections import Counter
import random


logger = logging.getLogger('deskew')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
minloss = 999

def visualizer_for_single_input(corrected_img, impath):
    out_dir = './result_images'
    name = os.path.basename(impath)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    cv2.imwrite(out_dir+'/'+name, corrected_img)

def getDegreeFromDeskewOuput(cls):
    cls = cls.squeeze()
    rad_range = np.deg2rad(90)
    degree = cls*rad_range*2/360-rad_range
    if isinstance(degree, torch.Tensor):
        degree = np.rad2deg(degree.item())
    return degree
    
def rotateImage(img, deg):
    if len(img.shape)==3:
        h,w,c = img.shape
    else:
        h,w = img.shape
    m = cv2.getRotationMatrix2D((w/2, h/2), deg, 1)
    dst = cv2.warpAffine(img, m, (w,h),borderValue=(255,255,255) if len(img.shape)==3 else 255)
    return dst


def getCharacterPatches(net, image, text_threshold, link_threshold, low_text):
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    # if len(image.shape)<3:
    #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)    # [h, w, c] to [b, c, h, w]
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
    # gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    if max(h,w)//canvas_size > 2:
        inv_gray = cv2.dilate(255-gray, k)
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio_gray(255-inv_gray, canvas_size, interpolation=cv2.INTER_AREA, mag_ratio=mag_ratio)
    else:
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio_gray(gray, canvas_size, interpolation=cv2.INTER_AREA, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    _, bin_img = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY)
    im2, contour, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, \
                                                cv2.CHAIN_APPROX_NONE)
    white_img = 255-np.zeros(img_resized.shape, dtype=np.uint8)
    mask = np.zeros(img_resized.shape, dtype=np.uint8)
    mask = cv2.fillPoly(mask, pts=contour, color=1, lineType=cv2.LINE_8)
    white_img[mask!=0] = bin_img[mask!=0]
  
    return img, white_img

def resize_and_normalize_for_deskew_model(img, target_size):
    h,w = img.shape
    # inverse color for dilation
    img = 255-img
    # resize
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img = cv2.dilate(img, k)

    while (max(h,w)//target_size) > 1:
            h, w = h//2, w//2
            img=cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            h, w = img.shape
    h, w = img.shape
    ratio = target_size / max(h, w)    
    
    target_h, target_w = int(h * ratio), int(w * ratio)
    #resize aspect ration and pad
    rsz_img = cv2.resize(img, (target_w, target_h), interpolation =  cv2.INTER_AREA)

    if len(rsz_img.shape)==2:
        h,w = rsz_img.shape
        padded_img = np.zeros((target_size, target_size),dtype=np.float32)
    elif len(rsz_img.shape)==3:
        h,w,c = rsz_img.shape
        padded_img = np.zeros((target_size, target_size,c), dtype=np.float32)
    else:
        pass
    padded_img[:h,:w]=rsz_img

    #normalize
    padded_img=padded_img/255.0

    # change [h,w] to [1, h, w]
    padded_img = np.expand_dims(padded_img, axis=0)
    return padded_img


def inference(img)->int:
    cls2degree = [0,90,180,270]
    org_img, rm_bg_img = preprocesssin(img,1280, 1.5)
    rsz_img = resize_and_normalize_for_deskew_model(rm_bg_img, 512)
    data={'img':rsz_img, 'imgpath': impath}
    data['img'] = torch.from_numpy(data['img']).float()
    if len(data['img'].shape)<4:
        data['img'] = data['img'].unsqueeze(0)
    with torch.no_grad():
        #deskew
        if torch.cuda.is_available():
            data['img'] = data['img'].cuda(non_blocking=True)
        deskew_cls = deskew_model.predict(data['img'])
        deskew_deg = getDegreeFromDeskewOuput(deskew_cls)
        dsk_img = rotateImage(org_img, -deskew_deg)
        #get character patches

        patches= getCharacterPatches(craft, dsk_img, 0.8, 0.5, 0.5)
        if patches:
            patches2tensor = [torch.from_numpy(patch).unsqueeze(0).float()/255.0 for patch in patches]
            in_img = torch.stack(patches2tensor, dim=0)
            if torch.cuda.is_available():
                in_img = in_img.cuda(non_blocking=True)
            dir_cls = stdNet.predict(in_img)
            candidates = Counter(dir_cls.tolist())
            pred = candidates.most_common(1)[0][0]
            degree = cls2degree[pred]
        else:
            degree = 0
    return -deskew_deg-degree
    
def parse_args():
    parser = argparse.ArgumentParser(description="handle arguments")
    parser.add_argument("--config", help="path to the configuration file", type=str)
    parser.add_argument("--input_data_dir", help="directory where the image data is.", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = json.load(f)
        cfg['config_file'] = args.config

    logger.info('create model')

    # load deskew model
    deskew_model = build_model(**cfg['deskew_model_cfg'])

    # ?????? ????????? "expected more than 1 value per channel when training, got input size" ?????? ??????

    # load character detection model
    logger.info('create text detection model')
    craft = build_model(**cfg['craft_model_cfg'])

    # load direction correction model
    logger.info('create direction correction model')
    stdNet = build_model(**cfg['direction_model_cfg'])

    if torch.cuda.is_available():
        deskew_model = deskew_model.cuda()
        craft = craft.cuda()
        stdNet = stdNet.cuda()  
    deskew_model.eval()
    craft.eval()
    stdNet.eval()
    # init logger
    logger.info('init logger')
    #mltracker = MLLogger(cfg, logger)
    img_dir = args.input_data_dir if args.input_data_dir.endswith('*') else args.input_data_dir+'/*'
    imgs = glob.glob(img_dir)
    

    for iter in range(1):
        random.seed(iter)
        random.shuffle(imgs)
        for impath in tqdm(imgs):
            img = cv2.imread(impath)#
            degree = inference(img)
            corrected_img = rotateImage(img, degree)
            visualizer_for_single_input(corrected_img, impath)





       
        