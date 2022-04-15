"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import os
import time
import argparse
import cv2
import numpy as np
import argparse
import logging
import json
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from modules.utils import build_transformer, MLLogger, build_lr_scheduler
from modules.dataset import build_dataloader
from modules.model import build_model

from utils import craft_utils
from utils import imgproc
from utils import file_utils
from modules.model import CRAFT
from glob import glob
from collections import OrderedDict, Counter
import random
from sklearn.metrics import precision_recall_fscore_support
import pickle
from sklearn.metrics import classification_report
logger = logging.getLogger('deskew')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='checkpoints/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.8, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.5, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.5, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
parser.add_argument("--config", help="path to the configuration file", type=str, default='config/renet_ocr.json')
parser.add_argument("--run_name", help="run name for mlflow tracking", type=str)
args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    # polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    # for k in range(len(polys)):
    #     if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    return boxes


def getRotated(img, deg):
    h,w,c = img.shape
    noise = 0#np.random.uniform(-2, 2)
    matrix = cv2.getRotationMatrix2D((w/2, h/2), deg+noise, 1)
    dst = cv2.warpAffine(img, matrix, (w, h),borderValue = (255,255,255))
    return dst

if __name__ == '__main__':
    degrees = [0,90,180,270]
    # load direction estimation model
    with open(args.config, 'r') as f:
        cfg = json.load(f)
        cfg['config_file']=args.config
        if args.run_name:
            cfg['mllogger_cfg']['run_name']=args.run_name
    
    logger.info('create model')
    stdNet = build_model(**cfg['model_cfg'])
    stdNet = stdNet.cuda()
    stdNet.eval()

    
    # load CRAFT
    net = CRAFT()     # initialize
    print('Loading weights from checkpoint (' + args.trained_model + ')')
    net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False
    net.eval()
    # LinkRefiner
    refine_net = None
    t = time.time()
    # load data
    data_path ='/train_data/valid/*'#os.getcwd()+'/need_to_be_check_files/*' #'/train_data/valid/*'
    img_pathes = glob(data_path, recursive=True)
    random.shuffle(img_pathes)
    gt_degrees = []
    pred_list = []

    for k, image_path in enumerate(img_pathes):
       
        if 1:
            print("Test image {:d}/{:d}: {:s}".format(k+1, len(img_pathes), image_path), end='\r')
            image = imgproc.loadImage(image_path)#image_path)
            for i, deg in enumerate(degrees):
                gt_degrees.append(i)
                rotated_image=getRotated(image.copy(), deg)
                #bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
                bboxes = test_net(net, rotated_image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
                file_utils.saveResult(image_path, rotated_image[:,:,::-1], bboxes, result_folder)
                patches = file_utils.getCharacterPatchFromImage(rotated_image[:,:,::-1], bboxes, max_num_patch=256)
                
                if patches:
                    with torch.no_grad():
                        patches2tensor = [torch.from_numpy(patch).unsqueeze(0).float()/255.0 for patch in patches]
                        in_img = torch.stack(patches2tensor, dim=0).cuda(non_blocking=True)
                        logits = stdNet(in_img)
                        prob = torch.softmax(logits, -1)
                        cls = torch.argmax(prob, -1)
                        if 0:
                            odir = f'./deg_{deg}_patches'
                            if not os.path.exists(odir):
                                os.mkdir(odir)
                            [cv2.imwrite(f'{odir}/false_{prob[i][data[1]]:.2f}_{i}.png', data[0]) for i, data in enumerate(zip(patches,cls)) if data[1]!=1]
                            [cv2.imwrite(f'{odir}/Positive_{prob[i][data[1]]:.2f}_{i}.png', data[0]) for i, data in enumerate(zip(patches,cls)) if data[1]==1]
                        candidates = Counter(cls.tolist())
                        pred = candidates.most_common(1)[0][0]
                        pred_list.append(pred)
                        degree = degrees[pred]
                        if pred != i:
                            filename = os.path.basename(image_path)
                            try:
                                cv2.imwrite(f'issuefiles/rotated_degree_{deg}_pred_{degree}_{filename}',image)
                            except OSError:
                                pass
                else:
                    pred_list.append(4)
            if (k+1)%100==0:
                try:
                    report = classification_report(gt_degrees, pred_list, target_names=['0', '1','2','3'])
                except ValueError:
                    report = classification_report(gt_degrees, pred_list, target_names=['0', '1','2','3','4'])
                with open('4class_low_text_0.5_link_0.5_text_score_0.8_report.txt', 'w') as f:
                    print(report, file=f)
        else:
            print("Test image {:d}/{:d}: {:s}".format(k+1, len(img_pathes), image_path), end='\r')
            image = imgproc.loadImage(image_path)#image_path)
            bboxes, polys = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
            patches = file_utils.getCharacterPatchFromImage(image[:,:,::-1], polys, max_num_patch=256)

            if patches:
                with torch.no_grad():
                    patches2tensor = [torch.from_numpy(patch).unsqueeze(0).float() for patch in patches]
                    in_img = torch.stack(patches2tensor, dim=0).cuda()
                    logits = stdNet(in_img)
                    cls = torch.argmax(torch.softmax(logits, -1), -1)
                    candidates = Counter(cls.tolist())
                    pred = candidates.most_common(1)[0][0]
                    pred_list.append(pred)
                    degree = degrees[pred]
                    h,w,c = image.shape
                    m = cv2.getRotationMatrix2D((w/2, h/2), -degree, 1)
                    dst = cv2.warpAffine(image, m, (w,h))
                    resimg=cv2.hconcat([image, dst])
                    name = os.path.basename(image_path)
                    cv2.imwrite(f'result/{name}', resimg)
                    
    #mltracker = MLLogger(cfg, logger)
    with open("pred_list.pickle","wb") as fw:
        pickle.dump(pred_list, fw)
    with open("gt_cls_list.pickle","wb") as fw:
        pickle.dump(gt_degrees, fw)
    try:
        report = classification_report(gt_degrees, pred_list, target_names=['0', '1','2','3'])
    except ValueError:
        report = classification_report(gt_degrees, pred_list, target_names=['0', '1','2','3','4'])
    with open('4class_low_text_0.5_link_0.5_text_score_0.8_report.txt', 'w') as f:
        print(report, file=f)

    # mltracker.log_metric('valid_precision',precision, 0),
    # mltracker.log_metric('valid_recall',recall, 0)
    # mltracker.log_metric('valid_f1_score',f1_score, 0)
    print("elapsed time : {}s".format(time.time() - t))
