import os
import torch
import numpy as np
import cv2

def parse_orientation_prediction_outputs(cls_logit):
    return (torch.sigmoid(cls_logit)>0.5).int()

def parse_rotation_prediction_outputs(cls_logit):
    return torch.argmax(torch.softmax(cls_logit, -1), -1)


def visualize_rotation_corrected_image(data, logit, logger, info, step=None):
    inds = [np.random.randint(0, data['img'].shape[0])] if step is not None else range(len(data['img']))
    for ind in inds:
        mean, std = data['mean'][ind], data['std'][ind]
        img = data['img'][ind].squeeze()
        img = img*std+mean
        img = img.data.cpu().numpy().copy().astype(np.uint8)
        rot_id = torch.argmax(torch.softmax(logit,-1),-1)[ind]
        rad_range = np.deg2rad(info['degree'])
        gt_deg = data['degree'][ind].item()
        rotation = rot_id*rad_range*2/info['buckets']-rad_range
        if isinstance(rotation, torch.Tensor):
            rotation = np.rad2deg(rotation.item())
        h,w = img.shape
        m = cv2.getRotationMatrix2D((w/2, h/2), -rotation, 1)
        dst = cv2.warpAffine(img, m, (w,h))
        resimg=cv2.hconcat([img, dst])

        name = os.path.basename(data['imgpath'][ind])
        save_name = f'pred_{rotation:.2f}_gt_{gt_deg:.2f}_{name}'
        if step:
            save_name=f'ep_{step}_sample_'+save_name
        logger.log_image(resimg, name=save_name)


def visualize_orientation_prediction_outputs(data, logit, logger, step=None):
    inds = [np.random.randint(0, data['img'].shape[0])] if step is not None else range(len(data['img']))
    for ind in inds:
        mean, std = data['mean'][ind], data['std'][ind]
        img = data['img'][ind].squeeze()
        img = img*std+mean
        img = img.data.cpu().numpy().copy().astype(np.uint8)
        gt_flip = data['flip'][ind].data.cpu().numpy()
        prob = torch.sigmoid(logit[ind])
        pred_flip = (prob>0.5).int()

        rotation = 180 if pred_flip else 0
        h,w = img.shape
        m = cv2.getRotationMatrix2D((w/2, h/2), rotation, 1)
        dst = cv2.warpAffine(img, m, (w,h))
        resimg = cv2.hconcat([img, dst])
        name = os.path.basename(data['imgpath'][ind])
        save_name = f'pred_{rotation:.2f}_gt_{gt_flip:.2f}_{name}'
        if step:
            save_name=f'ep_{step}_sample_'+save_name
        logger.log_image(resimg, name=save_name)

def visualize_rotation_corrected_image_compute_error(data, logit, logger, info, step=None, error=[]):
    inds = [np.random.randint(0, data['img'].shape[0])] if step is not None else range(len(data['img']))
    for ind in inds:
        mean, std = data['mean'][ind], data['std'][ind]
        img = data['img'][ind].squeeze()
        img = img*std+mean
        img = img.data.cpu().numpy().copy().astype(np.uint8)
        rot_id = torch.argmax(torch.softmax(logit,-1),-1)[ind]
        rad_range = np.deg2rad(info['degree'])
        gt_deg = data['degree'][ind].item()
        rotation = rot_id*rad_range*2/info['buckets']-rad_range
        if isinstance(rotation, torch.Tensor):
            rotation = np.rad2deg(rotation.item())
        error.append((gt_deg-rotation)**2)
        h,w = img.shape
        m = cv2.getRotationMatrix2D((w/2, h/2), -rotation, 1)
        dst = cv2.warpAffine(img, m, (w,h))
        resimg=cv2.hconcat([img, dst])

        name = os.path.basename(data['imgpath'][ind])
        save_name = f'pred_{rotation:.2f}_gt_{gt_deg:.2f}_{name}'
        if step:
            save_name=f'ep_{step}_sample_'+save_name
        logger.log_image(resimg, name=save_name)

   