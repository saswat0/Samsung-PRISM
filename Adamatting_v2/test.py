import os
import torch
import torchvision
import cv2 as cv
import numpy as np
import argparse
import cv2
import time
import torch.nn as nn
import torch.nn.functional as F
from net import adamatting
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

def gen_dataset(imgdir, trimapdir):
        sample_set = []
        img_ids = os.listdir(imgdir)
        img_ids.sort()
        cnt = len(img_ids)
        cur = 1
        for img_id in img_ids:
            img_name = os.path.join(imgdir, img_id)
            trimap_name = os.path.join(trimapdir, img_id)

            assert(os.path.exists(img_name))
            assert(os.path.exists(trimap_name))

            sample_set.append((img_name, trimap_name))

        return sample_set

def compute_gradient(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    grad = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    grad=cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
    return grad


# inference once for image, return numpy
def inference_once(args, model, scale_img, scale_trimap, aligned=True):

    if aligned:
        assert(scale_img.shape[0] == args.size_h)
        assert(scale_img.shape[1] == args.size_w)

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
    ])

    scale_img_rgb = cv2.cvtColor(scale_img, cv2.COLOR_BGR2RGB)
    # first, 0-255 to 0-1
    # second, x-mean/std and HWC to CHW
    tensor_img = normalize(scale_img_rgb).unsqueeze(0)

    scale_grad = compute_gradient(scale_img)
    #tensor_img = torch.from_numpy(scale_img.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
    tensor_trimap = torch.from_numpy(scale_trimap.astype(np.float32)[np.newaxis, np.newaxis, :, :])
    tensor_grad = torch.from_numpy(scale_grad.astype(np.float32)[np.newaxis, np.newaxis, :, :])

    if args.cuda:
        tensor_img = tensor_img.cuda()
        tensor_trimap = tensor_trimap.cuda()
        tensor_grad = tensor_grad.cuda()
    #print('Img Shape:{} Trimap Shape:{}'.format(img.shape, trimap.shape))

    input_t = torch.cat((tensor_img, tensor_trimap / 255.), 1)

    # forward
    if args.stage <= 1:
        # stage 1
        pred_mattes, _ = model(input_t)
    else:
        # stage 2, 3
        _, pred_mattes = model(input_t)
    pred_mattes = pred_mattes.data
    if args.cuda:
        pred_mattes = pred_mattes.cpu()
    pred_mattes = pred_mattes.numpy()[0, 0, :, :]
    return pred_mattes



# forward for a full image by crop method
def inference_img_by_crop(args, model, img, trimap):
    # crop the pictures, and forward one by one
    h, w, c = img.shape
    origin_pred_mattes = np.zeros((h, w), dtype=np.float32)
    marks = np.zeros((h, w), dtype=np.float32)

    for start_h in range(0, h, args.size_h):
        end_h = start_h + args.size_h
        for start_w in range(0, w, args.size_w):
        
            end_w = start_w + args.size_w
            crop_img = img[start_h: end_h, start_w: end_w, :]
            crop_trimap = trimap[start_h: end_h, start_w: end_w]
            
            crop_origin_h = crop_img.shape[0]
            crop_origin_w = crop_img.shape[1]

            #print("startH:{} startW:{} H:{} W:{}".format(start_h, start_w, crop_origin_h, crop_origin_w))

            if len(np.where(crop_trimap == 128)[0]) <= 0:
                continue

            # egde patch in the right or bottom
            if crop_origin_h != args.size_h or crop_origin_w != args.size_w:
                crop_img = cv2.resize(crop_img, (args.size_w, args.size_h), interpolation=cv2.INTER_LINEAR)
                crop_trimap = cv2.resize(crop_trimap, (args.size_w, args.size_h), interpolation=cv2.INTER_LINEAR)
            
            # inference for each crop image patch
            pred_mattes = inference_once(args, model, crop_img, crop_trimap)

            if crop_origin_h != args.size_h or crop_origin_w != args.size_w:
                pred_mattes = cv2.resize(pred_mattes, (crop_origin_w, crop_origin_h), interpolation=cv2.INTER_LINEAR)

            origin_pred_mattes[start_h: end_h, start_w: end_w] += pred_mattes
            marks[start_h: end_h, start_w: end_w] += 1

    # smooth for overlap part
    marks[marks <= 0] = 1.
    origin_pred_mattes /= marks
    return origin_pred_mattes


# forward for a full image by resize method
def inference_img_by_resize(args, model, img, trimap):
    h, w, c = img.shape
    # resize for network input, to Tensor
    scale_img = cv2.resize(img, (args.size_w, args.size_h), interpolation=cv2.INTER_LINEAR)
    scale_trimap = cv2.resize(trimap, (args.size_w, args.size_h), interpolation=cv2.INTER_LINEAR)

    pred_mattes = inference_once(args, model, scale_img, scale_trimap)

    # resize to origin size
    origin_pred_mattes = cv2.resize(pred_mattes, (w, h), interpolation = cv2.INTER_LINEAR)
    assert(origin_pred_mattes.shape == trimap.shape)
    return origin_pred_mattes


# forward a whole image
def inference_img_whole(args, model, img, trimap):
    h, w, c = img.shape
    new_h = min(args.max_size, h - (h % 32))
    new_w = min(args.max_size, w - (w % 32))

    # resize for network input, to Tensor
    scale_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scale_trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pred_mattes = inference_once(args, model, scale_img, scale_trimap, aligned=False)

    # resize to origin size
    origin_pred_mattes = cv2.resize(pred_mattes, (w, h), interpolation = cv2.INTER_LINEAR)
    assert(origin_pred_mattes.shape == trimap.shape)
    return origin_pred_mattes
