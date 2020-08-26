import argparse
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import net
from data import MatTransform, MatDatasetOffline
from torchvision import transforms
import time
import os
import cv2
import numpy as np
from test import inference_img_by_crop, inference_img_by_resize, inference_img_whole
from net.adamatting import AdaMatting
from ada_dataset import MatTransform, MatDatasetOffline
import math
from loss import trimap_adaptation_loss, alpha_estimation_loss, task_uncertainty_loss
import logging


def get_logger(fname):
    assert(fname != "")
    logger = logging.getLogger("AdaMatting")
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter("%(asctime)s-%(filename)s:%(lineno)d-%(levelname)s-%(message)s")

    # log file stream
    handler = logging.FileHandler(fname)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    # log console stream
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--size_h', type=int, default=320, required=True, help="height size of input image")
    parser.add_argument('--size_w', type=int, default=320, required=True, help="width size of input image")
    parser.add_argument('--crop_h', type=str, default='320,480,640', help="crop height size of input image")
    parser.add_argument('--crop_w', type=str, default='320,480,640', help="crop width size of input image")
    parser.add_argument('--alphaDir', type=str, default='D:/Samsung-PRISM/data/train/alpha', help="directory of alpha")
    parser.add_argument('--cur_iter', type=int, default=0, help="current iteration")
    parser.add_argument('--max_iter', type=int, default=0, help="maximum number of iterations")
    parser.add_argument('--fgDir', type=str, default='D:/Samsung-PRISM/data/train/fg', help="directory of fg")
    parser.add_argument('--bgDir', type=str, default='D:/Samsung-PRISM/data/train/bg', help="directory of bg")
    parser.add_argument('--imgDir', type=str, default='D:/Samsung-PRISM/data/train/merged', help="directory of img")
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=12, help='number of epochs to train for')
    parser.add_argument('--step', type=int, default=-1, help='epoch of learning decay')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--resume', type=str, help="checkpoint that model resume from")
    parser.add_argument('--pretrain', type=str, help="checkpoint that model pretrain from")
    parser.add_argument('--saveDir', type=str, default='D:/Samsung-PRISM/', help="checkpoint that model save to")
    parser.add_argument('--printFreq', type=int, default=10, help="checkpoint that model save to")
    parser.add_argument('--ckptSaveFreq', type=int, default=10, help="checkpoint that model save to")
    parser.add_argument('--testFreq', type=int, default=1, help="test frequency")
    parser.add_argument('--testImgDir', type=str, default='D:/Samsung-PRISM/data/test/merged', help="test image")
    parser.add_argument('--testTrimapDir', type=str, default='D:/Samsung-PRISM/data/test/mask', help="test trimap")
    parser.add_argument('--testAlphaDir', type=str, default='D:/Samsung-PRISM/data/test/alpha', help="test alpha ground truth")
    parser.add_argument('--testResDir', type=str, default='D:/Samsung-PRISM/data/test/', help="test result save to")
    parser.add_argument('--crop_or_resize', type=str, default="whole", choices=["resize", "crop", "whole"], help="how manipulate image before test")
    parser.add_argument('--max_size', type=int, default=1312, help="max size of test image")
    parser.add_argument('--log', type=str, default='tmplog.txt', help="log file")
    args = parser.parse_args()
    return args


def get_dataset(args):
    train_transform = MatTransform(flip=True)
    
    args.crop_h = [int(i) for i in args.crop_h.split(',')]
    args.crop_w = [int(i) for i in args.crop_w.split(',')]

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
    ])

    train_set = MatDatasetOffline(args, train_transform, normalize)
    train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)

    return train_loader

def build_model(args, logger):

    model = AdaMatting(in_channel=4)

    start_epoch = 1
    best_sad = 100000000.

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)

    if args.pretrain and os.path.isfile(args.pretrain):
        logger.info("loading pretrain '{}'".format(args.pretrain))
        ckpt = torch.load(args.pretrain)
        model.load_state_dict(ckpt['state_dict'],strict=False)
        logger.info("loaded pretrain '{}' (epoch {})".format(args.pretrain, ckpt['epoch']))
    
    if args.resume and os.path.isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        ckpt = torch.load(args.resume)
        start_epoch = ckpt['epoch']
        best_sad = ckpt['best_sad']
        model.load_state_dict(ckpt['state_dict'],strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {} bestSAD {:.3f})".format(args.resume, ckpt['epoch'], ckpt['best_sad']))
    
    return start_epoch, model, best_sad

def lr_scheduler(args, optimizer, init_lr, cur_iter, max_decay_times, decay_rate):
    lr = init_lr * decay_rate ** (cur_iter / args.max_iter * max_decay_times)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def format_second(secs):
    h = int(secs / 3600)
    m = int((secs % 3600) / 60)
    s = int(secs % 60)
    ss = "Exa(h:m:s):{:0>2}:{:0>2}:{:0>2}".format(h,m,s)
    return ss    

def train(args, model, optimizer, train_loader, epoch, logger):
    
    model.train()
    t0 = time.time()
    #fout = open("train_loss.txt",'w')
    for iteration, batch in enumerate(train_loader, 1):
        torch.cuda.empty_cache()
        img = Variable(batch[0])
        alpha = Variable(batch[1])
        fg = Variable(batch[2])
        bg = Variable(batch[3])
        trimap = Variable(batch[4])
        img_norm = Variable(batch[6])
        img_info = batch[-1]

        if args.cuda:
            img = img.cuda()
            alpha = alpha.cuda()
            fg = fg.cuda()
            bg = bg.cuda()
            trimap = trimap.cuda()
            img_norm = img_norm.cuda()

        print("Shape: Img:{} Alpha:{} Fg:{} Bg:{} Trimap:{}".format(img.shape, alpha.shape, fg.shape, bg.shape, trimap.shape))
        print("Val: Img:{} Alpha:{} Fg:{} Bg:{} Trimap:{} Img_info".format(img, alpha, fg, bg, trimap, img_info))

        lr_scheduler(args, optimizer=optimizer, init_lr=args.lr, cur_iter=args.cur_iter, max_decay_times=40, decay_rate=0.9)
        optimizer.zero_grad()

        trimap_adaption, t_argmax, alpha_estimation, log_sigma_t_sqr, log_sigma_a_sqr = model(torch.cat((img_norm, trimap / 255.), 1))

        L_overall, L_t, L_a = task_uncertainty_loss(pred_trimap=trimap_adaption, input_trimap_argmax=trimap, 
                                                        pred_alpha=alpha_estimation, gt_trimap=trimap, gt_alpha=alpha, 
                                                        log_sigma_t_sqr=log_sigma_t_sqr, log_sigma_a_sqr=log_sigma_a_sqr)

        sigma_t, sigma_a = torch.exp(log_sigma_t_sqr.mean() / 2), torch.exp(log_sigma_a_sqr.mean() / 2)
        
        optimizer.zero_grad()
        L_overall.backward()
        optimizer.step()

        # if args.cur_iter % args.printFreq ==  0:
        #     t1 = time.time()
        #     num_iter = len(train_loader)
        #     speed = (t1 - t0) / iteration
        #     exp_time = format_second(speed * (num_iter * (args.nEpochs - epoch + 1) - iteration))

        #     logger.info("Epoch: {:03d} | Iter: {:05d}/{} | Loss: {:.4e} | L_t: {:.4e} | L_a: {:.4e}"
        #                     .format(epoch, index, len(train_loader), avg_lo.avg, avg_lt.avg, avg_la.avg))
        args.cur_iter += 1


def test(args, model, logger):
    model.eval()
    sample_set = []
    img_ids = os.listdir(args.testImgDir)
    img_ids.sort()
    cnt = len(img_ids)
    mse_diffs = 0.
    sad_diffs = 0.
    cur = 0
    t0 = time.time()
    for img_id in img_ids:
        img_path = os.path.join(args.testImgDir, img_id)
        trimap_path = os.path.join(args.testTrimapDir, img_id)

        assert(os.path.exists(img_path))
        assert(os.path.exists(trimap_path))

        img = cv2.imread(img_path)
        trimap = cv2.imread(trimap_path)[:, :, 0]

        assert(img.shape[:2] == trimap.shape[:2])

        img_info = (img_path.split('/')[-1], img.shape[0], img.shape[1])

        cur += 1
        logger.info('[{}/{}] {}'.format(cur, cnt, img_info[0]))        

        with torch.no_grad():
            torch.cuda.empty_cache()

            if args.crop_or_resize == "whole":
                origin_pred_mattes = inference_img_whole(args, model, img, trimap)
            elif args.crop_or_resize == "crop":
                origin_pred_mattes = inference_img_by_crop(args, model, img, trimap)
            else:
                origin_pred_mattes = inference_img_by_resize(args, model, img, trimap)

        # only attention unknown region
        origin_pred_mattes[trimap == 255] = 1.
        origin_pred_mattes[trimap == 0  ] = 0.

        # origin trimap 
        pixel = float((trimap == 128).sum())
        
        # eval if gt alpha is given
        if args.testAlphaDir != '':
            alpha_name = os.path.join(args.testAlphaDir, img_info[0])
            assert(os.path.exists(alpha_name))
            alpha = cv2.imread(alpha_name)[:, :, 0] / 255.
            assert(alpha.shape == origin_pred_mattes.shape)

            mse_diff = ((origin_pred_mattes - alpha) ** 2).sum() / pixel
            sad_diff = np.abs(origin_pred_mattes - alpha).sum()
            mse_diffs += mse_diff
            sad_diffs += sad_diff
            logger.info("sad:{} mse:{}".format(sad_diff, mse_diff))

        origin_pred_mattes = (origin_pred_mattes * 255).astype(np.uint8)
        if not os.path.exists(args.testResDir):
            os.makedirs(args.testResDir)
        cv2.imwrite(os.path.join(args.testResDir, img_info[0]), origin_pred_mattes)

    logger.info("Avg-Cost: {} s/image".format((time.time() - t0) / cnt))
    if args.testAlphaDir != '':
        logger.info("Eval-MSE: {}".format(mse_diffs / cnt))
        logger.info("Eval-SAD: {}".format(sad_diffs / cnt))
    return sad_diffs / cnt


def checkpoint(epoch, save_dir, model, best_sad, logger, best=False):

    epoch_str = "best" if best else "e{}".format(epoch)
    model_out_path = "{}/ckpt_{}.pth".format(save_dir, epoch_str)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_sad': best_sad
    }, model_out_path )
    logger.info("Checkpoint saved to {}".format(model_out_path))


def main():

    args = get_args()
    logger = get_logger(args.log)
    logger.info("Loading args: \n{}".format(args))

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    logger.info("Loading dataset:")
    train_loader = get_dataset(args)

    logger.info("Building model:")
    start_epoch, model, best_sad = build_model(args, logger)
    args.cur_iter = 0

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)

    if args.cuda:
        model = model.cuda()

    args.max_iter = 43100 * (1 - args.valid_portion / 100) / args.batch_size * args.epochs

    logger.info("Starting Training")
    # training
    for epoch in range(start_epoch, args.epochs+1):
        # torch.set_grad_enabled(True)

        train(args, model, optimizer, train_loader, epoch, logger)
        if epoch > 0 and args.testFreq > 0 and epoch % args.testFreq == 0:
            cur_sad = test(args, model, logger)
            if cur_sad < best_sad:
                best_sad = cur_sad
                print("New best SAD: ", best_sad)
                checkpoint(epoch, args.saveDir, model, best_sad, logger, True)
        if epoch > 0 and epoch % args.ckptSaveFreq == 0:
            checkpoint(epoch, args.saveDir, model, best_sad, logger)


if __name__ == "__main__":
    main()
