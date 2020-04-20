import torch
import argparse
import logging
import numpy as np
import cv2 as cv


def gen_test_names():
    num_fgs = 50
    num_bgs_per_fg = 20

    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for _ in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    return names


def save_checkpoint(ckpt_path, is_best, is_alpha_best, logger, model, optimizer, epoch, cur_iter, peak_lr, best_loss, best_alpha_loss):
    state = {"state_dict": model.module.state_dict(),
             'optimizer': optimizer.state_dict(),
             "epoch": epoch,
             "cur_iter": cur_iter,
             "peak_lr": peak_lr,
             "best_loss": best_loss,
             "best_alpha_loss": best_alpha_loss}
    ckpt_fn = ckpt_path + "ckpt.tar"
    torch.save(state, ckpt_fn)
    logger.info("Checkpoint saved")
    if is_best:
        ckpt_fn = ckpt_path + "ckpt_best_overall.tar"
        torch.save(state, ckpt_fn)
        logger.info("Best checkpoint saved")
    if is_alpha_best:
        ckpt_fn = ckpt_path + "ckpt_best_alpha.tar"
        torch.save(state, ckpt_fn)
        logger.info("Best alpha loss checkpoint saved")


# def lr_scheduler(optimizer, cur_iter, peak_lr, end_lr, decay_iters, decay_power, power):
#     if cur_iter != 0 and cur_iter % decay_iters == 0:
#         peak_lr = peak_lr * decay_power

#     cycle_iter = cur_iter % decay_iters
#     lr = (peak_lr - end_lr) * ((1 - cycle_iter / decay_iters) ** power) + end_lr

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

#     return lr, peak_lr


def lr_scheduler(optimizer, init_lr, cur_iter, max_iter, max_decay_times, decay_rate):
    lr = init_lr * decay_rate ** (cur_iter / max_iter * max_decay_times)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AverageMeter(object):
    """
    Keeps track of the most recent value, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_sad(pred, alpha):
    pred = pred[0, 0, :, :].cpu().numpy()
    diff = np.abs(pred - alpha / 255)
    return np.sum(diff) / 1000


def compute_mse(pred, alpha, trimap):
    pred = pred[0, 0, :, :].cpu().numpy()
    num_pixels = float((trimap == 128).sum())
    return ((pred - alpha / 255) ** 2).sum() / num_pixels


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='set arguments')
    parser.add_argument('--mode', type=str, required=True, choices=["train", "test", "prep"], help="set the program to \'train\', \'test\', or \'prep\'")
    parser.add_argument('--valid_portion', type=int, required=True, help="percentage of valid data in all training samples")
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
    parser.add_argument('--decay_iters', type=int, help="Number of iterations every lr decay")
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda?')
    parser.add_argument('--gpu', type=str, default="0", help="choose gpus")
    parser.add_argument('--write_log', action="store_true", default=False, help="whether store log to log.txt")
    parser.add_argument('--raw_data_path', type=str, default="/data/datasets/im/AdaMatting/", help="dir where datasets are stored")
    parser.add_argument('--ckpt_path', type=str, default="./ckpts/", help="path to the saved checkpoint file")
    parser.add_argument('--save_ckpt', action="store_true", default=False, help="whether save checkpoint every epoch")
    parser.add_argument('--resume', type=str, default="", help="whether resume training from a ckpt")
    args = parser.parse_args()
    return args


def get_logger(flag):
    logger = logging.getLogger("AdaMatting")
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    # log file stream
    if (flag):
        handler = logging.FileHandler("log.txt")
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # log console stream
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(console)

    return logger
