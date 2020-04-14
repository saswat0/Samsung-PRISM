import torch
import torch.nn as nn


def trimap_adaptation_loss(pred_trimap, gt_trimap):
    loss = nn.CrossEntropyLoss()
    return loss(pred_trimap, gt_trimap)


def alpha_estimation_loss(pred_alpha, gt_alpha, pred_trimap_argmax):
    """
    pred_trimap_argmax
    0: background
    1: unknown
    2: foreground
    """
    mask = torch.zeros(pred_trimap_argmax.shape).cuda()
    mask[pred_trimap_argmax == 1] = 1.
    mask = mask.unsqueeze(dim=1)
    diff = (pred_alpha - gt_alpha + 1e-12).mul(mask)
    return torch.abs(diff).sum() / (mask.sum() + 1.)


def task_uncertainty_loss(pred_trimap, pred_trimap_argmax, pred_alpha, gt_trimap, gt_alpha, log_sigma_t_sqr, log_sigma_a_sqr):
    log_sigma_t_sqr = log_sigma_t_sqr.mean()
    log_sigma_a_sqr = log_sigma_a_sqr.mean()
    Lt = trimap_adaptation_loss(pred_trimap, gt_trimap)
    La = alpha_estimation_loss(pred_alpha, gt_alpha, pred_trimap_argmax)
    overall = 5e1 * Lt / (2 * torch.exp(log_sigma_t_sqr)) + 5e1 * La / torch.exp(log_sigma_a_sqr) + (log_sigma_t_sqr + log_sigma_a_sqr) / 2
    return overall, Lt, La
