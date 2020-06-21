import os
import torch
import torchvision
from tensorboardX import SummaryWriter
from tqdm import tqdm
import cv2 as cv
from torchvision import transforms
import numpy as np

from dataset.dataset import AdaMattingDataset
from dataset.pre_process import composite_dataset, gen_train_valid_names
from net.adamatting import AdaMatting
from loss import task_uncertainty_loss
from utility import get_args, get_logger, lr_scheduler, save_checkpoint, AverageMeter, \
                    compute_mse, compute_sad, gen_test_names, clip_gradient
from net.sync_batchnorm import convert_model