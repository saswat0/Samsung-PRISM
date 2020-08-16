import os
import torch
import torchvision
from tensorboardX import SummaryWriter
from tqdm import tqdm
import cv2 as cv
from torchvision import transforms
import numpy as np
from dataset.pre_process import composite_dataset, gen_train_valid_names

args = get_args()
logger = get_logger(args.write_log)
    
logger.info("Program runs in prep mode")
composite_dataset('/home/user/Samsung-PRISM/Datasets/', logger)
gen_train_valid_names(5, logger)