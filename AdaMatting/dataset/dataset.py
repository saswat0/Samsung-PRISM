import os
import math
import random
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AdaMattingDataset(Dataset):

    def __init__(self, raw_data_path, mode):
        self.crop_size = 320
        self.mode = mode
        self.raw_data_path = raw_data_path

        self.fg_path = os.path.join(self.raw_data_path, "train/fg/")
        self.bg_path = os.path.join(self.raw_data_path, "train/bg/")
        self.a_path = os.path.join(self.raw_data_path, "train/mask/")

        data_transforms = {
            # values from ImageNet
            'train': transforms.Compose([
                # transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'valid': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self.transformer = data_transforms[self.mode]

        with open(os.path.join(self.raw_data_path, "Combined_Dataset/Training_set/training_fg_names.txt")) as f:
            self.fg_files = f.read().splitlines()
        with open(os.path.join(self.raw_data_path, "Combined_Dataset/Training_set/training_bg_names.txt")) as f:
            self.bg_files = f.read().splitlines()

        filename = "dataset/{}_names.txt".format(self.mode)
        with open(filename, 'r') as file:
            self.names = file.read().splitlines()
    

    def __len__(self):
        return len(self.names)
    

    def __getitem__(self, index):
        name = self.names[index]
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        im_name = self.fg_files[fcount]
        bg_name = self.bg_files[bcount]
        fg = cv.imread(self.fg_path + im_name)
        alpha = cv.imread(self.a_path + im_name, 0)
        bg = cv.imread(self.bg_path + bg_name)

        # Resize and crop the bg to fit the fg
        h, w = fg.shape[:2]
        bh, bw = bg.shape[:2]
        wratio = w / bw
        hratio = h / bh
        ratio = wratio if wratio > hratio else hratio
        if ratio > 1:
            bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)
        bh, bw = bg.shape[:2]
        x = np.random.randint(0, bw - w) if bw > w else 0
        y = np.random.randint(0, bh - h) if bh > h else 0
        bg = np.array(bg[y:y + h, x:x + w], np.float32)

        # Composite the merged image
        normalized_alpha = np.zeros((h, w, 1), np.float32)
        normalized_alpha[:, :, 0] = alpha / 255.
        merged = normalized_alpha * fg + (1 - normalized_alpha) * bg
        merged = merged.astype(np.uint8)

        # Resize randomly if in training mode
        resize_factor = 0.75 * np.random.random_sample() + 0.75
        input_img = cv.resize(merged, None, fx=resize_factor, fy=resize_factor)
        gt_alpha = cv.resize(alpha, None, fx=resize_factor, fy=resize_factor)

        # Rotate randomly in training mode
        angle = np.random.randint(-45, 46)
        input_img = self.rotate_cv_image(input_img, angle)
        gt_alpha = self.rotate_cv_image(gt_alpha, angle)

        # Generate gt_trimap
        gt_trimap = np.zeros(gt_alpha.shape, np.float32)
        gt_trimap.fill(128)
        gt_trimap[gt_alpha == 0] = 0
        gt_trimap[gt_alpha == 255] = 255

        # Crop randomly if in training mode
        different_size = np.random.randint(self.crop_size, 801)
        crop_size = (different_size, different_size)
        x, y = self.random_crop_pos(gt_trimap, crop_size)
        input_img = self.do_crop(input_img, x, y, crop_size)
        gt_alpha = self.do_crop(gt_alpha, x, y, crop_size)
        gt_trimap = self.do_crop(gt_trimap, x, y, crop_size)

        # Flip randomly in training mode
        if np.random.random_sample() > 0.5:
            input_img = np.fliplr(input_img)
            gt_alpha = np.fliplr(gt_alpha)
            gt_trimap = np.fliplr(gt_trimap)

        # Generate input_trimap
        k_size = np.random.randint(10, 26)
        iterations = np.random.randint(1, 21)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
        eroded = cv.erode(gt_trimap, kernel, iterations)
        dilated = cv.dilate(gt_trimap, kernel, iterations)
        input_trimap = np.zeros(gt_trimap.shape, np.float32)
        input_trimap.fill(0.5)
        input_trimap[eroded == 255] = 1.0
        input_trimap[dilated == 0] = 0.0

        # cv.imwrite("temp/input_img.png", input_img)
        # cv.imwrite("temp/input_trimap.png", input_trimap)
        # cv.imwrite("temp/gt_alpha.png", gt_alpha)
        # cv.imwrite("temp/gt_trimap.png", gt_trimap)

        # Convert all from BGR to RGB and store as PIL image
        rgb_input_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)
        pil_input_img = transforms.ToPILImage()(rgb_input_img)
        pil_input_trimap = transforms.ToPILImage()(input_trimap)
        pil_gt_alpha = transforms.ToPILImage()(gt_alpha)

        display_rgb = transforms.ToTensor()(pil_input_img)

        inputs = torch.zeros((4, self.crop_size, self.crop_size), dtype=torch.float)
        inputs[0:3, :, :] = self.transformer(pil_input_img)
        inputs[3, :, :] = transforms.ToTensor()(pil_input_trimap)

        gts = torch.zeros((2, self.crop_size, self.crop_size), dtype=torch.float)
        gts[0, :, :] = transforms.ToTensor()(pil_gt_alpha)
        out_gt_trimap = np.zeros(gt_trimap.shape, np.float32)
        out_gt_trimap.fill(1.0)
        out_gt_trimap[gt_trimap == 255] = 2.0
        out_gt_trimap[gt_trimap == 0] = 0.0
        gts[1, :, :] = torch.from_numpy(out_gt_trimap)

        return display_rgb, inputs, gts


    def random_crop_pos(self, gt_trimap, crop_size=(320, 320)):
        crop_height, crop_width = crop_size
        y_indices, x_indices = np.where(gt_trimap == 128)
        num_unknowns = len(y_indices)
        x, y = 0, 0
        if num_unknowns > 0:
            ix = np.random.choice(range(num_unknowns))
            center_x = x_indices[ix]
            center_y = y_indices[ix]
            x = max(0, center_x - int(crop_width / 2))
            y = max(0, center_y - int(crop_height / 2))
        return x, y


    def do_crop(self, mat, x, y, crop_size):
        crop_height, crop_width = crop_size
        if len(mat.shape) == 2:
            ret = np.zeros((crop_height, crop_width), np.uint8)
        else:
            ret = np.zeros((crop_height, crop_width, 3), np.uint8)
        crop = mat[y:y + crop_height, x:x + crop_width]
        h, w = crop.shape[:2]
        ret[0:h, 0:w] = crop
        if crop_size != (self.crop_size, self.crop_size):
            ret = cv.resize(ret, dsize=(self.crop_size, self.crop_size), interpolation=cv.INTER_NEAREST)
        return ret


    def rotate_cv_image(self, image, angle=None, center=None, scale=1.0):
        (h, w) = image.shape[:2]

        if center is None:
            center = (w // 2, h // 2)
        if angle is None:
            angle = np.random.randint(-45, 46)

        M = cv.getRotationMatrix2D(center, angle, scale)
        rotated = cv.warpAffine(image, M, (w, h))
        return rotated
