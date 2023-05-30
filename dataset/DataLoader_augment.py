import os,sys
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import cv2
import numpy as np
import torch.nn.functional as F
import pdb

import random

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class myBrightness:

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return transforms.functional.adjust_brightness(x, self.factor)

class myContrast:

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return transforms.functional.adjust_contrast(x, self.factor)

class mySaturation:

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return transforms.functional.adjust_saturation(x, self.factor)

class myAffine:

    def __init__(self, angle, translate, scale, shear):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, x):
        return transforms.functional.affine(x, self.angle, self.translate, self.scale, self.shear)
class myCrop:

    def __init__(self, crop_params):
        self.crop_params = crop_params

    def __call__(self, x):
        return transforms.functional.crop(x, self.crop_params[0], self.crop_params[1], self.crop_params[2], self.crop_params[3])


def image_loader(path, transform):
    image = cv2.imread(path)
    image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    image = transform(image)
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

    image = np.float32(image) / 255.0
    image = cv2.resize(image, (256, 256))
    return image

def rgb_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transforms.ToTensor()(image)

def lab_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image = transforms.ToTensor()(image)
    # Normalize to range [-1, 1]
    image = transforms.Normalize([50,0,0], [50,127,127])(image)
    return image

class myImageFloder(data.Dataset):
    def __init__(self, filepath, filenames, training):

        self.refs = filenames
        self.filepath = filepath

    def __getitem__(self, index):
        refs = self.refs[index]     # ['00f88c4f0a/00004.jpg','00f88c4f0a/00006.jpg']

        image_pre = cv2.imread(os.path.join(self.filepath, refs[0]))
        image_pre = Image.fromarray(cv2.cvtColor(image_pre,cv2.COLOR_BGR2RGB))
        w, h = image_pre.size
        crop_params = transforms.RandomCrop.get_params(image_pre, (h*2//3,w*2//3))
        del image_pre

        random_hf = random.randint(0,1)
        random_vf = random.randint(0,1)
        random_degree = random.randint(-15,15)
        random_translate = (random.uniform(-0.1,0.1), random.uniform(-0.1,0.1))
        random_scale = random.uniform(0.6, 1.8)
        random_shear = random.randint(-20,20)
        random_b = random.uniform(0.7, 1.3)
        random_c = random.uniform(0.7, 1.3)
        random_s = random.uniform(0.7, 1.3)

        trans = transforms.Compose([
            myCrop(crop_params),
            transforms.RandomHorizontalFlip(random_hf),
            transforms.RandomVerticalFlip(random_vf),
            myBrightness(random_b),
            myContrast(random_c),
            mySaturation(random_s),
            myAffine(random_degree, random_translate, random_scale, random_shear)
        ])

        images = [image_loader(os.path.join(self.filepath, ref), trans) for ref in refs]

        images_lab = [lab_preprocess(ref) for ref in images]
        images_rgb = [rgb_preprocess(ref) for ref in images]

        return images_lab, images_rgb, 1

    def __len__(self):
        return len(self.refs)