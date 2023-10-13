# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import torch
import numpy as np
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize


class transform_sobel_edge(object):
    def __init__(self, args=None, upsample_size=0, p=0.1):
        self.gauss_ksize = 3 #args.sobel_gauss_ksize
        self.sobel_ksize = 3 #args.sobel_ksize
        self.upsample = 'True' #args.sobel_upsample
        self.upsample_size = 64 #upsample_size
        self.p = p

    def __call__(self, img, boxes=None, labels=None):

        if torch.rand(1) < self.p:
            return img

        else:
            if self.upsample == 'True':
                curr_size = img.size[0]
                resize_up = transforms.Resize(max(curr_size, self.upsample_size), 3)
                resize_down = transforms.Resize(curr_size, 3)
                rgb = np.array(resize_up(img))
            else:
                rgb = np.array(img)

            if len(rgb.shape) != 3:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)

            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            rgb = cv2.GaussianBlur(rgb, (self.gauss_ksize, self.gauss_ksize), self.gauss_ksize)
            sobelx = cv2.Sobel(rgb, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
            imgx = cv2.convertScaleAbs(sobelx)
            sobely = cv2.Sobel(rgb, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
            imgy = cv2.convertScaleAbs(sobely)
            tot = np.sqrt(np.square(sobelx) + np.square(sobely))
            imgtot = cv2.convertScaleAbs(tot)
            sobel_img = Image.fromarray(cv2.cvtColor(imgtot, cv2.COLOR_GRAY2BGR))

            sobel_img = resize_down(sobel_img) if self.upsample == 'True' else sobel_img


            return sobel_img


class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img

class SequentialCIFAR100Aug(ContinualDataset):

    NAME = 'seq-cifar100-aug'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 5
    IMG_SIZE = 32
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2470, 0.2435, 0.2615]

    def __init__(self, args):

        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args
        self.p = args.aug_prob

        IMG_SIZE = 32
        MEAN = [0.4914, 0.4822, 0.4465]
        STD = [0.2470, 0.2435, 0.2615]

        normalize = transforms.Normalize(mean=MEAN, std=STD)
        self.TRANSFORM = [transform_sobel_edge(p=self.p),
                     transforms.RandomCrop(IMG_SIZE, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor()]
        self.TRANSFORM_NORM = [transforms.RandomCrop(IMG_SIZE, padding=4),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          normalize]
        self.TRANSFORM_TEST = [transforms.ToTensor()]
        self.TRANSFORM_TEST_NORM = [transforms.ToTensor(), normalize]

    def get_data_loaders(self):

        if self.args.n_tasks is not None:
            self.N_CLASSES_PER_TASK = self.args.n_classes_per_task
            self.N_TASKS = self.args.n_tasks

        if self.args.aug_norm:
            transform = transforms.Compose(self.TRANSFORM_NORM)
            test_transform = transforms.Compose(self.TRANSFORM_TEST_NORM)
        else:
            transform = transforms.Compose(self.TRANSFORM)
            test_transform = transforms.Compose(self.TRANSFORM_TEST)

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = CIFAR100(base_path() + 'CIFAR100',train=False,
                                   download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    # @staticmethod
    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(),  transforms.Compose(self.TRANSFORM)])
        return transform

    @staticmethod
    def get_norm_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(),  transforms.Compose(SequentialCIFAR100Aug.TRANSFORM_NORM)])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR100Aug.N_CLASSES_PER_TASK
                        * SequentialCIFAR100Aug.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform
