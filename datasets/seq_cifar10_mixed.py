# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_mixed_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from utils.auxiliary import transform_sobel_edge

class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download)

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


class SequentialCIFAR10Mix(ContinualDataset):

    NAME = 'seq-cifar10-mixed'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    IMG_SIZE = 32
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2470, 0.2435, 0.2615]

    normalize = transforms.Normalize(mean=MEAN, std=STD)
    TRANSFORM = [transforms.RandomCrop(IMG_SIZE, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()]
    TRANSFORM_NORM = [transforms.RandomCrop(IMG_SIZE, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      normalize]
    TRANSFORM_TEST = [transforms.ToTensor()]
    TRANSFORM_TEST_NORM = [transforms.ToTensor(), normalize]

    def get_data_loaders(self):

        TRANSFORM_SHAPE = transforms.Compose([transform_sobel_edge(self.args, self.args.shape_upsample_size),
                           transforms.RandomCrop(self.IMG_SIZE, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           ])

        if self.args.aug_norm:
            transform = transforms.Compose(self.TRANSFORM_NORM)
            test_transform = transforms.Compose(self.TRANSFORM_TEST_NORM)
        else:
            transform = transforms.Compose(self.TRANSFORM)
            test_transform = transforms.Compose(self.TRANSFORM_TEST)

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        train_dataset_sh = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=TRANSFORM_SHAPE)

        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = CIFAR10(base_path() + 'CIFAR10',train=False,
                                   download=True, transform=test_transform)

        train, test = store_masked_mixed_loaders(train_dataset, train_dataset_sh, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader


    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(),  transforms.Compose(SequentialCIFAR10Mix.TRANSFORM)])
        return transform

    @staticmethod
    def get_norm_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(),  transforms.Compose(SequentialCIFAR10Mix.TRANSFORM_NORM)])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR10Mix.N_CLASSES_PER_TASK
                        * SequentialCIFAR10Mix.N_TASKS)

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

