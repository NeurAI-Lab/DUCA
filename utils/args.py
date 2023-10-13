# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, required=True,
                        help='The number of epochs for each task.')
    parser.add_argument('--n_tasks', type=int, required=False)
    parser.add_argument('--n_classes_per_task', type=int, required=False)

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--experiment_id', type=str, default='cl', help='experiment name')
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')
    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--aug_norm', action='store_true')
    parser.add_argument('--aug_shape', action='store_true')
    parser.add_argument('--save_model', action='store_true')

def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, required=True,
                        help='The batch size of the memory buffer.')

def add_auxiliary_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--aux', type=str, default='shape',
                        help='The type of auxiliary data')
    parser.add_argument('--img_size', type=int, default=32,
                        help='Image size.')
    parser.add_argument('--shape_filter', type=str, default='sobel',
                        help='The type of auxiliary data')
    parser.add_argument('--shape_upsample_size', type=int, default=64,
                        help='size to upsample for sobel filter')
    parser.add_argument('--sobel_gauss_ksize', default=3, type=int)
    parser.add_argument('--sobel_ksize', default=3, type=int)
    parser.add_argument('--sobel_upsample', type=str, default='True')
    parser.add_argument('--loss_type', nargs='*', type=str, default=['kl'], help="--loss_type kl at")
    parser.add_argument('--loss_wt', nargs='*', type=float, default=[1.0, 1.0, 1.0, 1.0])
    parser.add_argument('--dir_aux', action='store_true')
    parser.add_argument('--buf_aux', action='store_true')
    parser.add_argument('--aug_prob', default=0.0, type=float)
    parser.add_argument('--data_combine', action='store_true')

def add_gcil_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments required for GCIL-CIFAR100 Dataset.
    :param parser: the parser instance
    """
    # arguments for GCIL-CIFAR100
    parser.add_argument('--gil_seed', type=int, default=1993, help='Seed value for GIL-CIFAR task sampling')
    parser.add_argument('--pretrain', action='store_true', default=False, help='whether to use pretrain')
    parser.add_argument('--phase_class_upper', default=50, type=int, help='the maximum number of classes')
    parser.add_argument('--epoch_size', default=1000, type=int, help='Number of samples in one epoch')
    parser.add_argument('--pretrain_class_nb', default=0, type=int, help='the number of classes in first group')
    parser.add_argument('--weight_dist', default='unif', type=str, help='what type of weight distribution assigned to classes to sample (unif or longtail)')