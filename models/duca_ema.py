# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy

import torch
from copy import deepcopy
from utils.buffer import Buffer
from datasets import get_dataset
from utils.args import *
from utils.auxiliary import *
from models.utils.continual_model import ContinualModel
import torchvision.transforms as transforms

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' DUCA.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_auxiliary_args(parser)
    add_gcil_args(parser)
    parser.add_argument('--alpha_mm', nargs='*', type=float, required=False,
                        help='Penalty weight.')
    parser.add_argument('--ema_alpha', type=float, required=True,
                        help='ema decay weight.')
    parser.add_argument('--ema_update_freq', type=float, required=True,
                        help='frequency.')
    parser.add_argument('--beta_mm', nargs='*', type=float, required=False,
                        help='Penalty weight.')
    parser.add_argument('--abl_mode', type=str, default='None')
    return parser


class DUCAEMA(ContinualModel):
    NAME = 'duca-ema'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(DUCAEMA, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.dataset = get_dataset(args)
        self.aux = AuxiliaryNet(self.args, self.dataset, self.device)
        self.ema_net1 = deepcopy(self.net).to(self.device)
        self.global_step = 0
        self.current_task = 0

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        loss_dict = {}
        loss_aux_ema1 = loss_aux_ema2 = loss_log_12 = loss_log_21 = loss_buf_ce1 = loss_buf_ce2 = loss_aux12 = loss_aux21 = 0
        loss_aux12_buf = loss_aux21_buf = 0

        outputs1 = self.net(inputs)

        loss_ce1 = self.loss(outputs1, labels)
        loss1 = loss_ce1


        if not self.buffer.is_empty():
            buf_inputs1, buf_labels, buf_logits1 = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_inputs2 = self.buffer.get_data_aux(
                self.args.minibatch_size, transform=self.aux.transform)

            buf_outputs1 = self.net(buf_inputs1)
            buf_outputs_ema1 = self.ema_net1(buf_inputs1)

            loss_buf_ce1 = self.loss(buf_outputs1, buf_labels)
            loss1 += loss_buf_ce1

            if self.args.abl_mode == 'ib':
                loss_aux_ema1 = 0
                loss_aux_ema2 = 0
            else:
                loss_aux_ema1 = self.aux.loss(buf_outputs1, buf_outputs_ema1.detach())

            if self.args.abl_mode != 'nolog':
                loss_log_12 = F.mse_loss(buf_outputs1, buf_logits1)
                loss1 += self.args.alpha_mm[0] * loss_log_12

            loss1 += (loss_aux_ema1 * self.args.loss_wt[2])


        self.aux.collate_loss(loss_dict, loss_ce=loss_ce1, loss_buf_ce=loss_buf_ce1, loss_aux=loss_aux12, loss_aux_buf=loss_aux12_buf, loss_aux_mem=loss_aux_ema1,
                              loss_logit_mem=loss_log_12, m1=True)

        if hasattr(self, 'writer'):
            for loss_name, loss_item in loss_dict.items():
                self.writer.add_scalar('Task {}/{}'.format(self.current_task, loss_name), loss_item,
                                       global_step=self.iteration)

        loss1.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs1.data)

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.args.ema_update_freq:
            self.update_ema_model_variables()

        return loss1.item()


    def update_ema_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.args.ema_alpha)
        for ema_param, param in zip(self.ema_net1.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def end_task(self, dataset) -> None:
        self.current_task += 1
