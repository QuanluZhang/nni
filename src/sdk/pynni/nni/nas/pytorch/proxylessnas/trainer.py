# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import time
import json

import torch
from torch import nn as nn

from nni.nas.pytorch.base_trainer import BaseTrainer
#from nni.nas.pytorch.trainer import TorchTensorEncoder
from nni.nas.pytorch.utils import AverageMeter
from .mutator import ProxylessNasMutator
from .utils import cross_entropy_with_label_smoothing, accuracy

class TorchTensorEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=method-hidden
        if isinstance(o, torch.Tensor):
            olist = o.tolist()
            if "bool" not in o.type().lower() and all(map(lambda d: d == 0 or d == 1, olist)):
                print("Every element in %s is either 0 or 1. "
                                "You might consider convert it into bool.", olist)
            return olist
        return super().default(o)

class ProxylessNasTrainer(BaseTrainer):
    def __init__(self, model, model_optim, device,
                 train_loader, valid_loader, label_smoothing=0.1,
                 n_epochs=120, init_lr=0.025, binary_mode='full_v2',
                 arch_init_type='normal', arch_init_ratio=1e-3,
                 arch_optim_lr=1e-3, arch_weight_decay=0,
                 grad_update_arch_param_every=5, grad_update_steps=1,
                 warmup=True, warmup_epochs=25,
                 arch_valid_frequency=1,
                 load_ckpt=False, ckpt_path=None, arch_path=None):
        """
        Parameters
        ----------
        model : pytorch model
        model_optim : pytorch optimizer
        train_loader : pytorch data loader
        valid_loader : pytorch data loader
        device : device
        n_epochs : int
        init_lr : float
            init learning rate for training the model
        arch_init_type : str
            the way to init architecture parameters
        arch_init_ratio : float
            the ratio to init architecture parameters
        arch_optim_lr : float
            learning rate of the architecture parameters optimizer
        arch_weight_decay : float
            weight decay of the architecture parameters optimizer
        warmup : bool
            whether to do warmup
        warmup_epochs : int
            the number of epochs to do in warmup
        arch_valid_frequency : int
            frequency of printing validation result
        """
        self.model = model
        self.model_optim = model_optim
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.warmup = warmup
        self.warmup_epochs = warmup_epochs
        self.arch_valid_frequency = arch_valid_frequency
        self.label_smoothing = label_smoothing

        self.train_batch_size = train_loader.batch_sampler.batch_size
        self.valid_batch_size = valid_loader.batch_sampler.batch_size
        # update architecture parameters every this number of minibatches
        self.grad_update_arch_param_every = grad_update_arch_param_every
        # the number of steps per architecture parameter update
        self.grad_update_steps = grad_update_steps
        self.binary_mode = binary_mode

        self.load_ckpt = load_ckpt
        self.ckpt_path = ckpt_path
        self.arch_path = arch_path

        # init mutator
        self.mutator = ProxylessNasMutator(model)

        # DataParallel should be put behind the init of mutator
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        # iter of valid dataset for training architecture weights
        self._valid_iter = None
        # init architecture weights
        self._init_arch_params(arch_init_type, arch_init_ratio)
        # build architecture optimizer
        self.arch_optimizer = torch.optim.Adam(self.mutator.get_architecture_parameters(),
                                               arch_optim_lr,
                                               weight_decay=arch_weight_decay,
                                               betas=(0, 0.999),
                                               eps=1e-8)

        self.criterion = nn.CrossEntropyLoss()
        self.warmup_curr_epoch = 0
        self.train_curr_epoch = 0

    def _init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.mutator.get_architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def _validate(self):
        self.valid_loader.batch_sampler.batch_size = self.valid_batch_size
        self.valid_loader.batch_sampler.drop_last = False

        self.mutator.set_chosen_op_active()
        # remove unused modules to save memory
        self.mutator.unused_modules_off()
        # test on validation set under train mode
        self.model.train()
        batch_time = AverageMeter('batch_time')
        losses = AverageMeter('losses')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        end = time.time()
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.valid_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                loss = self.criterion(output, labels)
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0 or i + 1 == len(self.valid_loader):
                    test_log = 'Valid' + ': [{0}/{1}]\t'\
                                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'.\
                        format(i, len(self.valid_loader) - 1, batch_time=batch_time, loss=losses, top1=top1)
                    # return top5:
                    test_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
                    print(test_log)
        self.mutator.unused_modules_back()
        return losses.avg, top1.avg, top5.avg

    def _warm_up(self):
        lr_max = 0.05
        data_loader = self.train_loader
        nBatch = len(data_loader)
        T_total = self.warmup_epochs * nBatch # total num of batches

        for epoch in range(self.warmup_curr_epoch, self.warmup_epochs):
            print('\n', '-' * 30, 'Warmup epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter('batch_time')
            data_time = AverageMeter('data_time')
            losses = AverageMeter('losses')
            top1 = AverageMeter('top1')
            top5 = AverageMeter('top5')
            # switch to train mode
            self.model.train()

            end = time.time()
            print('=====================_warm_up, epoch: ', epoch)
            for i, (images, labels) in enumerate(data_loader):
                #print('=====================_warm_up, minibatch i: ', i)
                data_time.update(time.time() - end)
                # lr
                T_cur = epoch * nBatch + i
                warmup_lr = 0.5 * lr_max * (1 + math.cos(math.pi * T_cur / T_total))
                for param_group in self.model_optim.param_groups:
                    param_group['lr'] = warmup_lr
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                self.mutator.reset_binary_gates() # random sample binary gates
                # remove unused module for speedup
                self.mutator.unused_modules_off()
                output = self.model(images)
                if self.label_smoothing > 0:
                    loss = cross_entropy_with_label_smoothing(output, labels, self.label_smoothing)
                else:
                    loss = self.criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # compute gradient and do SGD step
                self.model.zero_grad()
                loss.backward()
                self.model_optim.step()
                # unused modules back
                self.mutator.unused_modules_back()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0 or i + 1 == nBatch:
                    batch_log = 'Warmup Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, top1=top1, top5=top5, lr=warmup_lr)
                    print(batch_log)
                self.save_checkpoint()
            val_loss, val_top1, val_top5 = self._validate()
            val_log = 'Warmup Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f}\ttop-5 acc {4:.3f}\t' \
                      'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}M'. \
                format(epoch + 1, self.warmup_epochs, val_loss, val_top1, val_top5, top1=top1, top5=top5)
            print(val_log)
            #self.save_checkpoint()
            self.warmup_curr_epoch += 1

    def _get_update_schedule(self, nBatch):
        schedule = {}
        for i in range(nBatch):
            if (i + 1) % self.grad_update_arch_param_every == 0:
                schedule[i] = self.grad_update_steps
        return schedule

    def _calc_learning_rate(self, epoch, batch=0, nBatch=None):
        T_total = self.n_epochs * nBatch
        T_cur = epoch * nBatch + batch
        lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
        return lr

    def _adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """
        Adjust learning of a given optimizer and return the new learning rate
        """
        new_lr = self._calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def _train(self):
        nBatch = len(self.train_loader)
        arch_param_num = self.mutator.num_arch_params()
        binary_gates_num = self.mutator.num_arch_params()
        #weight_param_num = len(list(self.net.weight_parameters()))
        print(
            '#arch_params: %d\t#binary_gates: %d\t#weight_params: xx' %
            (arch_param_num, binary_gates_num)
        )

        update_schedule = self._get_update_schedule(nBatch)

        for epoch in range(self.train_curr_epoch, self.n_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter('batch_time')
            data_time = AverageMeter('data_time')
            losses = AverageMeter('losses')
            top1 = AverageMeter('top1')
            top5 = AverageMeter('top5')
            entropy = AverageMeter('entropy')
            # switch to train mode
            self.model.train()

            end = time.time()
            for i, (images, labels) in enumerate(self.train_loader):
                data_time.update(time.time() - end)
                lr = self._adjust_learning_rate(self.model_optim, epoch, batch=i, nBatch=nBatch)
                # network entropy
                #net_entropy = self.mutator.entropy()
                #entropy.update(net_entropy.data.item() / arch_param_num, 1)
                # train weight parameters
                images, labels = images.to(self.device), labels.to(self.device)
                self.mutator.reset_binary_gates()
                self.mutator.unused_modules_off()
                output = self.model(images)
                if self.label_smoothing > 0:
                    loss = cross_entropy_with_label_smoothing(output, labels, self.label_smoothing)
                else:
                    loss = self.criterion(output, labels)
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                self.model.zero_grad()
                loss.backward()
                self.model_optim.step()
                self.mutator.unused_modules_back()
                if epoch > 0:
                    for _ in range(update_schedule.get(i, 0)):
                        start_time = time.time()
                        # GradientArchSearchConfig
                        self.mutator.arch_requires_grad()
                        arch_loss, exp_value = self._gradient_step()
                        self.mutator.arch_disable_grad()
                        used_time = time.time() - start_time
                        log_str = 'Architecture [%d-%d]\t Time %.4f\t Loss %.4f\t null %s' % \
                                    (epoch + 1, i, used_time, arch_loss, exp_value)
                        print(log_str)
                batch_time.update(time.time() - end)
                end = time.time()
                # training log
                if i % 10 == 0 or i + 1 == nBatch:
                    batch_log = 'Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Entropy {entropy.val:.5f} ({entropy.avg:.5f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, entropy=entropy, top1=top1, top5=top5, lr=lr)
                    print(batch_log)
            # TODO: print current network architecture
            # validate
            if (epoch + 1) % self.arch_valid_frequency == 0:
                val_loss, val_top1, val_top5 = self._validate()
                val_log = 'Valid [{0}]\tloss {1:.3f}\ttop-1 acc {2:.3f} \ttop-5 acc {3:.3f}\t' \
                          'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\t' \
                          'Entropy {entropy.val:.5f}M'. \
                    format(epoch + 1, val_loss, val_top1,
                           val_top5, entropy=entropy, top1=top1, top5=top5)
                print(val_log)
            self.save_checkpoint()
            self.train_curr_epoch += 1
        # convert to normal network according to architecture parameters

    def _valid_next_batch(self):
        if self._valid_iter is None:
            self._valid_iter = iter(self.valid_loader)
        try:
            data = next(self._valid_iter)
        except StopIteration:
            self._valid_iter = iter(self.valid_loader)
            data = next(self._valid_iter)
        return data

    def _gradient_step(self):
        # use the same batch size as train batch size for architecture weights
        self.valid_loader.batch_sampler.batch_size = self.train_batch_size
        self.valid_loader.batch_sampler.drop_last = True
        self.model.train()
        self.mutator.change_forward_mode(self.binary_mode)
        time1 = time.time()  # time
        # sample a batch of data from validation set
        images, labels = self._valid_next_batch()
        images, labels = images.to(self.device), labels.to(self.device)
        time2 = time.time()  # time
        self.mutator.reset_binary_gates()
        self.mutator.unused_modules_off()
        output = self.model(images)
        time3 = time.time()
        ce_loss = self.criterion(output, labels)
        expected_value = None
        loss = ce_loss
        self.model.zero_grad()
        loss.backward()
        self.mutator.set_arch_param_grad()
        self.arch_optimizer.step()
        if self.mutator.get_forward_mode() == 'two':
            self.mutator.rescale_updated_arch_param()
        self.mutator.unused_modules_back()
        self.mutator.change_forward_mode(None)
        time4 = time.time()
        print('(%.4f, %.4f, %.4f)' % (time2 - time1, time3 - time2, time4 - time3))
        return loss.data.item(), expected_value.item() if expected_value is not None else None

    def save_checkpoint(self):
        if self.ckpt_path:
            state = {
                'warmup_curr_epoch': self.warmup_curr_epoch,
                'train_curr_epoch': self.train_curr_epoch,
                'model': self.model.state_dict(),
                'optim': self.model_optim.state_dict(),
                'arch_optim': self.arch_optimizer.state_dict()
            }
            torch.save(state, self.ckpt_path)
        if self.arch_path:
            self.export(self.arch_path)

    def load_checkpoint(self):
        assert self.ckpt_path is not None, "If load_ckpt is not None, ckpt_path should not be None"
        ckpt = torch.load(self.ckpt_path)
        self.warmup_curr_epoch = ckpt['warmup_curr_epoch']
        self.train_curr_epoch = ckpt['train_curr_epoch']
        self.model.load_state_dict(ckpt['model'])
        self.model_optim.load_state_dict(ckpt['optim'])
        self.arch_optimizer.load_state_dict(ckpt['arch_optim'])

    def train(self):
        if self.load_ckpt:
            load_checkpoint()
        if self.warmup:
            self._warm_up()
        self._train()

    def export(self, file_name):
        exported_arch = self.mutator.sample_final()
        with open(file_name, 'w') as f:
            json.dump(exported_arch, f, indent=2, sort_keys=True, cls=TorchTensorEncoder)

    def validate(self):
        raise NotImplementedError

    def train_and_validate(self):
        raise NotImplementedError
