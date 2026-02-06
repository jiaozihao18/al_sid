# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import datetime
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.distributed as dist

from .dist_utils import is_dist_avail_and_initialized


def _get_device_for_tensor(device):
    """返回用于 all_reduce 等通信的 device，保证与后端一致。"""
    if device is None:
        return torch.device('cuda:0')
    return device


def get_device_memory_mb(device):
    """与设备无关：返回 (是否显示显存, 显存MB)。NPU/CUDA 可显示，CPU 不显示。"""
    if device is None:
        if torch.cuda.is_available():
            return True, torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        return False, 0.0
    if device.type == 'cuda':
        return True, torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    if device.type == 'npu':
        return True, torch.npu.max_memory_allocated(device) / (1024.0 * 1024.0)
    return False, 0.0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self, device=None):
        if not is_dist_avail_and_initialized():
            return
        dev = _get_device_for_tensor(device)
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device=dev)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('[Time]: {} | '.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, device=None):
        """
        Warning: does not synchronize the deque!
        device: 与训练 device 一致，用于 all_reduce 的 tensor 所在设备。
        """
        if not is_dist_avail_and_initialized():
            return
        dev = _get_device_for_tensor(device)
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=dev)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t", device=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.device = device

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self, device=None):
        dev = device if device is not None else self.device
        for meter in self.meters.values():
            meter.synchronize_between_processes(device=dev)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, num_batches_per_epoch, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(num_batches_per_epoch))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        show_mem, _ = get_device_memory_mb(getattr(self, 'device', None))
        if show_mem:
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0

        for i in range(num_batches_per_epoch):
            data = next(iterable)
            data_time.update(time.time() - end)
            yield data
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == num_batches_per_epoch - 1:
                eta_seconds = iter_time.global_avg * (num_batches_per_epoch - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                show_mem, memory_mb = get_device_memory_mb(getattr(self, 'device', None))
                if show_mem:
                    print(log_msg.format(
                        i, num_batches_per_epoch, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=memory_mb))
                else:
                    print(log_msg.format(
                        i, num_batches_per_epoch, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / num_batches_per_epoch))

    def log_every_list(self, iterable_list, num_batches_per_epoch_list, print_freq, epoch, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        num_batches_per_epoch = int(sum(num_batches_per_epoch_list))
        space_fmt = ':' + str(len(str(num_batches_per_epoch))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        show_mem, _ = get_device_memory_mb(getattr(self, 'device', None))
        if show_mem:
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)

        random_list = np.concatenate([[idx] * c for idx, c in enumerate(num_batches_per_epoch_list)])
        np.random.seed(epoch)
        np.random.shuffle(random_list)
        print(random_list[:50])

        for i in range(num_batches_per_epoch):
            iterable = iterable_list[random_list[i]]
            data = next(iterable)
            data_time.update(time.time() - end)
            yield (data, random_list[i])
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == num_batches_per_epoch - 1:
                eta_seconds = iter_time.global_avg * (num_batches_per_epoch - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                show_mem, memory_mb = get_device_memory_mb(getattr(self, 'device', None))
                if show_mem:
                    print(log_msg.format(
                        i, num_batches_per_epoch, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=memory_mb))
                else:
                    print(log_msg.format(
                        i, num_batches_per_epoch, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / num_batches_per_epoch))

    def log_every_list_with_datasetname(self, iterable_list, num_batches_per_epoch_list, dataset_names, print_freq, epoch, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        num_batches_per_epoch = int(sum(num_batches_per_epoch_list))
        space_fmt = ':' + str(len(str(num_batches_per_epoch))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        show_mem, _ = get_device_memory_mb(getattr(self, 'device', None))
        if show_mem:
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)

        random_list = np.concatenate([[idx] * c for idx, c in enumerate(num_batches_per_epoch_list)])
        np.random.seed(epoch)
        np.random.shuffle(random_list)
        print(random_list[:50])

        for i in range(num_batches_per_epoch):
            iterable = iterable_list[random_list[i]]
            data = next(iterable)
            data_time.update(time.time() - end)
            yield (data, random_list[i], dataset_names[random_list[i]])
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == num_batches_per_epoch - 1:
                eta_seconds = iter_time.global_avg * (num_batches_per_epoch - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                show_mem, memory_mb = get_device_memory_mb(getattr(self, 'device', None))
                if show_mem:
                    print(log_msg.format(
                        i, num_batches_per_epoch, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=memory_mb))
                else:
                    print(log_msg.format(
                        i, num_batches_per_epoch, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / num_batches_per_epoch))
