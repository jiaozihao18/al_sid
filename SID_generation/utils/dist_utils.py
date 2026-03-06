# encoding: utf-8
"""
@author: Yingwu.XSW
@date: 2022/3/9 下午6:35

NPU 训练：单卡时 cfg.dist.npu_id 来自 yml；多卡时由 torchrun 设置 RANK/WORLD_SIZE/LOCAL_RANK，
init_distributed_mode 从环境变量注入 rank/world_size/npu_id，不读 yml 的 dist（除单卡 npu_id）。
"""
import os

import torch
import torch.autograd as autograd
import torch.distributed as dist
from omegaconf import OmegaConf


def is_main_process():
    return (dist.get_rank() if dist.is_available() and dist.is_initialized() else 0) == 0


def main_print(*args, **kwargs):
    """只在主进程打印，避免多卡重复输出"""
    if is_main_process():
        print(*args, **kwargs)


def init_distributed_mode(cfg):
    """
    单卡：仅设 cfg.dist.distributed=False，npu_id 沿用 yml。
    多卡：由 torchrun 注入 RANK/WORLD_SIZE/LOCAL_RANK，据此设置 rank/world_size/npu_id 并初始化进程组。
    """
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_readonly(cfg, False)

    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        print('Not using distributed mode')
        cfg.dist.distributed = False
        OmegaConf.set_struct(cfg, True)
        OmegaConf.set_readonly(cfg, True)
        return

    cfg.dist.distributed = True
    cfg.dist.npu_id = int(os.environ.get('LOCAL_RANK', 0))

    torch.npu.set_device(cfg.dist.npu_id)
    torch.distributed.init_process_group(backend='hccl', init_method='env://')
    cfg.dist.rank = dist.get_rank()
    cfg.dist.world_size = dist.get_world_size()
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_readonly(cfg, True)


def scaled_all_reduce(tensors, is_scale=True):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    world size.
    """
    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    if world_size == 1:
        return tensors
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)  # default sum
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    if is_scale:
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


class GatherLayer(autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def all_reduce_mean(x, device=None):
    """device 由调用方传入（main 中根据 cfg.dist.npu_id 构造）。"""
    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    if world_size > 1:
        if device is None:
            device = torch.device('cuda:0')  # 兼容旧调用
        x_reduce = torch.tensor(x, dtype=torch.float32, device=device)
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    return x


def all_reduce_mean_batch(args, device=None):
    """
    对输入的多个变量（张量）进行全局均值汇聚。
    device 由调用方传入。
    """
    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    results = []
    if device is None:
        device = torch.device('cuda:0')
    for x in args:
        if world_size > 1:
            x_reduce = torch.tensor(x, dtype=torch.float32, device=device)
            dist.all_reduce(x_reduce)
            x_reduce /= world_size
            results.append(x_reduce.item())
        else:
            results.append(x)
    return results
