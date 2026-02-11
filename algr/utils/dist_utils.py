# encoding: utf-8
"""
设备抽象：支持 cuda 与 npu，用于 algr 分布式训练。
通过 device_type 指定 'cuda' 或 'npu'，后续统一使用 get_device 得到 device。
支持单进程直接运行（python runner.py）与 torchrun 分布式启动。
"""
import os


def setup_for_distributed(is_master: bool):
    """单进程时 is_master=True，所有 print 正常输出；多进程时仅主进程打印。"""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_device(device_type: str, local_rank: int = 0):
    """
    根据 device_type 和 local_rank 返回当前进程使用的 torch.device。
    """
    import torch
    dtype = (device_type or 'cuda').lower()
    device_id = max(local_rank, 0)
    if dtype == 'npu':
        return torch.device(f'npu:{device_id}')
    return torch.device(f'cuda:{device_id}')


def get_device_for_model_loading(device_type: str, local_rank: int = 0):
    """
    返回用于 model.from_pretrained(device_map=...) 的设备。
    对于单卡加载，返回设备字符串如 "npu:0" 或 "cuda:0"。
    """
    dtype = (device_type or 'cuda').lower()
    device_id = max(local_rank, 0)
    if dtype == 'npu':
        return f'npu:{device_id}'
    return f'cuda:{device_id}'


def init_distributed_mode(device_type: str, args):
    """
    初始化分布式环境。若 RANK、WORLD_SIZE 未设置（直接 python runner.py），则单进程运行；
    若已设置（torchrun 启动），则根据 device_type 初始化：npu 用 hccl，cuda 用 nccl。
    返回: True 表示分布式模式，False 表示单进程模式。
    """
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    dtype = (device_type or 'cuda').lower()
    local_rank = int(os.environ.get('LOCAL_RANK', getattr(args, 'local_rank', 0)))
    local_rank = max(local_rank, 0)  # 单进程时 local_rank 可能为 -1，统一为 0

    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        # 单进程直接运行，不初始化 process_group
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)
        if dtype == 'npu':
            try:
                import torch_npu  # noqa: F401
            except ImportError as e:
                raise ImportError('NPU 训练需要安装 torch_npu，且 device_type 为 npu。') from e
            torch.npu.set_device(local_rank)
        elif dtype == 'cuda':
            torch.cuda.set_device(local_rank)
        return False

    # 分布式模式
    if dtype == 'npu':
        try:
            import torch_npu  # noqa: F401
        except ImportError as e:
            raise ImportError('NPU 训练需要安装 torch_npu，且 device_type 为 npu。') from e
        torch.npu.set_device(local_rank)
        backend = 'hccl'
    else:
        torch.cuda.set_device(local_rank)
        backend = 'nccl'

    dist.init_process_group(backend=backend, timeout=timedelta(seconds=1800))
    return True
