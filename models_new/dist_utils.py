import os

def setup_for_distributed(is_master: bool):
    """单进程时 is_master=True，所有 print 正常输出；多进程时仅主进程打印。"""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(npu_id: int = 0):
    """
    初始化分布式环境（仅支持 NPU）：
    - 若 RANK / WORLD_SIZE 未设置，则单进程运行，不创建 process_group。
    - 若已设置，则使用 HCCL 初始化 NPU 多进程。
    """
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    local_rank = int(os.environ.get("LOCAL_RANK", npu_id))
    local_rank = max(local_rank, 0)

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)
        torch.npu.set_device(local_rank)
        return False

    # 分布式 NPU 模式
    torch.npu.set_device(local_rank)
    backend = "hccl"
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=1800))
    setup_for_distributed(is_master=(dist.get_rank() == 0))
    return True

