# 分布式 DDP 工具
import torch.distributed as dist

def is_dist_avail_and_initialized():
    """判断是否处于 DDP 分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    """判断当前是否为主进程 (Rank 0)，用于控制只打印一次日志和只保存一次模型"""
    return get_rank() == 0

def setup_for_distributed(is_master):
    """
    为了避免多卡训练时终端输出混杂，
    关闭非主进程的内建 print 函数。
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
