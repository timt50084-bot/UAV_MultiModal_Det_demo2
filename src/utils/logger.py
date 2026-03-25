# 规范化的 logging 封装
import logging
import sys
from .dist import is_main_process


def setup_logger(name="UAV_Det", save_dir=None, level=logging.INFO):
    """配置全局工业级 Logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    # 只有主进程输出日志到控制台
    if is_main_process():
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # 如果指定了目录，则主进程保存日志到文件
    if save_dir and is_main_process():
        fh = logging.FileHandler(f"{save_dir}/train.log", mode='a')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger