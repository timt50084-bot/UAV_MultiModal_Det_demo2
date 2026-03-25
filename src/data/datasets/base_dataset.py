from torch.utils.data import Dataset
from abc import abstractmethod

class BaseDataset(Dataset):
    """【数据集接口规范】强制所有的 Dataset 实现 set_epoch 接口以支持难度感知"""
    def __init__(self, root_dir, split, is_training):
        self.root_dir = root_dir
        self.split = split
        self.is_training = is_training
        self.current_epoch = 0
        self.max_epoch = 100

    def set_epoch(self, epoch, max_epoch):
        self.current_epoch = epoch
        self.max_epoch = max_epoch

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass