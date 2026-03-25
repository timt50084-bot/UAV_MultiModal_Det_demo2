from abc import ABC, abstractmethod

class BaseMetric(ABC):
    """
    【指标评估基类】
    所有的评估器 (如 OBBMetricsEvaluator) 必须继承此基类并实现以下方法，
    保证引擎的完全解耦。
    """
    @abstractmethod
    def reset(self):
        """清空缓存状态，在每个 Epoch 验证开始时调用"""
        pass

    @abstractmethod
    def add_batch(self, image_ids, batch_preds, batch_gts):
        """
        接收单个 batch 的预测结果和真实标签并存入内存
        """
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        """执行实际的评估计算逻辑"""
        pass

    @abstractmethod
    def get_full_metrics(self) -> dict:
        """
        返回包含所有指标的字典，例如 {'mAP_50': 0.8, 'mAP_S': 0.6}
        用于打印报告和触发早停保存
        """
        pass