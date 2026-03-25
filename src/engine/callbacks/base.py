# Callback 基类 (定义 on_epoch_end 等钩子)
class Callback:
    """标准生命周期回调基类，支持在训练/验证的不同阶段插入自定义逻辑"""
    def on_train_begin(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_epoch_begin(self, trainer): pass
    def on_epoch_end(self, trainer): pass
    def on_batch_begin(self, trainer): pass
    def on_batch_end(self, trainer): pass
    def on_val_begin(self, evaluator): pass
    def on_val_end(self, evaluator, metrics): pass