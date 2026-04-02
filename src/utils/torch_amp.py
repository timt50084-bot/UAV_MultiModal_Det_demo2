from contextlib import nullcontext

try:
    from torch.amp import GradScaler as _GradScaler
    from torch.amp import autocast as _autocast
    _USES_DEVICE_TYPE = True
except ImportError:  # pragma: no cover - older torch versions
    from torch.cuda.amp import GradScaler as _GradScaler
    from torch.cuda.amp import autocast as _cuda_autocast
    _USES_DEVICE_TYPE = False


def autocast(device_type='cuda', enabled=False):
    if _USES_DEVICE_TYPE:
        return _autocast(device_type=device_type, enabled=enabled)
    if str(device_type) != 'cuda':
        return nullcontext()
    return _cuda_autocast(enabled=enabled)


def make_grad_scaler(device_type='cuda', enabled=False):
    if _USES_DEVICE_TYPE:
        return _GradScaler(device_type, enabled=enabled)
    return _GradScaler(enabled=bool(enabled and str(device_type) == 'cuda'))
