try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight test envs
    torch = None


DETECTION_CUDA_REQUIRED_MESSAGE = (
    'This detection path requires CUDA. CPU detection reference has been removed.'
)


def resolve_detection_device(device_arg):
    if torch is None:
        raise RuntimeError(
            f'{DETECTION_CUDA_REQUIRED_MESSAGE} torch is not available in the current environment.'
        )

    device_index = int(device_arg)
    if device_index < 0:
        raise RuntimeError(f'{DETECTION_CUDA_REQUIRED_MESSAGE} Received --device={device_index}.')
    if not torch.cuda.is_available():
        raise RuntimeError(
            f'{DETECTION_CUDA_REQUIRED_MESSAGE} torch.cuda.is_available() is False.'
        )
    return torch.device(f'cuda:{device_index}')


def require_detection_cuda_device(device):
    device_type = str(getattr(device, 'type', device))
    if torch is None or device_type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError(
            f"{DETECTION_CUDA_REQUIRED_MESSAGE} Received device '{device}'."
        )
    return device
