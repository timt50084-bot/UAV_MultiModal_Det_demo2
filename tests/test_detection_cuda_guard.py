import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight test envs
    torch = None

from src.utils.detection_cuda import require_detection_cuda_device, resolve_detection_device


@unittest.skipUnless(torch is not None, 'torch is required for CUDA guard tests')
class DetectionCudaGuardTestCase(unittest.TestCase):
    def test_resolve_detection_device_rejects_cpu_flag(self):
        with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):
            resolve_detection_device(-1)

    def test_resolve_detection_device_rejects_missing_cuda(self):
        with patch('src.utils.detection_cuda.torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):
                resolve_detection_device(0)

    def test_resolve_detection_device_returns_cuda_device(self):
        with patch('src.utils.detection_cuda.torch.cuda.is_available', return_value=True):
            device = resolve_detection_device(1)

        self.assertEqual(str(device), 'cuda:1')

    def test_require_detection_cuda_device_rejects_cpu_device(self):
        with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):
            require_detection_cuda_device(torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
