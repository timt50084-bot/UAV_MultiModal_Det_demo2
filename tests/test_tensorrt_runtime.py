import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.deployment.tensorrt_runtime import (
    TensorRTEngineRunner,
    TensorRTRuntimeError,
    inspect_engine_io,
    select_builder_backend,
)
from tools.build_trt_engine import build_parser as build_trt_engine_parser
from tools.infer_trt import build_parser as infer_trt_parser


class _FakeTensorMode:
    INPUT = 'input'
    OUTPUT = 'output'


class _FakeTRT:
    TensorIOMode = _FakeTensorMode


class _FakeEngine:
    num_io_tensors = 3

    def get_tensor_name(self, index):
        return ['images_rgb', 'images_ir', 'outputs'][index]

    def get_tensor_mode(self, name):
        return {
            'images_rgb': _FakeTensorMode.INPUT,
            'images_ir': _FakeTensorMode.INPUT,
            'outputs': _FakeTensorMode.OUTPUT,
        }[name]

    def get_tensor_dtype(self, name):
        return {
            'images_rgb': 'float32',
            'images_ir': 'float32',
            'outputs': 'float32',
        }[name]

    def get_tensor_shape(self, name):
        return {
            'images_rgb': (1, 3, 640, 640),
            'images_ir': (1, 3, 640, 640),
            'outputs': (1, 8400, 10),
        }[name]


class TensorRTRuntimeTestCase(unittest.TestCase):
    def test_builder_backend_reports_missing_dependencies_clearly(self):
        with patch('src.deployment.tensorrt_runtime.load_tensorrt_module', side_effect=TensorRTRuntimeError('no trt')):
            with patch('src.deployment.tensorrt_runtime.shutil.which', return_value=None):
                with self.assertRaisesRegex(TensorRTRuntimeError, 'No TensorRT engine builder backend is available'):
                    select_builder_backend('auto')

    def test_build_trt_engine_parser_accepts_minimal_args(self):
        parser = build_trt_engine_parser()
        args = parser.parse_args(['--onnx', 'model.onnx', '--fp16'])
        self.assertEqual(args.onnx, 'model.onnx')
        self.assertTrue(args.fp16)
        self.assertEqual(args.backend, 'auto')

    def test_infer_trt_parser_accepts_minimal_args(self):
        parser = infer_trt_parser()
        args = parser.parse_args([
            '--engine', 'model.engine',
            '--source_rgb', 'rgb_dir',
            '--source_ir', 'ir_dir',
        ])
        self.assertEqual(args.engine, 'model.engine')
        self.assertEqual(args.source_rgb, 'rgb_dir')
        self.assertEqual(args.source_ir, 'ir_dir')

    def test_runner_reports_missing_tensorrt_runtime_clearly(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine_path = Path(tmp_dir) / 'model.engine'
            engine_path.write_bytes(b'fake-engine')
            with patch('src.deployment.tensorrt_runtime.torch.cuda.is_available', return_value=True):
                with patch('src.deployment.tensorrt_runtime.load_tensorrt_module', side_effect=TensorRTRuntimeError('TensorRT Python package is not installed')):
                    with self.assertRaisesRegex(TensorRTRuntimeError, 'TensorRT Python package is not installed'):
                        TensorRTEngineRunner(engine_path)

    def test_inspect_engine_io_supports_tensor_io_api(self):
        io_meta = inspect_engine_io(_FakeEngine(), trt_module=_FakeTRT)
        self.assertEqual([entry['name'] for entry in io_meta], ['images_rgb', 'images_ir', 'outputs'])
        self.assertEqual([entry['role'] for entry in io_meta], ['input', 'input', 'output'])
        self.assertEqual(io_meta[2]['shape'], (1, 8400, 10))


if __name__ == '__main__':
    unittest.main()
