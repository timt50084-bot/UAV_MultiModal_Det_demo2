import importlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch


TRT_INPUT_RGB_NAME = 'images_rgb'
TRT_INPUT_IR_NAME = 'images_ir'
TRT_OUTPUT_NAME = 'outputs'


class TensorRTRuntimeError(RuntimeError):
    pass


def load_tensorrt_module():
    try:
        return importlib.import_module('tensorrt')
    except ImportError as exc:
        raise TensorRTRuntimeError(
            'TensorRT Python package is not installed. '
            'Build/install TensorRT first, or use `tools/build_trt_engine.py --backend trtexec` '
            'for engine build only. TensorRT inference still requires the Python TensorRT runtime.'
        ) from exc


def parse_nchw_shape(text):
    if text is None:
        return None
    if isinstance(text, (tuple, list)):
        if len(text) != 4:
            raise ValueError(f'Expected 4D NCHW shape, got: {text}')
        return tuple(int(v) for v in text)

    normalized = str(text).strip().lower().replace('x', ' ').replace(',', ' ')
    parts = [part for part in normalized.split() if part]
    if len(parts) != 4:
        raise ValueError(f'Expected NCHW shape like `1x3x640x640`, got: {text!r}')
    return tuple(int(part) for part in parts)


def _shape_to_text(shape):
    return 'x'.join(str(int(v)) for v in shape)


def _resolve_profile_shapes(shape=None, min_shape=None, opt_shape=None, max_shape=None):
    base_shape = parse_nchw_shape(shape) if shape is not None else None
    min_shape = parse_nchw_shape(min_shape) if min_shape is not None else base_shape
    opt_shape = parse_nchw_shape(opt_shape) if opt_shape is not None else base_shape
    max_shape = parse_nchw_shape(max_shape) if max_shape is not None else base_shape

    populated = [value for value in (min_shape, opt_shape, max_shape) if value is not None]
    if populated and len(populated) != 3:
        raise ValueError('Dynamic TensorRT profile requires shape, or all of min_shape/opt_shape/max_shape.')
    return min_shape, opt_shape, max_shape


def select_builder_backend(preferred='auto', trtexec_path=None):
    preferred = str(preferred).lower()
    if preferred not in {'auto', 'python', 'trtexec'}:
        raise ValueError(f'Unsupported TensorRT builder backend: {preferred}')

    trtexec_path = str(trtexec_path) if trtexec_path else shutil.which('trtexec')

    if preferred in {'auto', 'python'}:
        try:
            load_tensorrt_module()
            return 'python'
        except TensorRTRuntimeError:
            if preferred == 'python':
                raise

    if preferred in {'auto', 'trtexec'} and trtexec_path:
        return 'trtexec'

    raise TensorRTRuntimeError(
        'No TensorRT engine builder backend is available. '
        'Install the TensorRT Python package, or make sure `trtexec` is on PATH.'
    )


def inspect_engine_io(engine, trt_module=None):
    io_meta = []
    if hasattr(engine, 'num_io_tensors'):
        if trt_module is None:
            raise ValueError('trt_module is required when inspecting a TensorRT engine with tensor I/O API.')
        for index in range(int(engine.num_io_tensors)):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            role = 'input' if mode == trt_module.TensorIOMode.INPUT else 'output'
            dtype = engine.get_tensor_dtype(name)
            shape = tuple(int(dim) for dim in engine.get_tensor_shape(name))
            io_meta.append({'name': name, 'role': role, 'dtype': dtype, 'shape': shape, 'index': index})
        return io_meta

    for index in range(int(engine.num_bindings)):
        name = engine.get_binding_name(index)
        role = 'input' if engine.binding_is_input(index) else 'output'
        dtype = engine.get_binding_dtype(index)
        shape = tuple(int(dim) for dim in engine.get_binding_shape(index))
        io_meta.append({'name': name, 'role': role, 'dtype': dtype, 'shape': shape, 'index': index})
    return io_meta


def _select_primary_name(io_meta, expected_name, role, position):
    names = [entry['name'] for entry in io_meta if entry['role'] == role]
    if not names:
        raise TensorRTRuntimeError(f'TensorRT engine has no {role} tensors.')
    if expected_name in names:
        return expected_name
    if position >= len(names):
        raise TensorRTRuntimeError(f'Could not resolve {role} tensor at position {position}. Available: {names}')
    return names[position]


def _trt_dtype_to_torch(dtype, trt_module):
    np_dtype = np.dtype(trt_module.nptype(dtype))
    mapping = {
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.bool_): torch.bool,
    }
    if np_dtype not in mapping:
        raise TensorRTRuntimeError(f'Unsupported TensorRT dtype: {np_dtype}')
    return mapping[np_dtype]


def build_engine_with_trtexec(
    onnx_path,
    engine_path,
    fp16=False,
    workspace_mb=2048,
    shape=None,
    min_shape=None,
    opt_shape=None,
    max_shape=None,
    trtexec_path=None,
    verbose=False,
):
    trtexec_path = str(trtexec_path) if trtexec_path else shutil.which('trtexec')
    if not trtexec_path:
        raise TensorRTRuntimeError('`trtexec` was not found on PATH.')

    min_shape, opt_shape, max_shape = _resolve_profile_shapes(
        shape=shape,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
    )

    command = [
        trtexec_path,
        f'--onnx={onnx_path}',
        f'--saveEngine={engine_path}',
        f'--workspace={int(workspace_mb)}',
    ]
    if fp16:
        command.append('--fp16')
    if verbose:
        command.append('--verbose')

    if min_shape and opt_shape and max_shape:
        command.extend([
            f'--minShapes={TRT_INPUT_RGB_NAME}:{_shape_to_text(min_shape)},{TRT_INPUT_IR_NAME}:{_shape_to_text(min_shape)}',
            f'--optShapes={TRT_INPUT_RGB_NAME}:{_shape_to_text(opt_shape)},{TRT_INPUT_IR_NAME}:{_shape_to_text(opt_shape)}',
            f'--maxShapes={TRT_INPUT_RGB_NAME}:{_shape_to_text(max_shape)},{TRT_INPUT_IR_NAME}:{_shape_to_text(max_shape)}',
        ])

    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise TensorRTRuntimeError(f'trtexec failed to build engine.\n{stderr}')
    return {'backend': 'trtexec', 'command': command, 'stdout': completed.stdout}


def build_engine_with_python(
    onnx_path,
    engine_path,
    fp16=False,
    workspace_mb=2048,
    shape=None,
    min_shape=None,
    opt_shape=None,
    max_shape=None,
    verbose=False,
):
    trt = load_tensorrt_module()
    min_shape, opt_shape, max_shape = _resolve_profile_shapes(
        shape=shape,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
    )

    logger_severity = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(logger_severity)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = Path(onnx_path).read_bytes()
    if not parser.parse(onnx_bytes):
        error_messages = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise TensorRTRuntimeError('TensorRT ONNX parser failed:\n' + '\n'.join(error_messages))

    config = builder.create_builder_config()
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb) * (1 << 20))
    else:
        config.max_workspace_size = int(workspace_mb) * (1 << 20)

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            raise TensorRTRuntimeError('FP16 engine requested, but TensorRT platform_has_fast_fp16 is false.')

    input_names = [network.get_input(i).name for i in range(network.num_inputs)]
    has_dynamic_input = any(any(int(dim) < 0 for dim in tuple(network.get_input(i).shape)) for i in range(network.num_inputs))
    if has_dynamic_input:
        if not (min_shape and opt_shape and max_shape):
            raise TensorRTRuntimeError(
                'Dynamic ONNX input detected. Provide --shape or explicit --min-shape/--opt-shape/--max-shape.'
            )
        profile = builder.create_optimization_profile()
        for name in input_names:
            profile.set_shape(name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    if hasattr(builder, 'build_serialized_network'):
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise TensorRTRuntimeError('TensorRT builder returned an empty serialized engine.')
        Path(engine_path).write_bytes(serialized_engine)
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise TensorRTRuntimeError('TensorRT builder returned an empty engine.')
        Path(engine_path).write_bytes(engine.serialize())

    return {'backend': 'python', 'input_names': input_names}


class TensorRTEngineRunner:
    def __init__(self, engine_path, device_id=0, logger_level='warning'):
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f'TensorRT engine file does not exist: {self.engine_path}')
        if not torch.cuda.is_available():
            raise TensorRTRuntimeError('TensorRT inference requires CUDA, but torch.cuda.is_available() is false.')

        self.trt = load_tensorrt_module()
        self.device = torch.device(f'cuda:{int(device_id)}')
        torch.cuda.set_device(self.device)

        self.logger = self.trt.Logger(getattr(self.trt.Logger, str(logger_level).upper(), self.trt.Logger.WARNING))
        self.runtime = self.trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise TensorRTRuntimeError(f'Failed to deserialize TensorRT engine: {self.engine_path}')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise TensorRTRuntimeError(f'Failed to create TensorRT execution context: {self.engine_path}')

        self.io_meta = inspect_engine_io(self.engine, trt_module=self.trt)
        self.input_names = [entry['name'] for entry in self.io_meta if entry['role'] == 'input']
        self.output_names = [entry['name'] for entry in self.io_meta if entry['role'] == 'output']
        self.rgb_input_name = _select_primary_name(self.io_meta, TRT_INPUT_RGB_NAME, 'input', 0)
        self.ir_input_name = _select_primary_name(self.io_meta, TRT_INPUT_IR_NAME, 'input', 1)
        self.primary_output_name = _select_primary_name(self.io_meta, TRT_OUTPUT_NAME, 'output', 0)

    def _get_io_dtype(self, name):
        for entry in self.io_meta:
            if entry['name'] == name:
                return _trt_dtype_to_torch(entry['dtype'], self.trt)
        raise KeyError(name)

    def _prepare_inputs(self, images_rgb, images_ir):
        tensors = {
            self.rgb_input_name: images_rgb,
            self.ir_input_name: images_ir,
        }
        prepared = {}
        for name, tensor in tensors.items():
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor)
            tensor = tensor.to(device=self.device, dtype=self._get_io_dtype(name), non_blocking=True).contiguous()
            prepared[name] = tensor
        return prepared

    def infer(self, images_rgb, images_ir):
        inputs = self._prepare_inputs(images_rgb, images_ir)
        if hasattr(self.engine, 'num_io_tensors'):
            return self._infer_tensor_api(inputs)
        return self._infer_legacy_bindings(inputs)

    def _infer_tensor_api(self, inputs):
        for name, tensor in inputs.items():
            self.context.set_input_shape(name, tuple(int(v) for v in tensor.shape))

        outputs = {}
        for name in self.output_names:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise TensorRTRuntimeError(f'Unresolved dynamic TensorRT output shape for {name}: {shape}')
            outputs[name] = torch.empty(shape, device=self.device, dtype=self._get_io_dtype(name))

        for name, tensor in {**inputs, **outputs}.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise TensorRTRuntimeError('TensorRT execute_async_v3 returned false.')
        stream.synchronize()
        return outputs

    def _infer_legacy_bindings(self, inputs):
        bindings = [0] * int(self.engine.num_bindings)

        for entry in self.io_meta:
            if entry['role'] == 'input':
                tensor = inputs[entry['name']]
                self.context.set_binding_shape(entry['index'], tuple(int(v) for v in tensor.shape))
                bindings[entry['index']] = int(tensor.data_ptr())

        outputs = {}
        for entry in self.io_meta:
            if entry['role'] != 'output':
                continue
            shape = tuple(int(dim) for dim in self.context.get_binding_shape(entry['index']))
            if any(dim < 0 for dim in shape):
                raise TensorRTRuntimeError(f'Unresolved dynamic TensorRT output shape for {entry["name"]}: {shape}')
            tensor = torch.empty(shape, device=self.device, dtype=self._get_io_dtype(entry['name']))
            outputs[entry['name']] = tensor
            bindings[entry['index']] = int(tensor.data_ptr())

        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream):
            raise TensorRTRuntimeError('TensorRT execute_async_v2 returned false.')
        stream.synchronize()
        return outputs
