import argparse
import sys
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deployment.tensorrt_runtime import (  # noqa: E402
    TensorRTRuntimeError,
    build_engine_with_python,
    build_engine_with_trtexec,
    select_builder_backend,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Build a TensorRT engine from an existing ONNX export. '
            'Typical workflow: tools/export.py -> tools/build_trt_engine.py -> tools/infer_trt.py'
        )
    )
    parser.add_argument('--onnx', type=str, required=True, help='Path to the exported ONNX model.')
    parser.add_argument('--engine', type=str, default='', help='Output TensorRT engine path. Defaults to <onnx>.engine')
    parser.add_argument(
        '--backend',
        type=str,
        default='auto',
        choices=['auto', 'python', 'trtexec'],
        help='TensorRT engine builder backend. `auto` prefers Python TensorRT, then falls back to trtexec.',
    )
    parser.add_argument('--trtexec-path', type=str, default='', help='Optional explicit trtexec executable path.')
    parser.add_argument('--workspace-mb', type=int, default=2048, help='TensorRT workspace size in MB.')
    parser.add_argument('--fp16', action='store_true', help='Build an FP16 engine when supported.')
    parser.add_argument(
        '--shape',
        type=str,
        default='',
        help='Optional NCHW shape like 1x3x640x640. Useful for dynamic ONNX exports when using a single fixed profile.',
    )
    parser.add_argument('--min-shape', type=str, default='', help='Optional min NCHW shape for dynamic TensorRT profile.')
    parser.add_argument('--opt-shape', type=str, default='', help='Optional opt NCHW shape for dynamic TensorRT profile.')
    parser.add_argument('--max-shape', type=str, default='', help='Optional max NCHW shape for dynamic TensorRT profile.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose TensorRT / trtexec build logging.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f'ONNX file does not exist: {onnx_path}')

    engine_path = Path(args.engine) if args.engine else onnx_path.with_suffix('.engine')
    backend = select_builder_backend(args.backend, trtexec_path=args.trtexec_path or None)

    print(f'Building TensorRT engine from ONNX: {onnx_path}')
    print(f'Output engine: {engine_path}')
    print(f'Builder backend: {backend}')

    kwargs = {
        'onnx_path': str(onnx_path),
        'engine_path': str(engine_path),
        'fp16': bool(args.fp16),
        'workspace_mb': int(args.workspace_mb),
        'shape': args.shape or None,
        'min_shape': args.min_shape or None,
        'opt_shape': args.opt_shape or None,
        'max_shape': args.max_shape or None,
        'verbose': bool(args.verbose),
    }
    if backend == 'python':
        result = build_engine_with_python(**kwargs)
    else:
        result = build_engine_with_trtexec(
            **kwargs,
            trtexec_path=args.trtexec_path or None,
        )

    print(f'TensorRT engine build succeeded: {engine_path}')
    print(f'Build summary: {result}')


if __name__ == '__main__':
    try:
        main()
    except (TensorRTRuntimeError, FileNotFoundError, ValueError) as exc:
        print(f'[TensorRT Build Error] {exc}', file=sys.stderr)
        raise SystemExit(1)
