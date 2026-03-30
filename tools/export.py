import argparse
import warnings

import torch

from src.model.builder import build_model
from src.model.output_adapter import flatten_predictions
from src.utils.config import load_config

warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rgb, ir):
        with torch.no_grad():
            out = self.model(rgb, ir, return_attention_map=False)
            outputs = out[0] if isinstance(out, tuple) else out
            flat_preds, _ = flatten_predictions(outputs)
            return flat_preds.contiguous()


def main():
    parser = argparse.ArgumentParser(description="Export the dual-modal OBB model to ONNX.")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--half', action='store_true', help='Export FP16 model when CUDA is available.')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic batch axis.')
    parser.add_argument('--opset', type=int, default=13)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cpu')

    base_model = build_model(cfg.model).to(device)
    base_model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    base_model.eval()

    model = ExportWrapper(base_model).eval()
    for p in model.parameters():
        p.requires_grad = False

    dummy_rgb = torch.randn(args.batch_size, 3, cfg.dataset.imgsz, cfg.dataset.imgsz).to(device)
    dummy_ir = torch.randn(args.batch_size, 3, cfg.dataset.imgsz, cfg.dataset.imgsz).to(device)
    export_path = args.weights.replace('.pt', '.onnx')

    if args.half and torch.cuda.is_available():
        model.cuda().half()
        dummy_rgb, dummy_ir = dummy_rgb.cuda().half(), dummy_ir.cuda().half()
        export_path = export_path.replace('.onnx', '_fp16.onnx')

    print("\nWarming up export graph...")
    with torch.no_grad():
        for _ in range(2):
            _ = model(dummy_rgb, dummy_ir)

    dynamic_axes = {
        'images_rgb': {0: 'batch'},
        'images_ir': {0: 'batch'},
        'outputs': {0: 'batch'}
    } if args.dynamic else None

    print(f"\nExporting ONNX (opset {args.opset})...")
    torch.onnx.export(
        model, (dummy_rgb, dummy_ir), export_path,
        input_names=['images_rgb', 'images_ir'], output_names=['outputs'],
        dynamic_axes=dynamic_axes, opset_version=args.opset, do_constant_folding=True
    )
    print(f"ONNX export succeeded: {export_path}")
    print('ONNX interface: inputs=(images_rgb, images_ir), outputs=(outputs flat predictions before Python-side OBB NMS)')

    try:
        import onnx
        import onnxsim

        model_simp, check = onnxsim.simplify(onnx.load(export_path), dynamic_input_shape=args.dynamic)
        if check:
            onnx.save(model_simp, export_path)
            print(f"onnxsim simplification succeeded: {export_path}")
    except ImportError:
        print("onnxsim is not installed, skipping simplification.")


if __name__ == '__main__':
    main()
