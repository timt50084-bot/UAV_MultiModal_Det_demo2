import math

import torch
import torch.nn.functional as F


def wrap_obb_angle(theta):
    return torch.remainder(theta + math.pi / 2.0, math.pi) - math.pi / 2.0


def build_tta_transforms(infer_cfg, base_size):
    transforms = []
    sizes = infer_cfg['multi_scale']['sizes'] if infer_cfg['multi_scale']['enabled'] else [int(base_size)]
    for size in sizes:
        transforms.append({'size': int(size), 'horizontal_flip': False})
        if infer_cfg['tta']['enabled'] and infer_cfg['tta'].get('horizontal_flip', False):
            transforms.append({'size': int(size), 'horizontal_flip': True})
    return transforms


def apply_tta_transforms(imgs_rgb, imgs_ir, prev_rgb, prev_ir, transform_cfg):
    size = int(transform_cfg['size'])
    transformed = []
    for tensor in [imgs_rgb, imgs_ir, prev_rgb, prev_ir]:
        tensor_out = tensor.clone()
        if tensor_out.shape[-1] != size or tensor_out.shape[-2] != size:
            tensor_out = F.interpolate(tensor_out, size=(size, size), mode='bilinear', align_corners=False)
        if transform_cfg.get('horizontal_flip', False):
            tensor_out = torch.flip(tensor_out, dims=[-1])
        transformed.append(tensor_out)
    return tuple(transformed)


def invert_tta_predictions(preds, transform_cfg, base_size):
    if preds is None or len(preds) == 0:
        if torch.is_tensor(preds):
            return preds.clone()
        return preds

    restored = preds.clone()
    size = float(transform_cfg['size'])
    if transform_cfg.get('horizontal_flip', False):
        restored[:, 0] = size - restored[:, 0]
        restored[:, 4] = wrap_obb_angle(-restored[:, 4])

    scale_factor = float(base_size) / size
    restored[:, :4] *= scale_factor
    return restored


def merge_tta_predictions(prediction_sets, merge_fn, **kwargs):
    return merge_fn(prediction_sets, **kwargs)
