import torch


def normalize_per_level_outputs(outputs):
    """Convert head outputs into per-level tensors shaped [B, 5 + nc, H, W]."""
    if outputs is None:
        return []

    if isinstance(outputs, torch.Tensor):
        return [outputs]

    normalized = []
    for out in outputs:
        if isinstance(out, dict):
            out = torch.cat((out['reg'], out['angle'], out['cls']), dim=1)
        normalized.append(out)

    return normalized


def flatten_predictions(outputs):
    per_level_outputs = normalize_per_level_outputs(outputs)
    if not per_level_outputs:
        raise ValueError("Model returned empty predictions.")

    flat_predictions = [
        out.reshape(out.shape[0], out.shape[1], -1).permute(0, 2, 1)
        for out in per_level_outputs
    ]
    return torch.cat(flat_predictions, dim=1), per_level_outputs
