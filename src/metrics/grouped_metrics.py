from collections import defaultdict


def compute_grouped_metrics(preds, gts, image_metadata, group_cfg, compute_metrics_fn):
    if not group_cfg.get('enabled', False):
        return {}
    if not image_metadata:
        return {}

    grouped_results = {}
    requested_keys = group_cfg.get('keys', []) or []

    for key in requested_keys:
        grouped_image_ids = defaultdict(set)
        for image_id, metadata in image_metadata.items():
            if not metadata:
                continue
            value = metadata.get(key)
            if value is None:
                continue
            grouped_image_ids[str(value)].add(image_id)

        if not grouped_image_ids:
            continue

        key_results = {}
        for value, image_ids in grouped_image_ids.items():
            sub_preds = [pred for pred in preds if pred['image_id'] in image_ids]
            sub_gts = [gt for gt in gts if gt['image_id'] in image_ids]
            sub_metadata = {image_id: image_metadata.get(image_id) for image_id in image_ids}
            key_results[value] = compute_metrics_fn(sub_preds, sub_gts, sub_metadata)

        if key_results:
            grouped_results[key] = key_results

    return grouped_results
