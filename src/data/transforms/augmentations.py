import random

import cv2
import numpy as np


class PC_MWA:
    """Physically consistent multi-modal weather augmentation."""

    def __init__(self, base_p=0.2, max_p=0.6):
        self.base_p = base_p
        self.max_p = max_p
        self.weather_types = ['low_light', 'fog', 'rain']
        self.weather_probs = [0.4, 0.4, 0.2]

    def adjust_gamma(self, image, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def apply_thermal_attenuation(self, img_ir, alpha=0.92, noise_std=3.0, seed=0):
        rng = np.random.default_rng(seed)
        img_ir_attenuated = cv2.convertScaleAbs(img_ir, alpha=alpha, beta=0)
        noise = rng.normal(0, noise_std, img_ir.shape).astype(np.float32)
        return np.clip(img_ir_attenuated.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def add_rain_streaks(self, image, drops_num=200, seed=0):
        rng = random.Random(seed)
        rain_layer = np.zeros_like(image)
        for _ in range(drops_num):
            x = rng.randint(0, image.shape[1] - 1)
            y = rng.randint(0, max(0, image.shape[0] - 20))
            dx = rng.randint(-3, 3)
            cv2.line(rain_layer, (x, y), (x + dx, y + 20), (200, 200, 200), 1)
        return cv2.addWeighted(image, 1.0, cv2.blur(rain_layer, (3, 3)), 0.7, 0)

    def sample_weather_config(self, progress=1.0):
        current_p = self.base_p + (self.max_p - self.base_p) * progress
        if random.random() > current_p:
            return None

        is_extreme = random.random() < (0.05 + 0.15 * progress)
        weather = random.choices(self.weather_types, weights=self.weather_probs, k=1)[0]
        config = {
            'weather': weather,
            'is_extreme': is_extreme,
            'seed': random.randint(0, 10_000_000),
        }

        if weather == 'low_light':
            config['gamma'] = random.uniform(4.5, 6.0) if is_extreme else random.uniform(2.5, 3.5)
            config['ir_alpha'] = random.uniform(0.75, 0.85) if is_extreme else random.uniform(0.90, 0.98)
        elif weather == 'fog':
            config['fog_int'] = 0.9 if is_extreme else random.uniform(0.4, 0.6)
            config['blur_ksize'] = 7 if is_extreme else 3
            config['ir_alpha'] = random.uniform(0.80, 0.95) if is_extreme else random.uniform(0.90, 0.95)
        elif weather == 'rain':
            config['drops'] = 1000 if is_extreme else 300
            config['extreme_blur'] = is_extreme
            config['ir_alpha'] = random.uniform(0.80, 0.95) if is_extreme else random.uniform(0.90, 0.95)

        return config

    def apply_weather_config(self, img_rgb, img_ir, config):
        if config is None:
            return img_rgb, img_ir

        weather = config['weather']
        seed = config['seed']

        if weather == 'low_light':
            img_rgb = self.adjust_gamma(img_rgb, config['gamma'])
            img_ir = self.apply_thermal_attenuation(img_ir, alpha=config['ir_alpha'], seed=seed)
        elif weather == 'fog':
            fog_int = config['fog_int']
            img_rgb = cv2.addWeighted(img_rgb, 1 - fog_int, np.ones_like(img_rgb) * 255, fog_int, 0)
            blur_ksize = config['blur_ksize']
            img_ir = cv2.GaussianBlur(img_ir, (blur_ksize, blur_ksize), 0)
            img_ir = self.apply_thermal_attenuation(img_ir, alpha=config['ir_alpha'], seed=seed)
        elif weather == 'rain':
            img_rgb = self.add_rain_streaks(img_rgb, config['drops'], seed=seed)
            if config['extreme_blur']:
                img_rgb = cv2.blur(img_rgb, (5, 5))
            img_ir = self.apply_thermal_attenuation(img_ir, alpha=config['ir_alpha'], seed=seed)

        return img_rgb, img_ir

    def __call__(self, img_rgb, img_ir, progress=1.0):
        config = self.sample_weather_config(progress)
        return self.apply_weather_config(img_rgb, img_ir, config)


class CrossModalMisalignment:
    """Simulate slight RGB/IR registration error without changing labels."""

    def __init__(
        self,
        enabled=False,
        prob=0.0,
        apply_to='ir',
        max_translate_ratio=0.01,
        max_rotate_deg=1.5,
        max_scale_delta=0.02,
        border_mode='reflect',
    ):
        self.enabled = enabled
        self.prob = prob
        self.apply_to = apply_to
        self.max_translate_ratio = max_translate_ratio
        self.max_rotate_deg = max_rotate_deg
        self.max_scale_delta = max_scale_delta
        self.border_mode = border_mode

    def sample_config(self, image_shape):
        if not self.enabled or random.random() > self.prob:
            return None

        height, width = image_shape[:2]
        target_modalities = ['rgb', 'ir'] if self.apply_to == 'random' else [self.apply_to]
        target_modality = random.choice(target_modalities)
        return {
            'target_modality': target_modality,
            'translate_x': random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width,
            'translate_y': random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height,
            'rotate_deg': random.uniform(-self.max_rotate_deg, self.max_rotate_deg),
            'scale': 1.0 + random.uniform(-self.max_scale_delta, self.max_scale_delta),
        }

    def _apply_affine(self, image, config):
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), config['rotate_deg'], config['scale'])
        matrix[0, 2] += config['translate_x']
        matrix[1, 2] += config['translate_y']
        border_mode = cv2.BORDER_REFLECT_101 if self.border_mode == 'reflect' else cv2.BORDER_CONSTANT
        return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode)

    def apply_config(self, img_rgb, img_ir, config):
        if config is None:
            return img_rgb, img_ir
        if config['target_modality'] == 'rgb':
            return self._apply_affine(img_rgb, config), img_ir
        return img_rgb, self._apply_affine(img_ir, config)

    def __call__(self, img_rgb, img_ir):
        config = self.sample_config(img_rgb.shape)
        return self.apply_config(img_rgb, img_ir, config)


class SensorDegradationAug:
    """Lightweight RGB/IR sensor degradation simulation."""

    def __init__(self, enabled=False, prob=0.0, rgb=None, ir=None):
        self.enabled = enabled
        self.prob = prob
        self.rgb_cfg = {
            'overexposure_prob': 0.3,
            'flare_prob': 0.15,
            'haze_prob': 0.1,
            'max_exposure_gain': 1.4,
        }
        self.ir_cfg = {
            'noise_prob': 0.3,
            'noise_std': 0.03,
            'drift_prob': 0.2,
            'max_drift': 0.08,
            'hotspot_prob': 0.15,
            'stripe_prob': 0.0,
        }
        if rgb is not None:
            self.rgb_cfg.update(rgb)
        if ir is not None:
            self.ir_cfg.update(ir)

    def _clip_uint8(self, image):
        return np.clip(image, 0, 255).astype(np.uint8)

    def _radial_mask(self, height, width, center_xy, radius_ratio):
        center_x = center_xy[0] * width
        center_y = center_xy[1] * height
        radius = max(1.0, radius_ratio * min(height, width))
        yy, xx = np.mgrid[0:height, 0:width]
        dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        mask = np.exp(-(dist ** 2) / (2.0 * radius ** 2))
        return mask[..., None].astype(np.float32)

    def sample_config(self):
        if not self.enabled or random.random() > self.prob:
            return None

        config = {
            'seed': random.randint(0, 10_000_000),
            'rgb': {
                'use_overexposure': random.random() < self.rgb_cfg['overexposure_prob'],
                'use_flare': random.random() < self.rgb_cfg['flare_prob'],
                'use_haze': random.random() < self.rgb_cfg['haze_prob'],
                'exposure_gain': random.uniform(1.05, self.rgb_cfg['max_exposure_gain']),
                'flare_strength': random.uniform(0.08, 0.20),
                'flare_center': (random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)),
                'flare_radius_ratio': random.uniform(0.10, 0.25),
                'exposure_center': (random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)),
                'exposure_radius_ratio': random.uniform(0.15, 0.35),
                'haze_strength': random.uniform(0.05, 0.18),
            },
            'ir': {
                'noise_prob': self.ir_cfg['noise_prob'],
                'noise_std': self.ir_cfg['noise_std'],
                'use_drift': random.random() < self.ir_cfg['drift_prob'],
                'drift_value': random.uniform(-self.ir_cfg['max_drift'], self.ir_cfg['max_drift']),
                'hotspot_prob': self.ir_cfg['hotspot_prob'],
                'stripe_prob': self.ir_cfg.get('stripe_prob', 0.0),
                'hotspot_strength': random.uniform(18.0, 42.0),
                'hotspot_radius_ratio': random.uniform(0.04, 0.10),
            },
        }
        return config

    def _apply_rgb_degradation(self, image, config):
        out = image.astype(np.float32)
        height, width = out.shape[:2]

        if config['use_haze']:
            haze_strength = config['haze_strength']
            out = out * (1.0 - haze_strength) + 255.0 * haze_strength
            out = (out - 127.5) * (1.0 - 0.25 * haze_strength) + 127.5

        if config['use_overexposure']:
            mask = self._radial_mask(height, width, config['exposure_center'], config['exposure_radius_ratio'])
            out = out * (1.0 + (config['exposure_gain'] - 1.0) * mask)

        if config['use_flare']:
            flare_mask = self._radial_mask(height, width, config['flare_center'], config['flare_radius_ratio'])
            out = out + 255.0 * config['flare_strength'] * flare_mask

        return self._clip_uint8(out)

    def _apply_ir_degradation(self, image, config, frame_seed_offset=0):
        out = image.astype(np.float32)
        height, width = out.shape[:2]
        base_seed = int(config['seed']) + int(frame_seed_offset) * 9973
        rng = np.random.default_rng(base_seed)

        if config['use_drift']:
            drift_value = config['drift_value'] * 255.0
            low_freq = rng.normal(0.0, 1.0, size=(max(2, height // 32), max(2, width // 32))).astype(np.float32)
            low_freq = cv2.resize(low_freq, (width, height), interpolation=cv2.INTER_CUBIC)
            low_freq = cv2.GaussianBlur(low_freq, (0, 0), sigmaX=3.0, sigmaY=3.0)
            low_freq = low_freq[..., None]
            out = out + drift_value + 0.15 * drift_value * low_freq

        if rng.random() < config['noise_prob']:
            noise = rng.normal(0.0, config['noise_std'] * 255.0, size=out.shape).astype(np.float32)
            out = out + noise

        if rng.random() < config['hotspot_prob']:
            hotspot_center = (rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85))
            hotspot_mask = self._radial_mask(height, width, hotspot_center, config['hotspot_radius_ratio'])
            out = out + config['hotspot_strength'] * hotspot_mask

        if rng.random() < config.get('stripe_prob', 0.0):
            stripes = rng.normal(0.0, 6.0, size=(1, width, 1)).astype(np.float32)
            out = out + stripes

        return self._clip_uint8(out)

    def apply_config(self, img_rgb, img_ir, config, frame_seed_offset=0):
        if config is None:
            return img_rgb, img_ir
        img_rgb = self._apply_rgb_degradation(img_rgb, config['rgb'])
        img_ir = self._apply_ir_degradation(img_ir, config['ir'], frame_seed_offset=frame_seed_offset)
        return img_rgb, img_ir

    def __call__(self, img_rgb, img_ir):
        config = self.sample_config()
        return self.apply_config(img_rgb, img_ir, config, frame_seed_offset=0)


class MultiModalAugmentationPipeline:
    """Augment current RGB-IR frame, with optional temporal consistency for previous frame."""

    def __init__(
        self,
        enable_cmcp=True,
        enable_mrre=True,
        enable_weather=True,
        enable_modality_dropout=True,
        cross_modal_misalignment=None,
        sensor_degradation=None,
    ):
        self.enable_cmcp = enable_cmcp
        self.enable_mrre = enable_mrre
        self.enable_modality_dropout = enable_modality_dropout
        self.weather_sim = PC_MWA(base_p=0.2, max_p=0.6) if enable_weather else None
        self.cross_modal_misalignment = CrossModalMisalignment(**cross_modal_misalignment) if cross_modal_misalignment else None
        self.sensor_degradation = SensorDegradationAug(**sensor_degradation) if sensor_degradation else None

    def check_iou_conflict(self, px, py, pw, ph, existing_labels, width, height, iou_thresh=0.1):
        box1 = [px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2]
        for lbl in existing_labels:
            _, cx, cy, w, h, _ = lbl
            box2 = [cx * width - w * width / 2, cy * height - h * height / 2,
                    cx * width + w * width / 2, cy * height + h * height / 2]
            ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
            ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
            if ix1 < ix2 and iy1 < iy2:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                iou = inter_area / float(pw * ph + (w * width) * (h * height) - inter_area + 1e-6)
                if iou > iou_thresh:
                    return True
        return False

    def apply_cmcp(self, labels, img_rgb, img_ir):
        if len(labels) == 0:
            return labels, img_rgb, img_ir
        height, width = img_rgb.shape[:2]
        new_labels = list(labels)

        for lbl in labels:
            cls_id, cx, cy, w, h, theta = lbl
            if w < 0.05 and h < 0.05:
                abs_w, abs_h = int(w * width), int(h * height)
                x1, y1 = max(0, int(cx * width) - abs_w // 2), max(0, int(cy * height) - abs_h // 2)
                x2, y2 = min(width, x1 + abs_w), min(height, y1 + abs_h)

                if x2 <= x1 or y2 <= y1:
                    continue
                patch_rgb, patch_ir = img_rgb[y1:y2, x1:x2].copy(), img_ir[y1:y2, x1:x2].copy()

                for _ in range(10):
                    paste_x, paste_y = random.randint(0, width - abs_w), random.randint(0, height - abs_h)
                    paste_cx, paste_cy = paste_x + abs_w / 2, paste_y + abs_h / 2
                    if not self.check_iou_conflict(paste_cx, paste_cy, abs_w, abs_h, new_labels, width, height):
                        img_rgb[paste_y:paste_y + patch_rgb.shape[0], paste_x:paste_x + patch_rgb.shape[1]] = patch_rgb
                        img_ir[paste_y:paste_y + patch_ir.shape[0], paste_x:paste_x + patch_ir.shape[1]] = patch_ir
                        new_labels.append([cls_id, paste_cx / width, paste_cy / height, w, h, theta])
                        break

        return np.array(new_labels, dtype=np.float32), img_rgb, img_ir

    def apply_mrre(self, labels, img_rgb, img_ir):
        if len(labels) == 0:
            return img_rgb, img_ir
        height, width = img_rgb.shape[:2]
        for lbl in labels:
            if random.random() > 0.5:
                continue
            _, cx, cy, w, h, _ = lbl
            abs_w, abs_h = int(w * width), int(h * height)
            bg_x, bg_y = random.randint(0, width - abs_w // 2), random.randint(0, height - abs_h // 2)
            bg_patch_rgb = img_rgb[bg_y:bg_y + abs_h // 2, bg_x:bg_x + abs_w // 2].copy()
            bg_patch_ir = img_ir[bg_y:bg_y + abs_h // 2, bg_x:bg_x + abs_w // 2].copy()
            tgt_x, tgt_y = int(cx * width) - abs_w // 4, int(cy * height) - abs_h // 4
            if 0 <= tgt_x < width - abs_w // 2 and 0 <= tgt_y < height - abs_h // 2:
                img_rgb[tgt_y:tgt_y + abs_h // 2, tgt_x:tgt_x + abs_w // 2] = bg_patch_rgb
                img_ir[tgt_y:tgt_y + abs_h // 2, tgt_x:tgt_x + abs_w // 2] = bg_patch_ir
        return img_rgb, img_ir

    def sample_modality_dropout_config(self):
        p = random.random()
        if p < 0.05:
            return {'mode': 'drop_rgb'}
        if p < 0.10:
            return {'mode': 'noise_ir', 'seed': random.randint(0, 10_000_000)}
        return None

    def apply_modality_dropout_config(self, img_rgb, img_ir, config):
        if config is None:
            return img_rgb, img_ir
        if config['mode'] == 'drop_rgb':
            img_rgb = np.zeros_like(img_rgb)
        elif config['mode'] == 'noise_ir':
            rng = np.random.default_rng(config['seed'])
            img_ir = rng.normal(128, 50, img_ir.shape).clip(0, 255).astype(np.uint8)
        return img_rgb, img_ir

    def __call__(self, img_rgb, img_ir, labels, epoch=0, max_epoch=100):
        progress = min(1.0, max(0.0, epoch / max_epoch))
        if self.enable_cmcp and random.random() < 0.5:
            labels, img_rgb, img_ir = self.apply_cmcp(labels, img_rgb, img_ir)
        if self.enable_mrre:
            img_rgb, img_ir = self.apply_mrre(labels, img_rgb, img_ir)

        misalignment_config = self.cross_modal_misalignment.sample_config(img_rgb.shape) if self.cross_modal_misalignment else None
        if self.cross_modal_misalignment:
            img_rgb, img_ir = self.cross_modal_misalignment.apply_config(img_rgb, img_ir, misalignment_config)

        weather_config = self.weather_sim.sample_weather_config(progress) if self.weather_sim else None
        img_rgb, img_ir = self.weather_sim.apply_weather_config(img_rgb, img_ir, weather_config) if self.weather_sim else (img_rgb, img_ir)

        sensor_config = self.sensor_degradation.sample_config() if self.sensor_degradation else None
        if self.sensor_degradation:
            img_rgb, img_ir = self.sensor_degradation.apply_config(img_rgb, img_ir, sensor_config, frame_seed_offset=0)

        modality_dropout = self.sample_modality_dropout_config() if self.enable_modality_dropout else None
        img_rgb, img_ir = self.apply_modality_dropout_config(img_rgb, img_ir, modality_dropout)
        return img_rgb, img_ir, labels

    def apply_temporal_pair(self, img_rgb, img_ir, prev_rgb, prev_ir, labels, epoch=0, max_epoch=100):
        progress = min(1.0, max(0.0, epoch / max_epoch))
        if self.enable_cmcp and random.random() < 0.5:
            labels, img_rgb, img_ir = self.apply_cmcp(labels, img_rgb, img_ir)
        if self.enable_mrre:
            img_rgb, img_ir = self.apply_mrre(labels, img_rgb, img_ir)

        misalignment_config = self.cross_modal_misalignment.sample_config(img_rgb.shape) if self.cross_modal_misalignment else None
        if self.cross_modal_misalignment:
            img_rgb, img_ir = self.cross_modal_misalignment.apply_config(img_rgb, img_ir, misalignment_config)
            prev_rgb, prev_ir = self.cross_modal_misalignment.apply_config(prev_rgb, prev_ir, misalignment_config)

        weather_config = self.weather_sim.sample_weather_config(progress) if self.weather_sim else None
        if self.weather_sim:
            img_rgb, img_ir = self.weather_sim.apply_weather_config(img_rgb, img_ir, weather_config)
            prev_rgb, prev_ir = self.weather_sim.apply_weather_config(prev_rgb, prev_ir, weather_config)

        sensor_config = self.sensor_degradation.sample_config() if self.sensor_degradation else None
        if self.sensor_degradation:
            img_rgb, img_ir = self.sensor_degradation.apply_config(img_rgb, img_ir, sensor_config, frame_seed_offset=0)
            prev_rgb, prev_ir = self.sensor_degradation.apply_config(prev_rgb, prev_ir, sensor_config, frame_seed_offset=1)

        modality_dropout = self.sample_modality_dropout_config() if self.enable_modality_dropout else None
        img_rgb, img_ir = self.apply_modality_dropout_config(img_rgb, img_ir, modality_dropout)
        prev_rgb, prev_ir = self.apply_modality_dropout_config(prev_rgb, prev_ir, modality_dropout)

        return img_rgb, img_ir, prev_rgb, prev_ir, labels
