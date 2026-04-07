import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from tools.infer import (
    build_single_input_modalities,
    classify_single_source,
    iter_video_frames,
    resolve_input_spec,
)


class InferSingleInputTestCase(unittest.TestCase):
    def test_single_input_image_and_video_helpers_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / 'single_input.jpg'
            image = np.zeros((24, 24, 3), dtype=np.uint8)
            image[..., 1] = 180
            image[..., 2] = 40
            self.assertTrue(cv2.imwrite(str(image_path), image))

            spec = resolve_input_spec(SimpleNamespace(
                source=str(image_path),
                source_rgb='',
                source_ir='',
                input_mode='auto',
            ))
            self.assertEqual(spec['mode'], 'single')
            self.assertEqual(spec['single_mode'], 'single_rgb')
            self.assertTrue(spec['assumed_single_mode'])
            self.assertEqual(classify_single_source(image_path), 'image')

            rgb_image, ir_image = build_single_input_modalities(image, spec['single_mode'], 'grayscale_to_ir')
            self.assertEqual(rgb_image.shape, ir_image.shape)
            self.assertFalse(np.array_equal(rgb_image, ir_image))

            video_path = tmp_path / 'single_input.avi'
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*'MJPG'),
                5.0,
                (24, 24),
            )
            if not writer.isOpened():
                self.skipTest('OpenCV video writer is unavailable in this environment.')
            for idx in range(3):
                frame = np.full((24, 24, 3), idx * 40, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            self.assertEqual(classify_single_source(video_path), 'video')
            frames = list(iter_video_frames(video_path))
            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[0][0], 0)
            self.assertEqual(frames[-1][0], 2)
            self.assertEqual(frames[0][1].shape, (24, 24, 3))


if __name__ == '__main__':
    unittest.main()
