import shutil
import tempfile
import unittest
from pathlib import Path

from tools.infer import IMAGE_SUFFIXES as INFER_IMAGE_SUFFIXES
from tools.infer import collect_frame_pairs
from tools.track import IMAGE_SUFFIXES as TRACK_IMAGE_SUFFIXES
from tools.track import collect_aligned_frame_pairs


class ToolInputPairsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp_root = Path(__file__).resolve().parent / '.tmp_tool_input_pairs'
        cls._temp_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._temp_root, ignore_errors=True)

    def _tempdir(self):
        return tempfile.TemporaryDirectory(dir=self._temp_root)

    def _touch_files(self, root, names):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        paths = []
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            paths.append(path)
        return paths

    def test_infer_missing_rgb_source_raises_clear_error(self):
        with self._tempdir() as tmpdir:
            ir_path = Path(tmpdir) / 'ir.jpg'
            ir_path.touch()
            with self.assertRaisesRegex(FileNotFoundError, 'RGB source does not exist'):
                collect_frame_pairs(str(Path(tmpdir) / 'missing_rgb.jpg'), str(ir_path))

    def test_infer_missing_ir_source_raises_clear_error(self):
        with self._tempdir() as tmpdir:
            rgb_path = Path(tmpdir) / 'rgb.jpg'
            rgb_path.touch()
            with self.assertRaisesRegex(FileNotFoundError, 'IR source does not exist'):
                collect_frame_pairs(str(rgb_path), str(Path(tmpdir) / 'missing_ir.jpg'))

    def test_infer_directory_inputs_require_aligned_pairs(self):
        with self._tempdir() as tmpdir:
            rgb_dir = Path(tmpdir) / 'rgb'
            ir_dir = Path(tmpdir) / 'ir'
            self._touch_files(rgb_dir, ['a.jpg'])
            self._touch_files(ir_dir, ['b.jpg'])
            with self.assertRaisesRegex(FileNotFoundError, 'No aligned RGB/IR image pairs found'):
                collect_frame_pairs(str(rgb_dir), str(ir_dir))

    def test_infer_single_file_inputs_pass_through_pair_collection(self):
        with self._tempdir() as tmpdir:
            rgb_path = Path(tmpdir) / 'rgb_frame.bmp'
            ir_path = Path(tmpdir) / 'ir_frame.tif'
            rgb_path.touch()
            ir_path.touch()
            pairs = collect_frame_pairs(str(rgb_path), str(ir_path))
            self.assertEqual(pairs, [(rgb_path, ir_path)])

    def test_infer_directory_inputs_collect_supported_suffix_pairs(self):
        with self._tempdir() as tmpdir:
            rgb_dir = Path(tmpdir) / 'rgb'
            ir_dir = Path(tmpdir) / 'ir'
            matched_names = ['a.jpg', 'b.png', 'c.bmp', 'd.tif', 'e.tiff']
            self._touch_files(rgb_dir, matched_names + ['ignored.gif'])
            self._touch_files(ir_dir, matched_names + ['ignored.gif'])

            pairs = collect_frame_pairs(str(rgb_dir), str(ir_dir))

            self.assertEqual([rgb.name for rgb, _ in pairs], matched_names)
            self.assertEqual([ir.name for _, ir in pairs], matched_names)

    def test_track_missing_rgb_dir_raises_clear_error(self):
        with self._tempdir() as tmpdir:
            ir_dir = Path(tmpdir) / 'ir'
            ir_dir.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, 'RGB frame directory does not exist'):
                collect_aligned_frame_pairs(str(Path(tmpdir) / 'missing_rgb'), str(ir_dir))

    def test_track_missing_ir_dir_raises_clear_error(self):
        with self._tempdir() as tmpdir:
            rgb_dir = Path(tmpdir) / 'rgb'
            rgb_dir.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, 'IR frame directory does not exist'):
                collect_aligned_frame_pairs(str(rgb_dir), str(Path(tmpdir) / 'missing_ir'))

    def test_track_inputs_must_be_directories(self):
        with self._tempdir() as tmpdir:
            rgb_file = Path(tmpdir) / 'rgb.jpg'
            ir_file = Path(tmpdir) / 'ir.jpg'
            rgb_file.touch()
            ir_file.touch()
            with self.assertRaisesRegex(FileNotFoundError, 'must both be directories with aligned filenames'):
                collect_aligned_frame_pairs(str(rgb_file), str(ir_file))

    def test_track_directory_inputs_require_aligned_pairs(self):
        with self._tempdir() as tmpdir:
            rgb_dir = Path(tmpdir) / 'rgb'
            ir_dir = Path(tmpdir) / 'ir'
            self._touch_files(rgb_dir, ['a.jpg'])
            self._touch_files(ir_dir, ['b.jpg'])
            with self.assertRaisesRegex(FileNotFoundError, 'No aligned RGB/IR frame pairs found'):
                collect_aligned_frame_pairs(str(rgb_dir), str(ir_dir))

    def test_track_directory_inputs_collect_supported_suffix_pairs(self):
        with self._tempdir() as tmpdir:
            rgb_dir = Path(tmpdir) / 'rgb'
            ir_dir = Path(tmpdir) / 'ir'
            matched_names = ['a.jpg', 'b.png', 'c.bmp', 'd.tif', 'e.tiff']
            self._touch_files(rgb_dir, matched_names + ['ignored.gif'])
            self._touch_files(ir_dir, matched_names + ['ignored.gif'])

            pairs = collect_aligned_frame_pairs(str(rgb_dir), str(ir_dir))

            self.assertEqual([rgb.name for rgb, _ in pairs], matched_names)
            self.assertEqual([ir.name for _, ir in pairs], matched_names)

    def test_track_suffixes_match_infer_suffixes(self):
        self.assertEqual(TRACK_IMAGE_SUFFIXES, INFER_IMAGE_SUFFIXES)


if __name__ == '__main__':
    unittest.main()
