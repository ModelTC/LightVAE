import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from vae_prune.data import read_clip


class NativeGeometryTest(unittest.TestCase):
    def test_native_first_clip_never_interpolates(self):
        with tempfile.TemporaryDirectory() as directory:
            for i in range(37):
                Image.new("RGB", (48, 32), (i, i, i)).save(Path(directory) / f"{i:05d}.png")
            with patch("vae_prune.data.F.interpolate", side_effect=AssertionError("resize forbidden")):
                x, meta = read_clip(
                    {"path": directory}, frames=33, native_resolution=True, validation=True, return_metadata=True
                )
            self.assertEqual(tuple(x.shape), (1, 3, 33, 32, 48))
            self.assertEqual(meta["start"], 0)

    def test_unaligned_native_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            Image.new("RGB", (48, 34)).save(Path(directory) / "00000.png")
            with self.assertRaisesRegex(ValueError, "align"):
                read_clip({"path": directory}, frames=1, native_resolution=True)
