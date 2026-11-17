import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from vae_prune.data import read_clip


class Container:
    def __init__(self, actual):
        self.actual = actual
        self.streams = SimpleNamespace(
            video=[SimpleNamespace(frames=22, codec_context=SimpleNamespace(thread_count=0))]
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def decode(self, video=0):
        for i in range(self.actual):
            yield SimpleNamespace(to_ndarray=lambda format, i=i: np.full((16, 16, 3), i, dtype=np.uint8))


class FrameCountTests(unittest.TestCase):
    def test_overcount_resamples_contiguous_frames(self):
        seed = next(s for s in range(1000) if random.Random(s).randrange(6) == 5)
        with patch("vae_prune.data.av.open", side_effect=lambda _: Container(21)):
            clip, meta = read_clip(
                {"path": "/not-a-real-video.avi"}, 17, native_resolution=True, seed=seed, return_metadata=True
            )
        start = random.Random(seed).randrange(5)
        self.assertEqual(meta["start"], start)
        self.assertEqual(tuple(clip.shape), (1, 3, 17, 16, 16))
        pixels = ((clip[0, 0, :, 0, 0] + 1) * 127.5).round().int().tolist()
        self.assertEqual(pixels, list(range(start, start + 17)))

    def test_truly_short_still_fails(self):
        with patch("vae_prune.data.av.open", side_effect=lambda _: Container(10)):
            with self.assertRaisesRegex(ValueError, "Empty/short video"):
                read_clip({"path": "/not-a-real-video.avi"}, 17, native_resolution=True)


if __name__ == "__main__":
    unittest.main()
