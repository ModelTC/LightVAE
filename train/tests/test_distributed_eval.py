import unittest
from unittest.mock import patch
from vae_prune.distributed_eval import local_indices, summarize


class DistributedEvalTests(unittest.TestCase):
    def test_unpadded_shards_even_uneven_empty(self):
        for count in (0, 1, 5, 30, 128, 158):
            for world in (1, 2, 4, 8):
                parts = []
                for rank in range(world):
                    with patch("vae_prune.distributed_eval.rank_world", return_value=(rank, world)):
                        parts.extend(local_indices(count))
                self.assertEqual(sorted(parts), list(range(count)))

    def test_video_weighted_mean(self):
        rows = [dict(index=i, dataset="ucf101", metrics={"student": {"psnr": float(i)}}) for i in range(5)]
        rows.append(dict(index=5, dataset="davis", metrics={"student": {"psnr": 99.0}}))
        result = summarize(rows)
        self.assertEqual(result["ucf101"], dict(count=5, student={"psnr": 2.0}))
        self.assertEqual(result["davis"]["student"]["psnr"], 99.0)


if __name__ == "__main__":
    # torchrun --standalone --nproc_per_node=2 tests/test_distributed_eval.py
    # Real CPU collectives; no GPU or model weights needed.
    import os
    import torch.distributed as dist
    from vae_prune.distributed_eval import gather_rows

    if "RANK" in os.environ:
        dist.init_process_group("gloo")
        try:
            for count in (0, 1, 5, 158):
                rows = [
                    dict(index=i, dataset="ucf101", metrics={"student": {"psnr": float(i)}})
                    for i in local_indices(count)
                ]
                gathered = gather_rows(rows)
                assert [r["index"] for r in gathered] == list(range(count))
                if count:
                    assert summarize(gathered)["ucf101"]["student"]["psnr"] == (count - 1) / 2
            print("distributed evaluation gather PASS", dist.get_rank(), flush=True)
        finally:
            dist.destroy_process_group()
    else:
        unittest.main()
