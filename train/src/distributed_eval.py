"""Exact, unpadded video sharding and globally ordered metric gathering."""

import torch.distributed as dist


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def local_indices(count):
    rank, world = rank_world()
    return range(rank, count, world)


def gather_rows(rows):
    _, world = rank_world()
    if world > 1:
        parts = [None] * world
        dist.all_gather_object(parts, rows)
        rows = [row for part in parts for row in part]
    return sorted(rows, key=lambda row: row["index"])


def summarize(rows):
    summary = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        selected = [r for r in rows if r["dataset"] == dataset]
        summary[dataset] = dict(count=len(selected))
        for name in selected[0]["metrics"]:
            values = [r["metrics"][name] for r in selected]
            summary[dataset][name] = (
                None
                if any(v is None for v in values)
                else {key: sum(v[key] for v in values) / len(values) for key in values[0]}
            )
    return summary
