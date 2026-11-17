"""Convert a Wan training checkpoint to the repository's inference format."""

import argparse
import json
from pathlib import Path

from safetensors.torch import save_file

from .checkpoints import load_student


def publish(source, destination):
    destination = Path(destination)
    config_path = destination.with_suffix(".json")
    for path in (destination, config_path):
        if path.exists():
            raise FileExistsError(path)
    model, saved = load_student(source)
    metadata = {key: saved[key] for key in ("family", "kept", "packed_records")}
    metadata.update(format_version=1, step=saved.get("step", 0), provenance=saved.get("provenance", {}))
    tensors = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata={"checkpoint_metadata": json.dumps(metadata)})
    config_path.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(destination)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = publish(args.checkpoint, args.output)
    print(f"Saved {metadata['family']} decoder: {args.output}", flush=True)


if __name__ == "__main__":
    main()
