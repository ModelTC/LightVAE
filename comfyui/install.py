"""Install the ComfyUI adapter; install the shared lightvae package with pip first."""

import argparse
from pathlib import Path
import shutil


FILES = ("__init__.py", "nodes.py", "vae.py", "components.py", "requirements.txt", "README.md")
DIRECTORIES = ("assets", "workflows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    root = args.comfy_root.resolve()
    if not (root / "folder_paths.py").is_file():
        parser.error("--comfy-root must point at ComfyUI (the folder containing main.py and folder_paths.py)")
    target = root / "custom_nodes" / "ComfyUI-LightVAE"
    if target.resolve() == source:
        parser.error("The node is already installed here.")
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(source / name, target / name)
    for name in DIRECTORIES:
        shutil.copytree(source / name, target / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]"))
    (root / "models" / "vae" / "LightVAE").mkdir(parents=True, exist_ok=True)
    print(f"Installed: {target}")
    print(f"Place *-comfy.safetensors in: {root / 'models/vae/LightVAE'}")
    print("Restart ComfyUI; search for 'Load LightVAE (TLR)'.")


if __name__ == "__main__":
    main()
