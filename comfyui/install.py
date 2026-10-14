"""Install the H3-only custom node. No extra Python packages are needed."""

import argparse
from pathlib import Path
import shutil


FILES = ("__init__.py", "nodes.py", "h3_vae.py", "README.md")
OBSOLETE = ("vae.py", "components.py", "requirements.txt",
            "workflows/wan21_t2v_lightvae.json", "workflows/wan22_5b_t2v_lightvae.json",
            "workflows/wan22_5b_i2v_lightvae.json", "workflows/wan22_14b_i2v_lightvae.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    root = args.comfy_root.resolve()
    if not (root / "folder_paths.py").is_file():
        parser.error("--comfy-root must point at ComfyUI (the folder containing main.py and folder_paths.py)")
    if not (root / "comfy/ldm/minimax/vae.py").is_file():
        parser.error("Update ComfyUI first: native MiniMax-H3 video VAE support is required.")
    target = root / "custom_nodes" / "ComfyUI-LightVAE"
    if target.resolve() == source:
        parser.error("The node is already installed here.")
    if target.resolve() == source.parent:
        parser.error("Use a separate LightVAE checkout; installation would overwrite the repository root.")
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(source / name, target / name)
    shutil.copytree(source / "workflows", target / "workflows", dirs_exist_ok=True)
    for name in OBSOLETE:
        (target / name).unlink(missing_ok=True)
    (root / "models" / "vae" / "LightVAE").mkdir(parents=True, exist_ok=True)
    print(f"Installed: {target}")
    print(f"Place H3 LightVAE .safetensors in: {root / 'models/vae/LightVAE'}")
    print("Restart ComfyUI; search for 'Load H3 LightVAE'.")


if __name__ == "__main__":
    main()
