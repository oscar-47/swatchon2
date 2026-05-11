"""
Build a clean staging tree and push it as a Hugging Face Space (Docker SDK).

Required env vars:
  HF_TOKEN       — write-scoped token from https://huggingface.co/settings/tokens
  HF_USERNAME    — your HF account name (e.g. "oscar47")
  HF_SPACE_NAME  — Space repo name (default: "fabricflow-demo")

Optional:
  OPENAI_API_KEY — if set, copied to Space Secrets so /assistant works

After upload, the Space URL will be:
  https://huggingface.co/spaces/<HF_USERNAME>/<HF_SPACE_NAME>
The live app lives at:
  https://<HF_USERNAME>-<HF_SPACE_NAME>.hf.space
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = Path(__file__).resolve().parent
STAGING = Path("/tmp/fabricflow_hf_space")
SAMPLE_SRC = Path(os.path.expanduser("~/Downloads/sample"))


def stage() -> Path:
    """Assemble the exact tree that HF Space expects, in /tmp."""
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    # Deployment-specific files at repo root
    for name in ("Dockerfile", "requirements.txt", "README.md",
                 ".gitattributes", ".dockerignore"):
        shutil.copy2(DEPLOY_DIR / name, STAGING / name)

    # Source tree
    shutil.copytree(REPO_ROOT / "server", STAGING / "server",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                  "fabricai_frontend",
                                                  "config.py"))
    shutil.copytree(REPO_ROOT / "web", STAGING / "web",
                    ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))

    # Models (LFS-tracked)
    runs_dst = STAGING / "runs"
    runs_dst.mkdir()
    for name in ("stage1_knit_vs_woven_vs_others_best.pth",
                 "stage2_woven_7class_best.pth",
                 "stage2_knit_6class_best.pth"):
        src = REPO_ROOT / "runs" / name
        if not src.exists():
            sys.exit(f"missing model checkpoint: {src}")
        shutil.copy2(src, runs_dst / name)

    # Sample images + passport PDFs (LFS-tracked)
    if not SAMPLE_SRC.is_dir():
        sys.exit(f"sample folder missing: {SAMPLE_SRC}")
    shutil.copytree(SAMPLE_SRC, STAGING / "sample",
                    ignore=shutil.ignore_patterns(".DS_Store"))

    return STAGING


def upload(staging: Path) -> str:
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN") or sys.exit("HF_TOKEN env var required")
    user = os.environ.get("HF_USERNAME") or sys.exit("HF_USERNAME env var required")
    space_name = os.environ.get("HF_SPACE_NAME", "fabricflow-demo")
    repo_id = f"{user}/{space_name}"

    api = HfApi(token=token)
    print(f"[hf] ensuring Space exists: {repo_id}")
    create_repo(repo_id, repo_type="space", space_sdk="docker",
                token=token, exist_ok=True)

    # OpenAI key → Space secret (so /assistant works)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        print("[hf] setting OPENAI_API_KEY Space secret")
        api.add_space_secret(repo_id=repo_id, key="OPENAI_API_KEY",
                             value=openai_key)

    print(f"[hf] uploading {staging} → {repo_id} (LFS auto-detected)")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(staging),
        commit_message="Deploy FabricFlow demo",
    )
    return f"https://{user}-{space_name}.hf.space"


if __name__ == "__main__":
    path = stage()
    print(f"[stage] tree ready at {path}")
    print(f"[stage] file count: {sum(1 for _ in path.rglob('*') if _.is_file())}")
    print(f"[stage] total size: "
          f"{sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) / 1e9:.2f} GB")
    if "--stage-only" in sys.argv:
        sys.exit(0)
    url = upload(path)
    print(f"\nDONE → Live demo: {url}")
    print(f"     Space repo : https://huggingface.co/spaces/"
          f"{os.environ['HF_USERNAME']}/"
          f"{os.environ.get('HF_SPACE_NAME', 'fabricflow-demo')}")
