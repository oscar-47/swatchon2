"""
Publish FabricFlow demo to Hugging Face: a model repo for the .pth checkpoints
plus a Docker-SDK Space that runs the FastAPI server and pulls those models
at build time.

Required env:
  HF_TOKEN       — write-scoped token (https://huggingface.co/settings/tokens)
  HF_USERNAME    — your HF account, e.g. "turbosun"

Optional env:
  HF_SPACE_NAME  — Space repo name (default: "fabricflow-demo")
  HF_MODELS_NAME — model repo name (default: "fabricflow-models")
  OPENAI_API_KEY — copied to Space Secrets so /assistant works

Result:
  Space repo  : https://huggingface.co/spaces/<USER>/<SPACE>
  Models repo : https://huggingface.co/<USER>/<MODELS>
  Live demo   : https://<USER>-<SPACE>.hf.space
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = Path(__file__).resolve().parent
STAGING_SPACE = Path("/tmp/fabricflow_hf_space")
STAGING_MODELS = Path("/tmp/fabricflow_hf_models")
SAMPLE_SRC = Path(os.path.expanduser("~/Downloads/sample"))

MODEL_FILES = (
    "stage1_knit_vs_woven_vs_others_best.pth",
    "stage2_woven_7class_best.pth",
    "stage2_knit_6class_best.pth",
)


def env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        sys.exit(f"missing env var: {name}")
    return val


def stage_models() -> Path:
    if STAGING_MODELS.exists():
        shutil.rmtree(STAGING_MODELS)
    STAGING_MODELS.mkdir(parents=True)

    # Models
    for name in MODEL_FILES:
        src = REPO_ROOT / "runs" / name
        if not src.exists():
            sys.exit(f"missing model checkpoint: {src}")
        shutil.copy2(src, STAGING_MODELS / name)

    # LFS rules + lightweight README
    (STAGING_MODELS / ".gitattributes").write_text(
        "*.pth filter=lfs diff=lfs merge=lfs -text\n"
    )
    (STAGING_MODELS / "README.md").write_text(
        "---\nlicense: mit\nlibrary_name: pytorch\ntags:\n"
        "  - image-classification\n  - fabric\n  - convnext\n---\n\n"
        "# FabricFlow Models\n\n"
        "ConvNeXt checkpoints used by the FabricFlow demo Space.\n\n"
        "- `stage1_knit_vs_woven_vs_others_best.pth` — 3-class top-level\n"
        "- `stage2_knit_6class_best.pth` — knit sub-categories\n"
        "- `stage2_woven_7class_best.pth` — woven sub-categories\n"
    )
    return STAGING_MODELS


def stage_space() -> Path:
    if STAGING_SPACE.exists():
        shutil.rmtree(STAGING_SPACE)
    STAGING_SPACE.mkdir(parents=True)

    # Deployment-specific files at Space repo root
    for name in ("Dockerfile", "requirements.txt", "README.md",
                 ".gitattributes", ".dockerignore"):
        shutil.copy2(DEPLOY_DIR / name, STAGING_SPACE / name)

    # Source tree (no models — those are pulled in Dockerfile at build time)
    shutil.copytree(REPO_ROOT / "server", STAGING_SPACE / "server",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                  "config.py"))
    shutil.copytree(REPO_ROOT / "web", STAGING_SPACE / "web",
                    ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))

    if not SAMPLE_SRC.is_dir():
        sys.exit(f"sample folder missing: {SAMPLE_SRC}")
    shutil.copytree(SAMPLE_SRC, STAGING_SPACE / "sample",
                    ignore=shutil.ignore_patterns(".DS_Store"))

    # FabricAI grounding knowledge doc lives at repo root; context_provider
    # reads it at runtime so the assistant has real content to cite.
    onboarding = REPO_ROOT / "onboarding_explain.md"
    if onboarding.exists():
        shutil.copy2(onboarding, STAGING_SPACE / "onboarding_explain.md")

    return STAGING_SPACE


def push_models(api, repo_id: str) -> None:
    from huggingface_hub import create_repo
    print(f"[hf] ensuring model repo: {repo_id}")
    create_repo(repo_id, repo_type="model", token=api.token, exist_ok=True)
    path = stage_models()
    print(f"[hf] uploading models from {path} → {repo_id}")
    api.upload_folder(repo_id=repo_id, repo_type="model",
                      folder_path=str(path),
                      commit_message="Upload FabricFlow ConvNeXt checkpoints")


def push_space(api, space_id: str, models_repo_id: str) -> str:
    from huggingface_hub import create_repo
    print(f"[hf] ensuring Space: {space_id}")
    create_repo(space_id, repo_type="space", space_sdk="docker",
                token=api.token, exist_ok=True)

    # Make sure Dockerfile knows where to pull models from
    api.add_space_variable(repo_id=space_id, key="MODELS_REPO",
                           value=models_repo_id)

    # Pass FabricAI credentials through Space Secrets. Two provider modes:
    #   - Azure OpenAI: any of AZURE_OPENAI_* env vars are forwarded.
    #   - Vanilla OpenAI: OPENAI_API_KEY / OPENAI_MODEL.
    # Whichever set is exported in the local shell wins. Values never touch
    # disk — they go straight from this process to the HF Secrets API.
    azure_secret_keys = (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
    )
    forwarded_any = False
    for key in azure_secret_keys:
        val = os.environ.get(key)
        if val:
            print(f"[hf] setting {key} Space secret")
            api.add_space_secret(repo_id=space_id, key=key, value=val)
            forwarded_any = True

    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_model = os.environ.get("OPENAI_MODEL")
    if openai_key and not forwarded_any:
        print("[hf] setting OPENAI_API_KEY Space secret")
        api.add_space_secret(repo_id=space_id, key="OPENAI_API_KEY",
                             value=openai_key)
    if openai_model:
        print("[hf] setting OPENAI_MODEL Space secret")
        api.add_space_secret(repo_id=space_id, key="OPENAI_MODEL",
                             value=openai_model)

    # snapshot_download in the build needs HF auth for private model repos.
    # Public model repos don't, but set anyway in case the user makes it private.
    api.add_space_secret(repo_id=space_id, key="HF_TOKEN", value=api.token)

    path = stage_space()
    size_gb = sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) / 1e9
    print(f"[hf] Space staging: {size_gb:.3f} GB")

    print(f"[hf] uploading Space from {path} → {space_id}")
    api.upload_folder(repo_id=space_id, repo_type="space",
                      folder_path=str(path),
                      commit_message="Deploy FabricFlow demo (Docker SDK)")

    user, name = space_id.split("/", 1)
    return f"https://{user}-{name}.hf.space"


def main() -> None:
    from huggingface_hub import HfApi

    token = env("HF_TOKEN")
    user = env("HF_USERNAME")
    space_name = os.environ.get("HF_SPACE_NAME", "fabricflow-demo")
    models_name = os.environ.get("HF_MODELS_NAME", "fabricflow-models")

    api = HfApi(token=token)
    me = api.whoami()
    print(f"[hf] authenticated as: {me.get('name')}")

    models_repo = f"{user}/{models_name}"
    space_repo = f"{user}/{space_name}"

    push_models(api, models_repo)
    url = push_space(api, space_repo, models_repo)

    print(f"\nDONE")
    print(f"  Models : https://huggingface.co/{models_repo}")
    print(f"  Space  : https://huggingface.co/spaces/{space_repo}")
    print(f"  Live   : {url}")
    print(f"\nFirst build takes ~10 min (pulls torch CPU wheel + models).")


if __name__ == "__main__":
    if "--stage-only" in sys.argv:
        stage_models()
        stage_space()
        sm = sum(p.stat().st_size for p in STAGING_MODELS.rglob('*') if p.is_file()) / 1e9
        ss = sum(p.stat().st_size for p in STAGING_SPACE.rglob('*') if p.is_file()) / 1e9
        print(f"[stage] models = {sm:.2f} GB, space = {ss:.3f} GB")
    else:
        main()
