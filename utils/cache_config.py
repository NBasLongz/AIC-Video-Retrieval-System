import os
import shutil
import string
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheConfigResult:
    cache_root: str
    hf_home: str
    hf_hub_cache: str
    transformers_cache: str
    torch_home: str


def _drive_roots_windows() -> list[str]:
    roots: list[str] = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.exists(root):
            roots.append(root)
    return roots


def _free_gb(path: str) -> float:
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)


def _pick_cache_root(project_root: str, min_free_gb: float) -> str:
    # 1) Explicit override per-machine/user.
    override = os.environ.get("AIC_CACHE_DIR", "").strip()
    if override:
        return override

    # 2) Prefer staying inside the repo drive if it has enough space.
    project_root = os.path.abspath(project_root)
    try:
        if _free_gb(project_root) >= min_free_gb:
            return os.path.join(project_root, ".aic_cache")
    except Exception:
        pass

    # 3) Otherwise, pick the drive root with the most free space (and >= threshold if possible).
    best_root = None
    best_free = -1.0
    candidates = _drive_roots_windows() if os.name == "nt" else [str(Path(project_root).anchor or "/")]
    for root in candidates:
        try:
            free = _free_gb(root)
        except Exception:
            continue
        if free > best_free:
            best_free = free
            best_root = root

    if best_root is None:
        return os.path.join(project_root, ".aic_cache")

    # Keep it deterministic and not too deep.
    return os.path.join(best_root, "aic_cache")


def auto_configure_model_caches(project_root: str) -> CacheConfigResult | None:
    """
    Configure HuggingFace + Torch caches to avoid filling up C: on Windows.

    Rules:
    - If user already set HF_HOME/HUGGINGFACE_HUB_CACHE/TRANSFORMERS_CACHE/TORCH_HOME, don't override.
    - Otherwise select a cache root:
        - AIC_CACHE_DIR (explicit) wins
        - else repo drive if free >= AIC_CACHE_MIN_GB (default 40)
        - else the drive with the most free space
    """
    # Default threshold: 40GB, but allow users to lower to 30GB etc.
    min_free_gb = float(os.environ.get("AIC_CACHE_MIN_GB", "40").strip() or "40")
    cache_root = _pick_cache_root(project_root, min_free_gb=min_free_gb)

    hf_home = os.path.join(cache_root, "huggingface")
    hf_hub_cache = os.path.join(hf_home, "hub")
    transformers_cache = hf_hub_cache
    torch_home = os.path.join(cache_root, "torch")

    # Only set env vars if they are not already present.
    changed = False
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = hf_home
        changed = True
    if not os.environ.get("HUGGINGFACE_HUB_CACHE"):
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache
        changed = True
    if not os.environ.get("TRANSFORMERS_CACHE"):
        os.environ["TRANSFORMERS_CACHE"] = transformers_cache
        changed = True
    if not os.environ.get("TORCH_HOME"):
        os.environ["TORCH_HOME"] = torch_home
        changed = True

    # Avoid pulling TensorFlow into Transformers on machines that don't need it.
    # This reduces import time and can avoid extra memory pressure.
    if not os.environ.get("TRANSFORMERS_NO_TF"):
        os.environ["TRANSFORMERS_NO_TF"] = "1"
        changed = True
    if not os.environ.get("USE_TF"):
        os.environ["USE_TF"] = "0"
        changed = True

    # Ensure directories exist if we changed anything (or if explicit AIC_CACHE_DIR is used).
    try:
        Path(os.environ.get("HF_HOME", hf_home)).mkdir(parents=True, exist_ok=True)
        Path(os.environ.get("HUGGINGFACE_HUB_CACHE", hf_hub_cache)).mkdir(parents=True, exist_ok=True)
        Path(os.environ.get("TORCH_HOME", torch_home)).mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we can't create, leave it to libraries to error with a clear message.
        pass

    if not changed and not os.environ.get("AIC_CACHE_DIR"):
        # Nothing to do and no explicit cache dir requested.
        return None

    return CacheConfigResult(
        cache_root=cache_root,
        hf_home=os.environ.get("HF_HOME", hf_home),
        hf_hub_cache=os.environ.get("HUGGINGFACE_HUB_CACHE", hf_hub_cache),
        transformers_cache=os.environ.get("TRANSFORMERS_CACHE", transformers_cache),
        torch_home=os.environ.get("TORCH_HOME", torch_home),
    )
