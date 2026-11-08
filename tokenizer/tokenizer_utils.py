"""Tokenizer/teacher 专用：只读历史数据，原子写入新实验产物。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "assets" / "legacy"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text().replace("__ROOT__", str(ROOT)))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def write_history(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))
    tmp.replace(path)


def save_torch(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def save_numpy(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, value)
    tmp.replace(path)


def save_npz(path, **values):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def start_run(cfg, smoke):
    out = Path(cfg.get("output_root", ROOT / "runs")) / cfg["run_id"]
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Nonempty run directory already exists: {out}; use a new run_id")
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg, smoke=bool(smoke), output_dir=str(out))
    write_json(out / "config.json", cfg)
    write_json(out / "training_started.json", {
        "run_id": cfg["run_id"], "pid": os.getpid(), "smoke": bool(smoke),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "initializing", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "first_parameter_update_verified": False,
        "parameter_change_verified": False,
    })
    return cfg, out


def record_update(out, loss, difference, **extra):
    row = read(out / "training_started.json")
    is_ema = extra.get("update_kind") == "ema_centroid_update"
    row.update(status="training", first_loss=float(loss),
               first_parameter_update_verified=bool(difference > 0) and not is_ema,
               parameter_change_verified=bool(difference > 0) and not is_ema,
               first_parameter_max_abs_update=0.0 if is_ema else float(difference),
               state_update_verified=bool(difference > 0),
               ema_update_verified=bool(difference > 0) and is_ema,
               first_ema_max_abs_update=float(difference) if is_ema else None, **extra)
    write_json(out / "training_started.json", row)
    if not difference > 0:
        raise RuntimeError("First training update did not change the checked parameter or EMA centroid")


def data_manifest(cfg):
    data = Path(cfg.get("data_dir", LEGACY / "data" / cfg["dataset"]))
    manifest = read(data / "manifest.json")
    if manifest["dataset"] != cfg["dataset"]:
        raise ValueError("Dataset/manifest mismatch")
    return data, manifest


def load_split(data, manifest, split, limit=None):
    path = data / f"{split}.pt"
    if sha(path) != manifest["sample_hashes"][split]:
        raise RuntimeError(f"Shared sample hash mismatch: {path}")
    values = torch.load(path, map_location="cpu", weights_only=True)
    if limit is not None:
        values = {k: v[:limit] for k, v in values.items()}
    return values


def item_inputs(cfg):
    data, manifest = data_manifest(cfg)
    embedding = Path(cfg.get("embedding_path", manifest["paths"]["embedding"]))  # 可随代码目录移动
    if sha(embedding) != manifest["embedding_fingerprint"]:
        raise RuntimeError("Embedding changed after audit")
    x = np.fromfile(embedding, dtype=np.float32).reshape(manifest["embedding_shape"])
    fit = np.load(data / "fit_item_rows.npy")
    # 训练 item 集由 train.pt 的输入和有效标签独立验证，不读取 test。
    train = load_split(data, manifest, "train")
    observed = np.unique(np.concatenate([train["input_ids"].numpy().ravel(),
                                         train["labels"].numpy().ravel()]))
    observed = observed[observed > 0] - 1
    if not np.array_equal(fit, observed):
        raise RuntimeError("Tokenizer fit items differ from training interactions")
    return x, fit, data, manifest


def provenance(manifest):
    return {"embedding_sha256": manifest["embedding_fingerprint"],
            "split_fingerprint": manifest["split_fingerprint"],
            "train_sha256": manifest["sample_hashes"]["train"],
            "val_sha256": manifest["sample_hashes"]["val"],
            "mapping_sha256": manifest["fingerprints"]["mapping"],
            "row_mapping": "codes/item_vectors row r corresponds to mapped item_id r+1"}


def report_failure(out, error, seconds):
    write_json(out / "result.json", {"status": "failed", "error": str(error),
                                      "error_type": type(error).__name__, "seconds": seconds})
