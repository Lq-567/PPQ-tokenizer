"""Shared-input FLASH reproduction helpers; upstream training model is unchanged."""
from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE
PROJECT = BUNDLE.parent
UPSTREAM = ROOT / "vendor" / "Flash"
sys.path.insert(0, str(UPSTREAM))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "hf_cache"))

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from genrec.datasets.AmazonReviews2014.dataset import AmazonReviews2014
from genrec.dataset import AbstractDataset
from genrec.models.Flash.model import Flash
from genrec.models.Flash.tokenizer import FlashTokenizer
from genrec.utils import get_config
from torch.utils.data import Dataset


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path, model):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(model.state_dict(), temp)
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    temp.replace(path)


def verify_source():
    lock = read(ROOT / "SOURCE_LOCK.json")
    for path, expected in lock["sha256"].items():
        assert sha(path) == expected, f"Source changed: {path}"
    return sha(ROOT / "SOURCE_LOCK.json")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)


def runtime():
    return {"at": now(), "hostname": socket.gethostname(), "server": os.environ.get("FLASH_SERVER", ".3"),
            "pid": os.getpid(), "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python": sys.version, "torch": torch.__version__}


class TensorRows(Dataset):
    def __init__(self, data):
        self.data = data
        self.n = len(data["input_ids"])
        assert all(len(value) == self.n for value in data.values())

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.data.items()}


def load_tensors(category, split):
    asset = read(os.environ.get("PPQ_ASSET_LOCK", str(ROOT / "ASSET_LOCK.json")))[category]
    path = Path(asset["tensor_paths"][split])
    expected = asset["tensor_file_sha256"][split]
    assert sha(path) == expected, f"Changed shared tensor: {path}"
    return torch.load(path, map_location="cpu", weights_only=True)


class FrozenOrderedTokenizer(FlashTokenizer):
    """Use the frozen Ordered PQ assignments with FLASH's native offset convention."""
    def _init_tokenizer(self, dataset):
        asset = read(os.environ.get("PPQ_ASSET_LOCK", str(ROOT / "ASSET_LOCK.json")))[dataset.category]
        if os.environ.get("GR_SID_LOCK"):
            sid = read(os.environ["GR_SID_LOCK"])
            assert sid["dataset"] == dataset.category and sid["m"] == self.n_digit
            asset = asset | sid
        assert sha(asset["sem_ids_path"]) == asset["sem_ids_sha256"]
        codes = np.load(asset["sem_ids_path"], allow_pickle=False)
        assert codes.shape == (dataset.n_items - 1, self.n_digit)
        assert np.issubdtype(codes.dtype, np.integer)
        assert int(codes.min()) >= 0 and int(codes.max()) < self.codebook_size
        assignments = {item: codes[item_id - 1].tolist()
                       for item, item_id in dataset.item2id.items() if item_id > 0}
        assert len(assignments) == dataset.n_items - 1
        return self._sem_ids_to_tokens(assignments)


def make_model(config, device):
    merged = get_config("Flash", "AmazonReviews2014", None, {
        **config, "cache_dir": str(ROOT / "cache"), "device": device, "use_ddp": False,
        "accelerator": Accelerator(cpu=device.type == "cpu"), "metadata": "none",
    })
    # The cached payloads were checked against the original loader/tokenizer for
    # every context. Read them directly so moving a frozen cache never triggers
    # unrelated raw-data downloads or writes to historical experiment folders.
    dataset = AmazonReviews2014.__new__(AmazonReviews2014)
    AbstractDataset.__init__(dataset, merged)
    dataset.category = config["category"]
    dataset.cache_dir = str(ROOT / "cache" / "AmazonReviews2014" / dataset.category)
    dataset.id_mapping = read(Path(dataset.cache_dir) / "processed" / "id_mapping.json")
    dataset.all_item_seqs = read(Path(dataset.cache_dir) / "processed" / "all_item_seqs.json")
    tokenizer = FrozenOrderedTokenizer(merged, dataset)
    seed_all(config["rand_seed"])
    # Preserve the baseline parameter initialization and RNG consumption exactly.
    # The unused text projector remains present; forward never calls it at weight 0.
    construction_config = dict(merged)
    construction_config["align_weight"] = 0.1
    model = SharedFlash(construction_config, dataset, tokenizer).to(device)
    model.align_weight = float(config["align_weight"])
    model.config["align_weight"] = model.align_weight
    return model


class SharedFlash(Flash):
    """Keep upstream losses/heads; give exact item-score ties a fixed item-ID order."""
    def item_scores(self, batch):
        assert "labels" not in batch, "Generation receives history only"
        outputs = self.forward(batch, return_loss=False)
        states = outputs.final_states.gather(
            1, (batch["seq_lens"] - 1).view(-1, 1, 1, 1).expand(
                -1, 1, self.n_pred_head, self.config["n_embd"]))
        states = F.normalize(states, dim=-1)
        embeddings = F.normalize(self.gpt2.wte.weight[1:-1], dim=-1)
        codebooks = torch.chunk(embeddings, self.n_pred_head, dim=0)
        temperatures = self.log_temperature.exp()
        if not self.per_head_temperature:
            temperatures = temperatures.expand(self.n_pred_head)
        log_probabilities = torch.cat([
            F.log_softmax(states[:, 0, index, :] @ codebooks[index].T / temperatures[index], dim=-1)
            for index in range(self.n_pred_head)
        ], dim=-1)
        return torch.gather(
            log_probabilities.unsqueeze(1).expand(-1, self.dataset.n_items - 1, -1),
            -1, (self.item_id2tokens[1:] - 1).unsqueeze(0).expand(len(states), -1, -1)
        ).sum(-1)

    def generate(self, batch, n_return_sequences=10):
        scores = self.item_scores(batch)
        # Catalog columns are already in ascending numeric item-ID order.
        ranking = torch.argsort(scores, descending=True, stable=True, dim=-1)
        return (ranking[:, :n_return_sequences] + 1).unsqueeze(-1)


def item_metrics(predictions, labels):
    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    assert predictions.shape == (len(labels), 10)
    hits = predictions == labels[:, None]
    values = {}
    for k in [5, 10]:
        values[f"recall@{k}"] = hits[:, :k].any(1).astype(np.float64)
        values[f"ndcg@{k}"] = (hits[:, :k] / np.log2(np.arange(2, k + 2))).sum(1)
    return {key: float(value.mean()) for key, value in values.items()}, values
