"""CPU 机制诊断；重建与 teacher 保真均是 oracle 诊断，不是推荐准确率。

Run: python -m src.diagnostics --config /absolute/path/config.json
See docs/hypotheses.md for input contracts and causal interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np


def _codes(value):
    a = np.asarray(value)
    if a.ndim != 2 or min(a.shape) == 0:
        raise ValueError("SID must be a nonempty [catalog_items, token_positions] matrix")
    if not np.issubdtype(a.dtype, np.integer) or np.any(a < 0):
        raise ValueError("SID must contain nonnegative integer local code indices")
    return a


def _matrix(value, name):
    a = np.asarray(value)
    if a.ndim != 2 or min(a.shape) == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite nonempty matrix")
    return a


def _entropy(masses):
    p = np.asarray(masses, dtype=np.float64)
    if np.any(p < 0) or not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError("Entropy masses must be finite, nonnegative, and nonzero")
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def bucket_statistics(codes, item_weights=None):
    """同时报告 unique ratio 与 item singleton ratio，避免误把两者当同一指标。"""
    a = _codes(codes)
    _, inverse, counts = np.unique(a, axis=0, return_inverse=True, return_counts=True)
    n = len(a)
    weights = np.ones(n, dtype=np.float64) if item_weights is None else np.asarray(item_weights, dtype=np.float64)
    if weights.shape != (n,) or np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("item_weights must align with catalog rows and have positive total mass")
    weights = weights / weights.sum()
    masses = np.bincount(inverse, weights=weights, minlength=len(counts))
    item_h, code_h = _entropy(weights), _entropy(masses)
    singleton = counts[inverse] == 1
    return {
        "catalog_items": n, "unique_buckets": len(counts),
        "unique_code_ratio": float(len(counts) / n),
        "collision_excess_ratio": float(1 - len(counts) / n),
        "singleton_item_fraction": float(singleton.mean()),
        "singleton_bucket_fraction": float((counts == 1).mean()),
        "singleton_probability_mass": float(weights[singleton].sum()),
        "mean_bucket_size_uniform_bucket": float(counts.mean()),
        "mean_bucket_size_uniform_item": float(counts[inverse].mean()),
        "bucket_size_quantiles": dict(zip(["p0", "p50", "p90", "p99", "p100"], map(float, np.quantile(counts, [0, .5, .9, .99, 1])))),
        "item_entropy_bits": item_h, "joint_code_entropy_bits": code_h,
        "item_given_code_entropy_bits": float(max(0., item_h - code_h)),
        "weighting": "uniform_catalog" if item_weights is None else "provided_item_mass",
        "interpretation": "catalog partition only; entropy is not conditional on user history",
    }


def token_statistics(codes, codebook_size=None):
    a = _codes(codes)
    sizes = [codebook_size] * a.shape[1] if np.isscalar(codebook_size) or codebook_size is None else list(codebook_size)
    if len(sizes) != a.shape[1]:
        raise ValueError("codebook_size must be scalar or one size per position")
    result = []
    for j, size in enumerate(sizes):
        values, counts = np.unique(a[:, j], return_counts=True)
        if size is not None and (int(size) <= 0 or values[-1] >= int(size)):
            raise ValueError("Code exceeds supplied local codebook size; remove position offsets first")
        h = _entropy(counts)
        result.append({"position": j, "used_codes": len(values), "codebook_size": size,
                       "utilization": None if size is None else float(len(values) / size),
                       "entropy_bits": h, "perplexity": float(2 ** h),
                       "max_token_frequency": float(counts.max() / len(a)),
                       "counts": {str(int(v)): int(c) for v, c in zip(values, counts)}})
    return result


def make_mask_plan(n_tokens, keep_lengths=None, subset_seeds=(101, 211, 307, 401, 503), group_size=8):
    """同一码率比较前缀、反向组前缀、随机组与任意 token 子集。保留原位置。"""
    if n_tokens < 1 or group_size < 1 or n_tokens % group_size:
        raise ValueError("n_tokens must be positive and divisible by group_size")
    lengths = list(range(group_size, n_tokens + 1, group_size)) if keep_lengths is None else sorted(set(map(int, keep_lengths)))
    if not lengths or any(k < 1 or k > n_tokens for k in lengths):
        raise ValueError("Invalid retained token count")
    seeds = list(map(int, subset_seeds))
    if len(set(seeds)) != len(seeds):
        raise ValueError("subset_seeds must be distinct")
    groups = np.arange(n_tokens).reshape(-1, group_size)
    result = []
    for k in lengths:
        result.append({"name": f"prefix_{k}", "kind": "prefix", "keep": k, "positions": list(range(k)), "seed": None})
        if k == n_tokens:
            continue
        if k % group_size == 0:
            result.append({"name": f"reverse_groups_{k}", "kind": "reverse_groups", "keep": k,
                           "positions": sorted(groups[::-1][:k // group_size].reshape(-1).tolist()), "seed": None})
        for seed in seeds:
            rng = np.random.default_rng(np.random.SeedSequence([seed, k, n_tokens]))
            result.append({"name": f"random_tokens_{k}_s{seed}", "kind": "random_tokens", "keep": k,
                           "positions": sorted(rng.choice(n_tokens, k, replace=False).tolist()), "seed": seed})
            if k % group_size == 0:
                chosen = rng.choice(len(groups), k // group_size, replace=False)
                result.append({"name": f"random_groups_{k}_s{seed}", "kind": "random_groups", "keep": k,
                               "positions": sorted(groups[chosen].reshape(-1).tolist()), "seed": seed})
    return result


def summarize_mask_metrics(rows, metric_paths):
    """随机 mask 的波动不是训练 seed 的置信区间。"""
    output = []
    for kind, k in sorted({(r["kind"], r["keep"]) for r in rows}):
        group = [r for r in rows if r["kind"] == kind and r["keep"] == k]
        summary = {"kind": kind, "keep": k, "mask_count": len(group), "metrics": {}}
        for path in metric_paths:
            values = []
            for row in group:
                v = row
                for part in path.split("."):
                    v = v[part]
                values.append(float(v))
            summary["metrics"][path] = {"mean": float(np.mean(values)), "std_across_masks": float(np.std(values)),
                                                "min": min(values), "max": max(values)}
        output.append(summary)
    return output


def structural_diagnostics(codes, masks=None, codebook_size=None, item_weights=None):
    a = _codes(codes)
    masks = make_mask_plan(a.shape[1], group_size=1) if masks is None else masks
    rows = [dict(mask, statistics=bucket_statistics(a[:, mask["positions"]], item_weights)) for mask in masks]
    return {"full_code": bucket_statistics(a, item_weights), "tokens": token_statistics(a, codebook_size),
            "masks": rows, "mask_summary": summarize_mask_metrics(rows, ["statistics.singleton_item_fraction", "statistics.item_given_code_entropy_bits"]),
            "claims": ["Full-code column permutation leaves the catalog partition invariant.",
                       "Restricting access to a prefix after permutation changes available information.",
                       "Marginal/catalog entropies do not estimate conditional total correlation given user history."]}


def score_fidelity(teacher_scores, student_scores, topks=(5, 10), temperature=1.):
    """固定候选和用户行序，报告 teacher 的决策保真；不接收真实目标 item。"""
    t = _matrix(teacher_scores, "teacher_scores").astype(np.float64)
    s = _matrix(student_scores, "student_scores").astype(np.float64)
    if t.shape != s.shape or temperature <= 0:
        raise ValueError("Aligned score shapes and positive temperature required")
    topks = sorted(set(int(k) for k in topks if 0 < int(k) <= t.shape[1]))
    if not topks:
        raise ValueError("At least one valid top-k is required")

    def log_probs(x):
        x = x / temperature
        x = x - x.max(axis=1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=1, keepdims=True))

    logq, logp = log_probs(t), log_probs(s)
    q, p = np.exp(logq), np.exp(logp)
    kl = np.maximum(0., (q * (logq - logp)).sum(axis=1))
    tv = np.abs(q - p).sum(axis=1) / 2
    error = t - s
    centered = error - error.mean(axis=1, keepdims=True)
    # Stable tie handling: row position is the candidate identity tie breaker.
    ti, si = np.argsort(-t, axis=1, kind="stable"), np.argsort(-s, axis=1, kind="stable")
    rows = np.arange(len(t))[:, None]
    result = {"users": len(t), "candidates": t.shape[1], "temperature": float(temperature),
              "raw_score_mse": float(np.mean(error ** 2)), "centered_score_mse": float(np.mean(centered ** 2)),
              "teacher_to_student_kl_nats": float(kl.mean()), "total_variation": float(tv.mean()),
              "topk": {}, "scope": "oracle_teacher_fidelity_on_supplied_candidates_not_recommendation_accuracy"}
    for k in topks:
        a, b = ti[:, :k], si[:, :k]
        overlap = np.array([np.intersect1d(x, y).size / k for x, y in zip(a, b)])
        regret = q[rows, a].sum(axis=1) - q[rows, b].sum(axis=1)
        epsilon = np.max(np.abs(centered), axis=1)
        margin = t[np.arange(len(t)), ti[:, k-1]] - t[np.arange(len(t)), ti[:, k]] if k < t.shape[1] else np.full(len(t), np.inf)
        result["topk"][str(k)] = {
            "set_overlap": float(overlap.mean()), "exact_set_fraction": float((overlap == 1.).mean()),
            "teacher_probability_mass_regret": float(regret.mean()),
            "mean_pinsker_regret_bound": float(np.minimum(1., np.sqrt(2 * kl)).mean()),
            "regret_bound_satisfied": bool(np.all(regret <= np.sqrt(2 * kl) + 1e-9)),
            "margin_certified_set_fraction": float((margin > 2 * epsilon).mean()),
        }
    return result


def reconstruction_diagnostics(codes, embeddings, reconstruct: Callable, masks, evaluation_rows=None,
                               teacher_users=None, teacher_scores=None, topks=(5, 10), temperature=1.):
    """回调 reconstruct(codes, original_keep_positions) 不可重新编号剩余位置。"""
    a, x = _codes(codes), _matrix(embeddings, "embeddings")
    if len(a) != len(x):
        raise ValueError("Embedding rows must match SID rows")
    selected = np.arange(len(a)) if evaluation_rows is None else np.asarray(evaluation_rows, dtype=np.int64)
    if selected.ndim != 1 or not len(selected) or selected.min() < 0 or selected.max() >= len(a) or len(np.unique(selected)) != len(selected):
        raise ValueError("evaluation_rows must be unique valid catalog rows")
    if (teacher_users is None) != (teacher_scores is None):
        raise ValueError("Provide both teacher_users and teacher_scores for dot-product fidelity")
    rows = []
    for mask in masks:
        reconstructed = _matrix(reconstruct(a, mask["positions"]), "reconstruction")
        if reconstructed.shape != x.shape:
            raise ValueError("Reconstructor output must match the complete embedding matrix")
        xx, yy = np.asarray(x[selected], dtype=np.float64), np.asarray(reconstructed[selected], dtype=np.float64)
        per_item = np.mean((xx - yy) ** 2, axis=1)
        norms = np.linalg.norm(xx, axis=1) * np.linalg.norm(yy, axis=1)
        cosine = (xx * yy).sum(axis=1) / np.maximum(norms, 1e-30)
        row = dict(mask, reconstruction={"mse": float(per_item.mean()), "median_item_mse": float(np.median(per_item)),
                   "p95_item_mse": float(np.quantile(per_item, .95)), "cosine_similarity": float(cosine.mean()),
                   "evaluated_items": len(selected), "zero_norm_pairs": int((norms == 0).sum())})
        if teacher_users is not None:
            users = _matrix(teacher_users, "teacher_users")
            if users.shape[1] != reconstructed.shape[1]:
                raise ValueError("Teacher users and decoder output must share the declared dot-product space")
            row["teacher_fidelity"] = score_fidelity(teacher_scores, users @ reconstructed.T, topks, temperature)
        rows.append(row)
    return {"scope": "oracle_item_reconstruction_not_history_conditioned_generation", "masks": rows,
            "mask_summary": summarize_mask_metrics(rows, ["reconstruction.mse", "reconstruction.cosine_similarity"]),
            "warning": "Better reconstruction does not establish higher recommendation accuracy; seen fit items are descriptive training diagnostics."}


def load_train_sequences(path, n_items):
    """只接受显式 train-only 的独立 artifact；绝不自动读取并猜测完整时间序列。"""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("split") != "train" or payload.get("item_index_base") != 0:
        raise ValueError("Train interactions require {split:'train', item_index_base:0, sequences:[[row,...],...]}")
    sequences = payload.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("sequences must be a list of train-only user sequences")
    for sequence in sequences:
        if not isinstance(sequence, list) or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= n_items for i in sequence):
            raise ValueError("Interaction rows must be integer catalog indices")
    return sequences


def collaborative_neighbors(codes, train_sequences, positions, k=10, max_queries=64, seed=2024):
    """离散 Hamming 近邻与 train 用户集合相似度；同频率区间随机物品为对照。"""
    a = _codes(codes)
    if not 0 < k < len(a) or max_queries < 1:
        raise ValueError("Invalid neighbor budget")
    users = [set() for _ in range(len(a))]
    events = np.zeros(len(a), dtype=np.int64)
    for u, sequence in enumerate(train_sequences):
        if any(i < 0 or i >= len(a) for i in sequence):
            raise ValueError("Invalid train item row")
        for i in sequence:
            events[i] += 1
            users[i].add(u)
    active = np.flatnonzero(events)
    if len(active) < 2:
        return {"status": "insufficient_train_items", "queries": 0}
    rng = np.random.default_rng(seed)
    query_ids = np.sort(rng.choice(active, min(max_queries, len(active)), replace=False))
    # Log frequency bins make controls match broad popularity without future labels.
    bins = np.floor(np.log2(np.maximum(1, events))).astype(int)
    pools = {int(b): np.flatnonzero((bins == b) & (events > 0)) for b in np.unique(bins[active])}
    similarity, control, tie_fraction, query_results = [], [], [], []
    aa = a[:, positions]

    def shared_user_cosine(i, j):
        return len(users[i] & users[j]) / max(1., np.sqrt(len(users[i]) * len(users[j])))

    for i in query_ids:
        score = np.mean(aa == aa[i], axis=1)
        score[i] = -1.
        ids = np.argsort(-score, kind="stable")[:k]
        vals, baseline = [], []
        for j in ids:
            vals.append(shared_user_cosine(int(i), int(j)))
            pool = pools.get(int(bins[j]), active)
            pool = pool[pool != i]
            baseline.append(shared_user_cosine(int(i), int(rng.choice(pool))) if len(pool) else 0.)
        similarity.extend(vals)
        control.extend(baseline)
        ties = float(np.sum(score == score[ids[-1]]) / (len(a) - 1))
        tie_fraction.append(ties)
        query_results.append({"row": int(i), "mean_neighbor_cosine": float(np.mean(vals)),
                              "mean_control_cosine": float(np.mean(baseline))})
    return {"status": "completed", "queries": len(query_ids), "k": k, "seed": seed,
            "mean_train_shared_user_cosine": float(np.mean(similarity)),
            "mean_popularity_bin_control_cosine": float(np.mean(control)),
            "mean_paired_difference": float(np.mean(np.asarray(similarity) - control)),
            "nonzero_shared_user_fraction": float((np.asarray(similarity) > 0).mean()),
            "mean_catalog_fraction_tied_at_neighbor_cutoff": float(np.mean(tie_fraction)),
            "query_results": query_results,
            "scope": "train_association_diagnostic_not_heldout_recommendation_or_causal_semantic_quality"}


def paired_seed_effect(baseline, treatment, seeds, bootstrap_samples=10000, bootstrap_seed=777, confidence=.95):
    """独立训练 seed 是抽样单位；不把 mask seed 或逐用户样本伪装成训练重复。"""
    b, t = np.asarray(baseline, dtype=float), np.asarray(treatment, dtype=float)
    seeds = list(seeds)
    if b.ndim != 1 or b.shape != t.shape or len(b) != len(seeds) or not len(b) or len(set(seeds)) != len(seeds):
        raise ValueError("One paired scalar per distinct training seed is required")
    if not np.isfinite(b).all() or not np.isfinite(t).all() or bootstrap_samples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid values or bootstrap settings")
    d = t - b
    result = {"training_seeds": seeds, "n_seed_pairs": len(d), "paired_differences": d.tolist(),
              "mean_absolute_effect": float(d.mean()), "relative_effect_of_means": None if b.mean() == 0 else float(d.mean() / b.mean()),
              "seed_bootstrap_interval": None, "confidence": confidence,
              "interpretation": "paired training seed variation; does not include population/user sampling uncertainty"}
    if len(d) > 1:
        rng = np.random.default_rng(bootstrap_seed)
        draws = d[rng.integers(0, len(d), size=(bootstrap_samples, len(d)))].mean(axis=1)
        result["seed_bootstrap_interval"] = np.quantile(draws, [(1-confidence)/2, (1+confidence)/2]).tolist()
    result["claim_limit"] = "single_seed_no_uncertainty_estimate" if len(d) == 1 else "few_seed_descriptive_interval_not_strong_significance_evidence" if len(d) < 5 else "bootstrap_under_independent_exchangeable_seed_assumption"
    return result


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_array(path, key=None, dimension=None):
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False)
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as data:
            if key is None and len(data.files) != 1:
                raise ValueError(f"Supply an array key for {p}")
            return data[key or data.files[0]]
    if dimension is not None:
        return np.fromfile(p, dtype=np.float32).reshape(-1, int(dimension))
    raise ValueError("Arrays must be .npy/.npz; raw float32 embeddings require embedding_dim")


def run_diagnostics(config):
    """写入单个原子 JSON；不训练模型，不修改源 artifact，不自动选择实验。"""
    cfg = dict(config)
    start = time.monotonic()
    a = _codes(_load_array(cfg["sid_path"], cfg.get("sid_key")))
    group_size = int(cfg.get("group_size", a.shape[1] // 4 if a.shape[1] % 4 == 0 else 1))
    masks = make_mask_plan(a.shape[1], cfg.get("keep_lengths"), cfg.get("subset_seeds", [101, 211, 307, 401, 503]), group_size)
    cfg.setdefault("group_size", group_size)
    cfg.setdefault("subset_seeds", [101, 211, 307, 401, 503])
    cfg.setdefault("keep_lengths", sorted({m["keep"] for m in masks}))
    result = {"status": "completed", "config": cfg, "diagnostic_version": "ordered-pq-mechanisms-v1",
              "structure": structural_diagnostics(a, masks, cfg.get("codebook_size")), "warnings": []}
    files = {k: v for k, v in cfg.items() if (k.endswith("_path") or k == "checkpoint") and k != "output_path" and isinstance(v, str)}
    if cfg.get("checkpoint"):
        if not cfg.get("reconstruction_adapter"):
            result["warnings"].append("checkpoint supplied but no reconstruction_adapter; only structural diagnostics executed")
        else:
            module, function = cfg["reconstruction_adapter"].split(":", 1)
            if not __package__:
                # 队列直接执行本文件时仍支持文档中的 src.module 工厂路径。
                root = str(Path(__file__).resolve().parents[1])
                if root not in sys.path:
                    sys.path.insert(0, root)
            factory = getattr(importlib.import_module(module), function)
            reconstruct = factory(cfg["checkpoint"], cfg)
            x = _load_array(cfg["embeddings_path"], cfg.get("embeddings_key"), cfg.get("embedding_dim"))
            selected = _load_array(cfg["evaluation_rows_path"]) if cfg.get("evaluation_rows_path") else None
            if cfg.get("teacher_scores_path") and cfg.get("teacher_training_split") != "train":
                raise ValueError("Teacher artifacts require explicit teacher_training_split='train'")
            tu = _load_array(cfg["teacher_users_path"]) if cfg.get("teacher_users_path") else None
            ts = _load_array(cfg["teacher_scores_path"]) if cfg.get("teacher_scores_path") else None
            result["oracle_reconstruction"] = reconstruction_diagnostics(a, x, reconstruct, masks, selected, tu, ts,
                                                                         cfg.get("topks", [5, 10]), cfg.get("temperature", 1.))
    if cfg.get("student_scores_path"):
        if cfg.get("teacher_training_split") != "train":
            raise ValueError("Teacher artifacts require explicit teacher_training_split='train'")
        result["oracle_teacher_fidelity"] = score_fidelity(_load_array(cfg["teacher_scores_path"]), _load_array(cfg["student_scores_path"]),
                                                           cfg.get("topks", [5, 10]), cfg.get("temperature", 1.))
    if cfg.get("train_sequences_path"):
        sequences = load_train_sequences(cfg["train_sequences_path"], len(a))
        event_counts = np.bincount([i for sequence in sequences for i in sequence], minlength=len(a))
        if event_counts.sum():
            result["train_event_weighted_prefixes"] = {str(m["keep"]): bucket_statistics(a[:, m["positions"]], event_counts) for m in masks if m["kind"] == "prefix"}
        result["train_collaborative_neighbors"] = [dict(m, diagnostics=collaborative_neighbors(a, sequences, m["positions"],
            int(cfg.get("neighbor_k", 10)), int(cfg.get("neighbor_queries", 64)), int(cfg.get("neighbor_seed", 2024))))
            for m in masks if m["kind"] == "prefix"]
    result["input_sha256"] = {k: {"path": str(Path(v).resolve()), "sha256": _sha(v)} for k, v in files.items()}
    result["implementation_sha256"] = _sha(__file__)
    result["seconds"] = time.monotonic() - start
    if result["structure"]["masks"][0]["statistics"]["singleton_item_fraction"] > .95:
        result["warnings"].append("Shortest prefix already singles out >95% of catalog items: truncation is not evidence of broad semantic buckets.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config.get("output_path", str(Path(args.config).with_suffix(".diagnostics.json"))))
    result = run_diagnostics(config)
    result["config_file_sha256"] = _sha(args.config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": result["status"], "output_path": str(output.resolve()), "seconds": result["seconds"], "warnings": result["warnings"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
