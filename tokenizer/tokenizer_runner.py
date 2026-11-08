"""固定 teacher 的多前缀任务感知量化；不启动/修改任何旧实验。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

if __package__:
    from .tokenizer_core import LearnablePQ, make_mask
    from .tokenizer_utils import (LEGACY, item_inputs, load_split, provenance, read, record_update,
        report_failure, save_numpy, save_torch, seed_all, sha, start_run, sync,
        write_history, write_json)
else:
    from tokenizer_core import LearnablePQ, make_mask
    from tokenizer_utils import (LEGACY, item_inputs, load_split, provenance, read, record_update,
        report_failure, save_numpy, save_torch, seed_all, sha, start_run, sync,
        write_history, write_json)

from prefix_objective import prefix_reconstruction, equal_weight_objective

MODES = {"optimized", "learned", "ordered", "random", "group_random", "hybrid",
         "task_ordered", "task_random", "task_full"}


def resolved_config(cfg):
    if cfg.get("training_variant", "standard") not in {"standard", "decoder_only", "ema_only", "decoder_frozen"}:
        raise ValueError("Unknown training_variant")
    aliases = {"optimized_pq": "optimized", "learnable_pq": "learned",
               "ordered_prefix": "ordered", "random_subset": "random"}
    mode = aliases.get(cfg.get("mode", "ordered"), cfg.get("mode", "ordered"))
    if mode != 'ordered':
        raise ValueError('This experiment authorizes only ordered equal-weight reconstruction')
    if cfg.get('prefix_aggregation', 'all_sum') != 'all_sum':
        raise ValueError('This authorized grid uses all_sum')
    tc = dict(lr=0.001, batch_size=1024, epochs=200, hidden_dim=256,
              beta=0.25, ema_decay=0.99, task_weight=1e-4,
              task_temperature=1.0, task_batch_size=64, task_candidates=256,
              task_hard_per_history=4, task_views=4, hybrid_probability=0.5,
              task_head_ridge=0.01, max_grad_norm=None,
              task_balance="fixed", target_task_grad_ratio=0.1,
              task_balance_max_weight=0.01, task_balance_eps=1e-12)
    if "lambda" in cfg.get("tokenizer", {}):
        raise ValueError("Mean all-sum has no tunable lambda")
    tc.update(cfg.get("tokenizer", {}))
    if tc["beta"] != 0.25:
        raise ValueError("Configuration is outside the authorized lambda/beta grid")
    for name in ("beta",):
        if not np.isfinite(tc[name]) or tc[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if mode.startswith("task_") and tc["task_views"] != 4:
        raise ValueError("Matched task ablation uses exactly four task views")
    if tc["task_balance"] not in {"fixed", "encoder_grad_ratio"}:
        raise ValueError("task_balance must be fixed or encoder_grad_ratio")
    if not np.isfinite(tc["target_task_grad_ratio"]) or tc["target_task_grad_ratio"] < 0:
        raise ValueError("target_task_grad_ratio must be finite and nonnegative")
    if not np.isfinite(tc["task_balance_max_weight"]) or tc["task_balance_max_weight"] <= 0:
        raise ValueError("task_balance_max_weight must be finite and positive")
    if not np.isfinite(tc["task_balance_eps"]) or tc["task_balance_eps"] <= 0:
        raise ValueError("task_balance_eps must be finite and positive")
    m = cfg.get("m", {"Beauty": 32, "Toys_and_Games": 16, "Sports_and_Outdoors": 16, "CDs_and_Vinyl": 64}[cfg["dataset"]])
    groups = cfg.get("groups", 4)
    if type(groups) is not int or groups not in {2,4,8,16} or m % groups:
        raise ValueError("Group count must be 2/4/8/16 and divide the unchanged full SID length")
    if mode.startswith('task_') and groups != 4:
        raise ValueError("Legacy teacher-task views are only defined for G4; use the separate alternating runner")
    if m != 32 or cfg["dataset"] not in ("Beauty", "Toys_and_Games"):
        raise ValueError("This authorized group study uses M32 in Beauty/Toys")
    retained = [j * (m // groups) for j in range(1, groups)]
    if "retained_blocks" in cfg and cfg["retained_blocks"] != retained:
        raise ValueError("retained_blocks must match groups; these are training views, not final SID lengths")
    return dict(cfg, mode=mode, tokenizer=tc, m=m, groups=groups, K=cfg.get("K", 256),
        prefix_aggregation='all_sum', objective='equal_mean_over_all_G_cumulative_reconstructions_plus_beta_commitment',
        latent_dim=cfg.get("latent_dim", 768), teacher_dim=cfg.get("teacher_dim", 256),
        retained_blocks=retained,
        task_lengths=retained + [m],
        reconstruction_mask=cfg.get("reconstruction_mask", "group_random" if mode == "group_random"
                                   else "random" if mode in {"random", "task_random"}
                                   else "hybrid" if mode == "hybrid" else "ordered"),
        checkpoint_selection="fixed final tokenizer epoch; generator selected independently on validation")


def initialization_paths(cfg):
    shared = Path(cfg.get("opq_initialization_dir", LEGACY / "tokenizers" / cfg["dataset"] / "shared_optimized_pq"))
    return shared / "initialization.npz", shared / "codes.npy"


def load_tokenizer(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir(): checkpoint = checkpoint / "best.pt"
    obj = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = obj["config"]
    init_path, code_path = initialization_paths(cfg)
    init = dict(np.load(init_path)); codes = np.load(code_path)
    # 只用于构建模型形状；EMA buffers 随后完整从 checkpoint 恢复。
    model = LearnablePQ(init, codes, cfg).to(device)
    model.load_state_dict(obj["model"])
    model.eval()
    return model, cfg


def load_diagnostic_reconstructor(checkpoint, config=None):
    config = config or {}
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    model, cfg = load_tokenizer(checkpoint, "cpu")
    model.requires_grad_(False)
    head = config.get("reconstruction_head", "semantic")
    if head not in {"semantic", "task"}:
        raise ValueError("reconstruction_head must be semantic or task")

    @torch.no_grad()
    def reconstruct(codes, keep_positions):
        # keep_positions 可为共享 position 列表，或 N×m bool mask。
        codes = np.asarray(codes)
        positions = np.asarray(keep_positions)
        output = []
        for start in range(0, len(codes), 1024):
            ids = torch.from_numpy(codes[start:start + 1024].astype(np.int64))
            q = model.codebooks[torch.arange(cfg["m"])[None], ids]
            if positions.ndim == 2:
                mask = torch.as_tensor(positions[start:start + len(ids)], dtype=torch.float32)
            else:
                mask = torch.zeros(len(ids), cfg["m"])
                mask[:, positions.astype(np.int64)] = 1.
            decoder = model.task_decode if head == "task" else model.decode
            output.append(decoder(q * mask[..., None]).numpy())
        return np.concatenate(output)
    return reconstruct


class FrozenTeacherCache:
    def __init__(self, path, manifest, device, smoke=False):
        self.path = Path(path)
        cache = read(self.path / "cache_manifest.json")
        for key, value in provenance(manifest).items():
            if cache.get(key) != value:
                raise ValueError(f"Teacher data provenance mismatch: {key}")
        if not cache.get("train_only"):
            raise ValueError("Teacher states must be explicitly marked train-only")
        if cache.get("smoke") and not smoke:
            raise ValueError("Formal tokenizer cannot consume a smoke teacher")
        for name, fingerprint in cache["files_sha256"].items():
            if sha(self.path / name) != fingerprint:
                raise ValueError(f"Teacher cache changed: {name}")
        if sha(self.path / "best.pt") != cache["teacher_checkpoint_sha256"]:
            raise ValueError("Teacher checkpoint changed after state export")
        self.states = torch.from_numpy(np.load(self.path / "train_states.npy")).to(device)
        self.targets = torch.from_numpy(np.load(self.path / "train_targets.npy")).long().to(device)
        self.items = torch.from_numpy(np.load(self.path / "item_vectors.npy")).to(device)
        self.manifest = cache
        for value in [self.states, self.targets, self.items]: value.requires_grad_(False)
        if self.targets.min() < 1 or self.targets.max() > len(self.items):
            raise ValueError("Teacher cached target item IDs are outside the catalog")


@torch.no_grad()
def initialize_task_head(model, initial_codes, fit, teacher, ridge):
    # 训练 item 的量化向量拟合冻结 CF item 向量，减少随机头初始失配。
    # 输入是硬量化 codebook lookup；没有原始 embedding 直达任务头的旁路。
    device = model.codebooks.device
    codes = torch.as_tensor(initial_codes[fit].astype(np.int64), device=device)
    q = model.codebooks[torch.arange(model.m, device=device)[None], codes].reshape(len(fit), -1)
    target = teacher.items[torch.as_tensor(fit, device=device)]
    xm = q.mean(0); ym = target.mean(0)
    x = q - xm; y = target - ym
    gram = x.T @ x + ridge * torch.eye(model.dim, device=device)
    coef = torch.linalg.solve(gram, x.T @ y)
    model.task_decoder.weight.copy_(coef.T)
    model.task_decoder.bias.copy_(ym - xm @ coef)


def task_loss(model, teacher, features, fit_rows, cfg, rng, position_rng):
    tc = cfg["tokenizer"]; device = features.device
    ix = torch.randint(len(teacher.states), (tc["task_batch_size"],), generator=rng, device=device)
    history = teacher.states[ix]
    positive = teacher.targets[ix] - 1
    random_rows = fit_rows[torch.randint(len(fit_rows), (tc["task_candidates"],), generator=rng, device=device)]
    # teacher hard negatives 只在 train item catalog 内选，避免训练未知 item 向量的随机偏好。
    with torch.no_grad():
        hard_scores = history @ teacher.items[fit_rows].T
        hard = fit_rows[hard_scores.topk(min(tc["task_hard_per_history"], len(fit_rows)), dim=1).indices]
        candidates = torch.unique(torch.cat([positive, random_rows, hard.flatten()]))
        target_log_probs = F.log_softmax(history @ teacher.items[candidates].T / tc["task_temperature"], dim=-1)
        target_probs = target_log_probs.exp()
    z, q, _ = model(features[candidates])
    if not q.requires_grad:
        raise RuntimeError("Task loss must pass through hard-quantized STE representation")
    views = []; lengths = cfg["task_lengths"]
    for length in lengths:
        # task_full 四次计算同一全长视图，严格匹配参数量、候选和 task decoder 次数。
        retained = model.m if cfg["mode"] == "task_full" else length
        count = torch.full((len(candidates),), retained, device=device)
        mask_mode = "random" if cfg["mode"] == "task_random" else "ordered"
        mask = make_mask(len(candidates), model.m, count, mask_mode, position_rng, device)
        item_vector = model.task_decode(q * mask[..., None])
        logits = history @ item_vector.T / tc["task_temperature"]
        views.append(F.kl_div(F.log_softmax(logits, dim=-1), target_probs, reduction="batchmean"))
    return torch.stack(views).mean(), views, len(candidates)


def gradient_balance(semantic, task, parameter, tc):
    """只测共享 encoder 参数；系数 stop-gradient，不引入二阶梯度或投影。"""
    task_gradient = torch.autograd.grad(task, parameter, retain_graph=True)[0]
    semantic_gradient = torch.autograd.grad(semantic, parameter, retain_graph=True)[0]
    task_norm = task_gradient.norm()
    semantic_norm = semantic_gradient.norm()
    if not torch.isfinite(task_norm) or not torch.isfinite(semantic_norm):
        raise RuntimeError("Nonfinite task/semantic gradient during balance calibration")
    eps = tc["task_balance_eps"]
    if tc["task_balance"] == "encoder_grad_ratio":
        raw_weight = tc["target_task_grad_ratio"] * semantic_norm / (task_norm + eps)
        effective_weight = raw_weight.clamp(max=tc["task_balance_max_weight"]).detach()
        capped = bool(raw_weight > tc["task_balance_max_weight"])
    else:
        # 固定权重保留 Python float，确保 loss 运算与第一轮逐位一致。
        effective_weight = tc["task_weight"]
        raw_weight = task_norm.new_tensor(effective_weight)
        capped = False
    actual_weight = float(effective_weight)
    weighted_task_norm = (actual_weight * task_gradient).norm()
    info = dict(task_encoder_norm=float(task_norm),
        weighted_task_encoder_norm=float(weighted_task_norm),
        semantic_encoder_norm=float(semantic_norm),
        task_ste_gradient_finite=True, task_ste_gradient_nonzero=bool(task_norm > 0),
        gradient_cosine=float(F.cosine_similarity(task_gradient.flatten(), semantic_gradient.flatten(), dim=0)),
        effective_task_weight=actual_weight, uncapped_task_weight=float(raw_weight),
        effective_gradient_ratio=float(weighted_task_norm / (semantic_norm + eps)),
        target_task_grad_ratio=tc["target_task_grad_ratio"] if tc["task_balance"] == "encoder_grad_ratio" else None,
        task_balance=tc["task_balance"], weight_capped=capped,
        near_zero_task_gradient=bool(task_norm <= eps),
        near_zero_semantic_gradient=bool(semantic_norm <= eps))
    return effective_weight, info


@torch.no_grad()
def diagnostics(model, features, cfg, rng):
    model.eval()
    lengths = cfg["task_lengths"]; errors = {str(k): 0. for k in lengths}
    random_errors = dict(errors); group_random_errors = dict(errors); codes = []
    for start in range(0, len(features), 1024):
        x = features[start:start + 1024]
        q, c = model.quantize(model.encode(x)); codes.append(c.cpu().numpy())
        for k in lengths:
            retained = torch.full((len(x),), k, device=x.device)
            for mode, result in [("ordered", errors), ("random", random_errors),
                                 ("group_random", group_random_errors)]:
                mask = make_mask(len(x), model.m, retained, mode, rng, x.device,
                                 groups=cfg.get('groups', 4))
                result[str(k)] += float((model.decode(q * mask[..., None]) - x).square().mean(1).sum())
    codes = np.concatenate(codes).astype(np.uint8 if model.k <= 256 else np.int32)
    prefixes = {}
    for k in lengths:
        _, counts = np.unique(codes[:, :k], axis=0, return_counts=True)
        prefixes[str(k)] = dict(distinct_prefixes=len(counts),
            distinct_prefixes_per_item=len(counts) / len(codes),
            collision=1 - len(counts) / len(codes),
            singleton_item_fraction=float((counts == 1).sum() / len(codes)), max_bucket=int(counts.max()))
    counts = np.stack([np.bincount(codes[:, j], minlength=model.k) for j in range(model.m)])
    p = counts / len(codes); entropy = -(p * np.log(np.maximum(p, 1e-30))).sum(1)
    result = dict(prefix=prefixes, reconstruction_mse={k: v / len(codes) for k, v in errors.items()},
        random_subset_reconstruction_mse={k: v / len(codes) for k, v in random_errors.items()},
        random_group_reconstruction_mse={k: v / len(codes) for k, v in group_random_errors.items()},
        used_codes=(counts > 0).sum(1).tolist(), mean_utilization=float((counts > 0).mean()),
        entropy=entropy.tolist(), mean_perplexity=float(np.exp(entropy).mean()),
        full_mse=errors[str(model.m)] / len(codes), ema_updates=int(model.ema_updates),
        row_mapping="array row r -> mapped item_id r+1", prefix_means_semantic_hierarchy=False)
    return codes, result


@torch.no_grad()
def validation_teacher_fidelity(model, features, fit, data, manifest, cfg, out, device):
    """固定验证历史的 oracle 任务保真；不读取验证标签，不参与 checkpoint 选择。"""
    if __package__:
        from .teacher_runner import load_teacher
    else:
        from teacher_runner import load_teacher
    started = time.monotonic()
    teacher_model, _ = load_teacher(cfg["teacher_path"], device)
    model.eval()
    limit = 32 if cfg["smoke"] else int(cfg["tokenizer"].get("fidelity_histories", 256))
    val = load_split(data, manifest, "val", limit)
    states = []
    for offset in range(0, len(val["input_ids"]), 128):
        inputs = val["input_ids"][offset:offset + 128].to(device)
        lengths = val["seq_lens"][offset:offset + 128].to(device)
        hidden = teacher_model(inputs)
        states.append(hidden[torch.arange(len(hidden), device=device), lengths - 1])
    users = torch.cat(states)
    items = teacher_model.item.weight[1:]
    fit_t = torch.as_tensor(fit, device=device)
    rng = np.random.default_rng(9112026)
    count = min(len(fit), int(cfg["tokenizer"].get("fidelity_candidates", 1024)))
    sampled = torch.as_tensor(rng.choice(fit, count, replace=False), device=device)
    hard = fit_t[(users @ items[fit_t].T).topk(min(4, len(fit)), dim=1).indices]
    candidates = torch.unique(torch.cat([sampled, hard.flatten()]))
    target_scores = users @ items[candidates].T
    temperature = cfg["tokenizer"]["task_temperature"]
    log_teacher = F.log_softmax(target_scores / temperature, dim=1)
    teacher_top = target_scores.argsort(dim=1, descending=True, stable=True)[:, :10]
    q_parts = []
    for part in candidates.split(1024):
        q, _ = model.quantize(model.encode(features[part]))
        q_parts.append(q)
    q = torch.cat(q_parts)
    rows = []
    for length in cfg["task_lengths"]:
        mask = (torch.arange(model.m, device=device) < length).float()[None, :, None]
        score = users @ model.task_decode(q * mask).T
        log_student = F.log_softmax(score / temperature, dim=1)
        student_top = score.argsort(dim=1, descending=True, stable=True)[:, :10]
        overlap = student_top[:, :, None].eq(teacher_top[:, None, :]).any(2).float().sum(1) / 10
        centered_error = (score - score.mean(1, keepdim=True)) - (target_scores - target_scores.mean(1, keepdim=True))
        rows.append(dict(prefix_length=length,
            kl_teacher_to_quantized=float((log_teacher.exp() * (log_teacher - log_student)).sum(1).mean()),
            teacher_top10_overlap=float(overlap.mean()), score_mse=float((score - target_scores).square().mean()),
            centered_score_mse=float(centered_error.square().mean())))
    save_numpy(out / "teacher_fidelity_candidates.npy", candidates.cpu().numpy())
    result = dict(protocol="oracle teacher ranking fidelity on fixed validation histories; not recommendation accuracy",
        used_for_selection=False, validation_targets_used=False, test_read=False,
        histories=len(users), candidate_count=len(candidates),
        candidate_id_base="0-based catalog rows", candidate_random_seed=9112026,
        candidate_sha256=sha(out / "teacher_fidelity_candidates.npy"),
        candidate_sampling="1024 fixed random train items plus union of teacher top4 train items per history",
        teacher_temperature=temperature, prefix_metrics=rows,
        task_readout="learned task decoder; only task methods are directly comparable",
        seconds=time.monotonic() - started, val_sha256=manifest["sample_hashes"]["val"])
    write_json(out / "val_teacher_fidelity.json", result)
    return result


def train(cfg, smoke=False):
    cfg = resolved_config(cfg)
    # teacher dimension is metadata, frozen before constructing the shared architecture.
    if cfg["mode"].startswith("task_"):
        cache = read(Path(cfg["teacher_path"]) / "cache_manifest.json")
        cfg["teacher_dim"] = cache["teacher_config"]["hidden_dim"]
    cfg, out = start_run(cfg, smoke)
    started = time.monotonic()
    try:
        seed_all(cfg["seed"])
        torch.set_num_threads(int(cfg.get("cpu_threads", 4)))
        device = torch.device(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        x, fit, data, manifest = item_inputs(cfg)
        init_path, code_path = initialization_paths(cfg)
        init = dict(np.load(init_path)); initial_codes = np.load(code_path)
        if initial_codes.shape != (len(x), cfg["m"]):
            raise ValueError("Initial codes have a different catalog or token count")
        model = LearnablePQ(init, initial_codes[fit], cfg).to(device)
        decoder_only = cfg.get("training_variant", "standard") == "decoder_only"
        ema_only = cfg.get("training_variant", "standard") == "ema_only"
        decoder_frozen = cfg.get("training_variant", "standard") == "decoder_frozen"
        if (ema_only or decoder_frozen) and cfg["mode"] != "learned":
            raise ValueError("Bridge controls must use full semantic reconstruction")
        if decoder_only:
            if cfg["mode"] != "learned":
                raise ValueError("Decoder-only negative control must use full semantic reconstruction")
            model.requires_grad_(False)
            model.decoder_linear.requires_grad_(True)
            model.decoder_residual.requires_grad_(True)
        if ema_only:
            model.requires_grad_(False)
        if decoder_frozen:
            model.decoder_linear.requires_grad_(False)
            model.decoder_residual.requires_grad_(False)
        features = torch.from_numpy(x).to(device)
        fit_rows = torch.as_tensor(fit, device=device)
        teacher = None
        task_mode = cfg["mode"].startswith("task_")
        if task_mode:
            teacher = FrozenTeacherCache(cfg["teacher_path"], manifest, device, smoke)
            if len(teacher.items) != len(features):
                raise ValueError("Teacher catalog mismatch")
            initialize_task_head(model, initial_codes, fit, teacher, cfg["tokenizer"]["task_head_ridge"])
        tc = cfg["tokenizer"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=0.)
        data_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 100)
        retained_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 200)
        position_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 300)
        task_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 700)
        task_position_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 701)
        diagnosis_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 500)
        retained_values = torch.as_tensor(cfg["retained_blocks"], device=device)
        source = LEGACY / "scripts" / "pq_codec.py"
        prov = dict(provenance(manifest), initialization=str(init_path), initialization_sha256=sha(init_path),
            initial_codes=str(code_path), initial_codes_sha256=sha(code_path),
            legacy_implementation=str(source), legacy_implementation_sha256=sha(source),
            source_sha256=sha(__file__), core_sha256=sha(Path(__file__).with_name("tokenizer_core.py")),
            teacher_path=str(cfg.get("teacher_path", "")),
            teacher_checkpoint_sha256=teacher.manifest["teacher_checkpoint_sha256"] if teacher else None)
        write_json(out / "provenance.json", prov)
        history = []; steps = 0; total_task_views = 0; first_grad = None
        epochs = 0 if cfg["mode"] == "optimized" else (1 if smoke else tc["epochs"])
        # Smoke executes one update at the formal batch size for real peak admission.
        train_fit = fit_rows[:min(tc['batch_size'], len(fit_rows))] if smoke else fit_rows
        training_started = time.monotonic()
        for epoch in range(1, epochs + 1):
            model.train(); totals = {k: 0. for k in ["loss", "full_mse", "masked_mse", "commitment",
                "semantic_weighted", "task_kl", "task_weighted", "mean_candidates"]}
            view_sums = np.zeros(4); batches = 0
            grad_rows = []
            order = train_fit[torch.randperm(len(train_fit), generator=data_rng, device=device)]
            for ix in order.split(tc["batch_size"]):
                batch = features[ix]
                retained = retained_values[torch.randint(len(retained_values), (len(ix),),
                                            generator=retained_rng, device=device)]
                mask = make_mask(len(ix), model.m, retained, cfg["reconstruction_mask"], position_rng,
                                 device, tc["hybrid_probability"], groups=cfg["groups"])
                z, q, codes = model(batch)
                full_mse = F.mse_loss(model.decode(q), batch)
                masked_mse = prefix_reconstruction(model.decode, q, batch, retained_values, mask, 'all_sum')
                # commitment 使用精确码本值，避免 STE 浮点相消误差改变历史基线。
                hard_q = model.codebooks[torch.arange(model.m, device=device)[None], codes]
                commitment = F.mse_loss(z, hard_q.detach())
                semantic = equal_weight_objective(full_mse, masked_mse, commitment, cfg['groups'], tc['beta'])
                task = full_mse.new_zeros(()); view_values = []; candidates = 0
                if task_mode:
                    task, view_values, candidates = task_loss(model, teacher, features, fit_rows, cfg, task_rng, task_position_rng)
                    total_task_views += 4
                effective_weight = tc["task_weight"]
                current_gradient = None
                if task_mode and (tc["task_balance"] == "encoder_grad_ratio" or steps == 0):
                    effective_weight, current_gradient = gradient_balance(semantic, task, model.encoder_linear.weight, tc)
                    grad_rows.append(current_gradient)
                loss = semantic + effective_weight * task
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite tokenizer objective")
                if steps == 0:
                    watched_parameter = model.codebooks if ema_only else model.decoder_linear.weight if decoder_only else model.encoder_linear.weight
                    before = watched_parameter.detach().clone()
                    if task_mode:
                        first_grad = current_gradient
                optimizer.zero_grad(set_to_none=True)
                if not ema_only:
                    loss.backward()
                if tc["max_grad_norm"] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"])
                # 与历史协议相同：每个重建 batch 仅一次完整 assignment EMA，task views 不额外更新。
                if not decoder_only:
                    model.update_ema(z, codes)
                if not ema_only:
                    optimizer.step()
                steps += 1
                if steps == 1:
                    record_update(out, loss.detach(), float((watched_parameter.detach() - before).abs().max()),
                                  stage="tokenizer", mode=cfg["mode"], device=str(device), task_gradient=first_grad,
                                  watched_parameter="codebooks" if ema_only else "decoder_linear.weight" if decoder_only else "encoder_linear.weight",
                                  update_kind="ema_centroid_update" if ema_only else "optimizer_update",
                                  optimizer_step_applicable=not ema_only)
                values = dict(loss=loss, full_mse=full_mse, masked_mse=masked_mse, commitment=commitment,
                    semantic_weighted=semantic, task_kl=task, task_weighted=effective_weight * task,
                    mean_candidates=candidates)
                for key, value in values.items(): totals[key] += float(value.detach()) if torch.is_tensor(value) else float(value)
                if view_values: view_sums += [float(v.detach()) for v in view_values]
                batches += 1
                if smoke: break
            history.append(dict(epoch=epoch, **{k: v / batches for k, v in totals.items()},
                task_kl_per_view=(view_sums / batches).tolist(), steps=steps,
                gradient_probe_batches=len(grad_rows),
                gradient_balance_summary={key: dict(mean=float(np.mean([r[key] for r in grad_rows])),
                    min=float(np.min([r[key] for r in grad_rows])), max=float(np.max([r[key] for r in grad_rows])))
                    for key in ["task_encoder_norm", "semantic_encoder_norm", "effective_task_weight",
                        "effective_gradient_ratio", "gradient_cosine", "weight_capped",
                        "near_zero_task_gradient", "near_zero_semantic_gradient"]} if grad_rows else {},
                seconds=time.monotonic() - started))
            write_history(out / "training.jsonl", history)
            if epoch == 1 or epoch % 25 == 0 or epoch == epochs: print(history[-1], flush=True)
        if cfg["mode"] == "optimized":
            row = read(out / "training_started.json")
            row.update(status="frozen_optimized_baseline", optimizer_step_not_applicable=True)
            write_json(out / "training_started.json", row)
            write_history(out / "training.jsonl", [])
        sync(device); training_seconds = time.monotonic() - training_started
        save_torch(out / "best.pt", dict(model=model.state_dict(), config=cfg, epoch=epochs,
            optimizer=optimizer.state_dict(), provenance=prov, selection="fixed final epoch"))
        codes, diag = diagnostics(model, features, cfg, diagnosis_rng)
        if cfg["mode"] == "optimized" and not np.array_equal(codes, initial_codes):
            raise RuntimeError("Frozen optimized baseline differs from original OPQ assignments")
        if not np.isfinite(diag["full_mse"]) or min(diag["used_codes"]) < 2:
            raise RuntimeError("Tokenizer collapsed or returned nonfinite diagnostics")
        save_numpy(out / "codes.npy", codes)
        if decoder_only:
            if not np.array_equal(codes, initial_codes) or int(model.ema_updates) != 0:
                raise RuntimeError("Decoder-only negative control changed the fixed SID")
            # 负对照直接保留源文件字节；不是新的 SID 训练结果。
            shutil.copyfile(code_path, out / "codes.npy")
            if sha(code_path) != sha(out / "codes.npy"):
                raise RuntimeError("Decoder-only SID byte identity failed")
        diag.update(first_task_gradient=first_grad, provenance=prov, codes_sha256=sha(out / "codes.npy"),
                    task_forward_views=total_task_views, training_seconds=training_seconds)
        write_json(out / "diagnostics.json", diag)
        if __package__:
            from .diagnostics import make_mask_plan, structural_diagnostics
        else:
            from diagnostics import make_mask_plan, structural_diagnostics
        plan = make_mask_plan(cfg["m"], cfg["task_lengths"], [101, 211, 307], cfg["m"] // cfg["groups"])
        write_json(out / "mechanism_diagnostics.json", structural_diagnostics(codes, plan, cfg["K"]))
        fidelity = validation_teacher_fidelity(model, features, fit, data, manifest, cfg, out, device) if task_mode else None
        # 保存后 CPU/GPU 共用 loader 可复现 hard codes，抽样验证不会加载 teacher。
        reloaded, _ = load_tokenizer(out / "best.pt", device)
        with torch.no_grad():
            _, loaded_codes = reloaded.quantize(reloaded.encode(features[:64]))
        if not np.array_equal(loaded_codes.cpu().numpy(), codes[:64]):
            raise RuntimeError("Saved tokenizer does not reproduce exported hard codes")
        result = dict(status="smoke_completed" if smoke else "completed", run_id=cfg["run_id"],
            dataset=cfg["dataset"], seed=cfg["seed"], mode=cfg["mode"], m=cfg["m"], K=cfg["K"],
            groups=cfg["groups"], tokens_per_group=cfg["m"] // cfg["groups"],
            final_sid_length=cfg["m"], masked_views_per_semantic_batch=len(retained_values),
            prefix_aggregation=cfg["prefix_aggregation"], prefix_denominator=cfg['groups'], lambda_=None,
            full_reconstruction_weight=1 / cfg['groups'], prefix_view_weight=1 / cfg['groups'], lambda_parameter_removed=True,
            retained_length_distribution="all proper cumulative group prefixes in every batch; full view and each proper prefix weighted 1/G",
            method=cfg.get("method", cfg["mode"]),
            epochs_completed=epochs, stop_reason="smoke_one_update" if smoke else "fixed_epoch_budget",
            converged=False, checkpoint_selection="fixed final epoch, no claim of convergence",
            seconds=time.monotonic() - started, training_seconds=training_seconds,
            training_steps=steps, task_forward_views=total_task_views,
            optimizer_steps=0 if ema_only else steps, ema_updates=int(model.ema_updates),
            task_compute_matching="task_full repeats full view four times; task_ordered/random use four resolutions",
            task_mode=task_mode, teacher_extra_seconds=read(Path(cfg["teacher_path"]) / "result.json").get("seconds") if teacher else 0.,
            task_balance=tc["task_balance"], target_task_grad_ratio=tc["target_task_grad_ratio"],
            balance_parameter="encoder_linear.weight", balance_stop_gradient=True,
            teacher_reused=True if teacher else False,
            parameter_count=sum(p.numel() for p in model.parameters()),
            trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
            network_architecture=cfg.get("network_architecture", "nonlinear"),
            training_variant=cfg.get("training_variant", "standard"),
            fixed_sid_verified=decoder_only,
            codes_path=str(out / "codes.npy"), checkpoint=str(out / "best.pt"),
            diagnostics=str(out / "diagnostics.json"), full_mse=diag["full_mse"],
            mechanism_diagnostics=str(out / "mechanism_diagnostics.json"),
            validation_teacher_fidelity=fidelity,
            save_reload_hard_codes_verified=True, provenance=prov)
        result['peak_cuda_memory_gb'] = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0.0
        write_json(out / "result.json", result)
        return result
    except Exception as error:
        report_failure(out, error, time.monotonic() - started)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train(read(args.config), args.smoke)
