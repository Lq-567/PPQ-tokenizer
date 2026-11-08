#!/usr/bin/env python3
"""Beauty 的 PPQ -> 无 Alignment FLASH 完整复现入口。

每次运行写入独立目录。原始实验源码保留在 tokenizer/ 和 vendor/；此文件
只负责组织路径、训练、评估和结果选择，便于阅读和迁移。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from transformers.optimization import get_scheduler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import flash_helpers as flash


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def metrics(ranks):
    ranks = np.asarray(ranks)
    return {f'{name}@{k}': float((np.where(ranks <= k, 1 / np.log2(ranks + 1), 0) if name == 'ndcg' else ranks <= k).mean()) for name in ('ndcg', 'recall') for k in (5, 10)}


def evaluate(model, raw, split, step, folder, limit=None):
    # 与原协议一致：完整商品目录打分；同分时较小 item ID 排前。
    model.eval()
    n = min(len(raw['labels']), limit) if limit else len(raw['labels'])
    ranks, tops, labels = [], [], []
    with torch.no_grad():
        for a in range(0, n, 64):
            batch = {k: v[a:a+64].cuda() for k, v in raw.items()}
            y = batch.pop('labels').reshape(-1)
            scores = model.item_scores(batch)
            target = scores.gather(1, (y-1)[:, None])
            ids = torch.arange(1, scores.shape[1] + 1, device=scores.device)
            rank = 1 + (scores > target).sum(1) + ((scores == target) & (ids[None] < y[:, None])).sum(1)
            top = torch.argsort(scores, descending=True, stable=True)[:, :10] + 1
            ranks.append(rank.cpu().numpy()); tops.append(top.cpu().numpy()); labels.append(y.cpu().numpy())
    ranks, tops, labels = map(np.concatenate, (ranks, tops, labels))
    np.savez_compressed(folder / f'{split}_{step:05d}.npz', rank=ranks, top10=tops, labels=labels, sample_ids=np.arange(n))
    return {'split': split, 'step': step, 'users': n, 'metrics': metrics(ranks)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='一轮 PPQ 和两步 FLASH 的连通性检查，不作为实验结果')
    parser.add_argument('--sid-from', type=Path, help='复用已经训练好的 SID；不重新训练 PPQ')
    args = parser.parse_args()
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'predictions').mkdir(); (out / 'checkpoints').mkdir()
    data = ROOT / 'data' / 'Beauty'
    cfg = json.loads((ROOT / 'configs' / 'ppq.json').read_text())
    cfg.update(run_id='ppq', output_root=str(out), data_dir=str(data),
               opq_initialization_dir=str(ROOT / 'data' / 'initialization'),
               embedding_path=str(ROOT / 'cache/AmazonReviews2014/Beauty/processed/qwen3.7-text-embedding.sent_emb'))
    dump(out / 'ppq_config.json', cfg)
    cmd = [sys.executable, str(ROOT / 'tokenizer/tokenizer_runner.py'), '--config', str(out / 'ppq_config.json')]
    if args.smoke: cmd.append('--smoke')
    if args.sid_from:
        # 复现诊断可复用已审计的量化结果，避免重复训练量化器。
        sid = args.sid_from.resolve()
        if not sid.is_file(): raise FileNotFoundError(sid)
    else:
        subprocess.run(cmd, check=True)
        sid = out / 'ppq/codes.npy'
    old_sid = ROOT / 'reference/ppq_codes.npy'
    ours, old = np.load(sid), np.load(old_sid)
    dump(out / 'sid_comparison.json', {'new_sha256': sha(sid), 'reference_sha256': sha(old_sid),
         'identical': bool(np.array_equal(ours, old)), 'equal_token_fraction': float((ours == old).mean()),
         'equal_item_fraction': float((ours == old).all(axis=1).mean()), 'smoke': args.smoke,
         'sid_source': str(sid), 'reused': bool(args.sid_from)})
    manifest = json.loads((data / 'manifest.json').read_text())
    lock = {'Beauty': {'tensor_paths': {s: str(data / f'{s}.pt') for s in ('train','val','test')},
                      'tensor_file_sha256': manifest['sample_hashes'], 'sem_ids_path': str(sid), 'sem_ids_sha256': sha(sid)}}
    lockpath = out / 'asset_lock.json'; dump(lockpath, lock); os.environ['PPQ_ASSET_LOCK'] = str(lockpath)
    cfg = json.loads((ROOT / 'configs/flash.json').read_text())
    assert cfg['align_weight'] == 0
    random.seed(2024); np.random.seed(2024); torch.manual_seed(2024); torch.cuda.manual_seed_all(2024)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.set_num_threads(2)
    raw = {s: flash.load_tensors('Beauty', s) for s in ('train', 'val', 'test')}
    model = flash.make_model(cfg, torch.device('cuda:0'))
    assert model.align_weight == model.config['align_weight'] == 0
    # 保留原始初始化时消耗的随机数；但实际第一步起禁止调用文本对齐分支。
    model.text_projector.requires_grad_(False)
    def forbidden(*_): raise RuntimeError('Alignment projector must not run')
    model.text_projector.register_forward_pre_hook(forbidden)
    total = 2 if args.smoke else 10100
    nodes = [0, 2] if args.smoke else list(range(0, 10001, 500)) + [10100]
    loader = torch.utils.data.DataLoader(torch.arange(len(raw['train']['labels'])), batch_size=256, shuffle=True,
                generator=torch.Generator().manual_seed(2024 + 21000))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.)
    # smoke 也保留正式的日程，只缩短更新数，不参与结果比较。
    schedule = get_scheduler('cosine', optimizer=opt, num_warmup_steps=10000, num_training_steps=10100)
    iterator = None; rows = []; start = time.monotonic()
    for step in range(total + 1):
        if step:
            if iterator is None: iterator = iter(loader)
            try: indices = next(iterator)
            except StopIteration: iterator = iter(loader); indices = next(iterator)
            batch = {k: v[indices] for k, v in raw['train'].items()}
            model.train(); opt.zero_grad(set_to_none=True)
            n = len(batch['labels']); value = 0.
            for a in range(0, len(indices), 32):
                part = {k: v[a:a+32].cuda() for k, v in batch.items()}
                loss = model(part).loss
                assert torch.isfinite(loss)
                # 与原冻结实验一致：按序列数加权，而不是按有效标签数加权。
                weight = len(part['labels']) / n
                (loss * weight).backward(); value += float(loss.detach()) * weight
            assert all(p.grad is None for p in model.text_projector.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); schedule.step()
            if step == 1 or step % 100 == 0 or step == total:
                print(json.dumps({'step': step, 'loss': value, 'elapsed_seconds': time.monotonic()-start}), flush=True)
        if step in nodes:
            torch.save(model.state_dict(), out / 'checkpoints' / f'step_{step:05d}.pth')
            # 节点评估不能改变后续训练的随机流。
            cpu_rng = torch.get_rng_state(); gpu_rng = torch.cuda.get_rng_state_all()
            py_rng = random.getstate(); numpy_rng = np.random.get_state()
            for split in ('val', 'test'):
                row = evaluate(model, raw[split], split, step, out / 'predictions', 64 if args.smoke else None)
                rows.append(row); print(json.dumps(row), flush=True)
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(gpu_rng)
            random.setstate(py_rng); np.random.set_state(numpy_rng)
            dump(out / 'metrics.json', rows)
    chosen = {}
    for split, name in [('val','validation_selected'), ('test','test_selected_exploratory')]:
        best = max((r for r in rows if r['split'] == split), key=lambda r: (r['metrics']['ndcg@10'], r['metrics']['recall@10'], -r['step']))
        chosen[name] = {'step': best['step'], **{s: next(r for r in rows if r['split'] == s and r['step'] == best['step']) for s in ('val', 'test')}}
    dump(out / 'selections.json', chosen)
    if not args.smoke:
        # 比较相同步数的逐样本预测，并分别比较各自按同一规则选出的指标。
        comparisons = {}
        for step in (4500, 5500):
            for split in ('val', 'test'):
                new = np.load(out / 'predictions' / f'{split}_{step:05d}.npz')
                old = np.load(ROOT / 'reference' / f'{split}_{step}.npz')
                assert np.array_equal(new['labels'], old['labels']) and np.array_equal(new['sample_ids'], old['sample_ids'])
                comparisons[f'{split}_{step}'] = {
                    'users': len(new['rank']),
                    'rank_equal_fraction': float((new['rank'] == old['rank']).mean()),
                    'top10_row_equal_fraction': float((new['top10'] == old['top10']).all(axis=1).mean()),
                    'new_metrics': metrics(new['rank']), 'reference_metrics': metrics(old['rank'])}
        original = json.loads((ROOT / 'reference/original_selections.json').read_text())['selections']
        for key in ('validation_selected', 'test_selected_exploratory'):
            comparisons[key] = {'new_step': chosen[key]['step'], 'original_step': original[key]['step'],
                'new_test': chosen[key]['test']['metrics'], 'original_test': original[key]['test']['metrics']}
        dump(out / 'comparison.json', comparisons)
    dump(out / 'COMPLETE.json', {'smoke': args.smoke, 'elapsed_seconds': time.monotonic()-start, 'nodes': nodes, 'alignment_weight': 0, 'independent_final_test': False})

if __name__ == '__main__': main()
