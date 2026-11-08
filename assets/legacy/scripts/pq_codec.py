"""Frozen RPG Optimized-PQ and matched learnable block quantizers with EMA."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import time

import faiss
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import ROOT, REPO, event, read, write, sha, seed_all, load_inputs


def fit_optimized_pq(dataset, m, threads=4):
    """Execute upstream OPQ method, retaining its index; no API/model loading."""
    out = ROOT / 'tokenizers' / dataset / 'shared_optimized_pq'
    out.mkdir(parents=True, exist_ok=True)
    x, fit, mapping, _ = load_inputs(dataset)
    source = REPO / 'genrec/models/RPG/tokenizer.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RPGTokenizer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_generate_semantic_id_opq')
    namespace = {'np': np, 'json': json}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), namespace)
    original = faiss.index_factory
    holder = {}

    def capture(*args):
        index = original(*args)
        holder['index'] = index
        return index

    adapter = SimpleNamespace(config={'opq_use_gpu': False, 'faiss_omp_num_threads': threads},
        index_factory=f'OPQ{m},IVF1,PQ{m}x8', n_digit=m, n_codebook_bits=8,
        id2item=mapping['id2item'], log=lambda msg: event('optimized_pq', dataset=dataset, message=msg))
    mask = np.zeros(len(x), dtype=bool)
    mask[fit] = True
    start = time.monotonic()
    faiss.index_factory = capture
    try:
        namespace[method.name](adapter, x, str(out / 'upstream_sid.json'), mask)
    finally:
        faiss.index_factory = original
    index = holder['index']
    faiss.write_index(index, str(out / 'index.faiss'))
    rotation = faiss.downcast_VectorTransform(index.chain.at(0))
    ivf = faiss.downcast_index(index.index)
    assert ivf.nlist == 1 and ivf.by_residual and ivf.code_size == m
    n = ivf.invlists.list_size(0)
    ids = faiss.rev_swig_ptr(ivf.invlists.get_ids(0), n).copy()
    packed = faiss.rev_swig_ptr(ivf.invlists.get_codes(0), n*m).reshape(n,m).copy()
    assert np.array_equal(np.sort(ids), np.arange(len(x)))
    codes = np.empty_like(packed)
    codes[ids] = packed
    upstream = read(out / 'upstream_sid.json')
    assert np.array_equal(codes, np.array([upstream[i] for i in mapping['id2item'][1:]]))
    a = faiss.vector_to_array(rotation.A).reshape(768,768)
    bias = faiss.vector_to_array(rotation.b) if rotation.have_bias else np.zeros(768, np.float32)
    center = ivf.quantizer.reconstruct(0)
    centroids = faiss.vector_to_array(ivf.pq.centroids).reshape(m,256,768//m)
    residual = np.ascontiguousarray(rotation.apply_py(x) - center)
    assert np.array_equal(ivf.pq.compute_codes(residual), codes)
    assert np.allclose(x @ a.T + bias, rotation.apply_py(x), atol=2e-6)
    assert np.allclose(a.T @ a, np.eye(768), atol=2e-5)
    recon = (ivf.pq.decode(codes) + center - bias) @ a
    ref = rotation.reverse_transform(np.ascontiguousarray(ivf.pq.decode(codes) + center))
    assert np.allclose(recon, ref, atol=3e-6)
    ivf.make_direct_map()
    sample = np.linspace(0,len(x)-1,64,dtype=int)
    exact = np.stack([index.reconstruct(int(i)) for i in sample])
    assert np.allclose(recon[sample], exact, atol=3e-6)
    np.save(out / 'codes.npy', codes)
    np.savez(out / 'initialization.npz', rotation=a, bias=bias, center=center, centroids=centroids)
    old = ROOT.parent / 'RPG/cache/AmazonReviews2014' / dataset / 'processed' / f'sentence-t5-base_OPQ{m},IVF1,PQ{m}x8.sem_ids'
    oldcodes = read(old)
    oldmatch = bool(np.array_equal(codes, np.array([oldcodes[i] for i in mapping['id2item'][1:]])))
    write(out / 'sanity.json', dict(full_mse=float(np.mean((recon-x)**2)),
        seconds=time.monotonic()-start, upstream_source_sha=sha(source), old_sid_exact_match=oldmatch,
        rotation_verified=True, residual_center_verified=True, nearest_assignments_exact=True,
        faiss_reconstruction_max_abs_error=float(np.max(np.abs(recon[sample]-exact))),
        opq_seed='upstream Faiss defaults (1234); no seed search', index_sha=sha(out/'index.faiss')))
    event('optimized_pq_complete', dataset=dataset, seconds=time.monotonic()-start, old_sid_exact_match=oldmatch)


class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim,hidden), nn.SiLU(), nn.Linear(hidden,dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class LearnablePQ(nn.Module):
    def __init__(self, initialization, fit_codes, cfg):
        super().__init__()
        self.m, self.k, self.dim = cfg['m'], cfg['K'], cfg['latent_dim']
        d = self.dim
        self.encoder_linear = nn.Linear(d,d)
        self.encoder_residual = ResidualMLP(d, cfg['tokenizer']['hidden_dim'])
        self.decoder_residual = ResidualMLP(d, cfg['tokenizer']['hidden_dim'])
        self.decoder_linear = nn.Linear(d,d)
        a = torch.from_numpy(initialization['rotation'])
        b = torch.from_numpy(initialization['bias'])
        c = torch.from_numpy(initialization['center'])
        with torch.no_grad():
            self.encoder_linear.weight.copy_(a)
            self.encoder_linear.bias.copy_(b-c)
            self.decoder_linear.weight.copy_(a.T)
            self.decoder_linear.bias.copy_((c-b) @ a)
        cb = torch.from_numpy(initialization['centroids'].copy())
        counts = torch.stack([torch.bincount(torch.from_numpy(fit_codes[:,j].astype(np.int64)), minlength=self.k) for j in range(self.m)]).float()
        counts = (counts * cfg['tokenizer']['batch_size'] / len(fit_codes)).clamp_min(0.1)
        self.register_buffer('codebooks', cb)
        self.register_buffer('ema_count', counts)
        self.register_buffer('ema_sum', cb * counts[...,None])
        self.register_buffer('ema_updates', torch.tensor(0, dtype=torch.long))
        self.decay = cfg['tokenizer']['ema_decay']

    def encode(self, x):
        return self.encoder_residual(self.encoder_linear(x)).reshape(-1,self.m,self.dim//self.m)

    def decode(self, q):
        return self.decoder_linear(self.decoder_residual(q.reshape(-1,self.dim)))

    def quantize(self, z):
        zt = z.transpose(0,1)
        dist = zt.square().sum(-1,keepdim=True) + self.codebooks.square().sum(-1)[:,None,:] - 2*torch.bmm(zt,self.codebooks.transpose(1,2))
        codes = dist.argmin(-1).T
        q = self.codebooks[torch.arange(self.m, device=z.device)[None,:], codes]
        return q, codes

    @torch.no_grad()
    def update_ema(self, z, codes):
        assignments = F.one_hot(codes, self.k).float().transpose(0,1)
        counts = assignments.sum(1)
        sums = torch.bmm(assignments.transpose(1,2), z.detach().transpose(0,1))
        self.ema_count.mul_(self.decay).add_(counts, alpha=1-self.decay)
        self.ema_sum.mul_(self.decay).add_(sums, alpha=1-self.decay)
        # Numerically preserve unused centroids; never mix gradient and EMA updates.
        valid = self.ema_count > 1e-6
        self.codebooks.copy_(torch.where(valid[...,None], self.ema_sum / self.ema_count.clamp_min(1e-6)[...,None], self.codebooks))
        self.ema_updates.add_(1)

    def forward(self, x, mask=None, update=True):
        z = self.encode(x)
        q, codes = self.quantize(z)
        straight = z + (q-z).detach()
        full = self.decode(straight)
        masked = self.decode(straight * mask[...,None]) if mask is not None else None
        commit = F.mse_loss(z, q.detach())
        if self.training and update:
            self.update_ema(z, codes)
        return full, masked, commit, codes


def make_mask(n, m, retained, mode, position_rng, device):
    positions = torch.arange(m,device=device)[None,:].expand(n,-1)
    if mode == 'random_subset':
        positions = torch.rand(n,m,generator=position_rng,device=device).argsort(dim=1).argsort(dim=1)
    return (positions < retained[:,None]).float()


def code_diagnostics(codes, k, prefixes):
    n,m = codes.shape
    counts = np.stack([np.bincount(codes[:,j],minlength=k) for j in range(m)])
    p = counts/n
    ent = -(p*np.log(np.maximum(p,1e-30))).sum(1)
    collision = {str(b): float(1-len(np.unique(codes[:,:b],axis=0))/n) for b in prefixes}
    return dict(collision=collision[str(m)], collision_definition='1 - number of unique SID tuples / number of catalog items',
        prefix_collision=collision, used_codes=(counts>0).sum(1).tolist(),
        utilization=((counts>0).sum(1)/k).tolist(), entropy=ent.tolist(), perplexity=np.exp(ent).tolist(),
        mean_utilization=float((counts>0).mean()), mean_perplexity=float(np.exp(ent).mean()))


def train_tokenizer(cfg, smoke=False):
    name = cfg['run_id']
    dest = ROOT / ('checks/smoke_tokenizers' if smoke else 'tokenizers') / name
    dest.mkdir(parents=True,exist_ok=True)
    sid_dest = ROOT / ('checks/smoke_sid_mappings' if smoke else 'sid_mappings') / name
    sid_dest.mkdir(parents=True,exist_ok=True)
    shared = ROOT / 'tokenizers' / cfg['dataset'] / 'shared_optimized_pq'
    x_np, fit, mapping, _ = load_inputs(cfg['dataset'])
    from common import dataset_paths
    manifest=read(ROOT/'data'/cfg['dataset']/'manifest.json')
    assert all(sha(path)==manifest['fingerprints'][key] for key,path in dataset_paths(cfg['dataset']).items()), 'Source data changed after audit'
    init = dict(np.load(shared/'initialization.npz'))
    initial_codes = np.load(shared/'codes.npy')
    device = torch.device('cuda:0')
    seed_all(cfg['seed'])
    start = time.monotonic()
    model = LearnablePQ(init, initial_codes[fit], cfg).to(device)
    torch.save(model.state_dict(),dest/'initial.pt')
    full_data = torch.from_numpy(x_np).to(device)
    data = full_data[torch.from_numpy(fit).to(device)]
    typ, lam = cfg['tokenizer_type'], cfg['lambda']
    tc = cfg['tokenizer']
    data_rng = torch.Generator(device=device).manual_seed(cfg['seed']+100)
    retain_rng = torch.Generator(device=device).manual_seed(cfg['seed']+200)
    position_rng = torch.Generator(device=device).manual_seed(cfg['seed']+300)
    retain_values = torch.tensor(cfg['retained_blocks'],device=device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=tc['lr'],weight_decay=0)
    epochs = (2 if smoke else tc['epochs']) if typ != 'optimized_pq' else 0
    if smoke:
        data = data[:2048]
    before_updates = int(model.ema_updates)
    with open(dest/'training.jsonl','w') as logfile:
        for epoch in range(1,epochs+1):
            model.train()
            totals = np.zeros(4)
            assignment_counts = torch.zeros(cfg['m'],cfg['K'],device=device)
            perm = torch.randperm(len(data),generator=data_rng,device=device)
            for ix in perm.split(tc['batch_size']):
                batch = data[ix]
                retained = retain_values[torch.randint(len(retain_values),(len(ix),),generator=retain_rng,device=device)]
                mask = make_mask(len(ix),cfg['m'],retained,typ,position_rng,device) if lam else None
                optimizer.zero_grad(set_to_none=True)
                full, masked, commit, batch_codes = model(batch,mask)
                assignment_counts += F.one_hot(batch_codes,cfg['K']).float().sum(0)
                lf = F.mse_loss(full,batch)
                lm = F.mse_loss(masked,batch) if masked is not None else lf.new_zeros(())
                loss = (lf+lam*lm)/(1+lam)+tc['beta']*commit
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Optimization failure: nonfinite tokenizer loss at epoch {epoch}')
                loss.backward()
                optimizer.step()
                totals += np.array([loss.item(),lf.item(),lm.item(),commit.item()])*len(ix)
            row = dict(epoch=epoch,loss=totals[0]/len(data),full_loss=totals[1]/len(data),masked_loss=totals[2]/len(data),commitment_loss=totals[3]/len(data))
            if epoch == 1 or epoch%25 == 0 or epoch == epochs:
                probs=assignment_counts/len(data)
                entropy=-(probs*probs.clamp_min(1e-30).log()).sum(1)
                row.update(used_codes=(assignment_counts>0).sum(1).int().cpu().tolist(),
                    perplexity=entropy.exp().cpu().tolist())
            logfile.write(json.dumps(row)+'\n'); logfile.flush()
            if epoch == 1 or epoch%25 == 0 or epoch == epochs:
                event('tokenizer_epoch',run_id=name,**row)
    expected = epochs*((len(data)+tc['batch_size']-1)//tc['batch_size'])
    assert int(model.ema_updates)-before_updates == expected
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=cfg,epochs=epochs), dest/'final.pt')
    # Exact save/reload verification, also done in the end-to-end smoke test.
    reloaded = LearnablePQ(init,initial_codes[fit],cfg).to(device)
    reloaded.load_state_dict(torch.load(dest/'final.pt',map_location=device,weights_only=False)['model'])
    model.eval(); reloaded.eval()
    with torch.no_grad():
        a=model(full_data[:128],update=False); b=reloaded(full_data[:128],update=False)
        assert torch.equal(a[3],b[3]) and torch.equal(a[0],b[0])
    all_codes=[]
    prefixes=cfg['retained_blocks']+[cfg['m']]
    prefix_error={str(b):0.0 for b in prefixes}
    random_error={str(b):0.0 for b in prefixes}
    prefix_fit={str(b):0.0 for b in prefixes}
    fit_mask=np.zeros(len(x_np),dtype=bool);fit_mask[fit]=True
    diagnostic_rng=torch.Generator(device=device).manual_seed(cfg['seed']+500)
    with torch.no_grad():
        for offset in range(0,len(full_data),1024):
            batch=full_data[offset:offset+1024]
            z=model.encode(batch); q,codes=model.quantize(z)
            all_codes.append(codes.cpu().numpy().astype(np.uint8))
            for b in prefixes:
                retain=torch.full((len(batch),),b,device=device)
                pm=make_mask(len(batch),cfg['m'],retain,'ordered_prefix',diagnostic_rng,device)
                rm=make_mask(len(batch),cfg['m'],retain,'random_subset',diagnostic_rng,device)
                err=(model.decode(q*pm[...,None])-batch).square().mean(1)
                prefix_error[str(b)]+=float(err.sum())
                local=torch.from_numpy(fit_mask[offset:offset+len(batch)]).to(device)
                prefix_fit[str(b)]+=float(err[local].sum())
                random_error[str(b)]+=float((model.decode(q*rm[...,None])-batch).square().mean(1).sum())
    codes=np.concatenate(all_codes)
    if typ=='optimized_pq':
        assert np.array_equal(codes,initial_codes), 'OPQ initialization changed nearest code assignment'
        recon_mse=prefix_error[str(cfg['m'])]/len(full_data)
        assert abs(recon_mse-read(shared/'sanity.json')['full_mse'])<1e-7
    diag=code_diagnostics(codes,cfg['K'],prefixes)
    diag.update(reconstruction={k:v/len(full_data) for k,v in prefix_error.items()},
        random_subset_reconstruction={k:v/len(full_data) for k,v in random_error.items()},
        fit_reconstruction={k:v/len(fit) for k,v in prefix_fit.items()},seconds=time.monotonic()-start,
        full_mse=prefix_error[str(cfg['m'])]/len(full_data),ema_updates=int(model.ema_updates),
        initial_state_sha=sha(dest/'initial.pt'),checkpoint=str(dest/'final.pt'),save_reload_exact=True,
        numerical_status='normal', prefix_mask_space='PQ residual blocks; shared inverse-OPQ decoder bias retained')
    if diag['collision']>0.5 or min(diag['used_codes'])<2 or diag['mean_perplexity']<4:
        diag['numerical_status']='optimization_failure_severe_collapse'
        write(dest/'diagnostics.json',diag)
        raise RuntimeError('Optimization failure: severe codebook/SID collapse; see diagnostics.json')
    np.save(sid_dest/'codes.npy',codes)
    write(sid_dest/'item_to_sid.json',{item:codes[i].tolist() for i,item in enumerate(mapping['id2item'][1:])})
    diag['sid_sha256']=sha(sid_dest/'codes.npy')
    write(sid_dest/'manifest.json',dict(cache_key=cfg['cache_key'],config=cfg,sha256=diag['sid_sha256'],
        row_mapping='row r -> RPG item_id r+1 -> id2item[r+1]',padding=0,eos=cfg['m']*cfg['K']+1,
        token_offset='sid[j] + j*K + 1',label_offset='token[j] - j*K - 1',smoke=smoke))
    write(dest/'diagnostics.json',diag)
    event('tokenizer_complete',run_id=name,**{k:diag[k] for k in ['full_mse','collision','mean_utilization','seconds','sid_sha256']})
    return codes,diag
