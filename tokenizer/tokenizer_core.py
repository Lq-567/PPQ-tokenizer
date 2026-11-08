"""复用历史 LearnablePQ 的结构、OPQ 初始化、STE 与单次 EMA 更新规则。"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class LearnablePQ(nn.Module):
    def __init__(self, initialization, fit_codes, cfg):
        super().__init__()
        self.m, self.k, self.dim = cfg["m"], cfg["K"], cfg["latent_dim"]
        d = self.dim; tc = cfg["tokenizer"]
        self.encoder_linear = nn.Linear(d, d)
        architecture = cfg.get("network_architecture", "nonlinear")
        if architecture not in {"nonlinear", "affine"}:
            raise ValueError("network_architecture must be nonlinear or affine")
        self.encoder_residual = ResidualMLP(d, tc["hidden_dim"]) if architecture == "nonlinear" else nn.Identity()
        self.decoder_residual = ResidualMLP(d, tc["hidden_dim"]) if architecture == "nonlinear" else nn.Identity()
        self.decoder_linear = nn.Linear(d, d)
        a = torch.from_numpy(initialization["rotation"])
        b = torch.from_numpy(initialization["bias"])
        center = torch.from_numpy(initialization["center"])
        with torch.no_grad():
            self.encoder_linear.weight.copy_(a)
            self.encoder_linear.bias.copy_(b - center)
            self.decoder_linear.weight.copy_(a.T)
            self.decoder_linear.bias.copy_((center - b) @ a)
        cb = torch.from_numpy(initialization["centroids"].copy())
        if tuple(cb.shape) != (self.m, self.k, self.dim // self.m):
            raise ValueError("Historical OPQ codebook shape differs from requested bitrate")
        counts = torch.stack([torch.bincount(torch.from_numpy(fit_codes[:, j].astype(np.int64)),
            minlength=self.k) for j in range(self.m)]).float()
        counts = (counts * tc["batch_size"] / len(fit_codes)).clamp_min(0.1)
        self.register_buffer("codebooks", cb)
        self.register_buffer("ema_count", counts)
        self.register_buffer("ema_sum", cb * counts[..., None])
        self.register_buffer("ema_updates", torch.tensor(0, dtype=torch.long))
        self.decay = tc["ema_decay"]
        # 所有模式都实例化同样的头，参数预算一致；非任务组不会使用这个头。
        self.task_decoder = nn.Linear(d, cfg["teacher_dim"])
        nn.init.zeros_(self.task_decoder.bias)

    def encode(self, x):
        return self.encoder_residual(self.encoder_linear(x)).reshape(-1, self.m, self.dim // self.m)

    def decode(self, q):
        return self.decoder_linear(self.decoder_residual(q.reshape(-1, self.dim)))

    def task_decode(self, q):
        return self.task_decoder(q.reshape(-1, self.dim))

    def quantize(self, z):
        zt = z.transpose(0, 1)
        dist = (zt.square().sum(-1, keepdim=True) + self.codebooks.square().sum(-1)[:, None, :]
                - 2 * torch.bmm(zt, self.codebooks.transpose(1, 2)))
        codes = dist.argmin(-1).T
        q = self.codebooks[torch.arange(self.m, device=z.device)[None, :], codes]
        return q, codes

    @torch.no_grad()
    def update_ema(self, z, codes):
        assignments = F.one_hot(codes, self.k).float().transpose(0, 1)
        counts = assignments.sum(1)
        sums = torch.bmm(assignments.transpose(1, 2), z.detach().transpose(0, 1))
        self.ema_count.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        valid = self.ema_count > 1e-6
        self.codebooks.copy_(torch.where(valid[..., None], self.ema_sum /
            self.ema_count.clamp_min(1e-6)[..., None], self.codebooks))
        self.ema_updates.add_(1)

    def forward(self, x):
        z = self.encode(x)
        q, codes = self.quantize(z)
        straight = z + (q - z).detach()
        return z, straight, codes


def make_mask(n, m, retained, mode, position_rng, device, hybrid_probability=0.5, groups=4):
    if mode == "group_random":
        if groups < 2 or m % groups:
            raise ValueError("Whole-group masking requires groups>=2 dividing the complete SID length")
        width = m // groups
        if torch.any(retained % width) or torch.any(retained < 0) or torch.any(retained > m):
            raise ValueError("Retained lengths must be whole-group counts within the full SID")
        ranks = torch.rand(n, groups, generator=position_rng, device=device).argsort(1).argsort(1)
        return (ranks < (retained // width)[:, None]).repeat_interleave(width, dim=1).float()
    if mode not in {"ordered", "random", "hybrid"}:
        raise ValueError("Unknown reconstruction mask")
    positions = torch.arange(m, device=device)[None, :].expand(n, -1)
    if mode in {"random", "hybrid"}:
        shuffled = torch.rand(n, m, generator=position_rng, device=device).argsort(1).argsort(1)
        if mode == "random":
            positions = shuffled
        else:
            use_random = torch.rand(n, 1, generator=position_rng, device=device) < hybrid_probability
            positions = torch.where(use_random, shuffled, positions)
    return (positions < retained[:, None]).float()
