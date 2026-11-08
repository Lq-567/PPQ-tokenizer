"""Historical prefix estimators plus the frozen equal-weight reconstruction."""
import torch
from torch.nn import functional as F

MODES = ('sample_one', 'all_mean', 'all_sum', 'sample_scaled')


def prefix_reconstruction(decode, q, target, lengths, sampled_mask, mode):
    if mode not in MODES or len(lengths) < 1:
        raise ValueError('Unknown prefix estimator or empty lengths')
    if mode.startswith('sample_'):
        loss = F.mse_loss(decode(q * sampled_mask[..., None]), target)
        return loss * (len(lengths) if mode == 'sample_scaled' else 1)
    losses = []
    for length in lengths:
        mask = torch.arange(q.shape[1], device=q.device) < length
        losses.append(F.mse_loss(decode(q * mask[None, :, None]), target))
    values = torch.stack(losses)
    return values.mean() if mode == 'all_mean' else values.sum()


def equal_weight_objective(full_mse, prefix_sum, commitment, groups, beta):
    """One equal weight 1/G for each proper cumulative prefix and the full view.

    Beta applies to the unscaled commitment term.  No lambda is accepted.
    """
    if groups not in (2,4,8,16):
        raise ValueError('This experiment authorizes G2/G4/G8/G16')
    return (full_mse + prefix_sum) / groups + beta * commitment


def all_sum_objective(full_mse, prefix_sum, commitment, lambda_, beta):
    return (full_mse + lambda_ * prefix_sum) / (1 + lambda_) + beta * commitment
