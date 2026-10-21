"""Finite legacy-action prior for migrating an existing policy to mask version 2.

The old mask is an initial preference, never a hard constraint. Network logits
can overcome the fixed penalty, and all new-mask actions retain gradients.
"""
import torch


def legacy_mask(tiles, ou, g):
    """Reconstruct mask 0 from the existing public/private feature tensors."""
    live = ou[..., 0] > 0
    x, y = ou[..., 1].long(), ou[..., 2].long()
    index = (y * 10 + x).clamp(0, 99)
    tile = tiles[:, 0].gather(1, index[..., None].expand(-1, -1, 13)).float()
    kind = (tile[..., 0] * 5).round().long()
    what = (tile[..., 1] * 12).round().long()
    has = tile[..., 6] > 0
    inv = ou[..., 3:]
    adj = ((x == 4) | (x == 5)) & ((y == 4) | (y == 5))
    free = (g[:, 41] * 100).round() < 100
    seeds = (g[:, 27:32] * 20).round() > 0
    shed = g[:, -12:] > 0
    out = torch.zeros((*live.shape, 44), dtype=torch.bool, device=ou.device)
    out[..., 0] = True
    out[..., 1] = y > 0
    out[..., 2] = y < 9
    out[..., 3] = x < 9
    out[..., 4] = x > 0
    out[..., 14] = adj & (inv.sum(-1) > 0)
    out[..., 20:32] = adj[..., None] & shed[:, None, :]
    place = adj[..., None] & (inv > 0) & free[:, None, None]
    # Tensor construction from host lists is forbidden during CUDA capture.
    structures = torch.stack((kind == 3, kind == 4, kind == 4), -1) & ~has[..., None]
    place[..., 9:] = torch.where(structures, inv[..., 9:] >= 1, place[..., 9:])
    out[..., 32:44] = place
    unlocked = kind != 1
    empty = kind == 0
    crop = kind == 5
    first = torch.where(what < 2, 2, torch.where(what == 2, 8, 10))
    mature = (tile[..., 10] * 30).round() >= first
    out[..., 5] = crop & (tile[..., 2] == 0)
    out[..., 6] = unlocked & ~empty & (kind != 2) & (tile[..., 4] > 0) & ((crop & mature) | (~crop & has))
    out[..., 7] = crop & (inv[..., 8] >= 1)
    out[..., 8] = unlocked & ~empty & ~has
    out[..., 9] = empty
    out[..., 10] = empty
    out[..., 11] = unlocked & has & (tile[..., 3] == 0) & (inv[..., 0] >= 1)
    out[..., 12] = unlocked & has & (tile[..., 5] > 0)
    out[..., 13] = unlocked & has & (tile[..., 7] == 0)
    out[..., 15:20] = empty[..., None] & seeds[:, None, :]
    return out & live[..., None]


def apply_prior(score, legacy, legal, penalty):
    # Keep FP32 logits for small but nonzero exploration probabilities under AMP.
    expanded = legal.bool() & ~legacy
    return score.float() - expanded.to(torch.float32) * penalty
