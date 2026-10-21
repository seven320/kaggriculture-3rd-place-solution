"""Unit-head position terms (policy.unitpos, 2026-09-26): D = destination tile for the move candidates, E1 = own-unit count
on a tile (0 / 1 / 2+), E2 = self-attention among the 20 own units with a learned relative-position bias.
Every output layer starts at zero, so a freshly attached module leaves all logits unchanged (v92 = same function).
The three model classes below are copied verbatim into the submission bundle's model_tf.py (export_bundle)."""
from pathlib import Path
import argparse
import inspect
import json
import math
import shutil
import types
import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_unitpos(config):
    """{"dest": bool, "ally": bool, "rel_attn": {"heads": int, "seed": int}}; missing keys = off."""
    if not isinstance(config, dict) or set(config) - {'dest', 'ally', 'rel_attn'}:
        raise ValueError(f'unsupported policy.unitpos configuration: {config!r}')
    for key in ('dest', 'ally'):
        if key in config and type(config[key]) is not bool:
            raise ValueError(f'policy.unitpos.{key} must be boolean')
    attn = config.get('rel_attn')
    if attn is not None:
        if not isinstance(attn, dict) or set(attn) - {'heads', 'seed'}:
            raise ValueError(f'unsupported policy.unitpos.rel_attn: {attn!r}')
        if any(type(attn.get(k, 1)) is not int for k in ('heads', 'seed')) or attn.get('heads', 4) < 1:
            raise ValueError('policy.unitpos.rel_attn heads/seed must be integers (heads >= 1)')
    if not (config.get('dest') or config.get('ally') or attn is not None):
        raise ValueError('policy.unitpos enables nothing; remove the key instead')
    return dict(config)


class UnitPosAttn(nn.Module):
    """E2: one self-attention layer over the own-unit tokens h[:,200:220]; score(u,v) += rel_bias[(x_v-x_u+9)*19 + (y_v-y_u+9), head].
    Dead units are masked as keys (a row without live units opens every key, so no NaN). out: Linear(d, 2d), zero."""
    def __init__(self, d, heads=4, seed=2026092602):
        super().__init__()
        if d % heads: raise ValueError('policy.unitpos.rel_attn heads must divide d')
        self.heads = int(heads)
        self.norm = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d)
        self.rel_bias = nn.Parameter(torch.zeros(19 * 19, self.heads))
        self.out = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, u, ux, uy, live):
        B, N, d = u.shape; H = self.heads
        q, k, v = self.qkv(self.norm(u)).view(B, N, 3, H, d // H).permute(2, 0, 3, 1, 4)
        rel = (ux[:, None, :] - ux[:, :, None] + 9) * 19 + (uy[:, None, :] - uy[:, :, None] + 9)   # [b, query u, key v]
        bias = F.embedding(rel, self.rel_bias).permute(0, 3, 1, 2)                                 # (B,H,N,N)
        keys = live | ~live.any(1, keepdim=True)
        bias = bias.masked_fill(~keys[:, None, None, :], float('-inf'))
        # Explicit attention (20 x 20): the same kernels with and without autograd (SDPA picks different CPU/GPU kernels).
        w = (torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(d // H)) + bias.to(q.dtype)).softmax(-1)
        a = torch.matmul(w.to(v.dtype), v)
        return self.out(a.transpose(1, 2).reshape(B, N, d))


class UnitPos(nn.Module):
    """D (dest: Linear(d, d, bias=False), zero; added to the first u_score layer's pre-activation of candidates 1..4 only),
    E1 (ally: Embedding(3, d), zero; added to the footing-tile half of the unit query and to D's destination feature),
    E2 (attn: UnitPosAttn). Candidates 1..4 = NORTH (0,-1), SOUTH (0,+1), EAST (+1,0), WEST (-1,0) (engine rules.rs)."""
    MOVES = ((0, -1), (0, 1), (1, 0), (-1, 0))

    def __init__(self, d, dest=False, ally=False, rel_attn=None):
        super().__init__()
        self.d = int(d)
        with torch.random.fork_rng(devices=[]):   # no draw from the caller's RNG; qkv/norm are reproducible from the seed
            torch.manual_seed(int((rel_attn or {}).get('seed', 2026092602)))
            self.ally = nn.Embedding(3, d) if ally else None
            self.dest = nn.Linear(d, d, bias=False) if dest else None
            self.attn = UnitPosAttn(d, **rel_attn) if rel_attn is not None else None
        for layer in (self.ally, self.dest):
            if layer is not None: nn.init.zeros_(layer.weight)
        self.register_buffer('moves', torch.tensor(self.MOVES), persistent=False)

    @staticmethod
    def geometry(ou):
        live = ou[..., 0] > 0
        ux = ou[..., 1].long().clamp(0, 9); uy = ou[..., 2].long().clamp(0, 9)
        return live, ux, uy, uy * 10 + ux

    @staticmethod
    def count(tiles, idx, live):
        """Live own units standing on each tile index of `tiles` (B, ...), capped at 2."""
        B = idx.shape[0]
        same = (tiles.reshape(B, -1, 1) == idx[:, None, :]) & live[:, None, :]
        return same.sum(-1).clamp(max=2).view(tiles.shape)

    def query_add(self, h, ou):
        """(B,20,2d) added to the unit query before uq_in, or None."""
        live, ux, uy, idx = self.geometry(ou); add = None
        if self.ally is not None:
            add = F.pad(self.ally(self.count(idx, idx, live)), (self.d, 0))
        if self.attn is not None:
            a = self.attn(h[:, 200:220], ux, uy, live)
            add = a if add is None else add.to(a.dtype) + a
        return add

    def dest_hidden(self, h, ou):
        """(B,20,4,d) added to the pre-activation of candidates 1..4 (off-board destination: zero feature)."""
        live, ux, uy, idx = self.geometry(ou)
        B, U = idx.shape
        nx = ux[..., None] + self.moves[:, 0]; ny = uy[..., None] + self.moves[:, 1]
        inside = (nx >= 0) & (nx <= 9) & (ny >= 0) & (ny <= 9)
        t = ny.clamp(0, 9) * 10 + nx.clamp(0, 9)
        f = torch.gather(h[:, :100], 1, t.view(B, U * 4, 1).expand(-1, -1, self.d)).view(B, U, 4, self.d)
        if self.ally is not None: f = f + self.ally(self.count(t, idx, live)).to(f.dtype)
        return self.dest(f * inside.unsqueeze(-1).to(f.dtype))


def add_dest(pre, dh):
    """pre (B,20,NC,d): only candidates 1..4 change."""
    return torch.cat([pre[:, :, :1], pre[:, :, 1:5] + dh.to(pre.dtype), pre[:, :, 5:]], 2)


def attach(sampler, config, settings=None):
    """Attach before optimizer creation / CUDA graph capture; records the config in sampler.cfg."""
    config = validate_unitpos(config)
    settings = settings or {}
    if settings.get('train_scope') is not None:
        raise ValueError('policy.unitpos cannot be combined with train_scope')
    if 'prune' in settings.get('policy', {}) or hasattr(sampler.net, 'prune_variant'):
        raise ValueError('policy.unitpos cannot be combined with policy pruning')
    if 'neighbor' in settings.get('policy', {}) or hasattr(sampler.net, 'nb_proj'):
        raise ValueError('policy.unitpos cannot be combined with policy.neighbor')
    if settings.get('remote_collector') is not None:
        raise ValueError('policy.unitpos does not support remote_collector')
    net = sampler.net
    if getattr(net, 'up', None) is None:
        if getattr(sampler, '_graph', None) is not None:
            raise ValueError('attach policy.unitpos before CUDA graph capture')
        parameter = next(net.parameters())
        net.up = UnitPos(net.d, **config).to(parameter).requires_grad_(any(p.requires_grad for p in net.parameters()))
    sampler.cfg['unitpos'] = config
    return sampler


def fill_missing(net, state_dict, iteration=0):
    """Only an iteration-0 checkpoint may omit the whole unitpos branch (filled with the fresh zero-output module)."""
    if getattr(net, 'up', None) is None or any(k.startswith('up.') for k in state_dict):
        return state_dict
    if iteration != 0:
        raise ValueError('noninitial checkpoint is missing policy.unitpos weights')
    return {**state_dict, **{'up.' + k: v.detach().clone() for k, v in net.up.state_dict().items()}}


def lr_warmup_config(settings):
    """base.policy.unitpos_lr_warmup = {"iters": W, "start": m0}: the unitpos parameters form their own actor param group with
    lr = actor_lr * clip(iteration / W, m0, 1) (user 2026-09-26 18:2x: start the new module's lr small and raise it). None = off."""
    policy = settings.get('policy', {})
    c = policy.get('unitpos_lr_warmup')
    if c is None: return None
    if 'unitpos' not in policy: raise ValueError('policy.unitpos_lr_warmup needs policy.unitpos')
    if not isinstance(c, dict) or set(c) - {'iters', 'start', 'max_lr'} or type(c.get('iters', 16)) is not int or c.get('iters', 16) < 1 \
            or not isinstance(c.get('start', .1), (int, float)) or not 0. < float(c.get('start', .1)) <= 1.:
        raise ValueError(f'unsupported policy.unitpos_lr_warmup: {c!r}')
    out=dict(iters=int(c.get('iters', 16)), start=float(c.get('start', .1)))
    if 'max_lr' in c:
        if isinstance(c['max_lr'],bool) or not isinstance(c['max_lr'],(int,float)) or not math.isfinite(c['max_lr']) or c['max_lr']<=0:
            raise ValueError('unitpos max_lr must be finite and positive')
        out['max_lr']=float(c['max_lr'])
    return out


def lr_multiplier(iteration, c):
    return min(max(iteration / c['iters'], c['start']), 1.)


def optimizer_groups(net):
    """[existing parameters in their original order, unitpos parameters] (the second group carries unitpos=True)."""
    up = {id(p) for p in net.up.parameters()}
    return [dict(params=[p for p in net.parameters() if id(p) not in up]), dict(params=list(net.up.parameters()), unitpos=True)]


def set_group_lr(optimizer, actor_lr, iteration, c):
    """Re-applied at every iteration start (a restored optimizer state would otherwise carry its saved lr). Returns the unitpos lr."""
    groups = optimizer.param_groups
    if len(groups) != 2 or groups[0].get('unitpos') or not groups[1].get('unitpos'):
        raise ValueError('unitpos lr warmup expects [existing, unitpos] actor param groups')
    lr = c.get('max_lr',actor_lr) * lr_multiplier(iteration, c)
    groups[0]['lr'] = actor_lr; groups[1]['lr'] = lr
    return lr


def bundle_model_source(source):
    """Patch the kit's original model_tf.py only; without cfg unitpos the patched code computes exactly the original."""
    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f'unsupported bundle model_tf.py (expected one {old!r})')
        source = source.replace(old, new, 1)
    replace_once('enc="tf", dec_layers=2):', 'enc="tf", dec_layers=2, unitpos=None):')
    replace_once('    def encode(self, tiles, ou, pu, g):',
        '        if unitpos is not None: self.up = UnitPos(d, **validate_unitpos(unitpos))\n\n'
        '    def encode(self, tiles, ou, pu, g):')
    replace_once('        q = self.uq_in(self.unit_query(h, ou))',
        '        query = self.unit_query(h, ou); up = getattr(self, "up", None)\n'
        '        add = up.query_add(h, ou) if up is not None else None\n'
        '        if add is not None: query = query + add.to(query.dtype)\n'
        '        q = self.uq_in(query)')
    replace_once('        s = self.u_score(torch.cat([q.unsqueeze(2).expand(-1, -1, NC, -1), ce], -1)).squeeze(-1)',
        '        x = torch.cat([q.unsqueeze(2).expand(-1, -1, NC, -1), ce], -1)\n'
        '        if up is None or up.dest is None: s = self.u_score(x).squeeze(-1)\n'
        '        else: s = self.u_score[2](F.relu(add_dest(self.u_score[0](x), up.dest_hidden(h, ou)))).squeeze(-1)')
    replace_once('class Policy(nn.Module):',
        '\n\n'.join(inspect.getsource(f) for f in (validate_unitpos, UnitPosAttn, UnitPos, add_dest)) + '\n\nclass Policy(nn.Module):')
    return source


def export_bundle(source, checkpoint, output):
    """Existing bundle dir + .pt (+ .pt.cfg.json) -> self-contained unitpos bundle (main.py/obs_feat.py/stock_target.json unchanged)."""
    source, checkpoint, output = (Path(p).resolve() for p in (source, checkpoint, output))
    if output.exists(): raise FileExistsError(output)
    cfg = json.loads(Path(str(checkpoint) + '.cfg.json').read_text())
    validate_unitpos(cfg.get('unitpos'))
    for name in ('main.py', 'obs_feat.py', 'stock_target.json'):
        if not (source / name).is_file(): raise FileNotFoundError(source / name)
    code = bundle_model_source((source / 'model_tf.py').read_text())
    module = types.ModuleType('_unitpos_bundle_check')
    exec(compile(code, str(output / 'model_tf.py'), 'exec'), module.__dict__)
    net = module.Policy(**cfg)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = state.get('net', state)
    net.load_state_dict(state, strict=True)
    from scripts.kagg_bc_kit.contracts import PolicyConfig, load_contract, save_contract
    contract_cfg = PolicyConfig(cfg, load_contract(checkpoint))
    if contract_cfg.feature_contract['mask_version'] != 0:
        raise ValueError('the v92 submission bundle supports mask0 only')
    output.mkdir(parents=True)
    for name in ('main.py', 'obs_feat.py', 'stock_target.json'):
        shutil.copy2(source / name, output / name)
    (output / 'model_tf.py').write_text(code)
    (output / 'cfg.json').write_text(json.dumps(cfg) + '\n')
    torch.save(state, output / 'policy.pt')
    save_contract(output / 'policy.pt', contract_cfg)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(export_bundle(args.source, args.checkpoint, args.out))


if __name__ == '__main__':
    main()
