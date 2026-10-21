"""Isolated, checkpoint-compatible optimizations for the downloaded BC kit."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
import os
# The teammate's BC kit (model_tf.py / obs_feat.py under original/code, bundle/) is not part of this repository yet.
OUT = Path(os.environ.get('KAGG_BC_KIT_DIR', ROOT / 'kagg_bc_kit'))
sys.path.insert(0, str(OUT / 'original/code'))
import model_tf as model
from obs_feat import featurize, to_action

class PriorTokenEnc(model.TokenEnc):
    def forward(self, tiles, ou, pu, g):
        from scripts.kagg_bc_kit.worker_prior import legacy_mask
        self.legacy_worker_legal = legacy_mask(tiles, ou, g)
        return super().forward(tiles, ou, pu, g)

class UnitAdapter(torch.nn.Module):
    """Unit-token adapter (train_scope worker_adapter, 2026-09-25): pre-norm self-attention among the 20 own-unit tokens
    (dead units masked as keys) + FFN, both residual. The attention output projection and the second FFN layer start at
    zero, so the adapter is the identity (bit-exact) until trained. Explicit SDPA: same path with and without grad."""
    def __init__(self, d, heads=8, ff=640, seed=2026092501):
        super().__init__()
        if d % heads: raise ValueError('unit_adapter heads must divide d')
        self.heads = int(heads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.n1 = torch.nn.LayerNorm(d); self.qkv = torch.nn.Linear(d, 3*d); self.out = torch.nn.Linear(d, d)
            self.n2 = torch.nn.LayerNorm(d); self.ff1 = torch.nn.Linear(d, int(ff)); self.ff2 = torch.nn.Linear(int(ff), d)
        for layer in (self.out, self.ff2):
            torch.nn.init.zeros_(layer.weight); torch.nn.init.zeros_(layer.bias)

    def forward(self, u, live):
        B, N, d = u.shape; H = self.heads
        q, k, v = self.qkv(self.n1(u)).view(B, N, 3, H, d//H).permute(2, 0, 3, 1, 4)
        keys = live | ~live.any(1, keepdim=True)   # a row without live units attends to all (its units are dead anyway)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keys[:, None, None, :])
        u = u + self.out(a.transpose(1, 2).reshape(B, N, d))
        return u + self.ff2(F.relu(self.ff1(self.n2(u))))


def validate_nmb(ptr, nmb_pen, nmb_mode):
    """cfg keys of the nmb submissions (main.py pops ptr and builds model_ptr_infer.PolicyPtrInfer(nmb_pen, nmb_mode)).
    Returns None (plain policy) or {'pen', 'mode'}; PolicyPtrInfer's defaults are pen 6.0, mode abs."""
    import math
    if not ptr:
        if nmb_pen is not None or nmb_mode is not None:
            raise ValueError('nmb_pen / nmb_mode need ptr: true')
        return None
    if ptr is not True:
        raise ValueError(f'ptr must be true or false: {ptr!r}')
    pen = 6.0 if nmb_pen is None else float(nmb_pen)
    mode = 'abs' if nmb_mode is None else str(nmb_mode)
    if mode not in ('abs', 'rel') or not math.isfinite(pen) or pen < 0:
        raise ValueError(f'unsupported nmb configuration: pen={nmb_pen!r} mode={nmb_mode!r}')
    return dict(pen=pen, mode=mode)


def nmb_apply(s, legal, legacy, pen_value, nmb):
    """PolicyPtrInfer.unit_logits on an already masked score: the submission's mark l (0 / 1 = mask 0 legal / 3 = mask 2 only)
    is rebuilt as l1 = legal & legacy, xp = legal & ~legacy (legal = mask 2, legacy = mask 0 from the features)."""
    xp = legal & ~legacy
    if nmb['pen'] >= 1e4: return s.masked_fill(xp, -1e4)
    p = pen_value.to(s.dtype)
    if nmb['mode'] == 'abs': return s - p * xp.to(s.dtype)
    m0 = s.masked_fill(~(legal & legacy), -1e4).amax(-1, keepdim=True)
    return torch.where(xp, torch.minimum(s, m0) - p, s)


class FactoredPolicy(model.Policy):
    """W[q,c]+b = Wq*q + Wc*c + b; keep all original state-dict keys."""
    def __init__(self, *args, worker_legacy_penalty=0., legacy_action_rows=False, unit_adapter=None, m12=None, unitpos=None,
                 ptr=False, nmb_pen=None, nmb_mode=None, **kwargs):
        import math
        if not math.isfinite(worker_legacy_penalty) or worker_legacy_penalty < 0:
            raise ValueError('worker_legacy_penalty must be finite and nonnegative')
        if legacy_action_rows and worker_legacy_penalty:
            raise ValueError('legacy_action_rows replaces worker_legacy_penalty; use one of them')
        super().__init__(*args, **kwargs)
        # Split output rows (mask 0 -> mask 2 migration): for actions 5..43, a second row a' = W'[a].q + b'[a]
        # gives the logit in the states where mask 0 forbids a (legal & ~legacy_mask); elsewhere the original
        # row is used unchanged. Absent unless the cfg flag is set (old state dicts and logits unchanged).
        self.legacy_action_rows = bool(legacy_action_rows)
        if self.legacy_action_rows:
            self.legacy_rows = torch.nn.Linear(2*self.d, model.NC - 5)
            torch.nn.init.zeros_(self.legacy_rows.weight); torch.nn.init.zeros_(self.legacy_rows.bias)
            self.enc_tok.__class__ = PriorTokenEnc
        # Optional unit adapter (cfg key unit_adapter = {heads, ff, seed}); absent: no module, keys and logits unchanged.
        self.unit_adapter = UnitAdapter(self.d, **unit_adapter) if unit_adapter else None
        # Optional m12 mask input (mask0 + M1 即植え直し; cfg key m12 = {free_input: true}, scripts/kagg_pfsp3/m12.py): the unit's "tile frees up
        # this hour" flag through a zero-initialized Linear added to the unit query (before uq_in), so logits stay bit-identical
        # until trained. Absent: no module, keys and logits unchanged.
        if m12 and (set(m12) - {'free_input'} or not m12.get('free_input')):
            raise ValueError(f'unsupported m12 configuration: {m12}')
        self.m12_free = torch.nn.Linear(1, 2*self.d) if m12 else None
        if self.m12_free is not None:
            torch.nn.init.zeros_(self.m12_free.weight); torch.nn.init.zeros_(self.m12_free.bias)
        self.worker_legacy_penalty = float(worker_legacy_penalty)
        self._worker_prior_enabled = self.worker_legacy_penalty > 0
        # Runtime coefficient only: keep the team's parameter names/shapes and
        # state_dict unchanged. Stable storage is read by CUDA graph replays.
        self.register_buffer('_worker_prior_value', torch.tensor(
            self.worker_legacy_penalty, dtype=torch.float32), persistent=False)
        if self._worker_prior_enabled:
            self.enc_tok.__class__ = PriorTokenEnc
        # Optional unit-head position terms (cfg key unitpos, scripts/kagg_pfsp3/policy_unitpos.py): zero-output at
        # creation. Absent: no attribute, keys and logits unchanged.
        if unitpos is not None:
            from scripts.kagg_pfsp3.policy_unitpos import UnitPos, validate_unitpos
            self.up = UnitPos(self.d, **validate_unitpos(unitpos))
        # Optional nmb head (cfg keys ptr / nmb_pen / nmb_mode, = the team submissions v96/v97 model_ptr_infer.PolicyPtrInfer,
        # 2026-09-26): the legal input is mask version 2; actions legal only under mask 2 (legal & ~mask 0, mask 0 rebuilt from the
        # features as legacy_action_rows does) get  rel: min(l, max over the unit's mask-0 logits) - pen  /  abs: l - pen  after the
        # final masking. Absent (ptr false, no nmb keys): no attribute change, logits unchanged. Needs a mask-2 contract (checked
        # by contracts.check_nmb where the contract is loaded).
        self.nmb = validate_nmb(ptr, nmb_pen, nmb_mode)
        if self.nmb is not None:
            self.register_buffer('_nmb_pen_value', torch.tensor(self.nmb['pen'], dtype=torch.float32), persistent=False)
            self.enc_tok.__class__ = PriorTokenEnc

    @torch.no_grad()
    def set_worker_legacy_penalty(self, value):
        import math
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError('worker_legacy_penalty must be finite and nonnegative')
        if value and self.legacy_action_rows:
            raise ValueError('legacy_action_rows replaces worker_legacy_penalty; use one of them')
        self.worker_legacy_penalty = value
        self._worker_prior_value.fill_(value)
        if value > 0 and not self._worker_prior_enabled:
            self._worker_prior_enabled = True
            self.enc_tok.__class__ = PriorTokenEnc

    def unit_query(self, h, ou):
        q = super().unit_query(h, ou)
        if self.unit_adapter is None: return q
        return torch.cat([self.unit_adapter(q[..., :self.d], ou[..., 0] > 0), q[..., self.d:]], -1)

    def unit_logits(self, h, ou, legal, free=None):
        query = self.unit_query(h, ou)
        if self.m12_free is not None:
            if free is None: raise ValueError('m12 policy: unit_logits needs the free flag (use scripts.kagg_pfsp3.m12)')
            query = query + self.m12_free(free.unsqueeze(-1).to(query.dtype))
        elif free is not None:
            raise ValueError('free flag given to a policy without m12')
        up = getattr(self, 'up', None)
        if up is not None:
            add = up.query_add(h, ou)
            if add is not None: query = query + add.to(query.dtype)
        q = self.uq_in(query)
        layer = self.u_score[0]
        query = F.linear(q, layer.weight[:, :2*self.d], layer.bias)
        candidate = F.linear(self.cand.weight, layer.weight[:, 2*self.d:])
        if up is None or up.dest is None:
            hidden = F.relu(query.unsqueeze(2) + candidate[None, None])
        else:
            from scripts.kagg_pfsp3.policy_unitpos import add_dest
            hidden = F.relu(add_dest(query.unsqueeze(2) + candidate[None, None], up.dest_hidden(h, ou)))
        score = self.u_score[2](hidden).squeeze(-1)
        if self._worker_prior_enabled:
            from scripts.kagg_bc_kit.worker_prior import apply_prior
            score = apply_prior(score, self.enc_tok.legacy_worker_legal, legal, self._worker_prior_value)
        if self.legacy_action_rows:
            new = legal.bool() & ~self.enc_tok.legacy_worker_legal
            alt = F.pad(self.legacy_rows(q).float(), (5, 0))  # moves/PASS (0..4) have no a' row
            score = torch.where(new, alt, score.float())
        score = score.masked_fill(~legal.bool(), -1e4)
        if self.nmb is not None:
            score = nmb_apply(score, legal.bool(), self.enc_tok.legacy_worker_legal, self._nmb_pen_value, self.nmb)
        return score, q

def reference_agent():
    from scripts.kagg_bc_kit.isolated_agent import load_agent
    return load_agent(OUT / 'bundle/main.py')

class BatchAgent:
    """Full observations -> full actions, including the unchanged v61 repair layer.

    Batch independent games/states; never batch sequential turns of a live game.
    Keep FP32, greedy decoding, seed allocation and market order unchanged.
    """
    def __init__(self, net, repair, mask_version=0):
        self.net = net.eval()
        self.dev = next(net.parameters()).device
        self.repair = repair
        self.mask_version = mask_version

    def feature_arrays(self, observations):
        from scripts.kagg_bc_kit.versioned_features import featurize
        return [np.stack(v) for v in zip(*(featurize(o,self.mask_version) for o in observations))]

    @torch.inference_mode()
    def __call__(self, observations):
        if not observations:
            return []
        arrays = self.feature_arrays(observations)
        tiles, ou, pu, g, legal = [torch.as_tensor(v,device=self.dev) for v in arrays]
        h, pad = self.net.encode(tiles.float(), ou.float(), pu.float(), g.float())
        from scripts.kagg_pfsp3 import m12
        if m12.enabled(self.net):
            ul, q, live_counts = self._m12_units(h, tiles, ou, g, legal, arrays)
            return self._finish(observations, h, pad, q, ul, live_counts)
        logits, q = self.net.unit_logits(h, ou, legal)
        # One transfer, preserving the original CPU argsort tie order.
        scores = logits.cpu()
        ordered = torch.argsort(scores, dim=-1, descending=True).numpy()
        scores = scores.numpy()
        ou_np, g_np = arrays[1], arrays[3]
        ul = np.full((len(observations),20), -1, np.int64)
        live_counts = []
        for b in range(len(observations)):
            live = np.flatnonzero(ou_np[b,:,0] > 0)
            live_counts.append(len(live))
            seeds = np.rint(g_np[b,27:32]*20).astype(int)
            used = np.zeros(5,int)
            for u in live:
                for c in ordered[b,u]:
                    if scores[b,u,c] < -1e3:
                        continue
                    if 15 <= c < 20 and used[c-15] >= seeds[c-15]:
                        continue
                    ul[b,u] = c
                    break
                if 15 <= ul[b,u] < 20:
                    used[ul[b,u]-15] += 1
        return self._finish(observations, h, pad, q, ul, live_counts)

    def _m12_units(self, h, tiles, ou, g, legal, arrays):
        """m12 greedy: same argsort order and seed skip as above, one unit at a time with the M1 exception
        (scripts/kagg_pfsp3/m12.py). Head evaluated with the free flag 0 and 1; each unit uses its own flag."""
        from scripts.kagg_pfsp3 import m12
        opener, same, plant, live_t = m12.context(tiles, ou, g, legal)
        B = len(ou); zeros = torch.zeros(B, 20, device=self.dev)
        lg0, q0 = self.net.unit_logits(h, ou, legal, free=zeros)
        ones = torch.ones(B, 20, device=self.dev)
        lg1, q1 = self.net.unit_logits(h, ou, m12.legal_for(legal, ones.bool(), plant), free=ones)
        s0, s1 = lg0.cpu(), lg1.cpu()
        o0 = torch.argsort(s0, dim=-1, descending=True).numpy(); o1 = torch.argsort(s1, dim=-1, descending=True).numpy()
        s0, s1 = s0.numpy(), s1.numpy()
        opener, same, plant = opener.cpu().numpy(), same.cpu().numpy(), plant.cpu().numpy()
        legal_np = legal.bool().cpu().numpy()
        ou_np, g_np = arrays[1], arrays[3]
        ul = np.full((B,20), -1, np.int64); free = np.zeros((B,20), bool); live_counts = []
        for b in range(B):
            live = np.flatnonzero(ou_np[b,:,0] > 0)
            live_counts.append(len(live))
            seeds = np.rint(g_np[b,27:32]*20).astype(int)
            used = np.zeros(5,int); opened = np.zeros(20,bool); planted = np.zeros(20,bool); emptied = np.zeros(20,bool)
            for u in live:
                f = bool((same[b,u,:u] & opened[:u]).any() and not (same[b,u,:u] & planted[:u]).any())
                allowed = legal_np[b,u] | (f & plant[b])
                order, score = (o1[b,u], s1[b,u]) if f else (o0[b,u], s0[b,u])
                for c in order:
                    if not allowed[c] or score[c] < -1e3:
                        continue
                    if 15 <= c < 20 and used[c-15] >= seeds[c-15]:
                        continue
                    ul[b,u] = c
                    break
                free[b,u] = f
                prior = bool((same[b,u,:u] & emptied[:u]).any())
                opened[u] = opener[b,u] and ul[b,u] == 6 and not prior
                emptied[u] = (opener[b,u] and ul[b,u] == 6) or ul[b,u] == 8
                planted[u] = 15 <= ul[b,u] < 20
                if planted[u]:
                    used[ul[b,u]-15] += 1
        q = torch.where(torch.as_tensor(free,device=self.dev)[...,None], q1, q0)
        return ul, q, live_counts

    def _finish(self, observations, h, pad, q, ul, live_counts):
        uq = self.net.unit_qty(q,torch.as_tensor(ul,device=self.dev)).argmax(-1).cpu().numpy()
        B = len(observations)
        prev = torch.full((B,1),23,dtype=torch.long,device=self.dev)
        done = torch.zeros(B,dtype=torch.bool,device=self.dev)
        choices, hidden = [], []
        for j in range(11):
            if j == 10:
                # Forced STOP has no quantity and needs no decoder call.
                break
            mo,y = self.net.market(h,pad,prev)
            c = mo[:,-1].argmax(-1).masked_fill(done,22)
            choices.append(c)
            hidden.append(y[:,-1])
            done |= c == 22
            if bool(done.all()):
                break
            prev = torch.cat([prev,c[:,None]],1)
        ms_t = torch.stack(choices,1)
        qty = self.net.market_qty(torch.stack(hidden,1),ms_t).argmax(-1).cpu().numpy()
        selected = ms_t.cpu().numpy()
        actions = []
        for b,obs in enumerate(observations):
            ms = np.full(11,22,np.int64)
            mq = np.zeros(11,np.int64)
            for j,c in enumerate(selected[b]):
                ms[j] = c
                if c < 19:
                    mq[j] = qty[b,j]
                if c == 22:
                    break
            a = to_action(ul[b],uq[b],ms,mq,live_counts[b])
            actions.append(self.repair(a,obs))
        return actions
