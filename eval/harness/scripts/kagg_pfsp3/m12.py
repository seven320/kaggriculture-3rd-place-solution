"""m12 mask = mask 0 + M1 「即植え直し」 (user 2026-09-25): a later unit PLANTs, in the same hour, the tile an earlier unit empties by harvesting.

M1: units act in order (farmer, then hands). When an earlier unit v on tile c chooses HARVEST, c holds a non-ongoing crop
(WHEAT / CARROT / MELON) and mask 0 allows v's HARVEST (yield > 0, age >= first_yield_day), the engine turns c into an empty
tile. A later unit u on c may then PLANT any crop with seeds at the start of the hour (the sampler's seed cap still applies),
unless a unit between them already planted on c. Nothing else can refill an empty tile in the same hour (BUILD_* need a tile
that is empty at the start, which c is not), so an opened PLANT always takes effect.
M2 (the market purchase picked up in the same hour) is not part of m12: the engine processes all unit actions before the
market (kaggriculture.py interpreter / rules.rs step_inner), so the purchase is never in the shed in time.

Policy input: the "tile frees up this hour" flag of each unit (FactoredPolicy.m12_free, a zero-initialized Linear added to
the unit query). Sampling evaluates the unit head once with flag 0 and once with flag 1 and picks per unit inside the
sequential loop; scoring recomputes the flags and the mask from the stored unit actions (no extra trajectory columns).
All functions use fixed shapes and no host reads (CUDA-graph and torch.compile safe).
"""
import torch
import torch.nn.functional as F

NU = 20


def enabled(net):
    return getattr(net, 'm12_free', None) is not None


def context(tiles, ou, g, legal):
    """Turn-start tensors -> (opener (B,20), same (B,20,20) [u,v], plant (B,44) PLANT bits of crops with seeds, live (B,20))."""
    live = ou[..., 0] > 0
    x = ou[..., 1].long().clamp(0, 9); y = ou[..., 2].long().clamp(0, 9); idx = y*10+x
    t = tiles[:, 0].gather(1, idx[..., None].expand(-1, -1, tiles.shape[-1])).float()
    kind = (t[..., 0]*5).round().long(); what = (t[..., 1]*12).round().long()
    crop = live & (kind == 5)
    opener = crop & ((what == 0) | (what == 1) | (what == 4)) & legal[..., 6].bool()
    same = (idx[:, :, None] == idx[:, None, :]) & crop[:, :, None] & live[:, None, :]
    seeds = (g[:, 27:32].float()*20).round() > 0
    plant = F.pad(seeds.to(torch.uint8), (15, 44-20)).bool()
    return opener, same, plant, live



def free_flags(opener, same, live, ul):
    """Vectorized flags from full unit actions (only earlier units matter). A HARVEST frees the tile only when it succeeds:
    the unit's tile passes the turn-start check (opener) and no earlier unit on the same tile already emptied it
    (a successful HARVEST or a DIG), so the first effective HARVEST on the tile is the one that opens it."""
    tri = torch.ones(NU, NU, dtype=torch.bool, device=ul.device).tril(-1)   # [u,v]: v < u
    rel = same & tri
    emptied = (opener & (ul == 6)) | (live & (ul == 8))
    prior = (rel & emptied[:, None, :]).any(-1)                              # tile already emptied before this unit
    success = opener & (ul == 6) & ~prior
    planted = live & (ul >= 15) & (ul < 20)
    return (rel & success[:, None, :]).any(-1) & ~(rel & planted[:, None, :]).any(-1)


def legal_for(legal, free, plant):
    return legal.bool() | (free[..., None] & plant[:, None, :])


def batch_units(net, h, T, ul):
    """Scoring path: unit logits, query and the m12 legal mask for stored actions. Returns (lg, q, legal, free)."""
    opener, same, plant, live = context(T['tiles'], T['ou'], T['g'], T['legal'])
    free = free_flags(opener, same, live, ul.long())
    legal = legal_for(T['legal'], free, plant)
    lg, q = net.unit_logits(h, T['ou'], legal, free=free.float())
    return lg, q, legal, free


def sample_units(net, h, T, categorical, *, greedy=False, seed_cap=True, amp=None):
    """Sequential unit sampling under m12. Returns ul, per-unit log-prob and entropy (not multiplied by live), q, free."""
    import contextlib
    amp = amp or contextlib.nullcontext
    opener, same, plant, live = context(T['tiles'], T['ou'], T['g'], T['legal'])
    B = live.shape[0]; dev = live.device
    zeros = torch.zeros(B, NU, device=dev); ones = torch.ones(B, NU, device=dev)
    with amp():
        lg0, q0 = net.unit_logits(h, T['ou'], T['legal'], free=zeros)
        lg1, q1 = net.unit_logits(h, T['ou'], legal_for(T['legal'], ones.bool(), plant), free=ones)
    lg0 = lg0.float(); lg1 = lg1.float()
    seeds = (T['g'][:, 27:32].float()*20).round().long(); used = torch.zeros_like(seeds)
    ul = torch.zeros(B, NU, dtype=torch.long, device=dev)
    lp_u = torch.zeros(B, NU, device=dev); en_u = torch.zeros(B, NU, device=dev)
    opened = torch.zeros(B, NU, dtype=torch.bool, device=dev); planted = torch.zeros_like(opened)
    emptied = torch.zeros_like(opened); free = torch.zeros_like(opened)
    legal = T['legal'].bool()
    for u in range(NU):
        if u == 0:
            f = torch.zeros(B, dtype=torch.bool, device=dev); prior = f
        else:
            s = same[:, u, :u]
            f = (s & opened[:, :u]).any(1) & ~(s & planted[:, :u]).any(1)
            prior = (s & emptied[:, :u]).any(1)
        mask = legal[:, u] | (f[:, None] & plant)
        mask[:, 0] |= ~live[:, u]
        if seed_cap: mask[:, 15:20] &= used < seeds
        c, lp, en = categorical(torch.where(f[:, None], lg1[:, u], lg0[:, u]), mask=mask, greedy=greedy)
        ul[:, u] = c; lp_u[:, u] = lp; en_u[:, u] = en; free[:, u] = f
        opened[:, u] = opener[:, u] & (c == 6) & ~prior       # the HARVEST that actually empties the tile
        emptied[:, u] = (opener[:, u] & (c == 6)) | (live[:, u] & (c == 8))
        p = (c >= 15) & (c < 20) & live[:, u]
        planted[:, u] = p
        used += F.one_hot((c-15).clamp(0, 4), 5)*p[:, None]
    q = torch.where(free[..., None], q1, q0)
    return ul, lp_u, en_u, q, free


def hour_window(g, hours):
    hour = (g[:, 35].float()*24).round().long()
    return (hour >= int(hours[0])) & (hour <= int(hours[1]))


def aux_terms(lsu, T, ul, hours):
    """m12_aux inputs on a scored batch: lsu (B,20,44) m12-masked log-softmax (seed cap included).
    Returns per-unit -log sum_{a in opened PLANTs} pi(a|s) and the eligible (row, unit) mask."""
    opener, same, plant, live = context(T['tiles'], T['ou'], T['g'], T['legal'])
    free = free_flags(opener, same, live, ul.long())
    actions = plant[:, None, 15:20] & (lsu[..., 15:20] > -1e3)
    units = free & live & actions.any(-1) & hour_window(T['g'], hours)[:, None]
    value = -torch.logsumexp(lsu[..., 15:20].float().masked_fill(~actions, -1e9), -1)
    return torch.where(units, value, torch.zeros_like(value)), units


def auxiliary(lsu, T, ul, hours, coefficient):
    per, units = aux_terms(lsu, T, ul, hours)
    count = units.sum()
    raw = per.sum()/count.clamp_min(1)
    return raw, raw*coefficient, count


@torch.no_grad()
def rollout_stats(data, stride, hours):
    """Learner rows of one collection: opened units (all hours / aux hours), PLANT selections on them, and selections the
    engine executed (next row of the same game/player: the tile is a PLANT of the chosen crop planted this hour)."""
    names = ('opened_units', 'opened_units_aux_hours', 'selections', 'selections_aux_hours', 'effective', 'effective_aux_hours',
             'weeded_at_day_end', 'next_row_missing')
    counts = dict.fromkeys(names, 0)
    N = len(data['g'])
    for start in range(0, N, 2048):
        stop = min(start+2048, N); ids = torch.arange(start, stop, device=data['g'].device)
        tiles, ou, g, legal = (data[k][ids] for k in ('tiles', 'ou', 'g', 'legal'))
        ul = data['ul'][ids].long()
        opener, same, plant, live = context(tiles, ou, g, legal)
        rows = data['learner'][ids].bool()
        free = free_flags(opener, same, live, ul) & live & plant.any(-1)[:, None] & rows[:, None]
        chosen = free & (ul >= 15) & (ul < 20)
        window = hour_window(g, hours)[:, None]
        nxt = (ids+stride).clamp_max(N-1)
        valid = (ids+stride < N) & (data['seed'][ids] == data['seed'][nxt]) & (data['player'][ids] == data['player'][nxt]) \
            & (data['step'][nxt] == data['step'][ids]+1)
        x = ou[..., 1].long().clamp(0, 9); y = ou[..., 2].long().clamp(0, 9); idx = y*10+x
        after = data['tiles'][nxt][:, 0].gather(1, idx[..., None].expand(-1, -1, tiles.shape[-1])).float()
        day_now = (g[:, 36].float()*30).round(); day_next = (data['g'][nxt][:, 36].float()*30).round()
        age = (after[..., 10]*30).round()
        done = (after[..., 0]*5).round().long().eq(5) & (after[..., 1]*12).round().long().eq(ul-15) \
            & age.eq((day_next-day_now)[:, None]) & valid[:, None]
        # hour 23: a plant not watered on its planting day turns into a WEED at the end of the day (official _new_plant
        # consecutive_unwatered = 1 -> _daily_refresh_plants), so the next row shows a WEED although the PLANT was executed
        weeded = (after[..., 0]*5).round().long().eq(2) & ((g[:, 35].float()*24).round().long() == 23)[:, None] & valid[:, None]
        values = [free, free & window, chosen, chosen & window, chosen & done, chosen & done & window, chosen & weeded, chosen & ~valid[:, None]]
        for k, v in zip(names, torch.stack([v.sum() for v in values]).cpu().tolist()): counts[k] += int(v)
    counts['hours'] = [int(hours[0]), int(hours[1])]
    return counts
