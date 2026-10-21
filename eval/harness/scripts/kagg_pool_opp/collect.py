"""Collection with mixed opponents: part of every wave is self-play (both players = learner), the rest is played
against frozen BC-pool nets (greedy, CUDA-graph replay, fp16). Only learner-controlled rows become transitions.

Layout: a wave is `active_envs` games; rows 2i (player 0) and 2i+1 (player 1). The learner samples all rows with its
usual CUDA-graph batch (its actions on opponent rows are discarded, which keeps the captured shape), then each opponent
net acts on its own rows. GAE runs per row as before; opponent rows are dropped before flattening, so the transition
order is (wave, time, kept row) with a constant number of kept rows per wave.
"""
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from scripts.kagg_ppo.native import NativePool
from scripts.kagg_ppo.core import gae
from scripts.kagg_ppo.policy import FEATURES,to_action
from scripts.kagg_fast_study.fast_sample_v2 import GraphSamplePolicy
try:
    from scripts.kagg_ppo.runtime import categorical_fast as categorical
except ImportError:
    from scripts.kagg_ppo.policy import categorical


class GreedyGraphOpponent(GraphSamplePolicy):
    """Frozen BC net acting greedily on a fixed number of rows per step through one captured CUDA graph."""

    def _core(self,T):
        net = self.net;B = T['g'].shape[0]
        from scripts.kagg_pfsp3 import m12
        amp = lambda: torch.autocast('cuda',dtype=self.autocast_dtype,enabled=self.autocast_bf16,cache_enabled=False)
        with amp():
            h,pad = net.encode(T['tiles'].float(),T['ou'],T['pu'],T['g'])
            if not m12.enabled(net): lg,q = net.unit_logits(h,T['ou'],T['legal'])
        live = T['ou'][...,0] > 0
        if m12.enabled(net):  # m12 mask (scripts/kagg_pfsp3/m12.py), greedy
            ul,_,_,q,_ = m12.sample_units(net,h,T,categorical,greedy=True,amp=amp)
        else:
            lg = lg.float()
        seeds = (T['g'][:,27:32]*20).round().long();used = torch.zeros_like(seeds)
        if not m12.enabled(net): ul = torch.zeros(B,20,dtype=torch.long,device=self.device)
        for u in range(0 if m12.enabled(net) else 20):
            mask = T['legal'][:,u].bool().clone();mask[:,0] |= ~live[:,u]
            mask[:,15:20] &= used < seeds
            c,_,_ = categorical(lg[:,u],mask=mask,greedy=True)
            ul[:,u] = c
            plant = (c >= 15)&(c < 20)&live[:,u]
            used += F.one_hot((c-15).clamp(0,4),5)*plant[:,None]
        with torch.autocast('cuda',dtype=self.autocast_dtype,enabled=self.autocast_bf16,cache_enabled=False):
            uq_logits = net.unit_qty(q,ul)
        uq,_,_ = categorical(uq_logits.float(),greedy=True)
        prev = torch.full((B,1),23,dtype=torch.long,device=self.device)
        ms = torch.full((B,10),22,dtype=torch.long,device=self.device)
        mq = torch.zeros_like(ms);active = torch.ones(B,dtype=torch.bool,device=self.device)
        net._incremental_cache = None
        for j in range(10):
            with torch.autocast('cuda',dtype=self.autocast_dtype,enabled=self.autocast_bf16,cache_enabled=False):
                logits,y = net.market(h,pad,prev)
            c,_,_ = categorical(logits[:,-1].float(),greedy=True)
            c = c.masked_fill(~active,22);ms[:,j] = c
            with torch.autocast('cuda',dtype=self.autocast_dtype,enabled=self.autocast_bf16,cache_enabled=False):
                q_logits = net.market_qty(y[:,-1:],c[:,None])[:,0]
            qty,_,_ = categorical(q_logits.float(),greedy=True)
            mq[:,j] = qty
            active &= c != 22
            prev = torch.cat([prev,c[:,None]],1)
        return dict(ul=ul,uq=uq,ms=ms,mq=mq,counts=live.sum(1))

    @torch.inference_mode()
    def act(self,pool,rows):
        """Greedy actions for the given rows of one native pool (len(rows) must equal the captured batch)."""
        from scripts.kagg_bc_kit.contracts import require_mask
        require_mask(pool,rows,self.mask_version)
        assert len(rows) == self.inference_batch,(len(rows),self.inference_batch)
        index = np.asarray(rows)
        T = {k:torch.as_tensor(pool.features[k][index],device=self.device) for k in FEATURES}
        if self._graph is None:
            self._capture(T)
        for k,v in T.items():
            self._static_in[k].copy_(v)
        self._graph.replay()
        out = self._static_out
        a = {k:out[k].cpu().numpy() for k in ('ul','uq','ms','mq')};counts = out['counts'].cpu().tolist()
        return [self.ref._pace(to_action(a['ul'][b],a['uq'][b],a['ms'][b],a['mq'][b],counts[b]),pool.observations[r]) for b,r in enumerate(rows)]


def wave_plan(spec,names,active_envs,wave_index):
    """Per game of one wave: None (self-play) or (opponent name, learner player). Deterministic in wave_index.

    spec: pool_fraction (share of games against the pool), opponents_per_wave (distinct nets per wave; rotate through
    `names` across waves so every net is used equally over time). The learner side alternates 0/1 within a wave.
    """
    pool_games = int(round(active_envs*spec['pool_fraction']));k = int(spec['opponents_per_wave'])
    if pool_games % k:
        raise ValueError('pool games per wave must divide by opponents_per_wave')
    per = pool_games//k;plan = [None]*(active_envs-pool_games)
    for j in range(k):
        name = names[(wave_index*k+j) % len(names)]
        plan += [(name,(len(plan)+g) % 2) for g in range(per)]
    return plan,per


def collect_mixed(policy,opponents,games_per_iteration,active_envs,seed_start,spec,wave_index_base,episode_steps=720,log=print):
    """opponents: dict name -> GreedyGraphOpponent (inference_batch == games per opponent per wave)."""
    if games_per_iteration < active_envs or games_per_iteration % active_envs:
        raise ValueError('collection must contain complete equal-size waves')
    if policy.inference_batch > active_envs*2 or active_envs*2 % policy.inference_batch:
        raise ValueError('active games must produce complete inference batches')
    names = sorted(opponents);started = time.perf_counter();parts,games,waves = [],[],[]
    for w,offset in enumerate(range(0,games_per_iteration,active_envs)):
        plan,per = wave_plan(spec,names,active_envs,wave_index_base+w)
        keep = [r for i,p in enumerate(plan) for r in ((2*i,2*i+1) if p is None else (2*i+p[1],))]
        by_name = {}
        for i,p in enumerate(plan):
            if p is not None: by_name.setdefault(p[0],[]).append(2*i+1-p[1])
        for name,rows in by_name.items():
            assert opponents[name].inference_batch == len(rows),(name,len(rows),opponents[name].inference_batch)
        tick = time.perf_counter();pool = NativePool(active_envs,episode_steps,seed_start+offset)
        init_s = time.perf_counter()-tick;sample_s = opp_s = env_s = 0.;records,rows_out = [],[]
        try:
            for t in range(episode_steps-1):
                tick = time.perf_counter()
                actions,record,_ = policy.sample(pool.observations)
                sample_s += time.perf_counter()-tick;tick = time.perf_counter()
                for name,rows in by_name.items():
                    for r,a in zip(rows,opponents[name].act(pool,rows)): actions[r] = a
                opp_s += time.perf_counter()-tick;tick = time.perf_counter()
                pool.step(actions)
                rewards = torch.zeros(2*active_envs);dones = torch.from_numpy(pool.done.copy()).bool()
                for i in range(active_envs):
                    if dones[2*i]:
                        money = pool.meta[2*i,4:6].tolist()
                        reward = math.log1p(max(0,money[0]))-math.log1p(max(0,money[1]))
                        rewards[2*i:2*i+2] = torch.tensor([reward,-reward]);dones[2*i:2*i+2] = True
                        p = plan[i];rows_out.append(dict(seed=seed_start+offset+i,money=money,reward_player0=reward,
                                                        opponent=None if p is None else p[0],learner_player=None if p is None else p[1]))
                env_s += time.perf_counter()-tick
                record['reward'] = rewards;record['done'] = dones;records.append(record)
                if (t+1) % 120 == 0: log(f'rollout {t+1}/{episode_steps-1}, games={active_envs}, pool games={len(by_name) and sum(map(len,by_name.values()))}')
            assert pool.done.all()
        finally:
            pool.close()
        tick = time.perf_counter()
        data = {k:torch.stack([r[k] for r in records]) for k in records[0]}
        advantages,returns = gae(data['reward'],data['old_value'],data['done'])
        data['advantage'] = advantages;data['return'] = returns
        keep_t = torch.tensor(keep,dtype=torch.long)
        flat = {k:v[:,keep_t].flatten(0,1) for k,v in data.items()}  # (time, kept row) -> transition
        parts.append(flat);games.extend(rows_out)
        waves.append(dict(game_offset=offset,games=active_envs,kept_rows=len(keep),plan=[None if p is None else list(p) for p in plan],
                          seconds=time.perf_counter()-tick+init_s+sample_s+opp_s+env_s,environment_init_seconds=init_s,sampling_seconds=sample_s,
                          opponent_seconds=opp_s,environment_step_seconds=env_s,buffer_and_gae_seconds=time.perf_counter()-tick))
    data = parts[0] if len(parts) == 1 else {k:torch.cat([p[k] for p in parts]) for k in parts[0]}
    assert [r['seed'] for r in games] == list(range(seed_start,seed_start+games_per_iteration))
    return data,games,dict(seconds=time.perf_counter()-started,waves=waves,games_per_iteration=games_per_iteration,active_envs=active_envs,
                           inference_batch=policy.inference_batch,transitions=len(data['return']),kept_rows_per_wave=waves[0]['kept_rows'],
                           pool_games=sum(1 for g in games if g['opponent']),spec=spec)


def diagnostic_indices_kept(waves,kept_rows,episode_steps=720):
    """Held-out diagnostic positions for the (wave, time, kept row) layout: two rows (first/last kept) at midday of each day."""
    positions = []
    for wave in range(waves):
        for day in range(30):
            t = min(day*24+12,episode_steps-2)
            for j in sorted({0,kept_rows-1}):
                positions.append(wave*(episode_steps-1)*kept_rows+t*kept_rows+j)
    return torch.tensor(sorted(set(positions)),dtype=torch.long)
