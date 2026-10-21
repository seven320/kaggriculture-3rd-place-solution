"""Collection where each wave's opponent plan comes from the caller (PFSP league): part self-play, part vs frozen opponents.
Opponents: GreedyGraphOpponent (BC nets, greedy) or SampledGraphOpponent (past learner checkpoints, stochastic), both one
captured CUDA graph at a fixed number of rows. Only learner rows become transitions (layout: wave, time, kept row)."""
import math
import time
import numpy as np
import torch
from scripts.kagg_ppo.native import NativePool
from scripts.kagg_ppo.core import gae
from scripts.kagg_ppo.policy import FEATURES,to_action
from scripts.kagg_fast_study.fast_sample_v2 import GraphSamplePolicy
from scripts.kagg_pool_opp.collect import GreedyGraphOpponent,diagnostic_indices_kept  # noqa: F401 (re-exported)


class SampledGraphOpponent(GraphSamplePolicy):
    """Past learner checkpoint acting stochastically (same sampler as training) on a fixed number of rows."""

    @torch.inference_mode()
    def act(self,pool,rows):
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


def collect_planned(policy,opponents,games_per_iteration,active_envs,seed_start,plan_fn,episode_steps=720,log=print,gamma=1.,lam=1.):
    """plan_fn(wave) -> list per game: None (self-play) or (opponent name, learner player). opponents: name -> object with act()."""
    if games_per_iteration < active_envs or games_per_iteration % active_envs:
        raise ValueError('collection must contain complete equal-size waves')
    if policy.inference_batch > active_envs*2 or active_envs*2 % policy.inference_batch:
        raise ValueError('active games must produce complete inference batches')
    started = time.perf_counter();parts,games,waves = [],[],[]
    for w,offset in enumerate(range(0,games_per_iteration,active_envs)):
        plan = plan_fn(w);assert len(plan) == active_envs
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
                if (t+1) % 120 == 0: log(f'rollout {t+1}/{episode_steps-1}, games={active_envs}, opponents={sorted(by_name)}')
            assert pool.done.all()
        finally:
            pool.close()
        tick = time.perf_counter()
        data = {k:torch.stack([r[k] for r in records]) for k in records[0]}
        advantages,returns = gae(data['reward'],data['old_value'],data['done'],gamma=gamma,lam=lam)
        data['advantage'] = advantages;data['return'] = returns
        keep_t = torch.tensor(keep,dtype=torch.long)
        parts.append({k:v[:,keep_t].flatten(0,1) for k,v in data.items()});games.extend(rows_out)
        waves.append(dict(game_offset=offset,games=active_envs,kept_rows=len(keep),plan=[None if p is None else list(p) for p in plan],
                          seconds=time.perf_counter()-tick+init_s+sample_s+opp_s+env_s,environment_init_seconds=init_s,sampling_seconds=sample_s,
                          opponent_seconds=opp_s,environment_step_seconds=env_s,buffer_and_gae_seconds=time.perf_counter()-tick))
    data = parts[0] if len(parts) == 1 else {k:torch.cat([p[k] for p in parts]) for k in parts[0]}
    assert [r['seed'] for r in games] == list(range(seed_start,seed_start+games_per_iteration))
    return data,games,dict(seconds=time.perf_counter()-started,waves=waves,games_per_iteration=games_per_iteration,active_envs=active_envs,
                           inference_batch=policy.inference_batch,transitions=len(data['return']),kept_rows_per_wave=waves[0]['kept_rows'],
                           pool_games=sum(1 for g in games if g['opponent']),gamma=gamma,gae_lambda=lam)
