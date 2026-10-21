"""Independent total collection size, active games and inference batch size."""
import time
import numpy as np
import torch
from scripts.kagg_ppo.policy import Policy, FEATURES
from scripts.kagg_ppo.core import collect as collect_wave


class StudyPolicy(Policy):
    def __init__(self, weights, device='gpu', inference_batch=32):
        super().__init__(weights, device)
        if inference_batch < 1:
            raise ValueError('inference_batch must be positive')
        self.inference_batch = inference_batch

    def features(self, observations):
        if not observations or '_native_pool' not in observations[0]:
            return super().features(observations)
        from scripts.kagg_bc_kit.contracts import require_mask
        pool = observations[0]['_native_pool']
        require_mask(pool,[o['_native_index'] for o in observations],self.mask_version)
        indexes = [o['_native_index'] for o in observations]
        if not all(o['_native_pool'] is pool for o in observations):
            raise ValueError('one native pool required per inference chunk')
        if indexes == list(range(pool.batch)):
            return super().features(observations)
        if indexes != list(range(indexes[0], indexes[0] + len(indexes))):
            raise ValueError('inference chunk must preserve environment/player order')
        selected = slice(indexes[0], indexes[-1] + 1)
        return {key: torch.as_tensor(pool.features[key][selected], device=self.device) for key in FEATURES}

    def sample(self, observations, greedy=False):
        if len(observations) <= self.inference_batch:
            return super().sample(observations, greedy)
        actions, records, entropies = [], [], []
        for offset in range(0, len(observations), self.inference_batch):
            a, r, e = super().sample(observations[offset:offset + self.inference_batch], greedy)
            actions.extend(a)
            records.append(r)
            entropies.append(e)
        return actions, {key: torch.cat([r[key] for r in records]) for key in records[0]}, torch.cat(entropies)


def collect(policy, games_per_iteration, active_envs, seed_start, episode_steps=720, log=print):
    """Freeze policy for all waves; each game's GAE is complete before concatenation."""
    if games_per_iteration < active_envs or games_per_iteration % active_envs:
        raise ValueError('collection must contain complete equal-size waves')
    if policy.inference_batch > active_envs * 2 or active_envs * 2 % policy.inference_batch:
        raise ValueError('active games must produce complete inference batches')
    started = time.perf_counter()
    data_parts, games, waves = [], [], []
    for offset in range(0, games_per_iteration, active_envs):
        timings = {}
        tick = time.perf_counter()
        data, outcomes = collect_wave(policy, active_envs, episode_steps, seed_start + offset,
                                      log=log, timings=timings)
        assert len(outcomes) == active_envs
        assert len(data['return']) == active_envs * 2 * (episode_steps - 1)
        data_parts.append(data)
        games.extend(outcomes)
        waves.append(dict(game_offset=offset, games=active_envs, seconds=time.perf_counter()-tick, **timings))
    data = data_parts[0] if len(data_parts) == 1 else {
        key: torch.cat([part[key] for part in data_parts]) for key in data_parts[0]}
    assert [r['seed'] for r in games] == list(range(seed_start, seed_start + games_per_iteration))
    return data, games, dict(seconds=time.perf_counter()-started, waves=waves,
                            games_per_iteration=games_per_iteration, active_envs=active_envs,
                            inference_batch=policy.inference_batch,
                            transitions=len(data['return']))
