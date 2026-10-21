"""Frozen NN opponent evaluation through the validated Rust in-process arrays."""
from contextlib import ExitStack
import time

from scripts.kagg_bc_kit.inference import Inference
from scripts.kagg_ppo.native import NativePool
from scripts.kagg_ppo_eval_native.batch import NativeBatchAgent


class NativeInference(Inference):
    def __init__(self, device, batch_size, weights):
        super().__init__(device, batch_size, weights)
        self._batch = NativeBatchAgent(self._batch.net, self._batch.repair, self.mask_version)


def matches(weights, opponent_weights, seeds, device='gpu', progress=None):
    if opponent_weights is None:
        raise ValueError('Native evaluation currently requires a fixed NN opponent')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Nonempty distinct seeds required')
    learner = NativeInference(device, 32, weights)
    frozen = NativeInference(device, 32, opponent_weights)
    start = time.perf_counter()
    # The FFI initializes consecutive seeds. Split only at gaps, and preserve
    # the historical NN batch order: (seed0,p0),(seed0,p1),(seed1,p0),...
    groups = []
    for seed in seeds:
        if groups and seed == groups[-1][-1] + 1:
            groups[-1].append(seed)
        else:
            groups.append([seed])
    traces = [[] for _ in range(2*len(seeds))]
    sample_seconds = step_seconds = 0.
    with ExitStack() as stack:
        pools, mapping = [], {}
        for group in groups:
            for player in (0, 1):
                pool = stack.enter_context(NativePool(len(group), 720, group[0], mask_versions=[learner.mask_version if p==player else frozen.mask_version for _ in group for p in (0,1)]))
                pools.append(pool)
                for index, seed in enumerate(group):
                    mapping[seed, player] = (pool, index, player)
        cases = [mapping[seed, p] for seed in seeds for p in (0, 1)]
        init_seconds = time.perf_counter()-start
        for t in range(719):
            tick = time.perf_counter()
            own = learner.act_batch([pool.observations[2*i+p] for pool, i, p in cases])
            other = frozen.act_batch([pool.observations[2*i+1-p] for pool, i, p in cases])
            sample_seconds += time.perf_counter()-tick
            tick = time.perf_counter()
            actions = {pool: [None]*pool.batch for pool in pools}
            for j, (pool, i, p) in enumerate(cases):
                both = [None, None]
                both[p], both[1-p] = own[j], other[j]
                actions[pool][2*i:2*i+2] = both
                traces[j].append(both)
            for pool in pools:
                pool.step(actions[pool])
            step_seconds += time.perf_counter()-tick
            if progress is not None and (t+1) % 120 == 0:
                progress(dict(step=t+1, steps=719, games=len(cases)))
        rows = []
        for (seed, p), (pool, i, _) in zip([(s, p) for s in seeds for p in (0, 1)], cases):
            assert pool.done[2*i:2*i+2].all() and int(pool.meta[2*i, 0]) == 719
            money = pool.meta[2*i, 4:6]
            own_money, opponent_money = float(money[p]), float(money[1-p])
            margin = own_money-opponent_money
            rows.append(dict(seed=seed, player=p, own_money=own_money, opponent_money=opponent_money,
                             margin=margin, result='win' if margin > 0 else 'loss' if margin < 0 else 'draw',
                             statuses=['DONE', 'DONE']))
    return dict(seconds=time.perf_counter()-start, outcomes=rows, backend='rust_native_arrays',
                wins=sum(r['result'] == 'win' for r in rows), draws=sum(r['result'] == 'draw' for r in rows),
                losses=sum(r['result'] == 'loss' for r in rows), n=len(rows),
                timings=dict(environment_init_seconds=init_seconds,
                             feature_transfer_inference_decode_seconds=sample_seconds,
                             action_pack_environment_step_features_seconds=step_seconds)), traces
