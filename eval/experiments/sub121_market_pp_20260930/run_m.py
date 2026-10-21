"""C1'(W3+W5+W6+W9+WT、sub121_c1c2_20260930 の C1p 腕)の上に、市場・雇用・土地の注文だけを変える後付け処理(market_pp.py)を足した腕。
条件は sub121_c1c2_20260930/protocol.json と同じ(seed・相手 7 本・両 player・job 構成・Rust・fp32・TF32 無効・学習側 64 詰め・相手 32)。
掛ける順序: 網 → 提出物の _pace(= _liq) → waste_replace2(W3/W5/W6 → W9 → WT、run_c.Top) → market_pp(本ファイルの MTop)。
対照: games/C1p(sub121_c1c2)と games/base(sub121_w9t)を再利用(回し直さない)。
使い方: run_m.py smoke <腕>  → smoke/<腕>/(seed 173600000〜03 × 7 相手 × 両 player、全 720 手)
        run_m.py arm <腕>    → games/<腕>/
整合: 初めて行動が変わる手(W3〜WT と market_pp の両方)までは元 sub121 の特徴+行動 hash と一致、変わらなかった局は全手 hash と結果が一致。
自動再起動なし。GPU lock は取らない(起動前の nvidia-smi 確認は起動者)。"""
import gzip, importlib.util, json, os, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repository root (the folder holding eval/, postprocess/, engine/)
C1C2 = ROOT / 'eval/experiments/sub121_c1c2_20260930'
_spec = importlib.util.spec_from_file_location('sub121_run_c', C1C2 / 'run_c.py')
C = importlib.util.module_from_spec(_spec); sys.modules['sub121_run_c'] = C; _spec.loader.exec_module(C)
S = C.S
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from scripts.kagg_pfsp3.evaluate_land import play_land_multi
import market_pp as MP

SCR = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'sub121_market_pp_20260930')
BASE_HASHES = S.SCR / 'hashes' / 'base'
ARMS = json.loads((HERE / 'arms.json').read_text())            # {腕: "W3+W5+W6+W9+WT|M1+M2"} 起動前に固定
sha, dump, smi, key = S.sha, S.dump, S.smi, S.key


def parse(arm):
    c1, m = ARMS[arm].split('|')
    spec = C.parse(c1); spec['market'] = tuple(m.split('+'))
    assert set(spec['market']) <= {'M1', 'M1x', 'M2', 'M3', 'M5'} and spec['flags'] == ('W3', 'W5', 'W6', 'W9', 'WT') and not spec['w8'], spec
    return spec


class MTop:
    def __init__(self, top, keys, spec, rules):
        self.top, self.keys, self.flags = top, keys, spec['market']
        self.mask_version = top.mask_version; self._batch = top._batch
        _sp = importlib.util.spec_from_file_location('kgr_official', S.OFFICIAL)
        K = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(K)
        self.market_price = K.market_price
        self.rules = rules; self.events = []; self.seconds = 0.; self.status = Counter()

    def act_batch(self, observations):
        acts = self.top.act_batch(observations)
        t0 = time.perf_counter(); step = int(observations[0]['step']); snaps = {}; out = []
        for j, (o, a) in enumerate(zip(observations, acts)):
            if not MP.needs_obs(a, step, self.flags):
                out.append(a); continue
            pool, i = o['_native_pool'], o['_native_index']
            if id(pool) not in snaps:
                snaps[id(pool)] = pool.debug()
            full = snaps[id(pool)][i // 2]['observations'][i % 2]
            assert int(full['step']) == step and full['player'] == self.keys[j][2]
            new, ev = MP.apply(full, a, self.flags, self.market_price)
            if 'M5' in self.flags and step % 24 == 23:
                new, ev5 = self.m5(full, new, step)
                ev += ev5
            assert [new.get('farmer')] + list(new.get('hands') or []) == [a.get('farmer')] + list(a.get('hands') or [])
            if step == 718:
                assert (new.get('market') or []) == (a.get('market') or [])
            for e in ev:
                e['game'] = j; e['quadrants'] = len(full['farms'][full['player']]['unlocked_quadrants'])
            self.events.extend(ev); out.append(new)
        self.seconds += time.perf_counter() - t0
        return out


    def m5(self, full, a, step):
        """[] を除いた注文列で W9 を計算し直し、足りない SELL を [] の枠へ入れる(unit の手は W9 では変わらない)。"""
        mk = list(a.get('market') or [])
        if not any(isinstance(o, list) and not o for o in mk):
            return a, []
        compact = [o for o in mk if not (isinstance(o, list) and not o)]
        ref, ev, st = C.WR2.replace_waste(full, S.CONFIG, dict(a, market=compact), [], flags=('W9',), rules=self.rules, budget_seconds=C.BUDGET)
        self.status[st] += 1
        if st != 'ok' or ref is a or not ev:
            return a, []
        assert [ref.get('farmer')] + list(ref.get('hands') or []) == [a.get('farmer')] + list(a.get('hands') or [])
        extra = ref['market'][len(compact):]
        assert ref['market'][:len(compact)] == compact and extra and all(o[0] == 'SELL' for o in extra)
        out = dict(a); out['market'] = MP.fill_empty(mk, extra)
        return out, [dict(kind='M5', step=step, added=extra, empties=sum(1 for o in mk if isinstance(o, list) and not o))]


def build(inf, keys, spec, rules):
    base, inner, top = C.build(inf, keys, spec, rules, 719)
    return base, inner, top, MTop(top, keys, spec, rules)


def summarize(top, mtop):
    summ, _ = C.summarize(top, top.inner, dict(w8=''), None)
    me = mtop.events
    summ.update(market_events=dict(Counter(e['kind'] for e in me)),
                market_orders_removed=dict(Counter(f"{e['kind']}:{e['order'][0]}:{e['order'][1] if len(e['order']) > 1 and e['order'][0] != 'HIRE' else ''}"
                                                   for e in me if e['kind'] in ('M2', 'M3'))),
                m1_dropped_buy=sum(len(e['dropped_buy']) for e in me if e['kind'] == 'M1'),
                m5_sell_units=sum(o[2] for e in me if e['kind'] == 'M5' for o in e['added']), m5_status=dict(mtop.status),
                market_seconds=mtop.seconds)
    return summ


def smoke(arm):
    spec = parse(arm)
    out = HERE / 'smoke' / arm; out.mkdir(parents=True, exist_ok=False)
    rules, learner, opps, mods, shas = C.load()
    seeds = list(range(173600000, 173600004)); names = list(S.OPPONENTS)
    keys = [(n, s, p) for n in names for s in seeds for p in (0, 1)]
    base, inner, top, mtop = build(learner, keys, spec, rules)
    t0 = time.time()
    rows = play_land_multi(mtop, opps, seeds)
    assert [key(g) for g in rows] == keys and base.calls == 719
    summ = summarize(top, mtop)
    res = dict(arm=arm, spec=spec, seconds=time.time() - t0, games=len(rows), wdl=dict(Counter(r['result'] for r in rows)), **summ,
               market_samples=[dict(e, key=list(keys[e['game']])) for e in mtop.events[:30]], code_sha256=sources(),
               gpu=smi('--query-gpu=memory.used,memory.total'))
    dump(out / 'smoke.json', res)
    print(json.dumps({k: v for k, v in res.items() if k not in ('market_samples', 'code_sha256')}, default=str), flush=True)


def sources():
    paths = C.sources()
    for p in [HERE / 'run_m.py', HERE / 'market_pp.py', HERE / 'arms.json']:
        paths[str(Path(p).resolve().relative_to(ROOT))] = sha(p)
    return paths


def run_arm(arm):
    spec = parse(arm)
    prot = json.loads((HERE / 'protocol.json').read_text())
    assert prot['arms'][arm]['spec'] == ARMS[arm]
    first, count = prot['seeds']['first'], prot['seeds']['count']
    adir = HERE / 'games' / arm; adir.mkdir(parents=True, exist_ok=False)
    hdir = SCR / 'hashes' / arm; hdir.mkdir(parents=True, exist_ok=False)
    rules, learner, opps, mods, shas = C.load()
    code = sources()
    shared = {key(g): g for g in json.loads((C.W9T / 'games/base/games_all.json').read_text())}; assert len(shared) == 1344
    c1p = {key(g): g for g in json.loads((C1C2 / 'games/C1p/games_all.json').read_text())}; assert len(c1p) == 1344
    dump(adir / 'manifest.json', dict(arm=arm, spec=spec, spec_str=ARMS[arm], seeds=[first, first + count - 1], waves=S.WAVES, pid=os.getpid(),
         source_sha256=shas, code_sha256=code, protocol_sha256=sha(HERE / 'protocol.json'), shared_base=str(C.W9T / 'games/base'),
         shared_c1p=str(C1C2 / 'games/C1p'),
         revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         gpu_before=smi('--query-gpu=memory.used,memory.total,power.draw'), gpu_apps_before=smi('--query-compute-apps=pid,used_memory'),
         precision='fp32', tf32=False, learner_batch=64, opponent_batch=32, configuration=S.CONFIG, budget_seconds_per_observation=C.BUDGET,
         automatic_restart=False))
    seeds_all = list(range(first, first + count)); saved = []; start = time.time()
    dump(adir / 'status.json', dict(status='running', pid=os.getpid(), started=start))
    jobs = [(i, w, names) for i in range(0, count, 16) for w, names in enumerate(S.WAVES)]
    for job, (i, w, names) in enumerate(jobs, 1):
        seeds = seeds_all[i:i + 16]
        keys = [(n, s, p) for n in names for s in seeds for p in (0, 1)]
        base, inner, top, mtop = build(learner, keys, spec, rules); tick = time.time()

        def progress(step):
            v = dict(status='running', arm=arm, job=job, jobs=len(jobs), step=step, finished=len(saved), seconds=time.time() - start)
            dump(adir / 'progress.json', v); print(json.dumps(v), flush=True)
        rows = play_land_multi(mtop, {n: opps[n] for n in names}, seeds, progress=progress)
        assert [key(g) for g in rows] == keys and base.calls == 719
        tag = f'seed{i:03d}_wave{w}'; target = adir / tag; target.mkdir()
        np.save(hdir / f'{tag}.npy', base.hashes)
        summ = summarize(top, mtop)
        with gzip.open(target / 'events.jsonl.gz', 'wt') as f:
            for e in top.events + mtop.events:
                f.write(json.dumps(dict(e, key=list(keys[e['game']])), separators=(',', ':'), default=str) + '\n')
        bh = np.load(BASE_HASHES / f'{tag}.npy')
        first_step = {}
        for e in top.events:
            if e['action'] is not None:
                first_step[e['game']] = min(first_step.get(e['game'], 10**9), e['step'])
        for e in mtop.events:
            first_step[e['game']] = min(first_step.get(e['game'], 10**9), e['step'])
        checked = 0; same_c1p = 0
        mgames = {e['game'] for e in mtop.events}
        for j, k in enumerate(keys):                        # market_pp が 1 度も変えなかった局は C1p 腕と結果が一致
            if j not in mgames:
                assert all(rows[j][x] == c1p[k][x] for x in ('own_money', 'opponent_money', 'result')), (k, 'C1p outcome differs')
                same_c1p += 1
        for j, k in enumerate(keys):
            f = first_step.get(j)
            if f is None:
                assert (bh[j] == base.hashes[j]).all(), (k, 'unchanged game trajectory differs')
                assert all(rows[j][x] == shared[k][x] for x in ('own_money', 'opponent_money', 'result')), (k, 'unchanged game outcome differs')
                checked += 719
            else:
                assert (bh[j, :f + 1] == base.hashes[j, :f + 1]).all(), (k, f, 'pre-divergence hash mismatch')
                checked += f + 1
        for r in rows:
            r['arm'] = arm
        dump(target / 'games.json', rows)
        dump(target / 'complete.json', dict(seconds=time.time() - tick, games=len(rows), **summ, games_changed=len(first_step),
             hash_steps_checked=checked, games_unchanged_by_market_equal_c1p=same_c1p, hashes=str(hdir / f'{tag}.npy'), hashes_sha256=sha(hdir / f'{tag}.npy')))
        saved.extend(rows); dump(adir / 'games_all.json', saved)
        print(json.dumps(dict(arm=arm, job_complete=job, seconds=round(time.time() - tick), wdl=dict(Counter(r['result'] for r in rows)),
                              replaced=summ['replaced'], market=summ['market_events'], status=summ['status'])), flush=True)
        del base, inner, top, mtop; torch.cuda.empty_cache()
    assert len(saved) == 1344 and {key(g) for g in saved} == set(shared)
    for p, d in code.items():
        assert sha(ROOT / p) == d, p
    for n in S.OPPONENTS:
        mp = getattr(mods[n], '_MP', None)
        if mp is not None and hasattr(mp, 'FIRE'):
            mp.FIRE.clear()
    dump(adir / 'status.json', dict(status='complete', seconds=time.time() - start, games=len(saved)))
    print(json.dumps(dict(arm_complete=arm, games=len(saved), seconds=time.time() - start)), flush=True)


if __name__ == '__main__':
    try:
        torch.manual_seed(20260930)
        if sys.argv[1] == 'smoke':
            smoke(sys.argv[2])
        else:
            assert sys.argv[1] == 'arm'
            run_arm(sys.argv[2])
    except BaseException as e:
        dump(HERE / f'failure_{"_".join(sys.argv[1:])}.json', dict(type=type(e).__name__, message=str(e), traceback=traceback.format_exc(), time=time.time(), argv=sys.argv))
        raise
