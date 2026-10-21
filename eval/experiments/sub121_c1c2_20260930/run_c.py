"""sub121(v109、v4x gen14 単体・4 面・_liq つき)に、v104 で効いた後付け処理の組み合わせ C1・C2(と W8 抜きの C1'・C2')を付けた腕。
元 sub121(games/base)と W9T 腕は artifacts/sub121_w9t_20260930 のものを再利用する(回し直さない)。
条件は sub121_w9t_20260930/protocol.json と同じ(seed・相手 7 本・両 player・job 構成・Rust・fp32・TF32 無効・学習側 64 詰め・相手 32)。

腕(v104 の arms_parallel2.py / confirm.py と同じ定義):
  C2p = W5+W6+W9+WT        C1p = W3+W5+W6+W9+WT
  C2  = W5+W6+W8+W9+WT     C1  = W3+W5+W6+W8+W9+WT     (W8 = ①②-b③、arms_parallel2.W8Parts と同じ構成)
掛ける順序(v104 と同じ): 網 → 提出物の _pace(= _liq、NativeBatchAgent の中) → W8(再 encode) → waste_replace2.replace_waste
(W3/W5/W6 の unit 置き換え → W9 → WT)。unit スコアは常に最初の forward の unit_logits(nmb の下駄の後)。
W3/W5/W6 の置き換え先 = 元の logits を「モデルが可(>= -1e3)かつ PASS・状態不変・W3〜W7/W14 に当たらない」候補に絞った最大。
使い方: run_c.py smoke <腕>  → smoke/<腕>/(seed 173600000〜03 × 7 相手 × 両 player、全 720 手)
        run_c.py arm <腕>    → games/<腕>/
自動再起動なし。GPU lock は取らない(起動前の nvidia-smi 確認は起動者)。"""
import gzip, importlib.util, json, os, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repository root (the folder holding eval/, postprocess/, engine/)
W9T = ROOT / 'eval/experiments/sub121_w9t_20260930'
_spec = importlib.util.spec_from_file_location('sub121_run', W9T / 'run.py')
S = importlib.util.module_from_spec(_spec); sys.modules['sub121_run'] = S; _spec.loader.exec_module(S)
for d in ('unit_work_cap_20260927_codex', 'combined_unit_work_20260927_codex', 'worker_motion_20260927_codex'):
    sys.path.insert(0, str(ROOT / 'eval/experiments' / d))
import numpy as np
import torch
from scripts.kagg_pfsp3.evaluate_land import play_land_multi
from scripts.kagg_bc_kit.versioned_features import featurize, _q
import waste_replace2 as WR2

SCR = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'sub121_c1c2_20260930')
BASE_HASHES = S.SCR / 'hashes' / 'base'
BUDGET = 1.0
ARMS = {'C2p': 'W5+W6+W9+WT', 'C1p': 'W3+W5+W6+W9+WT', 'C2': 'W5+W6+W8+W9+WT', 'C1': 'W3+W5+W6+W8+W9+WT'}
W8_FILES = ['eval/experiments/unit_work_cap_20260927_codex/combined_cap.py', 'eval/experiments/unit_work_cap_20260927_codex/cap_overlap.py',
            'eval/experiments/combined_unit_work_20260927_codex/replant.py', 'eval/experiments/worker_motion_20260927_codex/collect.py',
            'eval/experiments/worker_motion_20260927_codex/analyze.py', 'eval/experiments/worker_motion_20260927_codex/sequence_eval.py']
sha, dump, smi, key, h64, FEATS, CONFIG = S.sha, S.dump, S.smi, S.key, S.h64, S.FEATS, S.CONFIG
CAP = {}
QUAD = lambda x, y: ('N' if y < 5 else 'S') + ('W' if x < 5 else 'E')


def parse(spec):
    parts = spec.split('+')
    assert all(p in WR2.ALL or p == 'W8' for p in parts), spec
    return dict(flags=tuple(p for p in parts if p in WR2.ALL), w8='abc' if 'W8' in parts else '')


class Lazy:
    """arms.Lazy と同じ: unit u の {行動: スコア}(-1e3 未満はモデルが不可とした手)。"""
    def __init__(self, row, n):
        self.row, self.n = row, n

    def __len__(self):
        return self.n

    def __getitem__(self, u):
        r = self.row[u]
        return {WR2.nn_action(c): float(r[c]) for c in range(44) if r[c] >= -1e3}


class Base:
    """学習側 sub121: 64 に詰めて forward(_liq 込み)。最初の forward の unit スコアと q を控え、特徴+元行動の hash を
    sub121_w9t の Learner と同じ式で記録する(元 sub121 の hash と比べられる)。"""
    def __init__(self, inf, keys, steps):
        self.inf, self.keys = inf, keys; self.mask_version = inf.mask_version; self._batch = inf._batch
        self.hashes = np.zeros((len(keys), steps), np.uint64); self.calls = 0

    def act_batch(self, observations):
        step = int(observations[0]['step']); n = len(observations)
        assert n == len(self.keys) and all(int(o['step']) == step for o in observations)
        padded = list(observations)
        while len(padded) < 64:
            padded.append(observations[(len(padded) - n) % n])
        CAP.clear(); CAP['n'] = 0
        acts = self.inf.act_batch(padded)[:n]
        assert CAP['n'] == 1 and CAP['s'].shape[0] == 64, (CAP.get('n'), tuple(CAP['s'].shape))
        self.scores = CAP['s'][:n].float().cpu().numpy(); self.q = CAP['q'][:n]
        self.calls += 1
        for j, o in enumerate(observations):
            pool, i = o['_native_pool'], o['_native_index']
            self.hashes[j, step] = h64(*(np.ascontiguousarray(pool.features[k][i]).tobytes() for k in FEATS),
                                       json.dumps(acts[j], sort_keys=True).encode())
        return acts


class W8Parts:
    """arms_parallel2.W8Parts と同じ(①②-b③ を全部 ON にすると combined_cap.CombinedCap と同じ構成)。"""
    def __init__(self, learner, keys, parts):
        from cap_overlap import Overlap
        from replant import Sequential
        self.mask_version = learner.mask_version; self._batch = learner._batch
        self.work = self.replant = None; inner = learner
        pairs = [p for c, p in (('a', ['PLANT', 'WATER']), ('b', ['WATER', 'HARVEST'])) if c in parts]
        if pairs:
            self.work = Overlap(learner, keys, work_only=True, allowed_sources=[p[0] for p in pairs], allowed_pairs=pairs, harvest_cap_only=True)
            inner = SimpleNamespace(mask_version=learner.mask_version, _batch=learner._batch, act_batch=self.work.act_batch)
        if 'c' in parts:
            self.replant = Sequential(inner, keys, True); inner = self.replant
        self.inner = inner

    def act_batch(self, observations):
        return self.inner.act_batch(observations)          # 再 encode の unit_logits は Base が控えた後なので影響しない

    def events(self):
        out = [] if self.work is None else [dict(**e, rule='1' if e['previous_action'][0] == 'PLANT' else '2-b') for e in self.work.events]
        return out + ([] if self.replant is None else [dict(**e, rule='3') for e in self.replant.events])


class Top:
    """arms.Top と同じ(雑草の診断は省略): inner の行動に waste_replace2.replace_waste を掛ける。"""
    def __init__(self, base, inner, keys, spec, rules):
        self.base, self.inner, self.keys, self.spec, self.rules = base, inner, keys, spec, rules
        self.mask_version = base.mask_version; self._batch = base._batch
        self.events = []; self.status = Counter(); self.seconds = 0.; self.parity = 0; self.checks = Counter()

    @torch.inference_mode()
    def finalize(self, j, u, a):
        cls = torch.zeros(1, 20, dtype=torch.long, device=self.base.q.device); cls[0, u] = WR2.NN_CLASS[tuple(a)]
        qc = int(self._batch.net.unit_qty(self.base.q[j:j + 1], cls).argmax(-1)[0, u])
        qty = _q(qc, 20)
        return [a[0], a[1], 1 if qty is None else int(qty)]

    def act_batch(self, observations):
        acts = self.inner.act_batch(observations)
        t0 = time.perf_counter(); snaps = {}; out = []; step = int(observations[0]['step'])
        for j, (o, a) in enumerate(zip(observations, acts)):
            pool, i = o['_native_pool'], o['_native_index']
            if id(pool) not in snaps:
                snaps[id(pool)] = pool.debug()
            full = snaps[id(pool)][i // 2]['observations'][i % 2]
            assert int(full['step']) == step and full['player'] == self.keys[j][2]
            if self.parity < 32:
                for name, v in zip(FEATS, featurize(full, self.mask_version)):
                    assert np.array_equal(v, pool.features[name][i]), name
                self.parity += 1
            units = 1 + len(a.get('hands', []))
            scores = Lazy(self.base.scores[j], units)
            new, ev, st = WR2.replace_waste(full, CONFIG, a, scores, flags=self.spec['flags'],
                                            finalize=lambda u, x, j=j: self.finalize(j, u, x), rules=self.rules, budget_seconds=BUDGET)
            self.status[st] += 1
            nq = len(full['farms'][full['player']]['unlocked_quadrants'])
            for e in ev:
                e['game'] = j; e['quadrants'] = nq
                if e.get('pos') is not None:
                    x, y = e['pos']; e['tile_quadrant'] = QUAD(x, y)
                    t = full['farms'][full['player']]['tiles'][y][x]
                    assert t != 'LOCKED', (step, j, e)       # 置き換えの対象マスは解放済み
                if e['kind'] in ('W3', 'W5', 'W6') and e['action'] is not None:
                    c = WR2.NN_CLASS[tuple(e['action'][:2])]
                    assert self.base.scores[j][e['unit'], c] >= -1e3 and abs(self.base.scores[j][e['unit'], c] - e['score']) < 1e-6
                    self.checks['model_legal'] += 1
                    self.checks['legal_v0' if pool.features['legal'][i][e['unit'], c] else 'legal_xp_only'] += 1
            self.events.extend(ev)
            old_m, new_m = a.get('market') or [], new.get('market') or []
            assert new_m[:len(old_m)] == old_m and len(new_m) <= 10
            if 'W9' not in self.spec['flags']:
                assert new_m == old_m
            if step == 718:                                   # _liq の後なので WT は足さない(SELL 1000 で売り残し 0)
                assert old_m == [['SELL', it, 1000] for it in S.module_of(S.LEARNER['src'] / 'main.py')._LIQ_ITEMS]
            out.append(new)
        self.seconds += time.perf_counter() - t0
        return out


def install_capture(net):
    orig = net.unit_logits

    def unit_logits(h, ou, legal, free=None):
        s, q = orig(h, ou, legal) if free is None else orig(h, ou, legal, free)
        CAP['s'], CAP['q'] = s, q; CAP['n'] = CAP.get('n', 0) + 1
        return s, q
    net.unit_logits = unit_logits


def load():
    rules, learner, opps, mods, shas = S.load_all()
    install_capture(learner._batch.net)
    return rules, learner, opps, mods, shas


def build(inf, keys, spec, rules, steps):
    base = Base(inf, keys, steps)
    inner = W8Parts(base, keys, spec['w8']) if spec['w8'] else base
    return base, inner, Top(base, inner, keys, spec, rules)


def sources():
    paths = S.sources()
    for p in [HERE / 'run_c.py', WR2.__file__]:
        paths[str(Path(p).resolve().relative_to(ROOT))] = sha(p)
    for p in W8_FILES:
        paths[p] = sha(ROOT / p)
    return paths


def summarize(top, inner, spec, keys):
    ev = top.events
    rep = [e for e in ev if e['action'] is not None]
    w8 = [dict(e, key=list(e['key'])) for e in inner.events()] if spec['w8'] else []
    return dict(status=dict(top.status), events=dict(Counter(e['kind'] for e in ev)), replaced=dict(Counter(e['kind'] for e in rep)),
                replaced_by_quadrants={k: dict(Counter(e['quadrants'] for e in rep if e['kind'] == k)) for k in ('W3', 'W5', 'W6', 'W9', 'WT')},
                replaced_by_tile_quadrant={k: dict(Counter(e.get('tile_quadrant') for e in rep if e['kind'] == k)) for k in ('W3', 'W5', 'W6')},
                replaced_to=dict(Counter(f"{e['kind']}:{e['previous_action'][0]}->{e['action'][0]}" for e in rep if e['unit'] is not None)),
                w8_events=dict(Counter(e.get('rule') for e in w8)), candidate_checks=dict(top.checks),
                postprocess_seconds=top.seconds, feature_parity_checked=top.parity), w8


def smoke(arm):
    spec = parse(ARMS[arm])
    out = HERE / 'smoke' / arm; out.mkdir(parents=True, exist_ok=False)
    rules, learner, opps, mods, shas = load()
    seeds = list(range(173600000, 173600004)); names = list(S.OPPONENTS)
    keys = [(n, s, p) for n in names for s in seeds for p in (0, 1)]
    base, inner, top = build(learner, keys, spec, rules, 719)
    t0 = time.time()
    rows = play_land_multi(top, opps, seeds)
    assert [key(g) for g in rows] == keys and base.calls == 719
    summ, w8 = summarize(top, inner, spec, keys)
    ref = {key(g): g for g in json.loads((W9T / 'games/base/games_all.json').read_text())}
    wdl_base = Counter(ref[k]['result'] for k in keys)
    rep = [e for e in top.events if e['action'] is not None and e['unit'] is not None]
    res = dict(arm=arm, spec=spec, seconds=time.time() - t0, games=len(rows), wdl=dict(Counter(r['result'] for r in rows)),
               wdl_base_same_keys_other_batching=dict(wdl_base), **summ,
               samples={k: [dict(e, key=list(keys[e['game']])) for e in rep if e['kind'] == k][:8] for k in ('W3', 'W5', 'W6')},
               w8_samples=w8[:6], source_sha256=shas, code_sha256=sources(),
               gpu=smi('--query-gpu=memory.used,memory.total'))
    dump(out / 'smoke.json', res)
    print(json.dumps({k: v for k, v in res.items() if k not in ('samples', 'w8_samples', 'source_sha256', 'code_sha256')}, default=str), flush=True)


def run_arm(arm):
    spec = parse(ARMS[arm])
    prot = json.loads((HERE / 'protocol.json').read_text())
    assert prot['arms'][arm]['spec'] == ARMS[arm]
    first, count = prot['seeds']['first'], prot['seeds']['count']
    adir = HERE / 'games' / arm; adir.mkdir(parents=True, exist_ok=False)
    hdir = SCR / 'hashes' / arm; hdir.mkdir(parents=True, exist_ok=False)
    rules, learner, opps, mods, shas = load()
    code = sources()
    shared = {key(g): g for g in json.loads((W9T / 'games/base/games_all.json').read_text())}; assert len(shared) == 1344
    dump(adir / 'manifest.json', dict(arm=arm, spec=spec, spec_str=ARMS[arm], seeds=[first, first + count - 1], waves=S.WAVES, pid=os.getpid(),
         source_sha256=shas, code_sha256=code, protocol_sha256=sha(HERE / 'protocol.json'), shared_base=str(W9T / 'games/base'),
         revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         gpu_before=smi('--query-gpu=memory.used,memory.total,power.draw'), gpu_apps_before=smi('--query-compute-apps=pid,used_memory'),
         precision='fp32', tf32=False, learner_batch=64, opponent_batch=32, configuration=CONFIG, budget_seconds_per_observation=BUDGET,
         automatic_restart=False))
    seeds_all = list(range(first, first + count)); saved = []; start = time.time()
    dump(adir / 'status.json', dict(status='running', pid=os.getpid(), started=start))
    jobs = [(i, w, names) for i in range(0, count, 16) for w, names in enumerate(S.WAVES)]
    for job, (i, w, names) in enumerate(jobs, 1):
        seeds = seeds_all[i:i + 16]
        keys = [(n, s, p) for n in names for s in seeds for p in (0, 1)]
        base, inner, top = build(learner, keys, spec, rules, 719); tick = time.time()

        def progress(step):
            v = dict(status='running', arm=arm, job=job, jobs=len(jobs), step=step, finished=len(saved), seconds=time.time() - start)
            dump(adir / 'progress.json', v); print(json.dumps(v), flush=True)
        rows = play_land_multi(top, {n: opps[n] for n in names}, seeds, progress=progress)
        assert [key(g) for g in rows] == keys and base.calls == 719
        tag = f'seed{i:03d}_wave{w}'; target = adir / tag; target.mkdir()
        np.save(hdir / f'{tag}.npy', base.hashes)
        summ, w8 = summarize(top, inner, spec, keys)
        with gzip.open(target / 'events.jsonl.gz', 'wt') as f:
            for e in top.events:
                f.write(json.dumps(dict(e, key=list(keys[e['game']])), separators=(',', ':'), default=str) + '\n')
            for e in w8:
                f.write(json.dumps(dict(e, kind='W8'), separators=(',', ':'), default=str) + '\n')
        # 元 sub121 との整合: 初めて行動が変わる手まで特徴+元行動の hash が一致、変わらなかった局は全手 hash と結果が一致
        bh = np.load(BASE_HASHES / f'{tag}.npy')
        first_step = {}
        for e in top.events:
            if e['action'] is not None:
                first_step[e['game']] = min(first_step.get(e['game'], 10**9), e['step'])
        index = {k: j for j, k in enumerate(keys)}
        for e in w8:
            j = index[tuple(e['key'])]; first_step[j] = min(first_step.get(j, 10**9), e['step'])
        checked = 0
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
             hash_steps_checked=checked, hashes=str(hdir / f'{tag}.npy'), hashes_sha256=sha(hdir / f'{tag}.npy')))
        saved.extend(rows); dump(adir / 'games_all.json', saved)
        print(json.dumps(dict(arm=arm, job_complete=job, seconds=round(time.time() - tick), wdl=dict(Counter(r['result'] for r in rows)),
                              replaced=summ['replaced'], w8=summ['w8_events'], status=summ['status'])), flush=True)
        del base, inner, top; torch.cuda.empty_cache()
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
