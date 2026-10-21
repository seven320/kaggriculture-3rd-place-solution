"""(sub121_collect_early_deposit_20260930: sub121_collect_ab_20260930/run_p.py の複製。計画を planner_ed.plan_step(ALPHA = 環境変数 ED_ALPHA、持ち物を抱える時間の罰)に替えただけ。)
C1'+W9'(= sub121_market_pp_20260930 の M5 腕と同じ処理)の上に、最終日の回収計画(planner_cap.CollectPlanner と同じ plan_step: 倉庫の上限の同時制約・捨てない預け・2 手前からの先回り売り)を掛けた腕。
掛ける順序: 網 → _pace(_liq) → waste_replace2(W3/W5/W6 → W9 → WT) → market_pp(M5) → 本ファイルの PTop(day 29 の step >= 開始 の unit の手だけ置き換え、market は元のまま)。
条件は sub121_market_pp_20260930/protocol.json と同じ(seed 173600000〜095・相手 7 本・両 player・job 構成・Rust・fp32・TF32 無効・学習側 64 詰め・相手 32)。
計画は局ごとに状態(前回の計画の残り)を持ち、1 手ぶんの計算は CPU の子プロセス(spawn)で局ごとに並列。1 手ぶんの計算時間は子プロセス内で局ごとに測る。
倉庫の上限: 子プロセスで同じ観測に公式 _apply_unit_action(shedCapacity 100)を掛け、置き換えた手で捨てられる個数を毎手数える(0 のはず)。
使い方: run_p.py smoke <腕>  → smoke/<腕>/(seed 173600000〜01 × 7 相手 × 両 player)
        run_p.py arm <腕>    → games/<腕>/     腕 = P696 | P708
整合: 置き換えが最初に起きた手までは M5 腕の特徴+行動 hash と一致、置き換えがなかった局は結果が M5 腕と一致。自動再起動なし。"""
import copy, gzip, importlib.util, json, multiprocessing as mp, os, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repository root (the folder holding eval/, postprocess/, engine/)
MPP = ROOT / 'eval/experiments/sub121_market_pp_20260930'
OFFICIAL = Path(importlib.util.find_spec('kaggle_environments').origin).parent / 'envs/kaggriculture/kaggriculture.py'
sys.path.insert(0, str(HERE))
import planner_ed as PC
ALPHA = float(os.environ['ED_ALPHA'])

ARMS = {'P696': 696, 'P708': 708}
SCR = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'sub121_collect_early_deposit_20260930')
M5_HASHES = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'sub121_market_pp_20260930/hashes/M5')
PROCS = int(os.environ.get('PLAN_PROCS', '8'))
ALLOWED = {'NORTH', 'SOUTH', 'EAST', 'WEST', 'PASS', 'HARVEST', 'WATER', 'COLLECT_FERTILIZER', 'DROP', 'PLACE'}

_K = None


def _worker_init():
    global _K
    os.environ['OMP_NUM_THREADS'] = '1'
    sp = importlib.util.spec_from_file_location('kgr_official_w', OFFICIAL)
    _K = importlib.util.module_from_spec(sp); sp.loader.exec_module(_K)


def work(args):
    """1 局 1 手: 計画 → 公式 _apply_unit_action(倉庫 100)で捨てられる個数を数える。"""
    state, prev, prices = args
    t0 = time.perf_counter()
    acts, sells, nxt, info = PC.plan_step(state, prev, prices, alpha=ALPHA)
    sec = time.perf_counter() - t0
    farm = dict(farmer=list(state['farmer']), hands=[list(h) for h in state['hands']], tiles=copy.deepcopy(state['tiles']))
    priv = dict(inventories=copy.deepcopy(state['inventories']), shed=dict(state['shed']), seeds={})
    lost = 0
    for idx, a in enumerate(acts):
        assert isinstance(a, list) and a and a[0] in ALLOWED, a
        pos = farm['farmer'] if idx == 0 else farm['hands'][idx - 1]
        if a[0] == 'DROP' and _K._is_shed_adjacent(pos, 10):  # 公式 343〜356 行: 空きを超える分は捨てられる
            inv = priv['inventories'][idx] if idx < len(priv['inventories']) else {}
            lost += max(0, sum(n for n in inv.values() if n > 0) - max(0, 100 - sum(priv['shed'].values())))
        _K._apply_unit_action(farm, priv, idx, a, 10, state['step'] // 24, 24, 100)
    info['sec'] = sec; info['lost'] = lost
    info['shed_after_units'] = sum(priv['shed'].values())
    return acts, sells, nxt, info


class PTop:
    def __init__(self, mtop, keys, start, pool):
        self.mtop, self.keys, self.start, self.pool = mtop, keys, start, pool
        self.mask_version = mtop.mask_version; self._batch = mtop._batch
        self.prev = [None] * len(keys); self.prices = [None] * len(keys)
        self.events = []; self.max_sec = 0.; self.game_sec = [0.] * len(keys); self.seconds = 0.
        self.first = {}; self.discard = 0; self.overcap = 0; self.info = Counter()

    def act_batch(self, observations):
        acts = self.mtop.act_batch(observations)
        step = int(observations[0]['step'])
        if step < self.start or step > PC.LAST:
            return acts
        t0 = time.perf_counter(); snaps = {}; jobs = []
        for j, o in enumerate(observations):
            pool, i = o['_native_pool'], o['_native_index']
            if id(pool) not in snaps:
                snaps[id(pool)] = pool.debug()
            full = snaps[id(pool)][i // 2]['observations'][i % 2]
            p = self.keys[j][2]
            assert int(full['step']) == step and full['player'] == p
            if self.prices[j] is None:
                self.prices[j] = dict(full['market']['prices'])
            f = full['farms'][p]; pr = full['private']
            jobs.append((dict(step=step, farmer=f['farmer'], hands=f['hands'], tiles=f['tiles'], inventories=pr['inventories'],
                              shed=pr['shed']), self.prev[j], self.prices[j]))
        res = self.pool.map(work, jobs, chunksize=1)
        out = []
        for j, (a, (units, sells, nxt, info)) in enumerate(zip(acts, res)):
            self.prev[j] = nxt
            old = [a.get('farmer')] + list(a.get('hands') or [])
            new = dict(a); new['farmer'] = units[0]; new['hands'] = units[1:]
            removed = []
            if sells:
                new['market'], removed = PC.merge_market(a.get('market'), sells)
                self.first.setdefault(j, step)
                self.info.update(dict(sell_ahead_units=sum(o[2] for o in sells), sell_ahead_steps=1, removed_orders=len(removed)))
                for o in removed:
                    self.info['removed_' + str(o[0] if o else 'empty')] += 1
            assert len(units) == 1 + len(jobs[j][0]['hands'])
            changed = sum(1 for x, y in zip(old, units) if x != y) + abs(len(old) - len(units))
            if changed:
                self.first.setdefault(j, step)
            lost = info['lost']; self.discard += lost
            self.overcap += info['shed_after_units'] > 100
            self.max_sec = max(self.max_sec, info['sec']); self.game_sec[j] += info['sec']
            self.info.update(dict(deposit=info['deposit'], place=info['place'], wait_full=info['wait_full'], changed_units=changed))
            self.events.append(dict(kind='P', game=j, step=step, changed=changed, sec=round(info['sec'], 4), pred=info['pred'],
                                    deposit=info['deposit'], place=info['place'], wait_full=info['wait_full'],
                                    lost=lost, shed_after_units=info['shed_after_units'], sells=sells, removed=removed,
                                    dep_future=info['dep_future']))
            out.append(new)
        self.seconds += time.perf_counter() - t0
        return out


def load_m():
    sp = importlib.util.spec_from_file_location('sub121_run_m', MPP / 'run_m.py')
    M = importlib.util.module_from_spec(sp); sys.modules['sub121_run_m'] = M; sp.loader.exec_module(M)
    return M


def sources(M):
    paths = M.sources()
    for p in [HERE / 'run_p_ed.py', HERE / 'planner_ed.py']:
        paths[str(Path(p).resolve().relative_to(ROOT))] = M.sha(p)
    return paths


def summarize(pt):
    ev = pt.events
    return dict(plan_calls=len(ev), plan_max_sec=pt.max_sec, plan_game_sec_max=max(pt.game_sec), plan_game_sec_mean=sum(pt.game_sec) / len(pt.game_sec),
                plan_seconds_wall=pt.seconds, discarded_items=pt.discard, shed_over_100=pt.overcap, info=dict(pt.info),
                games_changed=len(pt.first))


def smoke(arm):
    M = load_m(); S = M.S; C = M.C
    spec = M.parse('M5'); start = ARMS[arm]
    out = HERE / 'smoke' / arm; out.mkdir(parents=True, exist_ok=False)
    rules, learner, opps, mods, shas = C.load()
    seeds = [173600000, 173600001]; names = list(S.OPPONENTS)
    keys = [(n, s, p) for n in names for s in seeds for p in (0, 1)]
    with mp.get_context('spawn').Pool(PROCS, initializer=_worker_init) as pool:
        base, inner, top, mtop = M.build(learner, keys, spec, rules)
        pt = PTop(mtop, keys, start, pool)
        t0 = time.time()
        from scripts.kagg_pfsp3.evaluate_land import play_land_multi
        rows = play_land_multi(pt, opps, seeds)
    assert [M.key(g) for g in rows] == keys and base.calls == 719
    m5 = {M.key(g): g for g in json.loads((MPP / 'games/M5/games_all.json').read_text())}
    diff = [dict(key=list(k), m5=[m5[k]['own_money'], m5[k]['result']], arm=[r['own_money'], r['result']]) for k, r in zip(keys, rows)]
    res = dict(arm=arm, start=start, seconds=time.time() - t0, games=len(rows), wdl=dict(Counter(r['result'] for r in rows)),
               **summarize(pt), own_money_diff_mean=sum(r['own_money'] - m5[k]['own_money'] for k, r in zip(keys, rows)) / len(rows),
               per_game=diff, samples=pt.events[:40], code_sha256=sources(M), gpu=M.smi('--query-gpu=memory.used,memory.total'))
    M.dump(out / 'smoke.json', res)
    print(json.dumps({k: v for k, v in res.items() if k not in ('samples', 'code_sha256', 'per_game')}, default=str), flush=True)


def run_arm(arm):
    M = load_m(); S = M.S; C = M.C
    spec = M.parse('M5'); start = ARMS[arm]
    prot = json.loads((HERE / 'protocol.json').read_text())
    assert prot['arms'][arm]['start_step'] == start
    first, count = prot['seeds']['first'], prot['seeds']['count']
    adir = HERE / 'games' / arm; adir.mkdir(parents=True, exist_ok=False)
    hdir = SCR / 'hashes' / arm; hdir.mkdir(parents=True, exist_ok=False)
    rules, learner, opps, mods, shas = C.load()
    code = sources(M)
    for p, d in prot['code_sha256'].items():
        assert code.get(p) == d, p
    m5 = {M.key(g): g for g in json.loads((MPP / 'games/M5/games_all.json').read_text())}; assert len(m5) == 1344
    M.dump(adir / 'manifest.json', dict(arm=arm, start_step=start, m5_spec=M.ARMS['M5'], seeds=[first, first + count - 1], waves=S.WAVES, pid=os.getpid(),
           source_sha256=shas, code_sha256=code, protocol_sha256=M.sha(HERE / 'protocol.json'), shared_m5=str(MPP / 'games/M5'),
           revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           gpu_before=M.smi('--query-gpu=memory.used,memory.total,power.draw'), plan_procs=PROCS, automatic_restart=False))
    seeds_all = list(range(first, first + count)); saved = []; t_start = time.time()
    M.dump(adir / 'status.json', dict(status='running', pid=os.getpid(), started=t_start))
    jobs = [(i, w, names) for i in range(0, count, 16) for w, names in enumerate(S.WAVES)]
    from scripts.kagg_pfsp3.evaluate_land import play_land_multi
    with mp.get_context('spawn').Pool(PROCS, initializer=_worker_init) as pool:
        for job, (i, w, names) in enumerate(jobs, 1):
            seeds = seeds_all[i:i + 16]
            keys = [(n, s, p) for n in names for s in seeds for p in (0, 1)]
            base, inner, top, mtop = M.build(learner, keys, spec, rules)
            pt = PTop(mtop, keys, start, pool); tick = time.time()
            rows = play_land_multi(pt, {n: opps[n] for n in names}, seeds)
            assert [M.key(g) for g in rows] == keys and base.calls == 719
            tag = f'seed{i:03d}_wave{w}'; target = adir / tag; target.mkdir()
            import numpy as np
            np.save(hdir / f'{tag}.npy', base.hashes)
            bh = np.load(M5_HASHES / f'{tag}.npy')
            checked = 0
            for j, k in enumerate(keys):
                f = pt.first.get(j)
                if f is None:
                    assert (bh[j] == base.hashes[j]).all(), (k, 'unchanged game trajectory differs from M5')
                    assert all(rows[j][x] == m5[k][x] for x in ('own_money', 'opponent_money', 'result')), (k, 'unchanged game outcome differs from M5')
                    checked += 719
                else:
                    assert (bh[j, :f + 1] == base.hashes[j, :f + 1]).all(), (k, f, 'pre-divergence hash mismatch vs M5')
                    checked += f + 1
            summ = summarize(pt)
            with gzip.open(target / 'events.jsonl.gz', 'wt') as fz:
                for e in top.events + mtop.events + pt.events:
                    fz.write(json.dumps(dict(e, key=list(keys[e['game']])), separators=(',', ':'), default=str) + '\n')
            for r in rows:
                r['arm'] = arm
            M.dump(target / 'games.json', rows)
            M.dump(target / 'complete.json', dict(seconds=time.time() - tick, games=len(rows), **summ, hash_steps_checked=checked,
                                                  game_sec=pt.game_sec))
            saved.extend(rows); M.dump(adir / 'games_all.json', saved)
            print(json.dumps(dict(arm=arm, job_complete=job, seconds=round(time.time() - tick), wdl=dict(Counter(r['result'] for r in rows)),
                                  **{k: v for k, v in summ.items() if k != 'info'})), flush=True)
            del base, inner, top, mtop, pt
    assert len(saved) == 1344 and {M.key(g) for g in saved} == set(m5)
    for p, d in code.items():
        assert M.sha(ROOT / p) == d, p
    M.dump(adir / 'status.json', dict(status='complete', seconds=time.time() - t_start, games=len(saved)))
    print(json.dumps(dict(arm_complete=arm, games=len(saved), seconds=time.time() - t_start)), flush=True)


if __name__ == '__main__':
    try:
        import torch
        torch.manual_seed(20260930)
        if sys.argv[1] == 'smoke':
            smoke(sys.argv[2])
        else:
            assert sys.argv[1] == 'arm'
            run_arm(sys.argv[2])
    except BaseException as e:
        (HERE / f'failure_{"_".join(sys.argv[1:])}.json').write_text(json.dumps(dict(type=type(e).__name__, message=str(e), traceback=traceback.format_exc(),
                                                                                   time=time.time(), argv=sys.argv), ensure_ascii=False, indent=1))
        raise
