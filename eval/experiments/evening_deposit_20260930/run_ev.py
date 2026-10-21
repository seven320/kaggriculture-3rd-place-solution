"""EV(夕方の寄せ売り)A/B(2026-10-01)。harness は w9d_drop_sell_20260930/run_w9d.py と同じ組み方(run_g5new を import、既存ファイルは変更しない)。
腕(対照 = games5/plan_s708a010 を再利用):
  k4 / k8 = 版 12 構成 2(gen35 + W3+W5+W6+W9'+WT(+M5)+ 回収計画 708・α=0.01)の M5 の後・回収計画の前に EV(postprocess_ev.apply_ev、K=4/8、V=EV_V)
掛ける順序: 網 → _land(_pace(.)) → Top35 → MTop(M5) → EVTop → PTop(day 29 の 708 以降)
計測: EV で行動が変わった手ごとに、見込みの倉庫(unit → 市場 → hour 23 なら日末搬入)を控え、次の手の実際の倉庫と照合する。
使い方: ED_START=708 ED_ALPHA=0.01 PLAN_PROCS=4 EV_V=<V> run_ev.py <k4|k8> <first_job> <last_job> [smoke]
  smoke: seed 174300000〜015、波 3 本目(gen34 だけ、32 局)を job 1 とし、smoke/<腕>_V<V>/ に出す(protocol の照合なし)。
自動再起動なし。"""
import gzip, json, multiprocessing as mp, os, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repository root (the folder holding eval/, postprocess/, engine/)
CSA = ROOT / 'eval/experiments/collect_start_alpha_20260930'
sys.path.insert(0, str(CSA)); sys.path.insert(0, str(HERE))
assert os.environ.get('ED_START') == '708' and os.environ.get('ED_ALPHA') == '0.01', 'ED_START=708 ED_ALPHA=0.01 を明示'
import run_g5new as G5
import postprocess_ev as EV
import numpy as np
import torch
from scripts.kagg_pfsp3.evaluate_land import play_land_multi

G, R, S = G5.G, G5.R, G5.S
CONTROL = CSA / 'games5' / 'plan_s708a010'
CHASH = G5.SCR / 'hashes5' / 'plan_s708a010'
V = float(os.environ['EV_V'])
KS = {'k4': 4, 'k8': 8, 'ctrl': -1, 'u2': 2, 'u4': 4, 'u2_d0_9': 2, 'u2_d7_20_key': 2, 'u2_d3_9': 2, 'u2_d10_20_nofert_no4q': 2}   # u* = EV21-1u(1 日 1 unit、K=2/4)
OPTS = {'u2_d0_9': dict(days=(0, 9)),
        'u2_d10_20_nofert_no4q': dict(days=(10, 20), target=['STRAWBERRY', 'CARROT', 'WHEAT', 'MILK'], no4q=True),  # 中盤だけ、肥料除外、4 面目の unit 除外
        'u2_d3_9': dict(days=(3, 9)),                                    # 確認用(u2_d0_9 と同じ規則、day 3〜9)                                    # 序盤だけ(ユーザー案)
        'u2_d7_20_key': dict(days=(7, 20), target=['STRAWBERRY', 'MILK', 'MELON', 'CARROT', 'TOMATO', 'FERTILIZER'], min_n=4)}   # ctrl: EV を掛けない計測だけ(smoke の照合用)


class EVTop:
    def __init__(self, inner, base, keys, K):
        self.inner, self.base, self.keys, self.K = inner, base, keys, K
        self.one_unit = False; self.opts = None
        self.mask_version = inner.mask_version; self._batch = inner._batch
        n = len(keys)
        self.st = [EV.GameState() for _ in range(n)]
        self.events = []; self.first = {}; self.pending = {}; self.checks = Counter(); self.mismatch = []
        self.changed = [0] * n; self.seconds = 0.; self.max_sec = 0.
        self.waste_units = [0] * n; self.waste_value = [0.] * n

    def act_batch(self, observations):
        acts = self.inner.act_batch(observations)
        step = int(observations[0]['step']); t0 = time.perf_counter()
        if self.pending:
            for j, (exp, eod_lost, drop_lost) in self.pending.items():
                full = self.base.full(observations[j])
                actual = {k: n for k, n in full['private']['shed'].items() if n > 0}
                self.checks['shed_match' if actual == exp else 'shed_mismatch'] += 1
                self.checks['drop_overflow_units'] += drop_lost
                if actual != exp and len(self.mismatch) < 20:
                    self.mismatch.append(dict(game=j, step=step, expected=exp, actual=actual))
            self.pending = {}
        out = []
        for j, (o, a) in enumerate(zip(observations, acts)):
            full = self.base.full(o)
            assert int(full['step']) == step and full['player'] == self.keys[j][2]
            ev = []
            new = EV.apply_ev(full, S.CONFIG, a, self.st[j], self.K, V, events=ev, one_unit=self.one_unit, opts=self.opts)
            if new is not a:
                self.changed[j] += 1; self.first.setdefault(j, step)
                self.pending[j] = EV.project_shed(full, S.CONFIG, new)
            if step % 24 == 23 and step // 24 <= 28:                 # 日末に捨てる見込み(W9D の DTop と同じ最終行動・同じ公式順)
                _, lost, dl = self.pending[j] if new is not a else EV.project_shed(full, S.CONFIG, new)
                pr = full['market']['prices']
                self.waste_units[j] += sum(lost.values()) + dl; self.waste_value[j] += sum(pr.get(k, 0) * v for k, v in lost.items())
            for e in ev:
                e['game'] = j
            self.events.extend(ev)
            out.append(new)
        dt = time.perf_counter() - t0
        self.seconds += dt; self.max_sec = max(self.max_sec, dt)
        return out


def sources():
    paths = G5.sources()
    for p in [HERE / 'run_ev.py', HERE / 'postprocess_ev.py', ROOT / 'postprocess/postprocess_c1p_w9p.py']:
        paths[str(p.resolve().relative_to(ROOT))] = G.sha(p)
    return paths


def per_game(events, n):
    rows = [Counter() for _ in range(n)]
    for e in events:
        r = rows[e['game']]
        if e['kind'] == 'EV_commit':
            r['commits'] += 1
        elif e['kind'] == 'EV_drop':
            if 'placed' in e:
                r['drops'] += 1; r['placed_units'] += sum(e['placed'].values())
            else:
                r['drop_' + e['result']] += 1
        elif e['kind'] == 'EV_replace':
            r['replaced_steps'] += 1
        elif e['kind'] == 'EV_sell':
            for how, item, k in e['added']:
                r['sell_' + how] += k
                if how != 'no_slot':
                    r['sell_value'] += e['price'][item] * k
        elif e['kind'] == 'EV_error':
            r['errors'] += 1
    return rows


def run(arm, jfirst, jlast, smoke=False):
    assert arm in KS
    if smoke:
        sopp = os.environ.get('EV_SMOKE_OPP', 'mmpqbcw_gen34')
        assert sopp in ('mmpqbcw_gen34', 'sub121_v109')                  # どちらも波の先頭(対照の hash の先頭 32 行に当たる)
        first, count, waves = 174300000, 16, [[sopp]]
        adir = HERE / 'smoke' / f'{arm}_{sopp.split("_")[0]}_V{int(V)}{"_nosell" if EV.NOSELL else ""}{"_keepwork" if EV.KEEPWORK else ""}'; hdir = adir / 'hashes'
        NOCTRL = False
    else:
        PROT = os.environ.get('EV_PROTOCOL', 'protocol.json')
        prot = json.loads((HERE / PROT).read_text())
        assert arm in prot['arms'], (arm, PROT)
        assert prot['V'] == V and not EV.NOSELL
        first, count, waves = prot['seeds']['first'], prot['seeds']['count'], G5.WAVES
        GD = prot.get('games_dir', 'games')                                # 対照の無い seed(protocol_d3_9_more)は games_more/、対照も同じ局を打つ
        adir = HERE / GD / arm; hdir = HERE / ('hashes' if GD == 'games' else 'hashes_' + GD) / arm
        NOCTRL = bool(prot.get('no_control'))
    adir.mkdir(parents=True, exist_ok=True); hdir.mkdir(parents=True, exist_ok=True)
    code = sources()
    if not smoke:
        for p, d in prot['code_sha256'].items():
            assert code.get(p) == d, p
    control = {}
    for f in CONTROL.glob('games_part_*.json'):
        control.update({G.key(g): g for g in json.loads(f.read_text())})
    assert len(control) == 1280
    if NOCTRL:
        control = None
    tagp = f'{jfirst}_{jlast}'
    rules, learner, opps, mods, shas = G5.load()
    G.dump(adir / f'manifest_part_{tagp}.json', dict(arm=arm, K=KS[arm], V=V, jobs=[jfirst, jlast], alpha=R.ALPHA, start_step=G5.START,
           seeds=[first, first + count - 1], waves=waves, pid=os.getpid(), source_sha256=shas, code_sha256=code,
           protocol_sha256=None if smoke else G.sha(HERE / PROT), opts=OPTS.get(arm),
           revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           gpu_before=G.smi('--query-gpu=memory.used,memory.total,power.draw'), plan_procs=R.PROCS, automatic_restart=False))
    seeds_all = list(range(first, first + count)); saved = []; t_start = time.time()
    jobs = [(i, w, names) for i in range(0, count, 16) for w, names in enumerate(waves)]
    with mp.get_context('spawn').Pool(R.PROCS, initializer=R._worker_init) as pool:
        for job, (i, w, names) in enumerate(jobs, 1):
            if not jfirst <= job <= jlast:
                continue
            seeds = seeds_all[i:i + 16]
            if smoke:
                w = [x for x, ws in enumerate(G5.WAVES) if names[0] in ws][0]
            keys = [(n, s, p) for n in names for s in seeds for p in (0, 1)]
            base, top, mtop, _ = G.build(learner, keys, 'pp', rules, mods['gen35'])
            et = EVTop(mtop, base, keys, KS[arm]); et.one_unit = arm.startswith('u'); et.opts = OPTS.get(arm)
            pt = R.PTop(et, keys, G5.START, pool); tick = time.time()
            rows = play_land_multi(pt, {n: opps[n] for n in names}, seeds)
            assert [G.key(g) for g in rows] == keys and base.calls == 719
            tag = f'seed{i:03d}_wave{w}'; target = adir / tag; target.mkdir()
            np.save(hdir / f'{tag}.npy', base.hashes)
            bh = None if control is None else np.load(CHASH / f'{tag}.npy')[:len(keys)]; checked = 0; same = 0
            for j, k in (enumerate(keys) if control is not None else []):
                f = et.first.get(j)
                if f is None:
                    assert (bh[j] == base.hashes[j]).all(), (k, 'unfired game trajectory differs from control')
                    assert all(rows[j][x] == control[k][x] for x in ('own_money', 'opponent_money', 'result')), (k, 'unfired outcome differs')
                    checked += 719; same += 1
                else:
                    assert (bh[j, :f + 1] == base.hashes[j, :f + 1]).all(), (k, f, 'pre-divergence hash mismatch')
                    checked += f + 1
            with gzip.open(target / 'events.jsonl.gz', 'wt') as fz:
                for e in et.events:
                    fz.write(json.dumps(dict(e, key=list(keys[e['game']])), separators=(',', ':'), default=str) + '\n')
            pg = per_game(et.events, len(keys))
            for j, r in enumerate(rows):
                r['waste_units'] = et.waste_units[j]; r['waste_value'] = et.waste_value[j]
                r['arm'] = arm; r['ev_changed_steps'] = et.changed[j]; r['ev_first_step'] = et.first.get(j); r['ev'] = dict(pg[j])
            G.dump(target / 'games.json', rows)
            summ = G.summarize(base, top, mtop); summ.update(R.summarize(pt))
            G.dump(target / 'complete.json', dict(seconds=time.time() - tick, games=len(rows), **summ, ev_checks=dict(et.checks),
                   shed_mismatch_samples=et.mismatch, ev_seconds=et.seconds, ev_max_batch_sec=et.max_sec, games_fired=len(et.first),
                   unfired_equal_control=same, hash_steps_checked=checked, ev_kinds=dict(Counter(e['kind'] for e in et.events))))
            saved.extend(rows); G.dump(adir / f'games_part_{tagp}.json', saved)
            tot = sum(pg, Counter())
            print(json.dumps(dict(arm=arm, job_complete=job, seconds=round(time.time() - tick), wdl=dict(Counter(r['result'] for r in rows)),
                                  fired_games=len(et.first), checks=dict(et.checks), ev=dict(tot), ev_max_batch_sec=round(et.max_sec, 4))),
                  flush=True)
            del base, top, mtop, pt, et; torch.cuda.empty_cache()
    for p, d in code.items():
        assert G.sha(ROOT / p) == d, p
    for n in mods:
        m = getattr(mods[n], '_MP', None)
        if m is not None and hasattr(m, 'FIRE'):
            m.FIRE.clear()
    G.dump(adir / f'status_part_{tagp}.json', dict(status='complete', seconds=time.time() - t_start, games=len(saved)))
    print(json.dumps(dict(part_complete=tagp, arm=arm, games=len(saved), seconds=time.time() - t_start)), flush=True)


if __name__ == '__main__':
    try:
        torch.manual_seed(20260930)
        run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), smoke=len(sys.argv) > 4 and sys.argv[4] == 'smoke')
    except BaseException as e:
        G.dump(HERE / f'failure_{"_".join(sys.argv[1:])}_V{int(V)}.json', dict(type=type(e).__name__, message=str(e),
               traceback=traceback.format_exc(), time=time.time(), argv=sys.argv))
        raise
