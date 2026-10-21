"""gen41 に版 13 の後処理を掛けたときの同一局 A/B(2026-10-01)。既存 harness は凍結扱いで import だけする(編集しない)。
  経路: evening_deposit_20260930/run_d23s.py(D23STop、min_ratio 1.0 = D23S_p1)→ run_ev → collect_start_alpha_20260930/run_g5new(相手の読み込み)
        → gen35_postproc_20260930/run_g35(Base35 / Top35 / build)→ sub121_collect_early_deposit_20260930/run_p_ed(PTop)。
腕:
  A = gen41 の main.py そのまま(網 → _postprocess = _land(_pace(.)))
  B = gen41 + W3+W5+W6+W9'+WT(Top35 → MTop の M5)+ D23S_p1 + 最終日の回収計画(開始 step 708、α=0.01)
      掛ける順序は版 13 の main.py(example_gen35_config3)と同じ: 網 → _postprocess → 土台 → D23S_p1 → 回収計画
学習側の差し替え: run_g35 の GEN35 / GEN35_POLICY_SHA / SCR を gen41 に向ける(gen41 の main.py は gen35 と同一ファイル)。
相手 7 本: sub121_v109, sub122_v4x20, sub123s_gen21, mmpqbcw_gen32, mmpqbcw_gen34(run_g5new と同じ)+ mmpqbcw_gen35(OppPP)
  + sub124(gen37 + 版 12 構成 2。main.py は example_gen35_base_search と同一 → 相手側にも Base35 → Top35 → MTop → PTop を掛ける)。
波 = [sub121,sub122] [sub123s,gen32] [gen34,gen35] [sub124]。job = 16 seed × 波、seed 塊が外側。
使い方: ED_START=708 ED_ALPHA=0.01 EV_V=0 PLAN_PROCS=<n> run_gen41.py stage
        ... run_gen41.py smoke <A|B>                 → smoke/<腕>/(seed 174300000〜015 × sub124 × 両 player = 32 局)
        ... run_gen41.py part <A|B> <k> <K>          → games/<腕>/(job 番号 j で (j-1) % K == k のものを打つ)
自動再起動なし。"""
import gzip, json, multiprocessing as mp, os, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repository root (the folder holding eval/, postprocess/, engine/)
os.environ.setdefault('EV_V', '0')
sys.path.insert(0, str(ROOT / 'eval/experiments/evening_deposit_20260930'))
import run_d23s as RD                                              # RE.EVTop = D23STop、G.sources の鍵を相対パスに
RE = RD.RE; G5 = RE.G5; G = RE.G; R = RE.R; S = RE.S; RM = G.RM; C = G.C
import numpy as np
import torch
from scripts.kagg_pfsp3.evaluate_land import play_land_multi

GEN41 = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'shimishige_teamabcw_20261001/teamabcw_gen41_paceland_liq')
GEN41_POLICY_SHA = 'd184128261351d3a222f284e1db91bee7726d59ade5b3760db7800c34fb0dbcc'
SCR = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'gen41_postproc_20261001')
MM = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'shimishige_teamabcw_gen32_gen34_20260930/extracted')
TS2 = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'team_subs_20261001')
SUB124 = dict(src=TS2 / 'sub124/source', receipt=TS2 / 'sub124/receipt.json', kind='single')
POOL = ['sub121_v109', 'sub122_v4x20', 'sub123s_gen21', 'mmpqbcw_gen32', 'mmpqbcw_gen34', 'mmpqbcw_gen35', 'sub124']
WAVES = [['sub121_v109', 'sub122_v4x20'], ['sub123s_gen21', 'mmpqbcw_gen32'], ['mmpqbcw_gen34', 'mmpqbcw_gen35'], ['sub124']]
SMOKE_OPP = 'sub124'
START = 708

# ---- 学習側を gen41 に、staging を本実験の scratch に(凍結ファイルは書き換えず、module の属性だけ差し替える) ----
G.GEN35 = GEN41; G.GEN35_POLICY_SHA = GEN41_POLICY_SHA; G.SCR = SCR / 'learner'
G5.SCR = SCR / 'opp'
G5.NEW['mmpqbcw_gen35'] = dict(src=MM / 'teamabcw_gen35_paceland_liq', receipt=None, kind='single', pp=True)
S.OPPONENTS = {'sub120_v108': S.OPPONENTS['sub120_v108']}      # 旧 7 本は使わない(sub120 は _pace の照合に要る)
G5.NAMES = [n for n in POOL if n != 'sub124'] + ['sub120_v108']
RD.D23STop.min_ratio = RD.MIN_RATIO['d23s_p1']
assert RD.D23STop.min_ratio == 1.0 and R.ALPHA == 0.01 and G5.START == START


def sources():
    paths = RD.sources()
    paths[str((HERE / 'run_gen41.py').relative_to(ROOT))] = G.sha(HERE / 'run_gen41.py')
    return paths


def load():
    rules, learner, opps, mods, shas = G5.load()
    old = S.SCR; S.SCR = SCR / 'opp'
    try:
        shas['sub124'] = S.verify_sources('sub124', SUB124)
        inf, mod = S.build_agent('sub124', SUB124, 64)
    finally:
        S.SCR = old
    assert hasattr(mod, '_postprocess') and mod._LIQUIDATE_FROM_STEP == 710 and mod._LAND_FROM_STEP == 240
    assert hasattr(mod, '_C1P_HOOK') and hasattr(mod, '_CP_PLAN')    # 版 12 構成 2 の main.py
    inf._batch.repair = lambda a, o: a
    C.install_capture(inf._batch.net)
    opps['sub124'], mods['sub124'] = inf, mod
    del opps['sub120_v108']
    assert set(opps) == set(POOL), sorted(opps)
    return rules, learner, opps, mods, shas


class Base41(G.Base35):
    """Base35 と同じ。_postprocess 後の行動(上の段を掛ける前)を控える。"""
    def act_batch(self, observations):
        acts = super().act_batch(observations)
        self.last = [json.dumps(a, sort_keys=True) for a in acts]
        return acts


class Final:
    """最外殻: 最終行動が Base の行動と違う最初の手、変わった手数、1 回の呼び出しの最大時間。"""
    def __init__(self, inner, base):
        self.inner, self.base = inner, base; self.mask_version = inner.mask_version; self._batch = inner._batch
        self.first = {}; self.changed = Counter(); self.max_sec = 0.; self.max_step = None

    def act_batch(self, observations):
        t0 = time.perf_counter()
        acts = self.inner.act_batch(observations)
        dt = time.perf_counter() - t0
        if dt > self.max_sec:
            self.max_sec, self.max_step = dt, int(observations[0]['step'])
        step = int(observations[0]['step'])
        for j, a in enumerate(acts):
            if json.dumps(a, sort_keys=True) != self.base.last[j]:
                self.first.setdefault(j, step); self.changed[j] += 1
        return acts


def chain(inf, keys, rules, mod, pool, d23s):
    """Base35 → Top35 → MTop(M5)→ [D23S_p1] → PTop(708、α=0.01)。"""
    base = Base41(inf, keys, 719, mod)
    top = G.Top35(base, base, keys, G.SPEC, rules)
    mtop = RM.MTop(top, keys, G.SPEC, rules)
    dt = RD.D23STop(mtop, base, keys, 0) if d23s else None
    pt = R.PTop(dt if d23s else mtop, keys, START, pool)
    return base, top, mtop, dt, pt


def run(mode, arm, k=0, K=1):
    assert arm in ('A', 'B')
    if mode == 'smoke':
        first, count, waves = 174300000, 16, [[SMOKE_OPP]]
        adir = HERE / 'smoke' / arm; hdir = SCR / 'smoke_hashes' / arm
        prot = None
    else:
        prot = json.loads((HERE / 'protocol.json').read_text())
        first, count, waves = prot['seeds']['first'], prot['seeds']['count'], WAVES
        assert prot['waves'] == WAVES and prot['opponents'] == POOL
        adir = HERE / 'games' / arm; hdir = SCR / 'hashes' / arm
    adir.mkdir(parents=True, exist_ok=True); hdir.mkdir(parents=True, exist_ok=True)
    code = sources()
    if prot is not None:
        for p, d in prot['code_sha256'].items():
            assert code.get(p) == d, p
    tagp = f'{k}of{K}'
    rules, learner, opps, mods, shas = load()
    G.dump(adir / f'manifest_part_{tagp}.json', dict(arm=arm, part=[k, K], seeds=[first, first + count - 1], waves=waves, pid=os.getpid(),
           source_sha256=shas, code_sha256=code, protocol_sha256=None if prot is None else G.sha(HERE / 'protocol.json'),
           revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), alpha=R.ALPHA, start_step=START,
           d23s_min_ratio=RD.D23STop.min_ratio, gpu_before=G.smi('--query-gpu=memory.used,memory.total,power.draw'), plan_procs=R.PROCS,
           automatic_restart=False))
    G.dump(adir / f'status_part_{tagp}.json', dict(status='running', pid=os.getpid(), started=time.time()))
    seeds_all = list(range(first, first + count)); saved = []; t_start = time.time()
    jobs = [(i, w, names) for i in range(0, count, 16) for w, names in enumerate(waves)]
    with mp.get_context('spawn').Pool(R.PROCS, initializer=R._worker_init) as pool:
        for job, (i, w, names) in enumerate(jobs, 1):
            if (job - 1) % K != k:
                continue
            seeds = seeds_all[i:i + 16]
            if mode == 'smoke':
                w = WAVES.index([SMOKE_OPP])
            keys = [(n, s, p) for n in names for s in seeds for p in (0, 1)]
            for n in names:
                if isinstance(opps[n], G5.OppPP):
                    opps[n].changed.clear()
            if arm == 'A':
                base = Base41(learner, keys, 719, mods['gen35']); top = mtop = dt = pt = None; agent = Final(base, base)
            else:
                base, top, mtop, dt, pt = chain(learner, keys, rules, mods['gen35'], pool, True); agent = Final(pt, base)
            oc = {}; play = {}
            for n in names:
                if n == 'sub124':
                    okeys = [(n, s, 1 - p) for s in seeds for p in (0, 1)]
                    oc[n] = chain(opps[n], okeys, rules, mods[n], pool, False); play[n] = oc[n][4]
                else:
                    play[n] = opps[n]
            tick = time.time()
            rows = play_land_multi(agent, play, seeds)
            assert [G.key(g) for g in rows] == keys and base.calls == 719
            for n, c in oc.items():
                assert c[0].calls == 719
            tag = f'seed{i:03d}_wave{w}'; target = adir / tag; target.mkdir()
            np.save(hdir / f'{tag}.npy', base.hashes)
            summ = dict(postprocess_gen41=dict(base.pp), chain_max_batch_sec=agent.max_sec, chain_max_batch_step=agent.max_step,
                        changed_games=len(agent.first))
            evs = []
            if arm == 'B':
                summ.update(G.summarize(base, top, mtop)); summ.update(R.summarize(pt))
                summ.update(d23s_seconds=dt.seconds, d23s_max_batch_sec=dt.max_sec, d23s_checks=dict(dt.checks), d23s_games_fired=len(dt.first),
                            d23s_kinds=dict(Counter(e['kind'] for e in dt.events)), shed_mismatch_samples=dt.mismatch)
                evs = top.events + mtop.events + dt.events + pt.events
                pg = RD.per_game(dt.events, len(keys))
            for n, c in oc.items():
                summ[f'opp_{n}'] = dict(G.summarize(c[0], c[1], c[2]), **R.summarize(c[4]))
            with gzip.open(target / 'events.jsonl.gz', 'wt') as fz:
                for e in evs:
                    fz.write(json.dumps(dict(e, key=list(keys[e['game']])), separators=(',', ':'), default=str) + '\n')
            for j, r in enumerate(rows):
                r['arm'] = arm; r['first_change_step'] = agent.first.get(j); r['changed_steps'] = agent.changed[j]
                if arm == 'B':
                    r['d23s_first_step'] = dt.first.get(j); r['d23s_changed_steps'] = dt.changed[j]; r['d23s'] = dict(pg[j])
                    r['plan_first_step'] = pt.first.get(j)
            G.dump(target / 'games.json', rows)
            G.dump(target / 'complete.json', dict(seconds=time.time() - tick, games=len(rows), **summ,
                   opp_postprocess={n: dict(v) for n, v in G5.opp_changed(opps).items() if n in names},
                   hashes=str(hdir / f'{tag}.npy'), hashes_sha256=G.sha(hdir / f'{tag}.npy')))
            saved.extend(rows); G.dump(adir / f'games_part_{tagp}.json', saved)
            print(json.dumps(dict(arm=arm, job_complete=job, seconds=round(time.time() - tick), wdl=dict(Counter(r['result'] for r in rows)),
                                  changed_games=len(agent.first), chain_max_batch_sec=round(agent.max_sec, 3),
                                  plan_max_sec=getattr(pt, 'max_sec', None), d23s_fired=None if dt is None else len(dt.first))), flush=True)
            del base, top, mtop, dt, pt, agent, oc, play; torch.cuda.empty_cache()
    for p, d in code.items():
        assert G.sha(ROOT / p) == d, p
    G.dump(adir / f'status_part_{tagp}.json', dict(status='complete', seconds=time.time() - t_start, games=len(saved)))
    print(json.dumps(dict(part_complete=tagp, arm=arm, games=len(saved), seconds=time.time() - t_start)), flush=True)


if __name__ == '__main__':
    try:
        torch.manual_seed(20260930)
        if sys.argv[1] == 'stage':
            load(); print(json.dumps(dict(staged=True, sources=len(sources()))), flush=True)
        elif sys.argv[1] == 'smoke':
            run('smoke', sys.argv[2])
        else:
            assert sys.argv[1] == 'part'
            run('part', sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    except BaseException as e:
        G.dump(HERE / f'failure_{"_".join(sys.argv[1:])}.json', dict(type=type(e).__name__, message=str(e), traceback=traceback.format_exc(),
               time=time.time(), argv=sys.argv))
        raise
