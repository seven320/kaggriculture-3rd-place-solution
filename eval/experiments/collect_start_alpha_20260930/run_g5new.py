"""(run_g12.py の複製: 相手を新しく足した 5 本だけに絞る(ユーザー指示 22:33)。波 = [sub121,sub122] [sub123s,gen32] [gen34]、
出力 games5/、hash は scratch の hashes5/、protocol は protocol_g5new.json。相手の読み込み(12 本すべて)と腕の掛け方は run_g12.py と同じ。)
元の説明: gen35 の全局対局、相手を最新版まで広げた 12 本(ユーザー指示 2026-09-30 22:0x)。
腕:
  base = gen35 の main.py そのまま(網 → _postprocess = _land(_pace(.))、gen35_postproc_20260930/run_g35.py の base 腕と同じ)
  plan = gen35 + 土台(W3/W5/W6 → W9 → WT → M5、run_g35.py の pp 腕)+ 最終日の回収計画(run_p_ed.PTop + planner_ed、
         開始 step = ED_START、α = ED_ALPHA)。run_g35_start.py の腕と同じ掛け方
相手 12 本: run_g35.py の 7 本(sub115_v103, sub116_v104, sub117_gen41, sub117_sw2_v105, sub118_v106, sub119_v107, sub120_v108)
  + sub121_v109, sub122_v4x20, sub123s_gen21, mmpqbcw_gen32, mmpqbcw_gen34(いずれも提出物の main.py そのまま)。
  sub121/sub122 の main.py は sub120 と同一(repair = _pace = _liq、既存 7 本と同じ扱い)。
  sub123s / gen32 / gen34 の main.py は gen35 と同一の _postprocess = _land(_pace(.)) を持つ → 網の repair を恒等にして、
  公式観測(pool.debug)に _postprocess を掛ける(OppPP、gen35 側の Base35 と同じ扱い)。
条件: Rust・fp32・TF32 無効・学習側 64 詰め・相手 32(16 seed × 両 player)。job = 16 seed × 波(相手 2 本)。
使い方: [ED_START=708] ED_ALPHA=0.01 run_g12.py smoke <base|plan>     → smoke12/<腕>/(seed 1 個 × 12 相手 × 両 player は batch 32 に合わないので 16 seed × 波 1 本)
        [ED_START=708] ED_ALPHA=0.01 run_g12.py part <base|plan> <first> <last> → games12/<腕>/(job 番号は 1 始まり)
自動再起動なし。"""
import gzip, json, multiprocessing as mp, os, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repository root (the folder holding eval/, postprocess/, engine/)
GP = ROOT / 'eval/experiments/gen35_postproc_20260930'
sys.path.insert(0, str(GP))
import run_g35 as G
ED = ROOT / 'eval/experiments/sub121_collect_early_deposit_20260930'
sys.path.insert(0, str(ED))
import run_p_ed as R
import numpy as np
import torch
from scripts.kagg_pfsp3.evaluate_land import play_land_multi

S = G.S
START = int(os.environ.get('ED_START', '708'))
TS = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'team_subs_20260930')
MM = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'shimishige_teamabcw_gen32_gen34_20260930/extracted')
NEW = {
    'sub121_v109': dict(src=TS / 'sub121_v109/source', receipt=TS / 'sub121_v109/receipt.json', kind='single', pp=False),
    'sub122_v4x20': dict(src=TS / 'sub122/source', receipt=TS / 'sub122/receipt.json', kind='single', pp=False),
    'sub123s_gen21': dict(src=TS / 'sub123s/source', receipt=TS / 'sub123s/receipt.json', kind='single', pp=True),
    'mmpqbcw_gen32': dict(src=MM / 'teamabcw_gen32_paceland_liq', receipt=None, kind='single', pp=True),
    'mmpqbcw_gen34': dict(src=MM / 'teamabcw_gen34_paceland_liq', receipt=None, kind='single', pp=True),
}
WAVES = [['sub115_v103', 'sub116_v104'], ['sub117_gen41', 'sub117_sw2_v105'], ['sub118_v106', 'sub119_v107'],
         ['sub120_v108', 'sub121_v109'], ['sub122_v4x20', 'sub123s_gen21'], ['mmpqbcw_gen32', 'mmpqbcw_gen34']]
NAMES = [n for w in WAVES for n in w]
WAVES = [['sub121_v109', 'sub122_v4x20'], ['sub123s_gen21', 'mmpqbcw_gen32'], ['mmpqbcw_gen34']]  # 本ファイルで打つ波(新 5 本)
SCR = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'collect_start_alpha_20260930')


class OppPP:
    """相手の網(repair 恒等)の行動に、提出物の main.py の _postprocess を公式観測(pool.debug)で掛ける。"""
    def __init__(self, inner, mod):
        self.inner, self.mod = inner, mod; self.mask_version = inner.mask_version; self.changed = Counter()

    def act_batch(self, observations):
        raw = self.inner.act_batch(observations); snaps = {}; out = []
        step = int(observations[0]['step'])
        for o, a in zip(observations, raw):
            pool, i = o['_native_pool'], o['_native_index']
            if id(pool) not in snaps:
                snaps[id(pool)] = pool.debug()
            full = snaps[id(pool)][i // 2]['observations'][i % 2]
            assert int(full['step']) == step
            new = self.mod._postprocess(json.loads(json.dumps(a)), full)
            if (new.get('market') or []) != (a.get('market') or []):
                self.changed['liq' if step >= 710 else 'pace_or_land'] += 1
            out.append(new)
        return out


def load():
    rules, learner, opps, mods, shas = G.load()
    SCR.mkdir(parents=True, exist_ok=True)
    old = S.SCR; S.SCR = SCR                                  # 新しい相手の staging は本実験の scratch に置く
    try:
        for n, s in NEW.items():
            shas[n] = S.verify_sources(n, s)
            inf, mod = S.build_agent(n, s, 32)
            if s['pp']:
                assert hasattr(mod, '_postprocess') and mod._LIQUIDATE_FROM_STEP == 710 and mod._LAND_FROM_STEP == 240, n
                inf._batch.repair = lambda a, o: a
                inf = OppPP(inf, mod)
            else:
                assert mod._pace.__code__.co_code == mods['sub120_v108']._pace.__code__.co_code, n
            opps[n], mods[n] = inf, mod
    finally:
        S.SCR = old
    assert set(opps) == set(NAMES), sorted(opps)
    return rules, learner, opps, mods, shas


def build(learner, keys, arm, rules, mod, pool):
    if arm == 'base':
        base, _, _, agent = G.build(learner, keys, 'base', rules, mod)
        return base, None, None, None, agent
    base, top, mtop, _ = G.build(learner, keys, 'pp', rules, mod)
    pt = R.PTop(mtop, keys, START, pool)
    return base, top, mtop, pt, pt


def arm_name(arm):
    return 'base' if arm == 'base' else f'plan_s{START}a{round(R.ALPHA * 1000):03d}'


def sources():
    paths = G.sources()
    for p in [HERE / 'run_g5new.py', ED / 'run_p_ed.py', ED / 'planner_ed.py']:
        paths[str(p.resolve().relative_to(ROOT))] = G.sha(p)
    return paths


def opp_changed(opps):
    return {n: dict(o.changed) for n, o in opps.items() if isinstance(o, OppPP)}


def run(mode, arm, jfirst=None, jlast=None):
    assert arm in ('base', 'plan')
    name = arm_name(arm)
    if mode == 'smoke':
        first, count, waves = 174300000, 16, [WAVES[-2]]
        adir = HERE / 'smoke12' / name; adir.mkdir(parents=True, exist_ok=False)
        hdir = SCR / 'smoke_hashes' / name; hdir.mkdir(parents=True, exist_ok=True)
        jfirst, jlast = 1, 1
    else:
        prot = json.loads((HERE / 'protocol_g5new.json').read_text())
        first, count, waves = prot['seeds']['first'], prot['seeds']['count'], WAVES
        adir = HERE / 'games5' / name; adir.mkdir(parents=True, exist_ok=True)
        hdir = SCR / 'hashes5' / name; hdir.mkdir(parents=True, exist_ok=True)
    code = sources()
    if mode != 'smoke':
        for p, d in prot['code_sha256'].items():
            assert code.get(p) == d, p
    tagp = f'{jfirst}_{jlast}'
    rules, learner, opps, mods, shas = load()
    G.dump(adir / f'manifest_part_{tagp}.json', dict(arm=name, jobs=[jfirst, jlast], alpha=R.ALPHA if arm == 'plan' else None,
           start_step=START if arm == 'plan' else None, seeds=[first, first + count - 1], waves=waves, pid=os.getpid(),
           source_sha256=shas, code_sha256=code, revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           gpu_before=G.smi('--query-gpu=memory.used,memory.total,power.draw'), plan_procs=R.PROCS, automatic_restart=False))
    G.dump(adir / f'status_part_{tagp}.json', dict(status='running', pid=os.getpid(), started=time.time()))
    seeds_all = list(range(first, first + count)); saved = []; t_start = time.time()
    jobs = [(i, w, names) for i in range(0, count, 16) for w, names in enumerate(waves)]
    with mp.get_context('spawn').Pool(R.PROCS if arm == 'plan' else 1, initializer=R._worker_init) as pool:
        for job, (i, w, names) in enumerate(jobs, 1):
            if not jfirst <= job <= jlast:
                continue
            seeds = seeds_all[i:i + 16]
            keys = [(n, s, p) for n in names for s in seeds for p in (0, 1)]
            for n in names:
                if isinstance(opps[n], OppPP):
                    opps[n].changed.clear()
            base, top, mtop, pt, agent = build(learner, keys, arm, rules, mods['gen35'], pool); tick = time.time()
            rows = play_land_multi(agent, {n: opps[n] for n in names}, seeds)
            assert [G.key(g) for g in rows] == keys and base.calls == 719
            tag = f'seed{i:03d}_wave{w}'; target = adir / tag; target.mkdir()
            np.save(hdir / f'{tag}.npy', base.hashes)
            summ = G.summarize(base, top, mtop)
            if pt is not None:
                summ.update(R.summarize(pt)); summ['plan_first_step'] = {json.dumps(list(keys[k])): v for k, v in pt.first.items()}
            with gzip.open(target / 'events.jsonl.gz', 'wt') as fz:
                for e in ([] if top is None else top.events + mtop.events + pt.events):
                    fz.write(json.dumps(dict(e, key=list(keys[e['game']])), separators=(',', ':'), default=str) + '\n')
            for r in rows:
                r['arm'] = name
            G.dump(target / 'games.json', rows)
            G.dump(target / 'complete.json', dict(seconds=time.time() - tick, games=len(rows), **summ,
                   opp_postprocess={n: v for n, v in opp_changed(opps).items() if n in names},
                   hashes=str(hdir / f'{tag}.npy'), hashes_sha256=G.sha(hdir / f'{tag}.npy')))
            saved.extend(rows); G.dump(adir / f'games_part_{tagp}.json', saved)
            print(json.dumps(dict(arm=name, job_complete=job, seconds=round(time.time() - tick), wdl=dict(Counter(r['result'] for r in rows)),
                                  opp_pp={n: v for n, v in opp_changed(opps).items() if n in names},
                                  plan_max_sec=getattr(pt, 'max_sec', None))), flush=True)
            del base, top, mtop, pt, agent; torch.cuda.empty_cache()
    for p, d in code.items():
        assert G.sha(ROOT / p) == d, p
    for n in mods:
        m = getattr(mods[n], '_MP', None)
        if m is not None and hasattr(m, 'FIRE'):
            m.FIRE.clear()
    G.dump(adir / f'status_part_{tagp}.json', dict(status='complete', seconds=time.time() - t_start, games=len(saved)))
    print(json.dumps(dict(part_complete=tagp, arm=name, games=len(saved), seconds=time.time() - t_start)), flush=True)


if __name__ == '__main__':
    try:
        torch.manual_seed(20260930)
        if sys.argv[1] == 'smoke':
            run('smoke', sys.argv[2])
        else:
            assert sys.argv[1] == 'part'
            run('part', sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    except BaseException as e:
        G.dump(HERE / f'failure_g5new_{"_".join(sys.argv[1:])}.json', dict(type=type(e).__name__, message=str(e), traceback=traceback.format_exc(),
               time=time.time(), argv=sys.argv))
        raise
