"""gen70(最終提出 sub125 / sub126 の方策)に gen41 と同じ後処理の同一局 A/B を掛け直す(2026-10-17)。
gen41_postproc_20261001/run_gen41.py(凍結)を import し、相手・波・seed・sub124 側の後処理・D23S_p1 の設定はそのまま使う。
学習側だけを gen70 に差し替える(module の属性を上書きするだけ。既存ファイルは編集しない。run_gen41_c.py と同じやり方)。
  学習側の 7 ファイル: gen41 の main.py / model_ptr_infer.py / model_tf.py / obs_feat.py / cfg.json / stock_target.json + sub125 の policy.pt
  ($KAGG_SCRATCH/gen70_postproc_20261017/gen70_src/、sha は protocol.json の learner)。
腕(掛ける順序は sub125 / sub126 の main.py の agent() と同じ):
  A = 網 → _postprocess(= _land(_pace(.)))                                  後処理なし
  C = A → W3+W5+W6+W9'+WT(Top35 → MTop M5)→ 最終日の回収計画(708、α=0.01)   = 版 12
  B = A → Top35 → MTop → D23S_p1 → 回収計画                                  = 版 13
  D = B → MELON16(sub125/source/melon16.py をそのまま読み込む)               = sub125 の agent()
  E = D → melon_replant(sub126/source/melon_replant.py をそのまま読み込む)   = sub126 の agent()
  MELON16 / melon_replant には main.py と同じく「直前の段の行動」と公式観測(Base の pool.debug、上の段と同じ物)を渡す。
使い方: ED_START=708 ED_ALPHA=0.01 EV_V=0 PLAN_PROCS=<n> run_gen70.py stage          → 学習側・相手の staging だけ(並列起動の前に 1 回)
        ... run_gen70.py smoke <A|C|B|D|E>   → smoke/<腕>/(sub124 × seed 174300000〜015 × 両 player = 32 局)
                                               + 先頭 2 seed × 両 player の 4 局で、全 719 手の最終行動を sub125 / sub126 の main.py の
                                               agent()(腕に無い段は恒等に差し替え)を同じ公式観測に掛けた行動と比べる
        ... run_gen70.py part <腕> <k> <K>   → games/<腕>/(job 番号 j で (j-1) % K == k のものを打つ)
自動再起動なし。"""
import gzip, importlib.util, json, multiprocessing as mp, os, subprocess, sys, time, traceback, types
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repository root (the folder holding eval/, postprocess/, engine/)
GEN41_DIR = ROOT / 'eval/experiments/gen41_postproc_20261001'
sys.path.insert(0, str(GEN41_DIR))
import run_gen41 as RG                                             # 凍結。import 時に相手 7 本・D23S_p1(min_ratio 1.0)・ED_START/ED_ALPHA を確かめる
RD, RE, G5, G, R, S, RM = RG.RD, RG.RE, RG.G5, RG.G, RG.R, RG.S, RG.RM
import numpy as np
import torch
from scripts.kagg_pfsp3.evaluate_land import play_land_multi

ARMS = ('A', 'C', 'B', 'D', 'E')
SCR = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'gen70_postproc_20261017')
GEN70 = SCR / 'gen70_src'
GEN70_POLICY_SHA = '76ec119f8513c8b728fbb0501b2839a0fe4556fdba96bbf6b8bd7e93b007f23c'
GEN41_SRC = RG.GEN41                                               # 学習側 6 ファイル(policy.pt 以外)の出どころ
GEN41_MAIN_SHA = '01968cd3f005bba338e84a293a6114c61fc845e07f07a6aa8ab052402419b2a7'
SUB125 = RG.TS2 / 'sub125/source'
SUB126 = RG.TS2 / 'sub126/source'
SUB_FILES = {SUB125: ('main.py', 'cfg.json', 'model_ptr_infer.py', 'model_tf.py', 'obs_feat.py', 'stock_target.json', 'policy.pt',
                      'postprocess_c1p_w9p.py', 'collect_plan_d29.py', 'postprocess_d23s_p1.py', 'melon16.py'),
             SUB126: ('main.py', 'cfg.json', 'model_ptr_infer.py', 'model_tf.py', 'obs_feat.py', 'stock_target.json', 'policy.pt',
                      'postprocess_c1p_w9p.py', 'collect_plan_d29.py', 'postprocess_d23s_p1.py', 'melon16.py', 'melon_replant.py')}
PARITY_SEEDS = int(os.environ.get('G70_PARITY_SEEDS', '2'))        # smoke の提出物 agent() との全手照合に使う先頭 seed 数(× 両 player)

# ---- 学習側を gen70 に、staging を本実験の scratch に(凍結ファイルは書き換えず、module の属性だけ差し替える) ----
RG.GEN41 = GEN70; RG.GEN41_POLICY_SHA = GEN70_POLICY_SHA; RG.SCR = SCR       # RG.load の sub124 staging は RG.SCR / 'opp'
G.GEN35 = GEN70; G.GEN35_POLICY_SHA = GEN70_POLICY_SHA; G.SCR = SCR / 'learner'
G5.SCR = SCR / 'opp'
assert RD.D23STop.min_ratio == 1.0 and R.ALPHA == 0.01 and G5.START == RG.START == 708


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


MELON16 = _load('gen70_melon16', SUB125 / 'melon16.py')            # 状態なし・import なし(22 行)
MREP = _load('gen70_melon_replant', SUB126 / 'melon_replant.py')    # 状態なし・import なし(22 行)


def _scratch_key(p):
    """Key for files under the scratch root: '$KAGG_SCRATCH/<relative path>' (no machine-specific absolute path in the record)."""
    return '$KAGG_SCRATCH/' + os.path.relpath(os.path.abspath(p), os.path.abspath(os.environ.get('KAGG_SCRATCH', 'scratch')))


def sources():
    paths = RG.sources()
    paths[str((HERE / 'run_gen70.py').relative_to(ROOT))] = G.sha(HERE / 'run_gen70.py')
    for f in S.FILES:
        paths[_scratch_key(GEN70 / f)] = G.sha(GEN70 / f)
    for d, fs in SUB_FILES.items():
        for f in fs:
            paths[_scratch_key(d / f)] = G.sha(d / f)
    return paths


def load():
    got = {f: G.sha(GEN70 / f) for f in S.FILES}
    assert got == dict({f: G.sha(GEN41_SRC / f) for f in S.FILES if f != 'policy.pt'}, **{'policy.pt': GEN70_POLICY_SHA}), got
    assert got['main.py'] == GEN41_MAIN_SHA
    assert G.sha(SUB125 / 'policy.pt') == GEN70_POLICY_SHA == G.sha(SUB126 / 'policy.pt')
    assert G.sha(SUB125 / 'melon16.py') == G.sha(SUB126 / 'melon16.py')
    SCR.mkdir(parents=True, exist_ok=True)
    return RG.load()


class Overlay:
    """提出物の main.py の最外段(MELON16 / melon_replant)。fn(obs, action) に公式観測と直前の段の行動を渡す。行動が変わった手を控える。"""
    def __init__(self, inner, base, keys, fn):
        self.inner, self.base, self.keys, self.fn = inner, base, keys, fn
        self.mask_version = inner.mask_version; self._batch = inner._batch
        self.first = {}; self.changed = Counter(); self.added = Counter(); self.samples = []; self.seconds = 0.

    def act_batch(self, observations):
        acts = self.inner.act_batch(observations)
        step = int(observations[0]['step']); t0 = time.perf_counter(); out = []
        for j, (o, a) in enumerate(zip(observations, acts)):
            full = self.base.full(o)
            assert int(full['step']) == step and full['player'] == self.keys[j][2]
            new = self.fn(full, a)
            if new is not a and json.dumps(new, sort_keys=True) != json.dumps(a, sort_keys=True):
                self.changed[j] += 1; self.first.setdefault(j, step)
                old_m, new_m = a.get('market') or [], new.get('market') or []
                self.added[j] += sum(int(x[2]) for x in new_m[len(old_m):] if len(x) >= 3 and x[0] == 'BUY_SEED')
                if len(self.samples) < 20:
                    self.samples.append(dict(game=j, step=step, before=a, after=new))
            out.append(new)
        self.seconds += time.perf_counter() - t0
        return out


class Cap:
    """smoke 用の最外殻: 選んだ局の各手で、段を掛ける前の公式観測(pool.debug)と最終行動を控える(行動は変えない)。"""
    def __init__(self, inner, sel):
        self.inner, self.sel = inner, sel; self.mask_version = inner.mask_version; self._batch = inner._batch
        self.obs = {j: [] for j in sel}; self.act = {j: [] for j in sel}

    def act_batch(self, observations):
        snaps = {}
        for j in self.sel:
            pool, i = observations[j]['_native_pool'], observations[j]['_native_index']
            if id(pool) not in snaps:
                snaps[id(pool)] = pool.debug()
            self.obs[j].append(json.dumps(snaps[id(pool)][i // 2]['observations'][i % 2]))
        acts = self.inner.act_batch(observations)
        for j in self.sel:
            self.act[j].append(json.dumps(acts[j]))
        return acts


def build(arm, inf, keys, rules, mod, pool):
    """腕の段を組む。返り値: base, top, mtop, dt, pt, m16, mrep, 最外段。"""
    if arm == 'A':                                                  # run_gen41.run の腕 A と同じ
        base = RG.Base41(inf, keys, 719, mod)
        return base, None, None, None, None, None, None, base
    base, top, mtop, dt, pt = RG.chain(inf, keys, rules, mod, pool, arm != 'C')   # Base41 → Top35 → MTop →[D23S_p1]→ PTop(gen41 の B / C と同じ)
    m16 = Overlay(pt, base, keys, MELON16.apply) if arm in ('D', 'E') else None
    mrep = Overlay(m16, base, keys, MREP.apply) if arm == 'E' else None
    return base, top, mtop, dt, pt, m16, mrep, (mrep or m16 or pt)


# smoke の照合に使う提出物と、腕に無い段(恒等に差し替える main.py の名前)
REF = {'A': (SUB125, ('_C1P_HOOK', '_D23S', '_CP_PLAN', '_MELON16')), 'C': (SUB125, ('_D23S', '_MELON16')), 'B': (SUB125, ('_MELON16',)),
       'D': (SUB125, ()), 'E': (SUB126, ())}
IDENT = {'_C1P_HOOK': types.SimpleNamespace(apply=lambda obs, config, action: action),
         '_D23S': types.SimpleNamespace(apply=lambda obs, config, action: action),
         '_CP_PLAN': lambda obs, action: action,
         '_MELON16': types.SimpleNamespace(apply=lambda obs, action: action)}
POST_MODULES = ('postprocess_c1p_w9p', 'collect_plan_d29', 'postprocess_d23s_p1', 'melon16', 'melon_replant')
# 10-17 修正(smoke try 1 の parity 不一致): 網・特徴の module も提出物のフォルダから読む。修正前は先に読まれていた
# masspeaks の BC kit の原本(original)の obs_feat(nmb の印なし)を使っていた。main.py が自フォルダから import する module はこれで全部。
NET_MODULES = ('obs_feat', 'model_tf', 'model_ptr_infer')


def submission_agent(arm):
    src, stubs = REF[arm]
    for n in NET_MODULES + POST_MODULES:                            # 相手・他の網が先に読んだ同名 module を使わない
        sys.modules.pop(n, None)
    sys.path.insert(0, str(src))
    try:
        mod = _load(f'gen70_ref_{src.parent.name}_main', src / 'main.py')
    finally:
        sys.path.remove(str(src))
    for n in NET_MODULES + POST_MODULES:
        if n == 'melon_replant' and src != SUB126:
            assert n not in sys.modules, sys.modules[n].__file__
            continue
        assert n in sys.modules, n
        assert Path(sys.modules[n].__file__).resolve().parent == src.resolve(), (n, sys.modules[n].__file__)
    import inspect
    for f in (inspect.getfile(mod.featurize), inspect.getfile(mod.to_action), mod.M.__file__, mod._MP.__file__, inspect.getfile(type(mod._NET))):
        assert Path(f).resolve().parent == src.resolve(), f
    for n in stubs:
        setattr(mod, n, IDENT[n])
    return mod, stubs


def parity(arm, cap, keys):
    """harness の最終行動と、同じ公式観測に提出物の agent()(CPU、1 局ずつ step 0 から順に)を掛けた行動を全手で比べる。"""
    mod, stubs = submission_agent(arm)
    tot = 0; mism = []; t0 = time.time()
    for j in cap.sel:
        if '_CP_PLAN' not in stubs:
            mod._CP_PLAN = mod._CP.CollectPlanner(start_step=708, alpha=0.01)      # main.py 42 行目と同じ引数、1 局 1 インスタンス
        assert len(cap.obs[j]) == len(cap.act[j]) == 719
        for so, sa in zip(cap.obs[j], cap.act[j]):
            o = json.loads(so)
            a = json.loads(json.dumps(mod.agent(o, S.CONFIG))); h = json.loads(sa); tot += 1
            if a != h:
                mism.append(dict(key=list(keys[j]), step=int(o['step']), harness=h, submission=a))
    return dict(reference=str(REF[arm][0] / 'main.py'), identity_stages=list(stubs), games=[list(keys[j]) for j in cap.sel],
                checked=tot, mismatch=len(mism), samples=mism[:10], seconds=round(time.time() - t0, 1))


def run(mode, arm, k=0, K=1):
    assert arm in ARMS
    if mode == 'smoke':
        first, count, waves = 174300000, 16, [[RG.SMOKE_OPP]]
        adir = HERE / 'smoke' / arm; hdir = SCR / 'smoke_hashes' / arm; prot = None
    else:
        prot = json.loads((HERE / 'protocol.json').read_text())
        first, count, waves = prot['seeds']['first'], prot['seeds']['count'], RG.WAVES
        assert prot['waves'] == RG.WAVES and prot['opponents'] == RG.POOL and arm in prot['arms']
        assert {f: G.sha(GEN70 / f) for f in S.FILES} == prot['learner']['file_sha256']
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
           revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), alpha=R.ALPHA, start_step=RG.START,
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
                w = RG.WAVES.index([RG.SMOKE_OPP])
            keys = [(n, s, p) for n in names for s in seeds for p in (0, 1)]
            for n in names:
                if isinstance(opps[n], G5.OppPP):
                    opps[n].changed.clear()
            base, top, mtop, dt, pt, m16, mrep, outer = build(arm, learner, keys, rules, mods['gen35'], pool)
            agent = RG.Final(outer, base)
            cap = None
            if mode == 'smoke':
                cap = Cap(agent, [j for j, kk in enumerate(keys) if kk[1] < seeds[0] + PARITY_SEEDS]); player = cap
            else:
                player = agent
            oc = {}; play = {}
            for n in names:
                if n == 'sub124':
                    okeys = [(n, s, 1 - p) for s in seeds for p in (0, 1)]
                    oc[n] = RG.chain(opps[n], okeys, rules, mods[n], pool, False); play[n] = oc[n][4]
                else:
                    play[n] = opps[n]
            tick = time.time()
            rows = play_land_multi(player, play, seeds)
            assert [G.key(g) for g in rows] == keys and base.calls == 719
            for n, c in oc.items():
                assert c[0].calls == 719
            tag = f'seed{i:03d}_wave{w}'; target = adir / tag; target.mkdir()
            np.save(hdir / f'{tag}.npy', base.hashes)
            summ = dict(postprocess_gen70=dict(base.pp), chain_max_batch_sec=agent.max_sec, chain_max_batch_step=agent.max_step,
                        changed_games=len(agent.first), play_seconds=time.time() - tick)
            evs = []
            if top is not None:
                summ.update(G.summarize(base, top, mtop)); summ.update(R.summarize(pt))
                evs = top.events + mtop.events + pt.events
            if dt is not None:
                summ.update(d23s_seconds=dt.seconds, d23s_max_batch_sec=dt.max_sec, d23s_checks=dict(dt.checks), d23s_games_fired=len(dt.first),
                            d23s_kinds=dict(Counter(e['kind'] for e in dt.events)), shed_mismatch_samples=dt.mismatch)
                evs = evs + dt.events
                pg = RD.per_game(dt.events, len(keys))
            for name, ov in (('melon16', m16), ('melon_replant', mrep)):
                if ov is not None:
                    summ[name] = dict(games_changed=len(ov.first), steps_changed=sum(ov.changed.values()), seed_units_added=sum(ov.added.values()),
                                      seconds=ov.seconds, samples=ov.samples[:5])
            for n, c in oc.items():
                summ[f'opp_{n}'] = dict(G.summarize(c[0], c[1], c[2]), **R.summarize(c[4]))
            if cap is not None:
                summ['parity_submission'] = parity(arm, cap, keys)
            with gzip.open(target / 'events.jsonl.gz', 'wt') as fz:
                for e in evs:
                    fz.write(json.dumps(dict(e, key=list(keys[e['game']])), separators=(',', ':'), default=str) + '\n')
            for j, r in enumerate(rows):
                r['arm'] = arm; r['first_change_step'] = agent.first.get(j); r['changed_steps'] = agent.changed[j]
                if pt is not None:
                    r['plan_first_step'] = pt.first.get(j)
                if dt is not None:
                    r['d23s_first_step'] = dt.first.get(j); r['d23s_changed_steps'] = dt.changed[j]; r['d23s'] = dict(pg[j])
                if m16 is not None:
                    r['melon16_first_step'] = m16.first.get(j); r['melon16_changed_steps'] = m16.changed[j]; r['melon16_seed_units'] = m16.added[j]
                if mrep is not None:
                    r['mrep_first_step'] = mrep.first.get(j); r['mrep_changed_steps'] = mrep.changed[j]
            G.dump(target / 'games.json', rows)
            G.dump(target / 'complete.json', dict(seconds=time.time() - tick, games=len(rows), **summ,
                   opp_postprocess={n: dict(v) for n, v in G5.opp_changed(opps).items() if n in names},
                   hashes=str(hdir / f'{tag}.npy'), hashes_sha256=G.sha(hdir / f'{tag}.npy')))
            saved.extend(rows); G.dump(adir / f'games_part_{tagp}.json', saved)
            print(json.dumps(dict(arm=arm, job_complete=job, seconds=round(time.time() - tick), wdl=dict(Counter(r['result'] for r in rows)),
                                  changed_games=len(agent.first), chain_max_batch_sec=round(agent.max_sec, 3),
                                  plan_max_sec=getattr(pt, 'max_sec', None), d23s_fired=None if dt is None else len(dt.first),
                                  melon16_games=None if m16 is None else len(m16.first), mrep_games=None if mrep is None else len(mrep.first),
                                  parity=None if cap is None else {x: summ['parity_submission'][x] for x in ('checked', 'mismatch')})), flush=True)
            del base, top, mtop, dt, pt, m16, mrep, outer, agent, cap, player, oc, play; torch.cuda.empty_cache()
    for p, d in code.items():
        assert G.sha(S.resolve_key(p)) == d, p
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
