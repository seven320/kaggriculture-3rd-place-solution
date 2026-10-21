"""shimishige の MMPQ-BCw gen35(PACE・LAND・段階売り・最終売り切りつき)に、版 11 の土台(W3+W5+W6+W9'+WT、
sub121_market_pp_20260930/run_m.py の M5 腕 = C1' + M5)を付けたときの同一局比較。
腕:
  base = gen35 の main.py そのまま(網 → main.py の _postprocess = _land(_pace(.)))
  pp   = base の行動に run_c.Top(W3/W5/W6 → W9 → WT、unit スコアは最初の forward の unit_logits)→ run_m.MTop(M5 = W9' の [] 枠詰め)
条件は sub121(sub121_w9t_20260930/protocol.json)と同じ: seed・相手 7 本・両 player・job 構成・Rust・fp32・TF32 無効・学習側 64 詰め・相手 32。
gen35 の _postprocess は unlocked_quadrants・market.inventory・money を読むので、NativePool の軽い観測ではなく pool.debug() の
公式観測に掛ける(NativeBatchAgent の repair は恒等にする)。
使い方: run_g35.py smoke <base|pp>  → smoke/<腕>/(seed 173600000〜03 × 7 相手 × 両 player、全 720 手、main.py 本体との一致の抜き取り)
        run_g35.py arm <base|pp>    → games/<腕>/
整合(report で確認): pp の初めて行動が変わる手までは base の特徴+行動 hash と一致、変わらなかった局は全手 hash と結果が一致。
自動再起動なし。GPU lock は取らない(起動前の nvidia-smi 確認は起動者)。"""
import gzip, importlib.util, json, os, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repository root (the folder holding eval/, postprocess/, engine/)
MPP = ROOT / 'eval/experiments/sub121_market_pp_20260930'
_spec = importlib.util.spec_from_file_location('sub121_run_m', MPP / 'run_m.py')
RM = importlib.util.module_from_spec(_spec); sys.modules['sub121_run_m'] = RM; _spec.loader.exec_module(RM)
C, S, MP = RM.C, RM.S, RM.MP
import numpy as np
import torch
from scripts.kagg_pfsp3.evaluate_land import play_land_multi
from scripts.kagg_bc_kit.versioned_features import featurize
WR2 = C.WR2

GEN35 = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'shimishige_teamabcw_gen32_gen34_20260930/extracted/teamabcw_gen35_paceland_liq')
GEN35_POLICY_SHA = 'deea2ca5c8c6610af6440db785dbf087019567e8546246126e1e19f253b51f8f'
SCR = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'gen35_postproc_20260930')
PORT = ROOT / 'postprocess'
SPEC = dict(flags=('W3', 'W5', 'W6', 'W9', 'WT'), w8='', market=('M5',))
sha, dump, smi, key, h64, FEATS, CONFIG = S.sha, S.dump, S.smi, S.key, S.h64, S.FEATS, S.CONFIG
CAP_STEPS = (0, 1, 23, 47, 100, 150, 200, 239, 240, 241, 260, 287, 300, 330, 400, 450, 500, 550, 600, 650, 695, 700,
             709, 710, 711, 714, 717, 718)


def load():
    rules = WR2.official_rules(S.OFFICIAL)
    torch.set_num_threads(1); torch.manual_seed(20260930)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    shas = {'gen35': {f: sha(GEN35 / f) for f in S.FILES}}
    assert shas['gen35']['policy.pt'] == GEN35_POLICY_SHA
    cfg = json.loads((GEN35 / 'cfg.json').read_text())
    assert cfg.get('ptr') and cfg['nmb_mode'] == 'rel' and float(cfg['nmb_pen']) == 6.0, cfg
    sd = torch.load(GEN35 / 'policy.pt', map_location='cpu', weights_only=True)
    assert not any(k.startswith(('a.', 'b.')) for k in sd)
    old = S.SCR; S.SCR = SCR                                  # 学習側の staging は本実験の scratch に置く(既存を上書きしない)
    try:
        p = S._stage('gen35', sd, cfg)
    finally:
        S.SCR = old
    learner = S.nmb_net(p, GEN35 / 'main.py', float(cfg['nmb_pen']), cfg['nmb_mode'], 64)
    learner._batch.repair = lambda a, o: a                  # gen35 の _postprocess は Base35 で公式観測に掛ける
    lmod = S.module_of(GEN35 / 'main.py')
    assert hasattr(lmod, '_postprocess') and lmod._LIQUIDATE_FROM_STEP == 710 and lmod._LAND_FROM_STEP == 240
    C.install_capture(learner._batch.net)
    opps, mods = {}, {'gen35': lmod}
    for n, s in S.OPPONENTS.items():
        shas[n] = S.verify_sources(n, s)
        opps[n], mods[n] = S.build_agent(n, s, 32)
    return rules, learner, opps, mods, shas


class Base35(C.Base):
    """64 詰めの forward(repair なし)→ 公式観測(pool.debug)に gen35 の _postprocess。hash は特徴 + _postprocess 後の行動。"""
    def __init__(self, inf, keys, steps, mod):
        super().__init__(inf, keys, steps); self.mod = mod; self.snaps = {}; self.pp = Counter()

    def act_batch(self, observations):
        step = int(observations[0]['step']); n = len(observations)
        assert n == len(self.keys) and all(int(o['step']) == step for o in observations)
        padded = list(observations)
        while len(padded) < 64:
            padded.append(observations[(len(padded) - n) % n])
        C.CAP.clear(); C.CAP['n'] = 0
        raw = self.inf.act_batch(padded)[:n]
        assert C.CAP['n'] == 1 and C.CAP['s'].shape[0] == 64
        self.scores = C.CAP['s'][:n].float().cpu().numpy(); self.q = C.CAP['q'][:n]
        self.calls += 1
        self.snaps = {}; acts = []
        for j, (o, a) in enumerate(zip(observations, raw)):
            pool, i = o['_native_pool'], o['_native_index']
            if id(pool) not in self.snaps:
                self.snaps[id(pool)] = pool.debug()
            full = self.snaps[id(pool)][i // 2]['observations'][i % 2]
            assert int(full['step']) == step and full['player'] == self.keys[j][2]
            new = self.mod._postprocess(json.loads(json.dumps(a)), full)
            om, nm = a.get('market') or [], new.get('market') or []
            if nm != om:
                self.pp['liq' if step >= 710 else 'pace_or_land'] += 1
                if any(x and x[0] == 'BUY_LAND' for x in nm) and not any(x and x[0] == 'BUY_LAND' for x in om):
                    self.pp['land_added'] += 1
            acts.append(new)
            self.hashes[j, step] = h64(*(np.ascontiguousarray(pool.features[k][i]).tobytes() for k in FEATS),
                                       json.dumps(new, sort_keys=True).encode())
        return acts

    def full(self, o):
        return self.snaps[id(o['_native_pool'])][o['_native_index'] // 2]['observations'][o['_native_index'] % 2]


class Top35(C.Top):
    """run_c.Top と同じ(公式観測は Base35 の snaps を使う)。step 718 の検査だけ gen35 の段階売り(SELL 1000 × 9 + 残し)に合わせる。"""
    def act_batch(self, observations):
        acts = self.inner.act_batch(observations)
        t0 = time.perf_counter(); out = []; step = int(observations[0]['step'])
        for j, (o, a) in enumerate(zip(observations, acts)):
            pool, i = o['_native_pool'], o['_native_index']
            full = self.base.full(o)
            assert int(full['step']) == step and full['player'] == self.keys[j][2]
            if self.parity < 32:
                for name, v in zip(FEATS, featurize(full, self.mask_version)):
                    assert np.array_equal(v, pool.features[name][i]), name
                self.parity += 1
            units = 1 + len(a.get('hands', []))
            scores = C.Lazy(self.base.scores[j], units)
            new, ev, st = WR2.replace_waste(full, CONFIG, a, scores, flags=self.spec['flags'],
                                            finalize=lambda u, x, j=j: self.finalize(j, u, x), rules=self.rules, budget_seconds=C.BUDGET)
            self.status[st] += 1
            nq = len(full['farms'][full['player']]['unlocked_quadrants'])
            for e in ev:
                e['game'] = j; e['quadrants'] = nq
                if e.get('pos') is not None:
                    x, y = e['pos']; e['tile_quadrant'] = C.QUAD(x, y)
                    assert full['farms'][full['player']]['tiles'][y][x] != 'LOCKED', (step, j, e)
                if e['kind'] in ('W3', 'W5', 'W6') and e['action'] is not None:
                    c = WR2.NN_CLASS[tuple(e['action'][:2])]
                    assert self.base.scores[j][e['unit'], c] >= -1e3 and abs(self.base.scores[j][e['unit'], c] - e['score']) < 1e-6
                    self.checks['model_legal'] += 1
                    self.checks['legal_v0' if pool.features['legal'][i][e['unit'], c] else 'legal_xp_only'] += 1
            self.events.extend(ev)
            old_m, new_m = a.get('market') or [], new.get('market') or []
            assert new_m[:len(old_m)] == old_m and len(new_m) <= 10
            if step == 718:
                assert old_m[:9] == [['SELL', it, 1000] for it in self.base.mod._VALUE_ORDER], old_m
            out.append(new)
        self.seconds += time.perf_counter() - t0
        return out


class Cap:
    """行動一致の抜き取り: 指定の手で、各相手の先頭 seed の両 player の公式観測と最終行動を控える。"""
    def __init__(self, inner, base, keys, seed0):
        self.inner, self.base = inner, base; self.mask_version = inner.mask_version; self._batch = inner._batch
        self.sel = [j for j, k in enumerate(keys) if k[1] == seed0]; self.rows = []

    def act_batch(self, observations):
        acts = self.inner.act_batch(observations); step = int(observations[0]['step'])
        if step in CAP_STEPS:
            for j in self.sel:
                self.rows.append(dict(step=step, row=j, obs=json.loads(json.dumps(self.base.full(observations[j]))),
                                      action=json.loads(json.dumps(acts[j]))))
        return acts


def build(inf, keys, arm, rules, mod):
    base = Base35(inf, keys, 719, mod)
    if arm == 'base':
        return base, None, None, base
    top = Top35(base, base, keys, SPEC, rules)
    mtop = RM.MTop(top, keys, SPEC, rules)
    return base, top, mtop, mtop


def summarize(base, top, mtop):
    out = dict(postprocess_gen35=dict(base.pp))
    if top is not None:
        summ, _ = C.summarize(top, top.inner, dict(w8=''), None)
        me = mtop.events
        summ.update(market_events=dict(Counter(e['kind'] for e in me)), m5_sell_units=sum(o[2] for e in me if e['kind'] == 'M5' for o in e['added']),
                    m5_status=dict(mtop.status), market_seconds=mtop.seconds)
        out.update(summ)
    return out


def sources():
    paths = {S.relkey(p): sha(p) for p in [HERE / 'run_g35.py', MPP / 'run_m.py', MPP / 'market_pp.py',
             C.HERE / 'run_c.py', S.HERE / 'run.py', Path(WR2.__file__), S.POOLDIR / 'nmb_eval.py', S.OFFICIAL, S.LIBRARY]}
    return paths


def parity(rows, mod, arm):
    """harness の最終行動と公式観測上の main.py(+ pp なら共有版 KitHook)の行動を比べる(CPU)。"""
    hook = None
    if arm == 'pp':
        sys.path.insert(0, str(PORT))
        import postprocess_c1p_w9p as PC
        hook = PC.KitHook(mod._NET)
    tot = 0; mism = []; hook_repl = 0
    for r in rows:
        with torch.no_grad():
            o = mod._todict(r['obs'])
            a = mod.agent(o) if hook is None else None
            if hook is not None:
                base_a = mod._postprocess(mod._act(o), o)
                ev = []
                a = hook.apply(o, CONFIG, base_a, events=ev)
                hook_repl += sum(1 for e in ev if e.get('action') is not None)
        a = json.loads(json.dumps(a)); tot += 1
        if a != r['action']:
            mism.append(dict(step=r['step'], row=r['row'], harness=r['action'], main_py=a))
    return dict(checked=tot, mismatch=len(mism), samples=mism[:10], portable_hook_replacements=hook_repl)


def smoke(arm):
    out = HERE / 'smoke' / arm; out.mkdir(parents=True, exist_ok=False)
    rules, learner, opps, mods, shas = load()
    seeds = list(range(173600000, 173600004)); names = list(S.OPPONENTS)
    keys = [(n, s, p) for n in names for s in seeds for p in (0, 1)]
    base, top, mtop, agent = build(learner, keys, arm, rules, mods['gen35'])
    cap = Cap(agent, base, keys, seeds[0])
    t0 = time.time()
    rows = play_land_multi(cap, opps, seeds)
    secs = time.time() - t0
    assert [key(g) for g in rows] == keys and base.calls == 719
    summ = summarize(base, top, mtop)
    par = parity(cap.rows, mods['gen35'], arm)
    for n in names:
        mp = getattr(mods[n], '_MP', None)
        if mp is not None and hasattr(mp, 'FIRE'):
            mp.FIRE.clear()
    res = dict(arm=arm, seconds=secs, games=len(rows), wdl=dict(Counter(r['result'] for r in rows)),
               wdl_by_opponent={n: dict(Counter(r['result'] for r in rows if r['opponent'] == n)) for n in names},
               margin_mean=float(np.mean([r['own_money'] - r['opponent_money'] for r in rows])), **summ, parity=par,
               samples=[] if top is None else [dict(e, key=list(keys[e['game']])) for e in top.events if e['action'] is not None][:20]
               + [dict(e, key=list(keys[e['game']])) for e in mtop.events][:10],
               source_sha256=shas, code_sha256=sources(), gpu=smi('--query-gpu=memory.used,memory.total'))
    dump(out / 'smoke.json', res)
    np.save(SCR / f'smoke_hashes_{arm}.npy', base.hashes)
    dump(out / 'games.json', rows)
    print(json.dumps({k: v for k, v in res.items() if k not in ('samples', 'source_sha256', 'code_sha256')}, default=str), flush=True)


def run_arm(arm):
    assert arm in ('base', 'pp')
    prot = json.loads((HERE / 'protocol.json').read_text())
    first, count = prot['seeds']['first'], prot['seeds']['count']
    adir = HERE / 'games' / arm; adir.mkdir(parents=True, exist_ok=False)
    hdir = SCR / 'hashes' / arm; hdir.mkdir(parents=True, exist_ok=False)
    rules, learner, opps, mods, shas = load()
    code = sources()
    for p, d in prot['code_sha256'].items():
        assert code.get(p) == d, p
    dump(adir / 'manifest.json', dict(arm=arm, spec=SPEC if arm == 'pp' else None, seeds=[first, first + count - 1], waves=S.WAVES, pid=os.getpid(),
         source_sha256=shas, code_sha256=code, protocol_sha256=sha(HERE / 'protocol.json'),
         revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         gpu_before=smi('--query-gpu=memory.used,memory.total,power.draw'), gpu_apps_before=smi('--query-compute-apps=pid,used_memory'),
         precision='fp32', tf32=False, learner_batch=64, opponent_batch=32, configuration=CONFIG, budget_seconds_per_observation=C.BUDGET,
         automatic_restart=False))
    seeds_all = list(range(first, first + count)); saved = []; start = time.time()
    dump(adir / 'status.json', dict(status='running', pid=os.getpid(), started=start))
    jobs = [(i, w, names) for i in range(0, count, 16) for w, names in enumerate(S.WAVES)]
    for job, (i, w, names) in enumerate(jobs, 1):
        seeds = seeds_all[i:i + 16]
        keys = [(n, s, p) for n in names for s in seeds for p in (0, 1)]
        base, top, mtop, agent = build(learner, keys, arm, rules, mods['gen35']); tick = time.time()

        def progress(step):
            v = dict(status='running', arm=arm, job=job, jobs=len(jobs), step=step, finished=len(saved), seconds=time.time() - start)
            dump(adir / 'progress.json', v); print(json.dumps(v), flush=True)
        rows = play_land_multi(agent, {n: opps[n] for n in names}, seeds, progress=progress)
        assert [key(g) for g in rows] == keys and base.calls == 719
        tag = f'seed{i:03d}_wave{w}'; target = adir / tag; target.mkdir()
        np.save(hdir / f'{tag}.npy', base.hashes)
        summ = summarize(base, top, mtop)
        with gzip.open(target / 'events.jsonl.gz', 'wt') as f:
            if top is not None:
                for e in top.events + mtop.events:
                    f.write(json.dumps(dict(e, key=list(keys[e['game']])), separators=(',', ':'), default=str) + '\n')
        for r in rows:
            r['arm'] = arm
        dump(target / 'games.json', rows)
        dump(target / 'complete.json', dict(seconds=time.time() - tick, games=len(rows), **summ,
             hashes=str(hdir / f'{tag}.npy'), hashes_sha256=sha(hdir / f'{tag}.npy')))
        saved.extend(rows); dump(adir / 'games_all.json', saved)
        print(json.dumps(dict(arm=arm, job_complete=job, seconds=round(time.time() - tick), wdl=dict(Counter(r['result'] for r in rows)),
                              replaced=summ.get('replaced'), market=summ.get('market_events'), pp=summ['postprocess_gen35'])), flush=True)
        del base, top, mtop, agent; torch.cuda.empty_cache()
    assert len(saved) == 1344
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
