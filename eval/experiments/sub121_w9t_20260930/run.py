"""sub121(v109、v4x gen14 単体・4 面・最終手の売り切り _liq つき)と、それに後付け処理 W9+WT を付けた版の同一局比較。
相手はチームの他の提出 7 つ(提出物の main.py の後処理・切り替え規則込み)。Rust 対局、fp32・TF32 無効。
  - 単体の網: nmb_eval と同じ網(mask_version 0 + 新マスク印 + rel 6 の下駄)、後処理は各 main.py の _pace(pace_from)。
  - 切り替えの網(sub117_sw2 / sub118 / sub119): policy.pt の a.* / b.* を fp32 の網 2 つに分け、各提出の model_ptr_infer.use_b を
    Rust の特徴 g に掛けて行ごとに A / B を選ぶ(v106_switch_tune の branches.py・gen28 eval と同じ作り)。
  - 後付け処理: share_w9t_20260929 の portable/postprocess_w9t.apply(dataset version 9 の版)を _liq の後に掛ける。
    hour 23 と step 718 以外では何も変えない(コード上 return action)ので、その 2 種の手でだけ呼ぶ。
    同じ観測で waste_replace2.replace_waste(flags W9,WT) の出力と一致することも毎回確かめる。
使い方: run.py smoke            → smoke/(4 seed × 7 相手、全 720 手、W9T、main.py 本体との行動一致の抜き取り)
        run.py arm <base|W9T>   → games/<arm>/(protocol.json の seed × 7 相手 × 両 player)
自動再起動なし。GPU lock は取らない(起動前の nvidia-smi 確認は起動者)。"""
import gzip, hashlib, importlib.util, json, os, re, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repository root (the folder holding eval/, postprocess/, engine/)
SCR = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'sub121_w9t_20260930')
POOLDIR = ROOT / 'eval/harness'
PNR = ROOT / 'eval/experiments/pass_noop_replace_v104_20260929'
PORT = ROOT / 'eval/experiments/share_w9t_20260929'
sys.path.insert(0, str(POOLDIR))
sys.path.insert(0, str(PNR)); sys.path.insert(0, str(PORT))
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch
from scripts.kagg_pfsp3.evaluate_land import inference, play_land_multi, LIBRARY
from scripts.kagg_pfsp3 import opponents as OPP
from scripts.kagg_ppo_eval_native.batch import NativeBatchAgent
from scripts.kagg_bc_kit.versioned_features import featurize
import nmb_eval
import waste_replace2 as WR2
import postprocess_w9t as PW

TS = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'team_subs_20260930')
A_ = ROOT / 'eval/experiments'
LEARNER = dict(name='sub121_v109', src=TS / 'sub121_v109/source', receipt=TS / 'sub121_v109/receipt.json', kind='single')
OPPONENTS = {
    'sub115_v103': dict(src=A_ / 'submission_strength_20260928_codex/v103/source', receipt=A_ / 'submission_strength_20260928_codex/v103/receipt.json', kind='single'),
    'sub116_v104': dict(src=A_ / 'latest_submission_base_20260928_codex/source', receipt=A_ / 'latest_submission_base_20260928_codex/receipt.json', kind='single'),
    'sub117_gen41': dict(src=TS / 'sub117_gen41/source', receipt=TS / 'sub117_gen41/receipt.json', kind='single'),
    'sub117_sw2_v105': dict(src=TS / 'sub117_sw2/source', receipt=TS / 'sub117_sw2/receipt.json', kind='switch'),
    'sub118_v106': dict(src=A_ / 'v106_switch_tune_20260929/source', receipt=A_ / 'v106_switch_tune_20260929/receipt.json', kind='switch'),
    'sub119_v107': dict(src=TS / 'sub119_v107/source', receipt=None, kind='switch'),
    'sub120_v108': dict(src=TS / 'sub120_v108/source', receipt=TS / 'sub120_v108/receipt.json', kind='single'),
}
WAVES = [['sub115_v103', 'sub116_v104'], ['sub117_gen41', 'sub117_sw2_v105'], ['sub118_v106', 'sub119_v107'], ['sub120_v108']]
CONFIG = dict(boardSize=10, turnsPerDay=24, shedCapacity=100, episodeSteps=720)
OFFICIAL = Path(importlib.util.find_spec('kaggle_environments').origin).parent / 'envs/kaggriculture/kaggriculture.py'
FEATS = ('tiles', 'ou', 'pu', 'g', 'legal')
SMI = os.environ.get('NVIDIA_SMI', 'nvidia-smi')
ARCH_DROP = ('ptr', 'nmb_pen', 'nmb_mode')
FILES = ('main.py', 'model_ptr_infer.py', 'model_tf.py', 'obs_feat.py', 'cfg.json', 'stock_target.json', 'policy.pt')


def relkey(p):
    """Key for code_sha256: path relative to the repository root; files inside site-packages are keyed from site-packages/."""
    a = os.path.abspath(p)
    if a.startswith(str(ROOT) + os.sep):
        return os.path.relpath(a, ROOT)
    i = a.find('site-packages' + os.sep)
    return a[i:] if i >= 0 else os.path.basename(a)


def resolve_key(k):
    """Inverse of relkey (plus the '$KAGG_SCRATCH/' keys written by run_gen70)."""
    if k.startswith('$KAGG_SCRATCH/'):
        return Path(os.environ.get('KAGG_SCRATCH', 'scratch'), k[len('$KAGG_SCRATCH/'):])
    if k.startswith('site-packages/'):
        return Path(importlib.util.find_spec('kaggle_environments').origin).parents[1] / k[len('site-packages/'):]
    return ROOT / k


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(p, v):
    p = Path(p); tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=1, default=str) + '\n'); tmp.replace(p)


def smi(q):
    return subprocess.check_output([SMI, q, '--format=csv,noheader'], text=True).strip()


def h64(*parts):
    h = hashlib.blake2b(digest_size=8)
    for x in parts:
        h.update(x)
    return int.from_bytes(h.digest(), 'little')


def key(g):
    return g['opponent'], g['seed'], g['learner_player']


def verify_sources(name, spec):
    """各提出の 7 ファイルの SHA を取り、receipt があれば照合する。"""
    got = {f: sha(Path(spec['src']) / f) for f in FILES}
    if spec.get('receipt'):
        rec = json.loads(Path(spec['receipt']).read_text())
        want = {Path(f.get('path') or f.get('name')).name: f['sha256'] for f in rec['files']}
        for f in FILES:
            assert want[f] == got[f], (name, f)
    return got


def _stage(name, sd, cfg):
    d = SCR / 'nets' / name; d.mkdir(parents=True, exist_ok=True)
    p = d / 'policy.pt'
    sd = {k: v.float().contiguous() for k, v in sd.items()}
    if not p.exists():
        torch.save(sd, p)
    else:                                                    # 既存の staging は中身が同じことだけ確かめる(上書きしない)
        old = torch.load(p, map_location='cpu', weights_only=True)
        assert set(old) == set(sd) and all(torch.equal(old[k], sd[k]) for k in sd), name
    arch = json.dumps({k: v for k, v in cfg.items() if k not in ARCH_DROP})
    c = d / 'policy.pt.cfg.json'
    if c.exists():
        assert c.read_text() == arch, name
    else:
        c.write_text(arch)
    return p


def nmb_net(path, main_py, pen, mode, batch):
    inf = inference(path, 'cuda', batch, main_py)
    assert inf.mask_version == 0
    net = nmb_eval.patch_net(inf._batch.net, pen, mode)
    inf._batch = nmb_eval._marked(NativeBatchAgent)(net, OPP.pace_from(main_py), 0)
    return inf


def module_of(main_py):
    OPP.pace_from(main_py)
    return OPP._MODULES[str(Path(main_py).resolve())]


class Switch:
    """切り替えの提出: 行ごとに提出自身の use_b(Rust の特徴 g)で A / B を選ぶ。"""
    def __init__(self, A, B, use_b):
        self.A, self.B, self.use_b = A, B, use_b; self.mask_version = 0; self.b_rows = 0; self.rows = 0; self.first_b = {}

    def act_batch(self, observations):
        gs = np.stack([o['_native_pool'].features['g'][o['_native_index']] for o in observations])
        m = self.use_b(torch.as_tensor(gs)).cpu().numpy().astype(bool)
        n = len(observations); acts = [None] * n
        if not m.all():
            aa = self.A.act_batch(observations)
            for j in range(n):
                if not m[j]:
                    acts[j] = aa[j]
        if m.any():
            ab = self.B.act_batch(observations)
            for j in range(n):
                if m[j]:
                    acts[j] = ab[j]
        self.b_rows += int(m.sum()); self.rows += n
        return acts


def build_agent(name, spec, batch):
    src = Path(spec['src']); main_py = src / 'main.py'
    cfg = json.loads((src / 'cfg.json').read_text())
    assert cfg.get('ptr') and cfg.get('nmb_mode') in ('abs', 'rel'), (name, cfg)
    pen, mode = float(cfg['nmb_pen']), str(cfg['nmb_mode'])
    sd = torch.load(src / 'policy.pt', map_location='cpu', weights_only=True)
    mod = module_of(main_py)
    if spec['kind'] == 'single':
        assert not any(k.startswith(('a.', 'b.')) for k in sd), name
        return nmb_net(_stage(name, sd, cfg), main_py, pen, mode, batch), mod
    parts = {}
    for pre in ('a.', 'b.'):
        part = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        assert len(part) == 150, (name, pre, len(part))
        parts[pre] = nmb_net(_stage(f'{name}_{pre[0].upper()}', part, cfg), main_py, pen, mode, batch)
    assert len(sd) == 300
    return Switch(parts['a.'], parts['b.'], mod._MP.use_b), mod


def switch_params(src):
    t = (Path(src) / 'model_ptr_infer.py').read_text()
    return {k: int(re.search(rf'^{k}\s*=\s*(\d+)', t, re.M).group(1)) for k in ('D_SW', 'K', 'M')}


class Learner:
    """学習側 sub121: 64 に詰めて forward(_liq 込み)。arm W9T なら hour 23 と step 718 で postprocess_w9t.apply を掛ける。"""
    def __init__(self, inf, keys, arm, steps, rules, capture=None):
        self.inf, self.keys, self.arm, self.rules = inf, keys, arm, rules
        self.mask_version = inf.mask_version
        self.hashes = np.zeros((len(keys), steps), np.uint64); self.calls = 0
        self.events = []; self.status = Counter(); self.seconds = 0.; self.parity = 0; self.capture = capture

    def act_batch(self, observations):
        step = int(observations[0]['step']); n = len(observations)
        assert n == len(self.keys) and all(int(o['step']) == step for o in observations)
        padded = list(observations)
        while len(padded) < 64:
            padded.append(observations[(len(padded) - n) % n])
        acts = self.inf.act_batch(padded)[:n]
        self.calls += 1
        for j, o in enumerate(observations):
            pool, i = o['_native_pool'], o['_native_index']
            self.hashes[j, step] = h64(*(np.ascontiguousarray(pool.features[k][i]).tobytes() for k in FEATS),
                                       json.dumps(acts[j], sort_keys=True).encode())
        if self.capture is not None:
            self.capture.take('learner', step, observations, acts, list(range(n)), [k[2] for k in self.keys])
        if self.arm == 'base' or not (step % 24 == 23 or step == 718):
            return acts
        t0 = time.perf_counter(); snaps = {}; out = []
        for j, (o, a) in enumerate(zip(observations, acts)):
            pool, i = o['_native_pool'], o['_native_index']
            if id(pool) not in snaps:
                snaps[id(pool)] = pool.debug()
            full = snaps[id(pool)][i // 2]['observations'][i % 2]
            assert int(full['step']) == step and full['player'] == self.keys[j][2]
            if self.parity < 64:
                for name, v in zip(FEATS, featurize(full, self.mask_version)):
                    assert np.array_equal(v, pool.features[name][i]), name
                self.parity += 1
            new = PW.apply(full, CONFIG, a)
            ref, ev, st = WR2.replace_waste(full, CONFIG, a, [], flags=('W9', 'WT'), rules=self.rules, budget_seconds=1.0)
            self.status[st] += 1
            assert st == 'ok' and repr(ref) == repr(new) and (ref is a) == (new is a), (step, j)
            old_m, new_m = a.get('market') or [], new.get('market') or []
            assert new_m[:len(old_m)] == old_m and len(new_m) <= 10
            assert [new.get('farmer')] + list(new.get('hands', [])) == [a.get('farmer')] + list(a.get('hands', []))
            added = new_m[len(old_m):]
            kinds = Counter(e['kind'] for e in ev if e.get('action') is not None)
            if added:
                self.events.append(dict(game=j, step=step, kind='WT' if step == 718 else 'W9', added=added, wr2_kinds=dict(kinds),
                                        quadrants=len(full['farms'][full['player']]['unlocked_quadrants']),
                                        shed_total=sum(full['private']['shed'].values()),
                                        carried=sum(sum(v.values()) for v in full['private']['inventories'])))
            out.append(new)
        self.seconds += time.perf_counter() - t0
        return out


class Capture:
    """main.py 本体との行動一致の抜き取り: 指定の手・局の公式観測と harness の行動を控え、後で main.py の agent に打たせて比べる。"""
    STEPS = (0, 23, 150, 239, 240, 260, 287, 288, 330, 450, 600, 695, 710, 718)

    def __init__(self, per_side=2):
        self.rows = []; self.per_side = per_side

    def take(self, who, step, observations, acts, idx, players):
        if step not in self.STEPS:
            return
        snaps = {}
        for c, (j, o) in enumerate(zip(idx, observations)):
            if c >= self.per_side * 2:
                break
            pool, i = o['_native_pool'], o['_native_index']
            if id(pool) not in snaps:
                snaps[id(pool)] = pool.debug()
            full = snaps[id(pool)][i // 2]['observations'][i % 2]
            assert int(full['step']) == step
            self.rows.append(dict(who=who, step=step, row=c, obs=json.loads(json.dumps(full)), action=json.loads(json.dumps(acts[c]))))


class CapOpp:
    def __init__(self, name, inner, capture):
        self.name, self.inner, self.capture = name, inner, capture; self.mask_version = inner.mask_version

    def act_batch(self, observations):
        acts = self.inner.act_batch(observations)
        self.capture.take(self.name, int(observations[0]['step']), observations, acts, list(range(len(observations))), None)
        return acts


def load_all(batch_opp=32):
    rules = WR2.official_rules(OFFICIAL)
    torch.set_num_threads(1); torch.manual_seed(20260930)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    shas = {LEARNER['name']: verify_sources(LEARNER['name'], LEARNER)}
    learner, lmod = build_agent(LEARNER['name'], LEARNER, 64)
    opps, mods = {}, {LEARNER['name']: lmod}
    for n, s in OPPONENTS.items():
        shas[n] = verify_sources(n, s)
        opps[n], mods[n] = build_agent(n, s, batch_opp)
    return rules, learner, opps, mods, shas


def sources():
    paths = {relkey(p): sha(p) for p in [HERE / 'run.py', PORT / 'postprocess_w9t.py', PNR / 'waste_replace2.py',
             POOLDIR / 'nmb_eval.py', OFFICIAL, LIBRARY]}
    return paths


def smoke():
    out = HERE / 'smoke'; out.mkdir(exist_ok=False)
    rules, learner, opps, mods, shas = load_all()
    cap = Capture()
    seeds = list(range(173600000, 173600004))
    names = list(OPPONENTS)
    keys = [(n, s, p) for n in names for s in seeds for p in (0, 1)]
    wrapped = {n: CapOpp(n, opps[n], cap) for n in names}
    # 学習側は 56 行(64 に詰める)。抜き取りは相手ごとに先頭 4 行(2 seed × 両 player)
    own = Learner(learner, keys, 'W9T', 719, rules)
    lcap = Capture(per_side=len(names) * 2)                # 学習側は各相手の先頭 1 seed 分を取るため全行から間引く
    def take_learner(who, step, observations, acts, idx, players):
        if step not in Capture.STEPS:
            return
        sel = [j for j, k in enumerate(keys) if k[1] == seeds[0]]
        lcap.take('learner', step, [observations[j] for j in sel], [acts[j] for j in sel], sel, None)
    own.capture = type('C', (), {'take': staticmethod(take_learner)})()
    t0 = time.time()
    rows = play_land_multi(own, wrapped, seeds)
    secs = time.time() - t0
    assert [key(g) for g in rows] == keys
    # 行動一致: main.py の agent(公式観測、CPU)
    mism = Counter(); tot = Counter(); samples = []
    for r in cap.rows + lcap.rows:
        who = r['who']; mod = mods[LEARNER['name'] if who == 'learner' else who]
        with torch.no_grad():
            py = mod.agent(r['obs'])
        py = json.loads(json.dumps(py))
        tot[who] += 1
        if py != r['action']:
            mism[who] += 1
            if len(samples) < 20:
                samples.append(dict(who=who, step=r['step'], row=r['row'], harness=r['action'], main_py=py))
    for n in names:                                            # 切り替えの main.py が終了時に source へ書く記録を消す(結果に影響しない)
        mp = getattr(mods[n], '_MP', None)
        if mp is not None and hasattr(mp, 'FIRE'):
            mp.FIRE.clear()
    ev = own.events
    res = dict(seconds=secs, games=len(rows), wdl=dict(Counter(r['result'] for r in rows)),
               wdl_by_opponent={n: dict(Counter(r['result'] for r in rows if r['opponent'] == n)) for n in names},
               postprocess_status=dict(own.status), feature_parity_checked=own.parity,
               events=dict(Counter(e['kind'] for e in ev)), games_changed=len({e['game'] for e in ev}),
               sell_orders_added=dict(Counter(e['kind'] for e in ev for _ in e['added'])),
               units_added=dict(Counter({k: sum(o[2] for e in ev if e['kind'] == k for o in e['added']) for k in ('W9', 'WT')})),
               w9_events_by_quadrants=dict(Counter(e['quadrants'] for e in ev if e['kind'] == 'W9')),
               switch_b_rows={n: [opps[n].b_rows, opps[n].rows] for n in names if isinstance(opps[n], Switch)},
               switch_params={n: switch_params(OPPONENTS[n]['src']) for n in names if OPPONENTS[n]['kind'] == 'switch'},
               parity_checked=dict(tot), parity_mismatch=dict(mism), parity_samples=samples,
               event_samples=ev[:30], source_sha256=shas, code_sha256=sources(), postprocess_seconds=own.seconds)
    dump(out / 'smoke.json', res)
    print(json.dumps({k: v for k, v in res.items() if k not in ('event_samples', 'source_sha256', 'parity_samples')}, default=str), flush=True)


def run_arm(arm):
    assert arm in ('base', 'W9T')
    prot = json.loads((HERE / 'protocol.json').read_text())
    first, count = prot['seeds']['first'], prot['seeds']['count']
    assert count % 16 == 0
    adir = HERE / 'games' / arm; adir.mkdir(parents=True, exist_ok=False)
    hdir = SCR / 'hashes' / arm; hdir.mkdir(parents=True, exist_ok=False)
    rules, learner, opps, mods, shas = load_all()
    code = sources()
    dump(adir / 'manifest.json', dict(arm=arm, seeds=[first, first + count - 1], waves=WAVES, pid=os.getpid(), source_sha256=shas,
         code_sha256=code, protocol_sha256=sha(HERE / 'protocol.json'),
         revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         gpu_before=smi('--query-gpu=memory.used,memory.total,power.draw'), gpu_apps_before=smi('--query-compute-apps=pid,used_memory'),
         precision='fp32', tf32=False, learner_batch=64, opponent_batch=32, configuration=CONFIG, automatic_restart=False,
         switch_params={n: switch_params(OPPONENTS[n]['src']) for n in OPPONENTS if OPPONENTS[n]['kind'] == 'switch'}))
    seeds_all = list(range(first, first + count)); saved = []; start = time.time()
    dump(adir / 'status.json', dict(status='running', pid=os.getpid(), started=start))
    jobs = [(i, w, names) for i in range(0, count, 16) for w, names in enumerate(WAVES)]
    for job, (i, w, names) in enumerate(jobs, 1):
        seeds = seeds_all[i:i + 16]
        keys = [(n, s, p) for n in names for s in seeds for p in (0, 1)]
        before = {n: (opps[n].b_rows, opps[n].rows) for n in names if isinstance(opps[n], Switch)}
        own = Learner(learner, keys, arm, 719, rules); tick = time.time()

        def progress(step):
            v = dict(status='running', arm=arm, job=job, jobs=len(jobs), step=step, finished=len(saved), seconds=time.time() - start)
            dump(adir / 'progress.json', v); print(json.dumps(v), flush=True)
        rows = play_land_multi(own, {n: opps[n] for n in names}, seeds, progress=progress)
        assert [key(g) for g in rows] == keys and own.calls == 719
        tag = f'seed{i:03d}_wave{w}'; target = adir / tag; target.mkdir()
        np.save(hdir / f'{tag}.npy', own.hashes)
        with gzip.open(target / 'events.jsonl.gz', 'wt') as f:
            for e in own.events:
                f.write(json.dumps(dict(e, key=list(keys[e['game']])), separators=(',', ':')) + '\n')
        for r in rows:
            r['arm'] = arm
        dump(target / 'games.json', rows)
        dump(target / 'complete.json', dict(seconds=time.time() - tick, games=len(rows), status=dict(own.status),
             events=dict(Counter(e['kind'] for e in own.events)), games_changed=len({e['game'] for e in own.events}),
             sell_orders_added=sum(len(e['added']) for e in own.events), postprocess_seconds=own.seconds,
             feature_parity_checked=own.parity, hashes=str(hdir / f'{tag}.npy'), hashes_sha256=sha(hdir / f'{tag}.npy'),
             switch_b_rows={n: [opps[n].b_rows - before[n][0], opps[n].rows - before[n][1]] for n in before}))
        saved.extend(rows); dump(adir / 'games_all.json', saved)
        print(json.dumps(dict(arm=arm, job_complete=job, seconds=round(time.time() - tick), wdl=dict(Counter(r['result'] for r in rows)),
                              events=len(own.events))), flush=True)
        del own; torch.cuda.empty_cache()
    for p, d in code.items():
        assert sha(ROOT / p) == d, p
    for n in OPPONENTS:
        mp = getattr(mods[n], '_MP', None)
        if mp is not None and hasattr(mp, 'FIRE'):
            mp.FIRE.clear()
    dump(adir / 'status.json', dict(status='complete', seconds=time.time() - start, games=len(saved)))
    print(json.dumps(dict(arm_complete=arm, games=len(saved), seconds=time.time() - start)), flush=True)


if __name__ == '__main__':
    try:
        if sys.argv[1] == 'smoke':
            smoke()
        else:
            assert sys.argv[1] == 'arm'
            run_arm(sys.argv[2])
    except BaseException as e:
        HERE.mkdir(exist_ok=True)
        dump(HERE / f'failure_{"_".join(sys.argv[1:])}.json', dict(type=type(e).__name__, message=str(e), traceback=traceback.format_exc(), time=time.time(), argv=sys.argv))
        raise
