"""postprocess_c1p_w9p(C1' + W9')と、実験で使った実装(run_m.py の C1p + M5 経路)との同等性試験(CPU のみ、2026-09-30)。
版 10 の test_postprocess_c1p.py に、W9'(run_m.MTop.m5)の参照と、空の注文 [] を含む観測の種類(empty_pad・overflow)を足したもの。

実験の実装(eval/experiments/sub121_c1c2_20260930/run_c.py)の C1p 経路は
  unit スコア = 網の最初の forward の unit_logits(nmb の下駄の後) → run_c.Lazy({行動: スコア}、-1e3 未満は除く)
  数量       = run_c.Top.finalize(網の unit_qty の argmax を obs_feat と同じ規則で数に直す)
  後処理     = waste_replace2.replace_waste(obs, configuration, action, scores, flags=('W3','W5','W6','W9','WT'), finalize=...)
  W9'        = その出力に run_m.MTop.m5(eval/experiments/sub121_market_pp_20260930/run_m.py)を掛けたもの: hour 23 で market に [] があれば、[] を除いた注文列で
               replace_waste(flags=('W9',)、unit スコアは空)を計算し直し、足した SELL を market_pp.fill_empty で [] の枠へ前から入れる。
               下の ref_m5 は MTop.m5 の本文の写し(self.status の集計を除く)で、fill_empty は market_pp.py そのもの(--mpp)を使う。
この試験では、run_c.Lazy と run_c.Top.finalize と同じ処理を下の RunCLazy / runc_finalize に書き(run_c.py は実験環境に依存して
単独では import できないため。run_c.py そのものを import した確認は非公開の実験記録にある)、waste_replace2 は eval/experiments/pass_noop_replace_v104_20260929/waste_replace2.py を使う。

観測: v104(チームの提出物の 1 つ)の実 LB 対局の replay の全手(step 0〜718、両 player の観測)。網は example_sub121(sub121 の提出物)。
  1. model   : 網 → _liq の行動と、その forward のスコア。portable 側は example の main.py と同じ KitHook.apply。
  2. recorded: replay に記録された行動(両 player)+ 同じ観測での網のスコア。portable 側は KitHook.apply。
  3. random  : 乱数の行動(不正な形を含む)+ 網のスコア。portable 側は apply(scores_from_logits, KitHook.finalize)。1 観測 1 個、hour 23 と 718 は 4 個。
  4. synth   : 乱数の行動 + 乱数のスコア(ランダムな手の部分集合と値)+ 決まった規則の finalize。1 観測 1 個。
  5. empty_pad: hour 23 の観測で、網 → _liq の行動の market に [] を乱数の位置へ足したもの(10 枠ちょうど・11 枠・全部 [])+ 網のスコア。
               1 観測 3 個。
  6. overflow : hour 23 の観測の複製で、倉庫を 90〜100 個の産物で埋め、unit の携行品を 5〜40 個にして日末にあふれる状況を作り、
               market を「乱数の注文 k 個 + [] (10−k) 個」(k = 3〜10、並びは乱数。1/4 は 11 枠)にしたもの + 網のスコア。1 観測 4 個。
  W9' が 1 回以上効いた呼び出しが empty_pad と overflow の両方にあることを合格の条件に含める。
一致の条件: 出力の repr が同じ、「渡した action と同じオブジェクトを返したか」が同じ、events(当たった手と置き換え先)が同じ、入力を書き換えない。
replay は対局の生データなのでこのリポジトリには含めていない。Kaggle の公開 episode データから <episode>.json.gz を --replays(既定は環境変数
KAGG_REPLAYS_DIR、無ければ ./replays)に置く。--example は Kit 系の提出物フォルダ(sub121 にこの段を付けたもの。組み立て方は postprocess/README.md、sub121 の重みは非公開。
既定は環境変数 KAGG_EXAMPLE_SUB121、無ければ ./example_sub121)。
使い方: python test_postprocess_c1p_w9p.py [--replays DIR] [--episodes E1,...] [--example DIR] [--wr2 PATH] [--mpp PATH]
        [--out validation.json] [--jobs 8]
"""
import argparse, copy, gzip, hashlib, importlib.util, json, os, platform, random, sys, time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
HERE = Path(__file__).resolve().parent
DEFAULT_REPLAYS = os.environ.get('KAGG_REPLAYS_DIR', 'replays')
DEFAULT_EPISODES = '114643260,114563237,114638818,114551580,114558870,114641776,114624074,114539949'
WR2_CANDIDATES = [HERE.parent / 'eval/experiments/pass_noop_replace_v104_20260929/waste_replace2.py']
MPP_CANDIDATES = [HERE.parent / 'eval/experiments/sub121_market_pp_20260930/market_pp.py']
FLAGS = ('W3', 'W5', 'W6', 'W9', 'WT')

MOVES = ['NORTH', 'SOUTH', 'EAST', 'WEST']
SIMPLE = ['PASS', 'WATER', 'HARVEST', 'FERTILIZE', 'DIG', 'BUILD_COOP', 'BUILD_PASTURE', 'FEED', 'COLLECT_FERTILIZER', 'CARE', 'DROP']
CROPS = ['WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON']
ITEMS = CROPS + ['EGG', 'MILK', 'WOOL', 'FERTILIZER', 'GOOSE', 'COW', 'SHEEP']
PRODUCTS = CROPS + ['EGG', 'MILK', 'WOOL', 'FERTILIZER']


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


class RunCLazy:
    """run_c.Lazy と同じ: unit u の {行動: スコア}(-1e3 未満はモデルが不可とした手)。"""
    def __init__(self, WR2, row, n):
        self.WR2, self.row, self.n = WR2, row, n

    def __len__(self):
        return self.n

    def __getitem__(self, u):
        r = self.row[u]
        return {self.WR2.nn_action(c): float(r[c]) for c in range(44) if r[c] >= -1e3}


def runc_finalize(WR2, net, q, u, a):
    """run_c.Top.finalize と同じ(j = 0): 網の unit_qty の argmax を versioned_features._q(qc, 20) で数に直す。"""
    import torch

    def _q(qc, qmax):                                      # scripts/kagg_bc_kit/versioned_features._q と同じ
        qc = int(qc)
        if qc == 0: return None
        if qc <= qmax + 1: return qc - 1
        return (999, 1000)[min(qc - (qmax + 2), 1)]
    with torch.inference_mode():
        cls = torch.zeros(1, 20, dtype=torch.long, device=q.device); cls[0, u] = WR2.NN_CLASS[tuple(a)]
        qc = int(net.unit_qty(q[0:1], cls).argmax(-1)[0, u])
    qty = _q(qc, 20)
    return [a[0], a[1], 1 if qty is None else int(qty)]


def ref_m5(WR2, MP, rules, full, cfg, a):
    """run_m.MTop.m5 と同じ(self.status の集計を除く。起動条件は run_m.MTop.act_batch と market_pp.needs_obs の M5 の条件)。
    events は waste_replace2 の W9 の event の kind を "W9'" にし、入れた枠の添字 slots を足したもの(portable の event の形)。"""
    step = int(full['step'])
    if step % 24 != 23:
        return a, []
    mk = list(a.get('market') or [])
    if not any(isinstance(o, list) and not o for o in mk):
        return a, []
    compact = [o for o in mk if not (isinstance(o, list) and not o)]
    ref, ev, st = WR2.replace_waste(full, cfg, dict(a, market=compact), [], flags=('W9',), rules=rules, budget_seconds=None)
    if st != 'ok' or ref is a or not ev:
        return a, []
    assert [ref.get('farmer')] + list(ref.get('hands') or []) == [a.get('farmer')] + list(a.get('hands') or [])
    extra = ref['market'][len(compact):]
    assert ref['market'][:len(compact)] == compact and extra and all(o[0] == 'SELL' for o in extra)
    out = dict(a); out['market'] = MP.fill_empty(mk, extra)
    assert len(ev) == 1 and ev[0]['kind'] == 'W9'
    slots = [i for i, o in enumerate(mk) if isinstance(o, list) and not o][:len(extra)]
    return out, [dict(ev[0], kind="W9'", slots=slots)]


def rand_unit(rng):
    r = rng.random()
    if r < 0.20:
        return [rng.choice(MOVES)]
    if r < 0.60:
        return [rng.choice(SIMPLE + ['CARE', 'CARE', 'WATER', 'WATER', 'FEED', 'FERTILIZE'])]
    if r < 0.75:
        return ['PLANT', rng.choice(CROPS)]
    if r < 0.93:
        a = [rng.choice(['PICKUP', 'PLACE']), rng.choice(ITEMS)]
        if rng.random() < 0.8:
            a.append(rng.choice([1, 1, 2, 3, 5, 10, 50, 100, 0, -1]))
        return a
    return rng.choice([[], ['PLANT'], ['PLANT', 'NOPE'], ['PICKUP'], ['UNKNOWN_OP'], ['PASS', 1]])


def rand_order(rng):
    r = rng.random()
    if r < 0.55:
        return ['SELL', rng.choice(PRODUCTS), rng.choice([1, 2, 3, 5, 10, 20, 40, 100, 1000])]
    if r < 0.75:
        return [rng.choice(['BUY_SEED', 'BUY_PRODUCT']), rng.choice(CROPS + ['FERTILIZER']), rng.choice([1, 2, 5])]
    if r < 0.85:
        return rng.choice([['HIRE'], ['BUY_LAND'], ['BUY_ANIMAL', 'GOOSE', 1], []])
    return rng.choice([['SELL', 'WHEAT', '5'], ['SELL', 'WHEAT', 2.0], ['SELL', 'WHEAT', 0], ['SELL', 'WHEAT', -3],
                       ['SELL', 'GOOSE', 1], ['SELL', 'WHEAT'], ['SELL', 'NOPE', 2], 'SELL WHEAT 1', ('SELL', 'EGG', 1)])


def rand_action(rng, obs):
    n_hands = min(19, max(0, len(obs['farms'][obs['player']]['hands']) + rng.choice([-1, 0, 0, 0, 1])))
    a = {'farmer': rand_unit(rng), 'hands': [rand_unit(rng) for _ in range(n_hands)],
         'market': [rand_order(rng) for _ in range(rng.choice([0, 0, 1, 2, 3, 5, 8, 9, 10, 11]))]}
    r = rng.random()
    if r < 0.03:
        del a['market']
    elif r < 0.05:
        a['market'] = None
    return a


def pad_empty(rng, market, total):
    """market に [] を乱数の位置へ足して total 個にする(total 以上なら [] を 1 個だけ足す)。"""
    out = [copy.deepcopy(o) for o in market]
    for _ in range(max(1, total - len(out))):
        out.insert(rng.randint(0, len(out)), [])
    return out


def overflow_obs(rng, obs):
    """hour 23 の観測の複製: 倉庫を 90〜100 個の産物で埋め、unit(farmer と hands)の携行品を 5〜40 個にする。"""
    o = copy.deepcopy(obs)
    pv = o['private']
    total = rng.randint(90, 100); items = rng.sample(PRODUCTS, rng.randint(1, 5))
    cuts = sorted(rng.sample(range(1, total), len(items) - 1))
    pv['shed'] = {item: b - a for item, a, b in zip(items, [0] + cuts, cuts + [total])}
    n_units = len(pv.get('inventories', []))
    for u in range(n_units):
        if u == 0 or rng.random() < 0.4:
            pv['inventories'][u] = {rng.choice(PRODUCTS): rng.randint(5, 40)}
    return o


def overflow_market(rng):
    k = rng.randint(3, 10)
    m = [rand_order(rng) for _ in range(k)] + [[] for _ in range(10 - k)]
    if rng.random() < 0.25:
        m.append([])
    rng.shuffle(m)
    return m


def rand_scores(rng, P, n):
    keys = [P.nn_action(c) for c in range(44)]
    out = []
    for _ in range(n):
        k = rng.randint(0, 44)
        out.append({a: rng.choice([round(rng.uniform(-20, 20), 3), 0.0, 1.0]) for a in rng.sample(keys, k)})
    return out


def synth_finalize(u, a):
    return [a[0], a[1], (len(a[1]) * 7 + u) % 23]         # 0 を含む決まった数(0 の時は両者とも数量 1 に直す)


def official_path():
    spec = importlib.util.find_spec('kaggle_environments')
    if spec is not None and spec.submodule_search_locations:
        p = Path(list(spec.submodule_search_locations)[0]) / 'envs/kaggriculture/kaggriculture.py'
        if p.exists():
            return p, 'installed kaggle_environments'
    raise SystemExit('kaggle-environments (1.32.7) must be installed: the test compares the embedded rule definitions with it')


def official_defs(path):
    import ast
    names = {'CROPS', 'ANIMALS', 'PRODUCTS', 'FARMER_MOVES', '_shed_access_tiles', '_is_shed_adjacent', '_new_plant', '_new_animal',
             '_farmer_position', '_set_farmer_position', '_farmer_inventory', '_inv_add', '_inv_take', '_apply_unit_action'}
    src = Path(path).read_text()
    lines = src.splitlines(keepends=True)
    out = {}
    for n in ast.parse(src).body:
        name = n.name if isinstance(n, ast.FunctionDef) else next(
            (t.id for t in getattr(n, 'targets', []) if isinstance(t, ast.Name)), None)
        if name in names:
            out[name] = ''.join(lines[n.lineno - 1:n.end_lineno])
    assert set(out) == names, sorted(names - set(out))
    return out


def run_episode(args_tuple):
    ep, replays, example, wr2_path, mpp_path, seed = args_tuple
    import torch
    torch.set_num_threads(1)
    ex = load_module('example_main', Path(example) / 'main.py')
    P = sys.modules['postprocess_c1p_w9p']
    port = load_module('postprocess_c1p_w9p_portable', HERE / 'postprocess_c1p_w9p.py')   # 試験の対象(portable/)。example の写しと同じ中身であること
    assert Path(P.__file__).read_bytes() == (HERE / 'postprocess_c1p_w9p.py').read_bytes()
    WR2 = load_module('waste_replace2', wr2_path)
    MP = load_module('market_pp', mpp_path)
    rules = WR2.official_rules(official_path()[0])
    hook = ex._C1P_HOOK
    rng = random.Random(seed * 1000 + int(ep) % 1000)
    tot = dict(calls=0, equal=0, mismatch=0, input_mutated=0, changed=0, wr_status={}, calls_with_empty=0, calls_with_empty_hour23=0,
               model_actions_with_empty=0, model_actions_with_empty_hour23=0)
    per_kind = {}; replaced = {}; mism = []
    rp = json.load(gzip.open(Path(replays) / f'{ep}.json.gz'))
    steps, cfg = rp['steps'], rp['configuration']

    def compare(kind, obs, action, portable_fn, ref_scores, ref_fin, step, who):
        o1, a1 = copy.deepcopy(obs), copy.deepcopy(action)
        o2, a2 = copy.deepcopy(obs), copy.deepcopy(action)
        before = (repr(o1), repr(a1))
        ev1 = []
        r1 = portable_fn(o1, a1, ev1)
        r2, ev2, st = WR2.replace_waste(o2, cfg, a2, ref_scores, flags=FLAGS, finalize=ref_fin, rules=rules, budget_seconds=None)
        r2, ev5 = ref_m5(WR2, MP, rules, o2, cfg, r2)                  # run_m の順序: C1p(run_c.Top)の後に M5
        ev2 = ev2 + ev5
        k = per_kind.setdefault(kind, dict(calls=0, equal=0, changed=0, with_empty=0, w9p=0))
        if isinstance(action, dict) and any(isinstance(x, list) and not x for x in (action.get('market') or [])):
            tot['calls_with_empty'] += 1; k['with_empty'] += 1
            tot['calls_with_empty_hour23'] += int(obs['step']) % 24 == 23
        tot['calls'] += 1; k['calls'] += 1
        tot['wr_status'][st] = tot['wr_status'].get(st, 0) + 1
        same = repr(r1) == repr(r2) and ((r1 is a1) == (r2 is a2)) and (st != 'ok' or repr(ev1) == repr(ev2))
        if (repr(o1), repr(a1)) != before or (repr(o2), repr(a2)) != before:
            tot['input_mutated'] += 1
        if same:
            tot['equal'] += 1; k['equal'] += 1
        else:
            tot['mismatch'] += 1
            if len(mism) < 10:
                mism.append(dict(kind=kind, episode=ep, step=step, who=who, portable=repr(r1)[:1500], reference=repr(r2)[:1500], status=st))
        if r1 is not a1:
            tot['changed'] += 1; k['changed'] += 1
            for e in ev1:
                if e['action'] is not None:
                    replaced[e['kind']] = replaced.get(e['kind'], 0) + 1
            k['w9p'] += any(e['kind'] == "W9'" for e in ev1)

    for s in range(len(steps) - 1):
        shared = steps[s][0]['observation']
        for me in (0, 1):
            obs = dict(steps[s][me]['observation'])
            for key in ('step', 'farms', 'market', 'town', 'day', 'hour'):
                if key not in obs and key in shared:
                    obs[key] = shared[key]
            obs['player'] = me
            with torch.no_grad():
                o = ex._todict(obs)
                hook.s = hook.q = None; hook.calls = 0
                a0 = ex._liq(ex._act(o), o)
            assert hook.calls == 1
            sc, q = hook.s, hook.q
            s_np = sc[:1].float().cpu().numpy()
            fin_r = lambda u, x: runc_finalize(WR2, ex._NET, q, u, x)

            def via_hook(o_, a_, ev_):                      # example の main.py と同じ経路(KitHook.apply)
                hook.s, hook.q, hook.calls = sc, q, 1
                with torch.no_grad():
                    return hook.apply(o_, cfg, a_, events=ev_)
            n0 = 1 + len(a0.get('hands', []))
            if any(isinstance(x, list) and not x for x in (a0.get('market') or [])):
                tot['model_actions_with_empty'] += 1; tot['model_actions_with_empty_hour23'] += s % 24 == 23
            compare('model', o, a0, via_hook, RunCLazy(WR2, s_np[0], n0), fin_r, s, me)
            rec = steps[s + 1][me].get('action')
            if isinstance(rec, dict):
                compare('recorded', o, rec, via_hook, RunCLazy(WR2, s_np[0], min(20, 1 + len(rec.get('hands', []) or []))), fin_r, s, me)
            hot = s % 24 == 23 or s == 718
            ns_p = SimpleNamespace(net=ex._NET, q=q)
            for _ in range(4 if hot else 1):
                ra = rand_action(rng, o)
                nr = min(20, 1 + len(ra.get('hands', [])))
                compare('random', o, ra, lambda o_, a_, ev_: port.apply(o_, cfg, a_, port.scores_from_logits(sc[0], nr),
                        finalize=lambda u, x: port.KitHook.finalize(ns_p, u, x), budget_seconds=None, events=ev_),
                        RunCLazy(WR2, s_np[0], nr), fin_r, s, me)
            if s % 24 == 23:
                for total in (10, 11, 0):
                    pa = copy.deepcopy(a0)
                    pa['market'] = [[] for _ in range(10)] if total == 0 else pad_empty(rng, a0.get('market') or [], total)
                    compare('empty_pad', o, pa, via_hook, RunCLazy(WR2, s_np[0], n0), fin_r, s, me)
                for _ in range(4):
                    oo = overflow_obs(rng, o)
                    pa = copy.deepcopy(a0); pa['market'] = overflow_market(rng)
                    compare('overflow', oo, pa, via_hook, RunCLazy(WR2, s_np[0], n0), fin_r, s, me)
            ra = rand_action(rng, o)
            rs = rand_scores(rng, port, 1 + len(ra.get('hands', [])))
            compare('synth', o, ra, lambda o_, a_, ev_: port.apply(o_, cfg, a_, rs, finalize=synth_finalize, budget_seconds=None, events=ev_),
                    rs, synth_finalize, s, me)
    return dict(episode=int(ep), steps=len(steps) - 1, totals=tot, by_kind=per_kind, replaced=replaced, mismatches=mism,
                replay_sha256=hashlib.sha256((Path(replays) / f'{ep}.json.gz').read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replays', default=DEFAULT_REPLAYS)
    ap.add_argument('--episodes', default=DEFAULT_EPISODES)
    ap.add_argument('--example', default=os.environ.get('KAGG_EXAMPLE_SUB121', 'example_sub121'))
    ap.add_argument('--wr2', default=next((str(p) for p in WR2_CANDIDATES if p.exists()), None))
    ap.add_argument('--mpp', default=next((str(p) for p in MPP_CANDIDATES if p.exists()), None))
    ap.add_argument('--out', default=str(HERE / 'validation.json'))
    ap.add_argument('--seed', type=int, default=20260930)
    ap.add_argument('--jobs', type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    eps = [e for e in args.episodes.split(',') if e]
    work = [(e, args.replays, args.example, args.wr2, args.mpp, args.seed) for e in eps]
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(min(args.jobs, len(work))) as pool:
            results = pool.map(run_episode, work)
    else:
        results = [run_episode(w) for w in work]
    total = dict(calls=0, equal=0, mismatch=0, input_mutated=0, changed=0, wr_status={}, calls_with_empty=0, calls_with_empty_hour23=0,
                 model_actions_with_empty=0, model_actions_with_empty_hour23=0)
    by_kind, replaced, mism = {}, {}, []
    for r in results:
        for k in ('calls', 'equal', 'mismatch', 'input_mutated', 'changed', 'calls_with_empty', 'calls_with_empty_hour23',
                  'model_actions_with_empty', 'model_actions_with_empty_hour23'):
            total[k] += r['totals'][k]
        for k, v in r['totals']['wr_status'].items():
            total['wr_status'][k] = total['wr_status'].get(k, 0) + v
        for kind, d in r['by_kind'].items():
            b = by_kind.setdefault(kind, dict(calls=0, equal=0, changed=0, with_empty=0, w9p=0))
            for k in b:
                b[k] += d[k]
        for k, v in r['replaced'].items():
            replaced[k] = replaced.get(k, 0) + v
        mism += r['mismatches']
    rules_path, rules_source = official_path()
    embedded_same = official_defs(HERE / 'postprocess_c1p_w9p.py') == official_defs(rules_path)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def rel(p):                                            # record paths relative to the repository root, never absolute
        p = Path(p).resolve(); root = HERE.parent.resolve()
        return str(p.relative_to(root)) if p.is_relative_to(root) else p.name
    passed = (embedded_same and total['mismatch'] == 0 and total['input_mutated'] == 0 and total['calls'] > 0
              and all(replaced.get(k, 0) > 0 for k in ('W3', 'W5', 'W6', 'W9', "W9'", 'WT'))
              and by_kind.get('empty_pad', {}).get('w9p', 0) > 0 and by_kind.get('overflow', {}).get('w9p', 0) > 0)
    out = dict(passed=passed, test="postprocess_c1p_w9p (KitHook.apply / apply) vs run_m.py C1p + M5 path (waste_replace2.replace_waste flags W3,W5,W6,W9,WT + run_c.Lazy + run_c.Top.finalize, then run_m.MTop.m5 with market_pp.fill_empty)",
               totals=total, by_kind=by_kind, replaced_in_changed_calls=replaced,
               episodes=[{k: r[k] for k in ('episode', 'steps', 'totals', 'replay_sha256')} for r in results], mismatches=mism[:20],
               embedded_official_defs_identical=embedded_same, rules_source=rules_source, rules_sha256=sha(rules_path),
               portable_sha256=sha(HERE / 'postprocess_c1p_w9p.py'), waste_replace2_path=rel(args.wr2), waste_replace2_sha256=sha(args.wr2),
               market_pp_path=rel(args.mpp), market_pp_sha256=sha(args.mpp),
               example_main_sha256=sha(Path(args.example) / 'main.py'), example_policy_sha256=sha(Path(args.example) / 'policy.pt'),
               random_seed=args.seed, python=platform.python_version(), seconds=round(time.time() - t0, 1))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(passed=passed, totals=total, replaced=replaced), ensure_ascii=False))
    if not passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
