"""gen70 の A / C / B / D / E の表と、同じ局どうしの差 C−A、B−C、D−B、E−D、D−A。使い方: report_gen70.py [games|smoke]
集計の関数は gen41 の report.py(boot・sign_p・load)と report_c.py(pair)を import して使う(編集しない)。
整合(gen41 の report.py / report_c.py と同じ検査):
  各腕 X ≠ A: X の最終行動が初めて変わる手(first_change_step)までは A の特徴+行動 hash と一致、変わらなかった局は全手 hash と結果が一致。
  参考(時間打ち切りのある段があるので診断扱い): D は MELON16 が初めて行動を変える手までは B と、E は melon_replant が初めて変える手までは D と hash 一致。
smoke では各腕の complete.json の parity_submission(提出物 sub125 / sub126 の agent() との全手照合)も並べる。"""
import json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
GEN41_DIR = HERE.parents[0] / 'gen41_postproc_20261001'
sys.path.insert(0, str(GEN41_DIR))
import report as RP                                                 # gen41 の report.py
import report_c as RC                                               # gen41 の report_c.py(pair)
assert Path(RP.__file__).resolve().parent == GEN41_DIR and Path(RC.__file__).resolve().parent == GEN41_DIR

SCR = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'gen70_postproc_20261017')
ARMS = ('A', 'C', 'B', 'D', 'E')
PAIRS = (('C', 'A'), ('B', 'C'), ('D', 'B'), ('E', 'D'), ('D', 'A'))
pts, margin, key = RP.pts, RP.margin, RP.key


def hashes(sub, arm, tag):
    return np.load(SCR / ('smoke_hashes' if sub == 'smoke' else 'hashes') / arm / f'{tag}.npy')


def order(sub, arm, tag):
    return {key(g): j for j, g in enumerate(json.loads((HERE / sub / arm / tag / 'games.json').read_text()))}


def integrity(sub, X, cX, Y, cY, x, y, field):
    """X の行が field の手まで Y と hash 一致(field が None の局は全手 hash と結果が一致)。"""
    bad = []; checked = unchanged = 0
    for t in sorted(set(cX) & set(cY)):
        hx, hy = hashes(sub, x, t), hashes(sub, y, t); ox, oy = order(sub, x, t), order(sub, y, t)
        assert ox == oy, t
        for k, j in ox.items():
            if k not in X or k not in Y:
                continue
            f = X[k].get(field); checked += 1
            if f is None:
                unchanged += 1
                ok = (hx[j] == hy[j]).all() and all(X[k][c] == Y[k][c] for c in ('own_money', 'opponent_money', 'result'))
            else:
                ok = (hx[j, :f + 1] == hy[j, :f + 1]).all()
            if not ok:
                bad.append(str((k, f)))
    return dict(checked=checked, unchanged_games=unchanged, mismatches=len(bad), samples=bad[:10])


def pair(X, Y, rng):
    keys = sorted(set(X) & set(Y))
    if not keys:
        return None
    d = RC.pair(Y, X, keys, rng)                                     # report_c.pair(X0, X1) は X1 − X0
    d['flips'] = f"+{d['loss_to_win']} / -{d['win_to_loss']}"
    return d


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else 'games'
    data = {a: RP.load(HERE / sub / a) for a in ARMS if (HERE / sub / a).is_dir()}
    data = {a: v for a, v in data.items() if v[0]}
    rows = {a: v[0] for a, v in data.items()}; comp = {a: v[1] for a, v in data.items()}
    common = sorted(set.intersection(*(set(v) for v in rows.values()))) if rows else []
    rng = np.random.default_rng(0)
    out = dict(sub=sub, arms_present=list(rows), games_common=len(common), integrity={}, diagnostic={})
    for a, X in rows.items():
        out[a] = dict(games=len(X), wdl_common=dict(Counter(X[k]['result'] for k in common)),
                      win_rate_common=round(float(np.mean([pts(X[k]) for k in common])), 4) if common else None,
                      margin_mean_common=round(float(np.mean([margin(X[k]) for k in common])), 1) if common else None,
                      changed_games=sum(X[k]['first_change_step'] is not None for k in X))
        if a != 'A' and 'A' in rows:
            out['integrity'][f'{a}_vs_A'] = integrity(sub, X, comp[a], rows['A'], comp['A'], a, 'A', 'first_change_step')
    for x, y, f in (('D', 'B', 'melon16_first_step'), ('E', 'D', 'mrep_first_step')):
        if x in rows and y in rows:
            out['diagnostic'][f'{x}_vs_{y}'] = integrity(sub, rows[x], comp[x], rows[y], comp[y], x, y, f)
    for x, y in PAIRS:
        if x in rows and y in rows:
            out[f'{x}_minus_{y}'] = pair(rows[x], rows[y], rng)
    for x, y in (('D', 'B'), ('D', 'A')):
        if x in rows and y in rows:
            ks = sorted(set(rows[x]) & set(rows[y]))
            out[f'{x}_minus_{y}_by_opponent'] = {n: RC.pair(rows[y], rows[x], [k for k in ks if k[0] == n], rng) for n in sorted({k[0] for k in ks})}
    for a, f, u in (('D', 'melon16_first_step', 'melon16_seed_units'), ('E', 'melon16_first_step', 'melon16_seed_units'), ('E', 'mrep_first_step', None)):
        if a in rows:
            X = rows[a]
            out.setdefault(f'{a}_overlays', {})[f] = dict(games_fired=sum(X[k].get(f) is not None for k in X),
                                                           seed_units_per_game=None if u is None else round(float(np.mean([X[k].get(u, 0) for k in X])), 3))
    out['time'] = {a: dict(chain_max_batch_sec=round(max(c['chain_max_batch_sec'] for c in cs.values()), 4),
                           plan_max_sec_per_game=round(max((c.get('plan_max_sec') or 0) for c in cs.values()), 4),
                           job_play_seconds_mean=round(float(np.mean([c.get('play_seconds', c['seconds']) for c in cs.values()])), 1))
                   for a, cs in comp.items()}
    if sub == 'smoke':
        out['parity_submission'] = {a: [dict((k, v) for k, v in c['parity_submission'].items() if k != 'samples') | dict(samples=c['parity_submission']['samples'][:3])
                                        for c in cs.values() if 'parity_submission' in c] for a, cs in comp.items()}
    (HERE / f'summary_{sub}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str) + '\n')
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
