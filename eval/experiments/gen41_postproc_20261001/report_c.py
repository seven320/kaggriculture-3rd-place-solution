"""A / C / B の 3 行表と、同じ局どうしの差 C−A、B−C、B−A。使い方: report_c.py [games|smoke]
整合: C の最終行動が初めて変わる手までは A の特徴+行動 hash と一致(report.py と同じ検査)。"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import report as RP

HERE = RP.HERE; SCR = RP.SCR; pts = RP.pts; margin = RP.margin; key = RP.key


def pair(X, Y, keys, rng):
    fl = Counter((X[k]['result'], Y[k]['result']) for k in keys)
    up = sum(v for (a, b), v in fl.items() if pts({'result': b}) > pts({'result': a}))
    down = sum(v for (a, b), v in fl.items() if pts({'result': b}) < pts({'result': a}))
    return dict(games=len(keys), loss_to_win=fl[('loss', 'win')], win_to_loss=fl[('win', 'loss')], improved=up, worsened=down,
                sign_p=RP.sign_p(up, down), win_rate_diff=RP.boot(keys, lambda k: pts(Y[k]) - pts(X[k]), rng),
                margin_diff=RP.boot(keys, lambda k: margin(Y[k]) - margin(X[k]), rng))


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else 'games'
    arms = {a: RP.load(HERE / sub / a) for a in ('A', 'B', 'C')}
    A, B, C = (arms[a][0] for a in ('A', 'B', 'C'))
    keys = sorted(set(A) & set(C) & (set(B) if sub == 'games' else set(C)))
    rng = np.random.default_rng(0)
    hs = 'smoke_hashes' if sub == 'smoke' else 'hashes'
    bad = []; checked = 0
    for t in sorted(set(arms['A'][1]) & set(arms['C'][1])):
        ha = np.load(SCR / hs / 'A' / f'{t}.npy'); hc = np.load(SCR / hs / 'C' / f'{t}.npy')
        order = {key(g): j for j, g in enumerate(json.loads((HERE / sub / 'C' / t / 'games.json').read_text()))}
        for k, j in order.items():
            f = C[k]['first_change_step']
            ok = (ha[j] == hc[j]).all() if f is None else (ha[j, :f + 1] == hc[j, :f + 1]).all()
            checked += 1
            if not ok:
                bad.append(str((k, f)))
    out = dict(games=len(keys), integrity_C_vs_A=dict(checked=checked, mismatches=len(bad), samples=bad[:10]))
    for a, X in (('A', A), ('C', C), ('B', B)):
        ks = [k for k in keys if k in X]
        out[a] = dict(wdl=dict(Counter(X[k]['result'] for k in ks)), margin_mean=round(float(np.mean([margin(X[k]) for k in ks])), 1))
    if sub == 'games' and keys:
        out['C_minus_A'] = pair(A, C, keys, rng); out['B_minus_C'] = pair(C, B, keys, rng); out['B_minus_A'] = pair(A, B, keys, rng)
        out['B_minus_C_by_opponent'] = {n: pair(C, B, [k for k in keys if k[0] == n], rng) for n in sorted({k[0] for k in keys})}
    else:
        out['C_minus_A'] = pair(A, C, keys, rng)
    out['C_waste'] = dict(units_per_game=round(float(np.mean([C[k]['waste_units'] for k in keys])), 3),
                          value_per_game=round(float(np.mean([C[k]['waste_value'] for k in keys])), 1),
                          games_with_waste=sum(C[k]['waste_units'] > 0 for k in keys))
    cc = list(arms['C'][1].values())
    out['C_time'] = dict(chain_max_batch_sec=round(max(c['chain_max_batch_sec'] for c in cc), 4),
                         plan_max_sec_per_game=round(max(c['plan_max_sec'] for c in cc), 4))
    (HERE / f'summary_c_{sub}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
