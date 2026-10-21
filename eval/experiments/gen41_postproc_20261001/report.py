"""gen41 A/B の集計。使い方: report.py [games|smoke] → summary_<dir>.json と標準出力。
A と B の両方がそろった局だけを対にする(同じ opponent・seed・learner_player)。
整合: B で最終行動が初めて変わる手までは A の特徴+行動 hash と一致、変わらなかった局は全手 hash と結果が一致。"""
import glob, json, os, sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SCR = Path(os.environ.get('KAGG_SCRATCH', 'scratch'), 'gen41_postproc_20261001')
key = lambda g: (g['opponent'], g['seed'], g['learner_player'])
pts = lambda g: {'win': 1., 'draw': .5, 'loss': 0.}[g['result']]
margin = lambda g: g['own_money'] - g['opponent_money']
SOLD = ('slot', 'append', 'merge')


def boot(keys, f, rng, n=10000):
    seeds = sorted({k[1] for k in keys}); by = defaultdict(list)
    for k in keys:
        by[k[1]].append(f(k))
    sums = np.array([sum(by[s]) for s in seeds]); cnt = np.array([len(by[s]) for s in seeds])
    idx = rng.integers(0, len(seeds), size=(n, len(seeds)))
    bs = sums[idx].sum(1) / cnt[idx].sum(1)
    return dict(mean=round(float(sums.sum() / cnt.sum()), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                seeds=len(seeds))


def sign_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(d):
    rows, comp = {}, {}
    for f in glob.glob(str(d / 'seed*_wave*/games.json')):
        tag = Path(f).parent.name
        comp[tag] = json.loads((Path(f).parent / 'complete.json').read_text())
        for g in json.loads(Path(f).read_text()):
            g['_tag'] = tag; rows[key(g)] = g
    return rows, comp


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else 'games'
    A, cA = load(HERE / sub / 'A'); B, cB = load(HERE / sub / 'B')
    keys = sorted(set(A) & set(B)); rng = np.random.default_rng(0)
    tags = sorted(set(cA) & set(cB))
    hs = 'smoke_hashes' if sub == 'smoke' else 'hashes'
    # 整合
    bad = []; checked = 0; unchanged = 0; unchanged_same = 0
    for t in tags:
        ha = np.load(SCR / hs / 'A' / f'{t}.npy'); hb = np.load(SCR / hs / 'B' / f'{t}.npy')
        ks = [k for k in keys if B[k]['_tag'] == t]
        order = {key(g): j for j, g in enumerate(json.loads((HERE / sub / 'B' / t / 'games.json').read_text()))}
        for k in ks:
            j = order[k]; f = B[k]['first_change_step']
            if f is None:
                unchanged += 1
                ok = (ha[j] == hb[j]).all() and all(A[k][x] == B[k][x] for x in ('own_money', 'opponent_money', 'result'))
                unchanged_same += ok
                if not ok:
                    bad.append((k, 'unchanged differs'))
            elif not (ha[j, :f + 1] == hb[j, :f + 1]).all():
                bad.append((k, f, 'pre-divergence hash mismatch'))
            checked += 1
    out = dict(games=len(keys), jobs_paired=len(tags), integrity=dict(checked=checked, unchanged_games=unchanged, unchanged_equal=unchanged_same,
               mismatches=len(bad), samples=[str(b) for b in bad[:10]]))
    for arm, X in (('A', A), ('B', B)):
        out[arm] = dict(wdl=dict(Counter(X[k]['result'] for k in keys)), win_rate=round(float(np.mean([pts(X[k]) for k in keys])), 4) if keys else None,
                        margin_mean=round(float(np.mean([margin(X[k]) for k in keys])), 1) if keys else None)
    if keys:
        fl = Counter((A[k]['result'], B[k]['result']) for k in keys)
        up = sum(v for (a, b), v in fl.items() if pts({'result': b}) > pts({'result': a}))
        down = sum(v for (a, b), v in fl.items() if pts({'result': b}) < pts({'result': a}))
        out['flips'] = dict(loss_to_win=fl[('loss', 'win')], win_to_loss=fl[('win', 'loss')], improved=up, worsened=down,
                            other={f'{a}->{b}': v for (a, b), v in fl.items() if a != b and (a, b) not in (('loss', 'win'), ('win', 'loss'))})
        out['sign_p'] = sign_p(up, down)
        out['win_rate_diff'] = boot(keys, lambda k: pts(B[k]) - pts(A[k]), rng)
        out['margin_diff'] = boot(keys, lambda k: margin(B[k]) - margin(A[k]), rng)
        out['own_money_diff'] = boot(keys, lambda k: B[k]['own_money'] - A[k]['own_money'], rng)
        out['by_opponent'] = {}
        for n in sorted({k[0] for k in keys}):
            ks = [k for k in keys if k[0] == n]
            f2 = Counter((A[k]['result'], B[k]['result']) for k in ks)
            out['by_opponent'][n] = dict(games=len(ks), A=dict(Counter(A[k]['result'] for k in ks)), B=dict(Counter(B[k]['result'] for k in ks)),
                                         win_rate_diff=boot(ks, lambda k: pts(B[k]) - pts(A[k]), rng),
                                         margin_diff=round(float(np.mean([margin(B[k]) - margin(A[k]) for k in ks])), 1),
                                         loss_to_win=f2[('loss', 'win')], win_to_loss=f2[('win', 'loss')])
        d = [B[k].get('d23s') or {} for k in keys]
        out['d23s'] = dict(fired_games=sum(B[k].get('d23s_first_step') is not None for k in keys),
                           sold_units_per_game=round(sum(sum(r.get('sell_' + h, 0) for h in SOLD) for r in d) / len(keys), 3),
                           sell_value_per_game=round(sum(r.get('sell_value', 0) for r in d) / len(keys), 1),
                           drops_per_game=round(sum(r.get('drops', 0) for r in d) / len(keys), 3),
                           placed_units_per_game=round(sum(r.get('placed_units', 0) for r in d) / len(keys), 3),
                           not_sold_below_ratio_units_per_game=round(sum(r.get('sell_below_ratio', 0) for r in d) / len(keys), 3))
        out['changed_games_B'] = sum(B[k]['first_change_step'] is not None for k in keys)
        out['plan_fired_games_B'] = sum(B[k].get('plan_first_step') is not None for k in keys)
    cb = [cB[t] for t in tags]; ca = [cA[t] for t in tags]
    if cb:
        out['time'] = dict(B_chain_max_batch_sec=round(max(c['chain_max_batch_sec'] for c in cb), 4),
                           A_chain_max_batch_sec=round(max(c['chain_max_batch_sec'] for c in ca), 4),
                           B_d23s_max_batch_sec=round(max(c['d23s_max_batch_sec'] for c in cb), 4),
                           B_plan_max_sec_per_game=round(max(c.get('plan_max_sec') or 0 for c in cb), 4),
                           d23s_checks=dict(sum((Counter(c['d23s_checks']) for c in cb), Counter())))
    (HERE / f'summary_{sub}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
