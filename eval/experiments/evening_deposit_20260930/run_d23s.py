"""D23S の A/B(2026-10-01)。run_ev.py(走行中の実験が凍結)を import し、行動を変える段だけ D23S に差し替える。
使い方: EV_PROTOCOL=protocol_d23s.json ED_START=708 ED_ALPHA=0.01 PLAN_PROCS=1 EV_V=0 run_d23s.py d23s <first_job> <last_job> [smoke]
  smoke は run_ev と同じ(EV_SMOKE_OPP の 32 局、smoke/ に出す)。"""
import sys, traceback, time
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_ev as RE
import postprocess_d23s as D
import os


def _g_sources():
    """run_g35.sources と同じ一覧。鍵はリポジトリ root からの相対パス(site-packages の中は site-packages/ から。run.relkey)。"""
    g = RE.G; root = RE.ROOT
    ps = [g.HERE / 'run_g35.py', g.MPP / 'run_m.py', g.MPP / 'market_pp.py', g.C.HERE / 'run_c.py', g.S.HERE / 'run.py', Path(g.WR2.__file__),
          g.S.POOLDIR / 'nmb_eval.py', g.S.OFFICIAL, g.S.LIBRARY]
    return {g.S.relkey(p): g.sha(p) for p in ps}


RE.G.sources = _g_sources

RE.KS['d23s'] = 0; RE.KS['d23s_p1'] = 0
MIN_RATIO = {'d23s': None, 'd23s_p1': 1.0}


class D23STop(RE.EVTop):
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
            new = D.apply_d23s(full, RE.S.CONFIG, a, events=ev, min_ratio=self.min_ratio)
            if new is not a:
                self.changed[j] += 1; self.first.setdefault(j, step)
                self.pending[j] = RE.EV.project_shed(full, RE.S.CONFIG, new)
            if step % 24 == 23 and step // 24 <= 28:
                _, lost, dl = self.pending[j] if new is not a else RE.EV.project_shed(full, RE.S.CONFIG, new)
                pr = full['market']['prices']
                self.waste_units[j] += sum(lost.values()) + dl; self.waste_value[j] += sum(pr.get(k, 0) * v for k, v in lost.items())
            for e in ev:
                e['game'] = j
            self.events.extend(ev)
            out.append(new)
        dt = time.perf_counter() - t0
        self.seconds += dt; self.max_sec = max(self.max_sec, dt)
        return out


_per_game = RE.per_game


def per_game(events, n):
    rows = _per_game(events, n)
    for e in events:
        if e['kind'] == 'EV_sell':
            for how, item, k in e['added']:
                if how == 'below_ratio':                                  # run_ev.per_game は no_slot 以外を売った額に数えるので戻す
                    rows[e['game']]['sell_value'] -= e['price'][item] * k
        if e['kind'] == 'D23S_unit':
            r = rows[e['game']]; r['d23s_' + e['result']] += 1
            if 'placed' in e:
                r['placed_units'] += sum(e['placed'].values()); r['drops'] += 1
    return rows


_sources = RE.sources


def sources():
    paths = _sources()
    for p in [HERE / 'run_d23s.py', HERE / 'postprocess_d23s.py']:
        paths[str(p.resolve().relative_to(RE.ROOT))] = RE.G.sha(p)
    return paths


RE.EVTop = D23STop; RE.per_game = per_game; RE.sources = sources

if __name__ == '__main__':
    try:
        RE.torch.manual_seed(20260930)
        assert sys.argv[1] in MIN_RATIO
        D23STop.min_ratio = MIN_RATIO[sys.argv[1]]
        RE.run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), smoke=len(sys.argv) > 4 and sys.argv[4] == 'smoke')
    except BaseException as e:
        RE.G.dump(HERE / f'failure_d23s_{"_".join(sys.argv[1:])}.json', dict(type=type(e).__name__, message=str(e),
                  traceback=traceback.format_exc(), time=time.time(), argv=sys.argv))
        raise
