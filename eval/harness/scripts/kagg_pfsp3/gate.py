"""Gate against the current best version (pfsp3). `head_to_head_graph` plays current vs best with two GreedyGraphOpponent nets in fp32
(autocast off, TF32 off for the duration), greedy, the reference bundle's post-processing on both sides, on `seeds` x both players;
this replaces the BatchAgent path of scripts/kagg_vs_submission/compare.py (review item: Python decode per row) and was checked
to give the same outcomes (see docs). `head_to_head` (the original path) is kept for parity tests."""
from contextlib import ExitStack
from pathlib import Path
import torch
from scripts.kagg_ppo.native import NativePool
from scripts.kagg_pool_opp.collect import GreedyGraphOpponent
from scripts.kagg_pfsp.league import League


def head_to_head(current_weights,best_weights,seeds,batch,repair=None):
    from scripts.kagg_vs_submission.compare import inference,play
    rows = play(inference(str(current_weights),repair,batch),inference(str(best_weights),repair,batch),list(seeds))
    return rows,summarize(rows)


def summarize(rows):
    w = sum(r['result'] == 'win' for r in rows);d = sum(r['result'] == 'draw' for r in rows)
    return dict(n=len(rows),wins=w,draws=d,losses=len(rows)-w-d,win_rate=(w+.5*d)/len(rows),margin_mean=sum(r['margin'] for r in rows)/len(rows))


def head_to_head_graph(current_weights,best_weights,seeds,tf32_after=True):
    seeds = list(seeds);n = len(seeds)
    assert seeds == list(range(seeds[0],seeds[0]+n)),'consecutive seeds (one native pool per player assignment)'
    tf32 = torch.backends.cuda.matmul.allow_tf32;cur = best = None
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    try:
        cur = GreedyGraphOpponent(Path(current_weights),'gpu',n,autocast_bf16=False);best = GreedyGraphOpponent(Path(best_weights),'gpu',n,autocast_bf16=False)
        rows = []
        with ExitStack() as stack:
            pools = [stack.enter_context(NativePool(n,720,seeds[0],mask_versions=[cur.mask_version if player==p else best.mask_version for _ in range(n) for player in (0,1)])) for p in (0,1)]  # pool p: current is player p
            for t in range(719):
                for p,pool in enumerate(pools):
                    mine = [2*i+p for i in range(n)];theirs = [2*i+1-p for i in range(n)]
                    actions = [None]*(2*n)
                    for r,a in zip(mine,cur.act(pool,mine)): actions[r] = a
                    for r,a in zip(theirs,best.act(pool,theirs)): actions[r] = a
                    pool.step(actions)
            for p,pool in enumerate(pools):
                for i,seed in enumerate(seeds):
                    money = pool.meta[2*i,4:6].tolist();own,other = money[p],money[1-p]
                    rows.append(dict(seed=seed,player=p,own_money=own,opponent_money=other,margin=own-other,result='win' if own > other else 'loss' if own < other else 'draw'))
        rows.sort(key=lambda r:(r['seed'],r['player']))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32 and tf32_after;torch.backends.cudnn.allow_tf32 = tf32 and tf32_after
        cur = best = None;torch.cuda.empty_cache()
    return rows,summarize(rows)


class GateLeague(League):
    def remove(self,name):
        self.opponents.pop(name,None)

    def generation_names(self):
        return sorted(n for n in self.opponents if n.startswith('gen_'))

    @classmethod
    def restore(cls,snap):
        return cls(dict(snap['opponents']),snap['p'],snap['floor'],snap['prior_games'],snap.get('mode','hard'))
