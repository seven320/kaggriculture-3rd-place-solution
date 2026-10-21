"""PFSP league: opponents (frozen BC nets + the learner's own past checkpoints) with the learner's running win-rate estimate.
Prioritised fictitious self-play (AlphaStar): pick opponent i with probability proportional to f_hard(x_i) = (1 - x_i)^p, where x_i is
the estimated probability that the learner beats i. Hard opponents are played more; solved ones fade but keep a floor."""
import json
import numpy as np


class League:
    def __init__(self,opponents=None,p=2.,floor=.01,prior_games=2,mode="hard"):
        self.opponents = opponents or {}  # name -> dict(path, kind='greedy'|'sampled', games, wins, draws, added_iteration)
        if mode not in ("hard","variance"): raise ValueError(f"unknown PFSP mode: {mode}")
        self.mode=mode
        self.p = p;self.floor = floor;self.prior_games = prior_games

    def add(self,name,path,kind,iteration):
        if name not in self.opponents:
            self.opponents[name] = dict(path=str(path),kind=kind,games=0,wins=0,draws=0,added_iteration=iteration)

    def win_rate(self,name):
        o = self.opponents[name];return (o['wins']+.5*o['draws']+.5*self.prior_games)/(o['games']+self.prior_games)

    def weights(self):
        names = sorted(self.opponents);rates = np.array([self.win_rate(n) for n in names]);w = ((1-rates)**self.p if self.mode=='hard' else rates*(1-rates))+self.floor
        return names,w/w.sum()

    def sample(self,k,rng):
        """k distinct opponents, successive draws without replacement under the PFSP weights."""
        names,w = self.weights();chosen = []
        for _ in range(min(k,len(names))):
            idx = rng.choice(len(names),p=w);chosen.append(names[idx]);w[idx] = 0.;w = w/w.sum() if w.sum() > 0 else w
        return chosen

    def record(self,name,learner_won,draw):
        o = self.opponents[name];o['games'] += 1;o['wins'] += int(learner_won and not draw);o['draws'] += int(draw)

    def snapshot(self):
        names,w = self.weights()
        return dict(mode=self.mode,p=self.p,floor=self.floor,prior_games=self.prior_games,opponents=json.loads(json.dumps(self.opponents)),
                    weights={n:float(x) for n,x in zip(names,w)},win_rates={n:self.win_rate(n) for n in names})

    @classmethod
    def restore(cls,snap):
        return cls(dict(snap['opponents']),snap['p'],snap['floor'],snap['prior_games'],snap.get('mode','hard'))
