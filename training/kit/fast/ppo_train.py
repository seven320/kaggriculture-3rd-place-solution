"""方針ベクトル(11次元)を PPO で学習する。

階層の上段だけを学習させる。1局=30決定なので信用割当が短く、
生の行動空間(720x10ユニット)で回すのと違って PPO が機能する形になっている。
報酬は終端の金差のみ(shaping しない)。評価は別途 勝率 で行う。
"""
import os, sys, time, json, argparse
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
from kagenv import KagEnv, OBS_DIM
from kagenv2 import KagEnv2
from kagenv3 import KagEnv3
from agent3 import N_STRAT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LN4 = float(np.log(4.0))


class AC(nn.Module):
    def __init__(self, obs=OBS_DIM, act=N_STRAT, h=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.mu = nn.Linear(h, act); self.v = nn.Linear(h, 1)
        self.logstd = nn.Parameter(torch.full((act,), -0.7))
        nn.init.orthogonal_(self.mu.weight, 0.01); nn.init.zeros_(self.mu.bias)

    def forward(self, x):
        z = self.body(x)
        return self.mu(z), self.logstd.expand_as(self.mu(z)), self.v(z).squeeze(-1)

    def act(self, x):
        mu, ls, v = self(x)
        d = torch.distributions.Normal(mu, ls.exp())
        a = d.sample()
        return a, d.log_prob(a).sum(-1), v


def to_strategy(a):
    """生の行動を方針ベクトル [0.25, 4] に写す。"""
    return np.exp(np.tanh(a) * LN4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nenv", type=int, default=48)
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--opps", type=str, default="0,4,9,15,20,25")
    ap.add_argument("--out", type=str, default="/content/drive/MyDrive/kaggriculture/runs/ppo1")
    ap.add_argument("--seedbase", type=int, default=900000)
    ap.add_argument("--vec", type=int, default=0, help="1=numpyベクトル実行層 2=GPU実行層(大バッチで4倍速)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    opps = [int(x) for x in a.opps.split(",")]
    net = AC().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    ck = os.path.join(a.out, "ck.pt")
    it0 = 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location=DEV)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); it0 = st["it"]
        print("再開 iter=%d" % it0, flush=True)
    log = open(os.path.join(a.out, "log.txt"), "a")
    rng = np.random.default_rng(0)
    for it in range(it0, a.iters):
        opp = opps[it % len(opps)]
        base = a.seedbase + (it // len(opps)) * a.nenv * 4      # シードは毎回ずらす
        EnvC = KagEnv3 if a.vec == 2 else (KagEnv2 if a.vec else KagEnv)
        env = EnvC([base + j for j in range(a.nenv)], opp_idx=opp)
        ob = env.reset()
        OB, AC_, LP, VL, = [], [], [], []
        t0 = time.time()
        for d in range(30):
            x = torch.as_tensor(ob, dtype=torch.float32, device=DEV)
            with torch.no_grad(): act, lp, v = net.act(x)
            OB.append(ob.copy()); AC_.append(act.cpu().numpy())
            LP.append(lp.cpu().numpy()); VL.append(v.cpu().numpy())
            ob, rew, done = env.step(to_strategy(act.cpu().numpy()))
        dt = time.time() - t0
        # 終端のみ報酬。30ステップなので割引はせず、優位は (R - V) で作る。
        R = rew.astype(np.float32)
        OB = np.stack(OB); AC_ = np.stack(AC_); LP = np.stack(LP); VL = np.stack(VL)
        adv = (R[None, :] - VL)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = np.broadcast_to(R[None, :], VL.shape).copy()
        T, N = VL.shape
        ob_t = torch.as_tensor(OB.reshape(T*N, -1), device=DEV)
        ac_t = torch.as_tensor(AC_.reshape(T*N, -1), device=DEV)
        lp_t = torch.as_tensor(LP.reshape(-1), device=DEV)
        ad_t = torch.as_tensor(adv.reshape(-1), device=DEV)
        rt_t = torch.as_tensor(ret.reshape(-1), device=DEV)
        idx = np.arange(T*N)
        for _ in range(a.epochs):
            rng.shuffle(idx)
            for s in range(0, len(idx), 256):
                mb = torch.as_tensor(idx[s:s+256], device=DEV)
                mu, ls, v = net(ob_t[mb])
                d = torch.distributions.Normal(mu, ls.exp())
                nlp = d.log_prob(ac_t[mb]).sum(-1)
                ratio = (nlp - lp_t[mb]).exp()
                l1 = ratio * ad_t[mb]
                l2 = torch.clamp(ratio, 1-a.clip, 1+a.clip) * ad_t[mb]
                pol = -torch.min(l1, l2).mean()
                vf = ((v - rt_t[mb])**2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pol + 0.5*vf - 0.003*ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        m = env.final
        wr = float((m[:, 0] > m[:, 1]).mean())
        line = ("iter%5d 相手%2d 自分%8.0f 相手%8.0f 差%9.0f 勝率%.3f  %.1fs %.2f局/秒"
                % (it, opp, m[:, 0].mean(), m[:, 1].mean(), (m[:, 0]-m[:, 1]).mean(), wr,
                   dt, a.nenv/dt))
        print(line, flush=True); log.write(line+"\n"); log.flush()
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it+1}, ck)
        # 成長を後から測り直せるように、50周ごとに番号つきで残す(上書きしない)。
        if (it + 1) % 50 == 0:
            torch.save({"net": net.state_dict(), "it": it+1},
                       os.path.join(a.out, "ck_%05d.pt" % (it+1)))


if __name__ == "__main__":
    main()
