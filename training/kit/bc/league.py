"""総当たりリーグ(反応する強い相手どうし)。チームの Rust エンジンの batch2 で、両席とも我々の Python エージェントを打たせる。
   エージェント指定: 名前:種類:フォルダ(種類 bc = main.py+policy.pt の BC(修理層込み)、py = ルールだけの main.py(局ごとに別の実体))
   各組を SEEDS 局(半分ずつ席を入れ替え)。使い方: AGENTS="..." NSEED=48 python3 league.py 出力.json"""
import os, sys, json, time, types, subprocess, itertools
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_tf as M
from rust_eval2 import greedy_batch, greedy_batch_ens
# 09-22: 合法判定の版は各エージェントの obs_feat.py の既定(旧 = 版0、新マスクで組んだもの = 版2)で決める。
#   環境変数 LEGAL_OPT が入っていると新マスクの体だけ黙って上書きされるので、ここで外す。
if os.environ.pop("LEGAL_OPT", None) is not None: print("!! 環境変数 LEGAL_OPT を外した(版は各エージェントの既定で決める)", flush=True)
ARENA = "/content/lab/rust_arena/target/release/arena"; DEV = "cuda"

# チームの Rust の読み取りは市場の空の枠 [] や読めない注文を受け付けない。公式環境ではそれらは「何もせず枠を1つ占める」ので、
# 同じ意味になる「数量0の SELL」(Rust でも何もせず枠を占める。rules.rs の process_market で確認)に置き換える。
_ITEMS = {"WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER","GOOSE","COW","SHEEP"}
_UNIT = {"PASS","NORTH","SOUTH","EAST","WEST","PLANT","WATER","HARVEST","FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","PICKUP","DROP","PLACE","FEED","COLLECT_FERTILIZER","CARE"}
NOOP = ["SELL", "WHEAT", 0]
def _okq(v): return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 2 ** 32
def sanitize(a):
    if not isinstance(a, dict): return {"farmer": ["PASS"], "hands": [], "market": []}
    def unit(u): return u if isinstance(u, list) and u and u[0] in _UNIT else ["PASS"]
    mk = []
    for o in (a.get("market") or [])[:10]:
        if isinstance(o, list) and o and o[0] in ("HIRE", "BUY_LAND"): mk.append([o[0]])
        elif isinstance(o, list) and len(o) >= 3 and o[0] in ("SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL") and o[1] in _ITEMS and _okq(o[2]): mk.append(o[:3])
        else: mk.append(NOOP)
    return {"farmer": unit(a.get("farmer") or ["PASS"]), "hands": [unit(h) for h in (a.get("hands") or [])], "market": mk}

def load_mod(folder, name):
    # 09-22: 同じプロセスで2体を読むと、2体目の `from obs_feat import ...` が1体目の obs_feat(sys.modules のキャッシュ)を掴み、
    #   2体とも**先に読んだ方の合法判定(マスク)**で打っていた。旧マスク 対 新マスクの評価(g117l2 0.486 など)は全部これで無効。
    #   フォルダ内の .py はエージェントごとに読み直し、読み終えたら元のキャッシュに戻す。
    #   main.py 自身も sys.path に足すので、sys.path は丸ごと元に戻す。フォルダの市場モジュール(model_mkt_infer 等)は
    #   mod._loc に取っておき、BC が後から import せずにそれを使う(後から import すると1体目の版を掴む)。
    tp = os.path.join(folder, "main.py"); sp_ = sys.path[:]; sys.path.insert(0, folder)
    loc = [f[:-3] for f in os.listdir(folder) if f.endswith(".py") and f != "main.py"]
    saved = {k: sys.modules.pop(k) for k in loc if k in sys.modules}
    mod = types.ModuleType(name); mod.__dict__["__file__"] = tp
    try:
        exec(compile(open(tp).read(), tp, "exec"), mod.__dict__)
        for k in ("model_mkt_infer", "mkt_infer_feat", "model_ptr_infer"):   # main.py が読まなかった分もこのフォルダの版で読む
            if k in loc and k not in sys.modules: __import__(k)
        mod._loc = {k: sys.modules[k] for k in loc if k in sys.modules}
    finally:
        sys.path[:] = sp_
        for k in loc: sys.modules.pop(k, None)
        sys.modules.update(saved)
    return mod
class BC:
    def __init__(self, folder):
        self.m = load_mod(folder, "bc_%d" % id(self))
        cfg = json.load(open(os.path.join(folder, "cfg.json")))
        self.mkt = cfg.pop("mkt", False); ptr = cfg.pop("ptr", False)
        if self.mkt:                          # exp266: 市場16特徴つき(推論でも特徴を作る)
            import json as _j
            MK = self.m._loc.get("model_mkt_infer") or __import__("model_mkt_infer"); MF = self.m._loc.get("mkt_infer_feat") or __import__("mkt_infer_feat")
            self.net = MK.PolicyMktInfer(ptr=ptr, **cfg).to(DEV)
            tgt = _j.load(open(os.path.join(folder, "stock_target.json")))
            self.MF = MF; self.tgt = tgt; self.feats = {}
        elif ptr:                             # exp264: 市場の頭がポインタ頭の版
            MP = self.m._loc.get("model_ptr_infer") or __import__("model_ptr_infer"); self.net = MP.PolicyPtrInfer(**cfg).to(DEV)
        else: self.net = M.Policy(**cfg).to(DEV)
        self.net.load_state_dict(torch.load(os.path.join(folder, "policy.pt"), map_location=DEV)); self.net.eval()
        # exp309: 同じ folder に policy_a.pt … があれば logit アンサンブルにする(無ければ今までと完全に同じ1体)
        self.nets = [self.net]
        for _s in "abcdefg":
            _f = os.path.join(folder, "policy_%s.pt" % _s)
            if not os.path.exists(_f): continue
            _n = M.Policy(**cfg).to(DEV); _sd = torch.load(_f, map_location=DEV)
            _n.load_state_dict(_sd.get("net", _sd) if isinstance(_sd, dict) and "net" in _sd else _sd); _n.eval(); self.nets.append(_n)
        self.pace = getattr(self.m, "_pace", None)
        self.land = getattr(self.m, "_land", None)      # exp290: 4区画目のガードレール(ある場合だけ)
    def start(self, n):
        self.feats = [None] * n

    def act(self, obs_list, idx=None):
        F = [self.m.featurize(o) for o in obs_list]; st = tuple(np.stack([f[k] for f in F]) for k in range(5))
        mk = None
        if getattr(self, "mkt", False):
            rows = []
            for j, o in enumerate(obs_list):
                gi = idx[j] if idx is not None else j
                if self.feats[gi] is None: self.feats[gi] = self.MF.Feats(self.tgt, self.tgt["products"])
                t = o.get("step")
                if t is None: t = int(o.get("day", 0)) * 24 + int(o.get("hour", 0))
                rows.append(self.feats[gi].push(o, self._seat(o), int(t)))
            mk = np.array(rows, np.float32)
        ul, uq, ms, mq = (greedy_batch(self.net, st, mkt=mk) if len(self.nets) == 1 else greedy_batch_ens(self.nets, st, mkt=mk)); out = []
        for j, o in enumerate(obs_list):
            a = self.m.to_action(ul[j], uq[j], ms[j], mq[j], int((st[1][j][:, 0] > 0).sum()))
            a = self.pace(a, o) if self.pace is not None else a
            out.append(self.land(a, o) if self.land is not None else a)
        return out

    @staticmethod
    def _seat(o):
        p = o.get("index")
        if p is None: p = o.get("player", 0)
        return int(p or 0)


class PY:
    def __init__(self, folder): self.folder = folder
    def start(self, n): self.inst = [load_mod(self.folder, "py_%d_%d" % (id(self), i)) for i in range(n)]
    def act(self, obs_list, idx): return [self.inst[i].agent(o, {"episodeSteps": 720}) for i, o in zip(idx, obs_list)]
def play(A, B, seeds, seats):
    n = len(seeds); A.start(n); B.start(n)
    pr = subprocess.Popen([ARENA, "batch2"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    pr.stdin.write(json.dumps({"games": [dict(seed=s) for s in seeds]}) + "\n"); pr.stdin.flush()
    play.shops = [None] * n                                                   # exp246: 局ごとの開いた店(最後に見えたもの)
    MIDT = [int(x) for x in os.environ.get("MIDT", "300,500").split(",") if x]   # exp316: 途中の所持金(終盤の伸びを見るため)
    play.mid = {t: [None] * n for t in MIDT}
    tt = -1
    while True:
        msg = json.loads(pr.stdout.readline())
        if msg.get("done"): res = msg["results"]; break
        tt += 1
        o = (msg["obs0"], msg["obs1"]); act = ([None] * n, [None] * n)
        for i in range(n):
            if o[0][i] is not None:
                try: play.shops[i] = list((o[0][i].get("town") or {}).get("unlocked_shops") or [])
                except Exception: pass
                if tt in play.mid:
                    try: play.mid[tt][i] = [float(f["money"]) for f in o[0][i]["farms"]]
                    except Exception: pass
        for who, ag in ((0, A), (1, B)):
            idx = [i for i in range(n) if o[0][i] is not None]
            if not idx: continue
            sidx = [(i, seats[i] if who == 0 else 1 - seats[i]) for i in idx]          # その局でこのエージェントが座る席
            obs = [o[s][i] for i, s in sidx]
            acts = ag.act(obs, idx)
            for (i, s), a in zip(sidx, acts): act[s][i] = sanitize(a)
        pr.stdin.write(json.dumps({"a0": act[0], "a1": act[1]}) + "\n"); pr.stdin.flush()
    pr.wait()
    return [(r[seats[i]], r[1 - seats[i]]) for i, r in enumerate(res)]          # (A の所持金, B の所持金)
if __name__ == "__main__":
    out = sys.argv[1]; R = json.load(open(out)) if os.path.exists(out) else {}
    spec = [x.split(":", 2) for x in os.environ["AGENTS"].split(",")]
    AG = {n: (BC(f) if t == "bc" else PY(f)) for n, t, f in spec}
    NS = int(os.environ.get("NSEED", 48)); seeds = [int(os.environ.get("SEED0", 880001)) + 7919 * k for k in range(NS)]; seats = [k % 2 for k in range(NS)]   # SEED0: 別の種で追加の局を打つとき
    for (a, _, _), (b, _, _) in itertools.combinations(spec, 2):
        key = "%s|%s" % (a, b)
        if key in R: continue
        t = time.time(); r = play(AG[a], AG[b], seeds, seats)
        R[key] = r; R[key + "#shops"] = play.shops
        for t_, v_ in play.mid.items():                                        # exp316: 途中の所持金(絶対座席)。席の入れ替えは読む側で seats を使う
            R["%s#money%d" % (key, t_)] = v_
        R[key + "#seats"] = seats
        json.dump(R, open(out, "w"))
        w = np.mean([x > y for x, y in r]); d = np.mean([x - y for x, y in r])
        print("%-14s 対 %-14s %d局  %s の勝率 %.3f 差 %+6.0f(%s %.0f / %s %.0f)%.0fs" % (a, b, len(r), a, w, d, a, np.median([x for x, _ in r]), b, np.median([y for _, y in r]), time.time() - t), flush=True)
    names = [n for n, _, _ in spec]; S = {n: [] for n in names}
    for key, r in R.items():
        a, b = key.split("|")
        if a in S and b in S:
            for x, y in r: S[a].append(float(x > y) + 0.5 * (x == y)); S[b].append(float(y > x) + 0.5 * (x == y))
    print("\n=== 総合(全相手への平均勝率)===")
    for n in sorted(names, key=lambda n: -np.mean(S[n]) if S[n] else 0):
        if S[n]: print("  %-14s %.3f ± %.3f(%d局)" % (n, np.mean(S[n]), np.std(S[n]) / max(1, len(S[n])) ** .5, len(S[n])))
