"""ラダーを「測定器」に変える: Bradley-Terry 最尤で提出ごとの強さを推定する。

なぜ生の Elo とラダー順位を指標にできないか(両方ともここで潰す):

  1. 生の勝率は相手の強さで交絡する。実測: v26 は勝率 0.8750 でスコア 2273.8、
     v31 は勝率 0.6144 でスコア 2480.5。低レートのエージェントは弱い相手としか
     当たらないので勝率が上がる。勝率は強さの単調関数ではない。
  2. Elo は経路依存で、K が減衰するため序盤の引きが凍結される(pokemon 側で
     バイト同一の再提出が 1059.1 vs 952.3、run 間オフセット約105点)。
     さらに収束に 150-300 局かかるので、若い提出のスコアは「弱い」ではなく「未定」。

BT は両方を同時に外す。相手の公開レートを固定オフセットとして式に入れる
(→交絡が消える)、全局を等重みで一括最尤(→経路依存も K 減衰も無い)、
Fisher 情報から SE が出る(→何局で何点の差が有意かが言える)。

  P(勝ち | 相手レート R) = sigmoid( (theta - R) * ln10/400 )

theta が提出の強さ。引き分けは 0.5 で扱う。相手レートは lb_snapshot の
現在値で、対戦時点の値ではない(pokemon 側と同じ既知の留保)。
"""
import os, sys, json, math, collections
BASE = "/content/drive/MyDrive/kaggriculture"
OUT  = BASE + "/runs"
TEAM = "masspeaks"
C    = math.log(10) / 400.0

def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi(); a.authenticate(); return a

def load_lb():
    p = OUT + "/lb_snapshot.json"
    rows = json.load(open(p))
    return {r[0]: float(r[1]) for r in rows}

def fetch(a, refresh=True):
    cache = OUT + "/bt_games.json"
    if not refresh and os.path.exists(cache):
        return json.load(open(cache))
    subs = a.competition_submissions("kaggriculture")
    out = []
    for s in subs:
        sid = int(getattr(s, "ref", None) or s.id)
        desc = (getattr(s, "description", "") or "")
        tag = desc.split(":")[0][:6] if desc else str(sid)
        try:
            eps = a.competition_list_episodes(sid)
        except Exception as e:
            print(f"  {sid} {tag}: {e}"); continue
        n = 0
        for e in eps:
            d = json.loads(e.to_json()) if hasattr(e, "to_json") else e
            if d.get("state") != "COMPLETED": continue
            ags = d.get("agents") or []
            me = [x for x in ags if x.get("teamName") == TEAM]
            op = [x for x in ags if x.get("teamName") != TEAM]
            if len(me) != 1 or len(op) != 1: continue
            me, op = me[0], op[0]
            if int(me.get("submissionId") or 0) != sid: continue
            rm, ro = me.get("reward"), op.get("reward")
            if rm is None or ro is None: continue
            out.append({"sub": sid, "tag": tag, "ep": d["id"],
                        "opp": op.get("teamName"), "my": float(rm), "or": float(ro),
                        "t": d.get("endTime", "")})
            n += 1
        print(f"  {sid} {tag:<7} {n:4d} 局", flush=True)
    json.dump(out, open(cache, "w"))
    return out

def fit(games, lb, mean_r):
    """1提出ぶんの theta を Newton 法で最尤推定。戻り値 (theta, se, n, wr, meanR)."""
    ys, rs = [], []
    for g in games:
        y = 1.0 if g["my"] > g["or"] else (0.0 if g["my"] < g["or"] else 0.5)
        ys.append(y); rs.append(lb.get(g["opp"], mean_r))
    n = len(ys)
    if n == 0: return None
    th = sum(rs) / n
    for _ in range(200):
        gr = 0.0; H = 0.0
        for y, r in zip(ys, rs):
            p = 1.0 / (1.0 + math.exp(-C * (th - r)))
            gr += C * (y - p); H += C * C * p * (1 - p)
        if H < 1e-12: break
        step = gr / H
        th += max(-400.0, min(400.0, step))
        if abs(step) < 1e-7: break
    I = sum(C * C * (lambda p: p * (1 - p))(1.0 / (1.0 + math.exp(-C * (th - r)))) for r in rs)
    se = (1.0 / math.sqrt(I)) if I > 1e-12 else float("inf")
    sep = th - sum(rs) / n
    # 完全分離(全勝/全敗)は theta が発散する。上限を付けて印を残す。
    return dict(theta=th, se=se, n=n, wr=sum(ys) / n, meanR=sum(rs) / n,
                sep=(sum(ys) in (0.0, float(n))))

def main():
    refresh = "--cache" not in sys.argv
    # 相手レートは「現在」の値なので、古い対局ほど交絡する: 当時弱かった相手が
    # 今2500点なら、その勝ちを過大評価する。窓を切ると当時値≒現在値になる。
    hours = 1e9
    for i, x in enumerate(sys.argv):
        if x == "--since": hours = float(sys.argv[i + 1])
    lb = load_lb()
    mean_r = sum(lb.values()) / len(lb)
    a = api() if refresh else None
    games = fetch(a, refresh)
    if hours < 1e8:
        import datetime as _dt
        cut = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=hours)
        keep = []
        for g in games:
            try:
                t = _dt.datetime.fromisoformat(g["t"].replace("Z", "+00:00"))
            except Exception:
                continue
            if t >= cut: keep.append(g)
        print(f"直近 {hours:.0f}h に絞り込み: {len(games)} -> {len(keep)} 局")
        games = keep
    by = collections.defaultdict(list)
    for g in games: by[(g["sub"], g["tag"])].append(g)
    rows = []
    for (sid, tag), gs in by.items():
        f = fit(gs, lb, mean_r)
        if f: f.update(sub=sid, tag=tag); rows.append(f)
    rows.sort(key=lambda r: -r["theta"])
    print(f"\n相手レート未収録 {sum(1 for g in games if g['opp'] not in lb)}/{len(games)} 局 "
          f"(全体平均 {mean_r:.0f} で代替)\n")
    print(f"{'提出':<9}{'BT強さ':>9}{'±SE':>7}{'95%CI':>17}{'局数':>6}{'勝率':>7}{'相手平均':>9}")
    for r in rows:
        lo, hi = r["theta"] - 1.96 * r["se"], r["theta"] + 1.96 * r["se"]
        mark = " *分離" if r["sep"] else ""
        print(f"{r['tag']:<9}{r['theta']:9.1f}{r['se']:7.1f}"
              f"{('[%.0f, %.0f]' % (lo, hi)):>17}{r['n']:6d}{r['wr']:7.3f}{r['meanR']:9.1f}{mark}")
    # 対比: 共通相手だけに絞った差分(BT の仮定を使わない頑健性チェック)
    print("\n共通相手ペア比較(BTの仮定を使わない対照):")
    idx = {r["tag"]: by[(r["sub"], r["tag"])] for r in rows}
    tags = [r["tag"] for r in rows]
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            A, B = idx[tags[i]], idx[tags[j]]
            oa = collections.defaultdict(list); ob = collections.defaultdict(list)
            for g in A: oa[g["opp"]].append(1.0 if g["my"] > g["or"] else (0.0 if g["my"] < g["or"] else .5))
            for g in B: ob[g["opp"]].append(1.0 if g["my"] > g["or"] else (0.0 if g["my"] < g["or"] else .5))
            common = set(oa) & set(ob)
            if len(common) < 8: continue
            d = [sum(oa[o]) / len(oa[o]) - sum(ob[o]) / len(ob[o]) for o in common]
            m = sum(d) / len(d)
            sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1))
            se = sd / math.sqrt(len(d))
            print(f"  {tags[i]:<8} - {tags[j]:<8} 勝率差 {m:+.3f} ± {se:.3f} "
                  f"(共通相手 {len(common)}, z={m/se if se>0 else 0:+.2f})")
    json.dump(rows, open(OUT + "/bt_strength.json", "w"), indent=1)
    print(f"\n-> {OUT}/bt_strength.json")

if __name__ == "__main__":
    main()
