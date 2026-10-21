"""最新の棋譜を常に取る(ユーザー 09-15「最新の棋譜は常に手に入れてくださいね」)。何度でも実行でき、既にある局は飛ばす(新しい局から取る)。
  1) 我々の提出(直近 OURN 個の COMPLETE)の全局 → data/rp/our<提出ID>/
  2) リーダーボード上位 TOPN チームの、公開スコア上位 SUBN 個の提出の最新 PER 局 → data/rp/latest/<チーム>_<提出ID>/
  429 は待って取り直す。進みは data/rp/latest/manifest.json とログに。
  使い方: TOPN=12 SUBN=2 PER=60 OURN=3 python3 fetch_latest.py
"""
import json, os, sys, time, re
from kaggle.api.kaggle_api_extended import KaggleApi
import kagglesdk.competitions.types.competition_api_service as C
W = "/workspace/kaggriculture/data/rp"
TOPN, SUBN, PER, OURN = (int(os.environ.get(k, d)) for k, d in (("TOPN", 12), ("SUBN", 2), ("PER", 60), ("OURN", 3)))
GAP, RETRY = float(os.environ.get("GAP", 1.5)), int(os.environ.get("RETRY", 10))
safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:32]
MAN = W + "/latest/manifest.json"; os.makedirs(W + "/latest", exist_ok=True)
man = json.load(open(MAN)) if os.path.exists(MAN) else {}
t0 = time.time(); tot = 0


def grab(cc, eps, d):
    """eps(新しい順)を d に。既にあるものは飛ばす。取った数を返す。"""
    global tot
    os.makedirs(d, exist_ok=True); got = 0
    for e in eps:
        p = "%s/%d.json" % (d, e)
        if os.path.exists(p): continue
        wait = 30.0
        for tr in range(RETRY):
            try:
                rr = C.ApiGetEpisodeReplayRequest(); rr.episode_id = e
                c = cc.get_episode_replay(rr).content
                open(p + ".tmp", "wb").write(c); os.replace(p + ".tmp", p); got += 1; tot += 1; break
            except Exception as ex:
                s = str(ex)
                if "429" in s or "Too Many" in s: time.sleep(wait); wait = min(wait * 2, 300)
                else: print("   失敗 %s %s" % (e, s[:80]), flush=True); time.sleep(5); break
        time.sleep(GAP)
    return got


def episodes(cc, sid):
    er = C.ApiListSubmissionEpisodesRequest(); er.submission_id = sid
    eps = cc.list_submission_episodes(er).episodes
    return [int(e.id) for e in sorted(eps, key=lambda e: str(e.end_time), reverse=True)]


a = KaggleApi(); a.authenticate()
with a.build_kaggle_client() as k:
    cc = k.competitions.competition_api_client
    # 1) 我々の提出
    ours = [s for s in a.competition_submissions("kaggriculture") if str(getattr(s, "status", "")).lower().endswith("complete")][:OURN]
    for s in ours:
        sid = int(s.ref); eps = episodes(cc, sid); g = grab(cc, eps, "%s/our%d" % (W, sid))
        man["our%d" % sid] = dict(n=len(eps), desc=str(getattr(s, "description", ""))[:60], t=time.strftime("%Y-%m-%dT%H:%M"))
        print("我々 %d(%s): 局 %d、新規 %d" % (sid, str(getattr(s, "description", ""))[:30], len(eps), g), flush=True)
    # 2) 上位チーム
    rq = C.ApiGetLeaderboardRequest(); rq.competition_name = "kaggriculture"; rq.page_size = 100
    lb = cc.get_leaderboard(rq).submissions
    for i, row in enumerate(lb[:TOPN]):
        try:
            sr = C.ApiListTeamPublicSubmissionsRequest(); sr.team_id = row.team_id
            subs = sorted(cc.list_team_public_submissions(sr).submissions, key=lambda z: float(getattr(z, "public_score", 0) or 0), reverse=True)[:SUBN]
            for z in subs:
                sid = int(z.id); eps = episodes(cc, sid)[:PER]; d = "%s/latest/%s_%d" % (W, safe(row.team_name), sid)
                g = grab(cc, eps, d)
                man[os.path.basename(d)] = dict(team=row.team_name, rank=i + 1, score=float(row.score), sub_score=float(getattr(z, "public_score", 0) or 0), t=time.strftime("%Y-%m-%dT%H:%M"))
                json.dump(man, open(MAN, "w"), ensure_ascii=False, indent=0)
                print("%2d %-22s 提出 %d(%.1f): 最新 %d 局のうち新規 %d(%.0f分)" % (i + 1, row.team_name[:22], sid, man[os.path.basename(d)]["sub_score"], len(eps), g, (time.time() - t0) / 60), flush=True)
        except Exception as ex:
            print("%2d %-22s ERR %s" % (i + 1, row.team_name[:22], str(ex)[:80]), flush=True)
json.dump(man, open(MAN, "w"), ensure_ascii=False, indent=0)
print("DONE 新規 %d 局(%.0f分)" % (tot, (time.time() - t0) / 60), flush=True)
