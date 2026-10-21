import collections
from kaggle.api.kaggle_api_extended import KaggleApi
import kagglesdk.competitions.types.competition_api_service as C
a = KaggleApi(); a.authenticate()
SUBS = [(56664798,"v107 rs21"),(56661176,"v106 rs05liq")]
with a.build_kaggle_client() as k:
    cc = k.competitions.competition_api_client
    for sid, nm in SUBS:
        er = C.ApiListSubmissionEpisodesRequest(); er.submission_id = sid; by = collections.defaultdict(list)
        for e in cc.list_submission_episodes(er).episodes:
            ag = list(e.agents); m=[x for x in ag if x.team_name=="DSM"]; o=[x for x in ag if x.team_name=="M & M & P & Q"]
            if len(ag)!=2 or not m or not o or m[0].reward is None or o[0].reward is None: continue
            by[o[0].submission_id].append((m[0].reward>o[0].reward, m[0].reward-o[0].reward))
        print(nm, " / ".join("PFN %s: %d/%d 差%+.0f" % (s, sum(w for w,_ in v), len(v), sum(d for _,d in v)/len(v)) for s,v in sorted(by.items())))
