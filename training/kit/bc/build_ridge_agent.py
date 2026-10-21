"""exp197: fam_router.py の出力(リッジの規則)から、build_mix2.py に渡せる「家族エージェント」のフォルダを作る。
   出力フォルダ: actions.json(テープの束。0本目 = 固定の1本)と main.py(`_RULE = {店の組: テープ番号}` の1行を含む)。
   その後 `build_mix2.py <mix_ug のフォルダ> <このフォルダ> <出力> wsl` で r53 の修理つきの提出フォルダになる。
   使い方: python3 build_ridge_agent.py <fam_router.json> <家族名 GermanJurado1|ultimatum_game> <リプレイdir> <出力dir>
"""
import json, os, sys

res, fam, rpdir, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
R = json.load(open(res))[fam]
rule_e = R["rule_ridge"]                                   # 組 → テープの eid
used = [R["fixed"]] + sorted(set(rule_e.values()) - {R["fixed"]})   # 0本目は固定の1本(規則に無い組の既定)


def acts_of(d, seat):
    return [{"farmer": (d["steps"][i][seat].get("action") or {}).get("farmer") or ["PASS"],
             "hands": list((d["steps"][i][seat].get("action") or {}).get("hands") or []),
             "market": list((d["steps"][i][seat].get("action") or {}).get("market") or [])}
            for i in range(1, len(d["steps"]))]


T = []
for e in used:
    d = json.load(open(os.path.join(rpdir, e + ".json")))
    seat = d["info"]["TeamNames"].index(fam)
    T.append(acts_of(d, seat)[:719])
rule = {k: used.index(v) for k, v in rule_e.items() if used.index(v) != 0}   # 0 は既定なので書かない
os.makedirs(out, exist_ok=True)
json.dump(T, open(os.path.join(out, "actions.json"), "w"))
open(os.path.join(out, "main.py"), "w").write(
    '"""exp197 のリッジ家族ルーター(%s 家族、%d本)。店の組 → テープ番号。0 = %s(固定)。"""\n_RULE = %r\n' % (fam, len(used), R["fixed"], rule))
print("作成 %s: テープ %d 本 %s、規則 %d 組(残り %d 組は0本目)" % (out, len(used), used, len(rule), len(rule_e) - len(rule)))
