"""公式リプレイからの BC 抽出(正しい経路)。

旧抽出(extract2)は片側のテープを**固定のシード・固定の相手**で再生していたため、
観測が教師の見た対局と別物になっていた(200本中0本が元の店の組と一致、実験37)。
ここでは公式リプレイの **元のシード + 両者の元の行動** で対局を再現する。
再現した状態は公式の観測と全項目・全手で一致する(verify_extract.py、不一致0)。

対応: 公式の observation steps[i] は action steps[i] 適用後 = action steps[i+1] を選ぶ時の状態。
      よって (特徴 = 状態 before 行動 i+1, ラベル = steps[i+1].action) を組にする。
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
from codec import ITEMS
from deploy import wide_blank, units_to_codes, mkt_to_codes, encode_true, NU, NSLOT
import kagsim, kagpack
NITEM = len(ITEMS)


def acts_of(d, seat):
    out = []
    for i in range(1, len(d["steps"])):
        a = d["steps"][i][seat].get("action") or {}
        out.append({"farmer": a.get("farmer") or ["PASS"], "hands": a.get("hands") or [],
                    "market": a.get("market") or []})
    return out


def make_env(d, seat):
    """教師の座席に幅広の空テープ、相手の座席に相手の元の行動、元のシード。"""
    op = kagpack.pack(acts_of(d, 1 - seat)); w = wide_blank()
    A, Bt = (w, op) if seat == 0 else (op, w)
    env = kagsim.VecMarket(A[0], A[1], Bt[0], Bt[1], [int(d["info"]["seed"])], 720, seat)
    env.keep_nonsell = False; env.reset()
    return env


def features(env):
    """方策が見る入力。公式 obs と同型(教師自身の視点、相手の private は含まない)。"""
    b = np.concatenate([np.asarray(env.tiles_base(0)), np.asarray(env.tiles_dyn(0)),
                        np.asarray(env.tiles_base(1)), np.asarray(env.tiles_dyn(1))], 3)
    board = b.transpose(0, 3, 1, 2).astype(np.float32)[0]
    g = np.concatenate([np.asarray(env.obs_legal()), np.asarray(env.farm_pub(0)),
                        np.asarray(env.farm_pub(1)),
                        np.asarray(env.shops(), np.float32) / 8.0], 1).astype(np.float32)[0]
    inv = np.asarray(env.unit_inv(NU))[0]; up = np.asarray(env.unit_pos2(0))[0]
    return board, g, inv, up


def unit_rows(inv, up):
    rows = []
    for u in range(NU):
        x, y = int(up[u, 0]), int(up[u, 1])
        if x < 0: continue
        rows.append((u, np.concatenate([inv[u], [x/9.0, y/9.0, u/19.0]]).astype(np.float32)))
    return rows


def extract_replay(d, seat):
    """1局・1座席ぶんの (特徴, ラベル)。ボードは手ごとに1枚だけ保存し、ユニットは添字で参照する
    (旧抽出はユニットごとにボードを複製して1シャード1.4GBに膨らみ、メモリ不足の一因になっていた)。"""
    me = acts_of(d, seat); env = make_env(d, seat)
    BD, G, US, UI, YV, YI, YQ, MY = [], [], [], [], [], [], [], []
    for i in range(len(me)):
        board, g, inv, up = features(env)
        uv, ui, uq, slot, nu = encode_true(me[i])
        BD.append(board); G.append(g); MY.append(slot)
        for u, su in unit_rows(inv, up):
            # 盤面に居るのに教師が指示していないユニットは PASS(エンジンでも既定は PASS)
            v, it, q = (uv[u], ui[u], uq[u]) if u < nu else (0, NITEM, 0)
            US.append(su); UI.append(i); YV.append(v); YI.append(it); YQ.append(q)
        env.step_units(units_to_codes(uv, ui, uq, nu)[None], mkt_to_codes(slot)[None])
    m = env.money()[0]
    return dict(board=np.asarray(BD, np.float16), g=np.asarray(G, np.float32),
                su=np.asarray(US, np.float32), ustep=np.asarray(UI, np.int32),
                yv=np.asarray(YV, np.int64), yi=np.asarray(YI, np.int64), yq=np.asarray(YQ, np.int64),
                my=np.asarray(MY, np.int64), final=(float(m[0]), float(m[1])))
