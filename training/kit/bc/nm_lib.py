"""新マスク(LEGAL_OPT=2)移行の共通部品(newmask-distill-plan.md)。**新しいファイル。走っているジョブはこれを読まない。**

- mask2_np / mask2_t: 特徴(ou, g)だけから新マスク(LEGAL_OPT=2)を作る。kagsim.legal_mask(opt=2)・obs_feat(LEGAL_OPT=2)と同じ定義:
    待機は常に可 / 移動は盤の中なら可 / 水やり〜全部置く(5〜14)・取り出す(20〜31)・置く(32〜43)は常に可 /
    植える(15〜19)はその作物の種が手番開始に1つ以上あれば可。生きていないユニットは全部 0。
  位置は ou[..., 1:3]、種は g[27:32]*20(obs_legal の並び)。
- keep_np / keep_t: 増えた手(mask2 − mask0)のうち「手番内の連携で本当に成立しうる型」。蒸留の目標で ε を配る先。
    pickup … 納屋の隣(先のユニットが同じ手番に入れた品を取る)
    place  … 納屋の隣で、その品を持っている(先のユニットの取り出しで空きができる)
    tile   … 自分の立っているマスに**先に動く自分のユニット(番号が小さい)も立っている**ときのマスへの手(水やり〜世話 5〜13、植える 15〜19)。
             先のユニットの収穫・掘る・植える・建てる・家畜を置く で、そのマスの状態が変わりうる
  (seqmask_nm.py の実測: 逐次で本当に増える手の 83〜84% が納屋の隣の取り出し、残りがほぼ同じマスへの手。型を変えたら計画書の表も直すこと)
"""
import numpy as np
import torch

NU, NC = 20, 44
SHED_TI = (44, 45, 54, 55)     # 納屋の隣 (x,y)=(4,4)(5,4)(4,5)(5,5) → y*10+x
ALWAYS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14] + list(range(20, 44))


def mask2_np(ou, g):
    """ou (..., 20, 15), g (..., 75) -> uint8 (..., 20, 44)。"""
    ou = np.asarray(ou, np.float32); g = np.asarray(g, np.float32)
    live = ou[..., 0] > 0
    x = ou[..., 1]; y = ou[..., 2]
    L = np.zeros(ou.shape[:-1] + (NC,), np.uint8)
    L[..., 0] = 1
    L[..., 1] = y - 1 >= 0; L[..., 2] = y + 1 <= 9; L[..., 3] = x + 1 <= 9; L[..., 4] = x - 1 >= 0
    L[..., ALWAYS] = 1
    seeds = np.rint(g[..., 27:32] * 20) > 0                       # (..., 5)
    L[..., 15:20] = seeds[..., None, :]
    L[~live] = 0
    return L


def mask2_t(ou, g):
    """torch 版(学習の中で使う)。ou (B,20,15), g (B,75) -> bool (B,20,44)。"""
    B = ou.shape[0]; dev = ou.device
    live = ou[..., 0] > 0
    x = ou[..., 1]; y = ou[..., 2]
    L = torch.zeros(B, NU, NC, dtype=torch.bool, device=dev)
    L[..., 0] = True
    L[..., 1] = y - 1 >= 0; L[..., 2] = y + 1 <= 9; L[..., 3] = x + 1 <= 9; L[..., 4] = x - 1 >= 0
    L[..., ALWAYS] = True
    seeds = torch.round(g[:, 27:32].float() * 20) > 0
    L[..., 15:20] = seeds.unsqueeze(1).expand(B, NU, 5)
    return L & live.unsqueeze(-1)


def keep_t(tiles, ou):
    """増えた手のうち ε を配る型(bool (B,20,44))。mask2 & ~mask0 と掛けて使う。"""
    B = ou.shape[0]; dev = ou.device
    live = ou[..., 0] > 0
    ux = ou[..., 1].long().clamp(0, 9); uy = ou[..., 2].long().clamp(0, 9); ti = uy * 10 + ux          # (B,20)
    adj = torch.zeros_like(live)
    for s in SHED_TI: adj |= ti == s
    have = ou[..., 3:15] > 0.5                                                                          # (B,20,12) 携行品
    # 同じマスに**先に動く**自分のユニット(番号が小さい)が立っている(そのユニットがマスを変えうる)
    same = (ti.unsqueeze(2) == ti.unsqueeze(1)) & live.unsqueeze(1) & live.unsqueeze(2)          # [b, u, v]
    earlier = torch.ones(NU, NU, dtype=torch.bool, device=dev).tril(-1)                          # v < u
    shared = (same & earlier).any(-1)
    K = torch.zeros(B, NU, NC, dtype=torch.bool, device=dev)
    K[..., 20:32] = adj.unsqueeze(-1)
    K[..., 32:44] = adj.unsqueeze(-1) & have
    K[..., 5:14] = shared.unsqueeze(-1)
    K[..., 15:20] = shared.unsqueeze(-1)
    return K & live.unsqueeze(-1)


def keep_np(tiles, ou):
    return keep_t(None, torch.as_tensor(np.asarray(ou, np.float32))).numpy()      # tiles は使わない(型をそろえるための引数)
