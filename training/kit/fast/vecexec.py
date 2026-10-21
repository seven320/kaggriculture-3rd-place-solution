"""N シードぶんの実行層を一括で計算する(タイル仕事の判定 + ユニット割当)。

なぜ: プロファイルで act 本体が 13.4s/35s、tile_jobs が 260万回、dict.get が 2,290万回。
シードごとの Python ループが 140倍の損失を出しており、PPO を回せない主因になっている。
ここでは全シードを (N,10,10) の配列で同時に処理し、シード方向のループを消す。

仕事の符号:  0=なし 1=収穫 2=水やり 3=掘る 4=植える
優先度は agent2 と同じ 収穫0 < 水やり1 < 掘る2 < 植える3。
"""
import numpy as np
from agent2 import (OP, BS, CROP, FYD, ONGOING, LASTD, K_EMPTY, K_LOCKED, K_WEED,
                    K_COOP, K_PASTURE, K_PLANT)

J_NONE, J_HARV, J_WATER, J_DIG, J_PLANT = 0, 1, 2, 3, 4
J_PRI = np.array([9, 0, 1, 2, 3], np.int32)
J_OP = np.array([OP["PASS"], OP["HARVEST"], OP["WATER"], OP["DIG"], OP["PLANT"]], np.int32)

NC = 5
C_ONG = np.array([c in ONGOING for c in range(NC)], bool)
C_MYD = np.array([(LASTD[c] if c in ONGOING else CROP[c][0]) for c in range(NC)], np.int32)
C_MAXY = np.array([(ONGOING[c][2] if c in ONGOING else CROP[c][1]) for c in range(NC)], np.int32)
C_FYD = np.array([(ONGOING[c][0] if c in ONGOING else FYD.get(c, LASTD[c])) for c in range(NC)], np.int32)
C_IV = np.array([(ONGOING[c][1] if c in ONGOING else 1) for c in range(NC)], np.int32)
C_LAST = np.array([LASTD[c] for c in range(NC)], np.int32)
C_W0 = (C_MYD + 1) // 2


def tile_jobs_vec(kind, what, watd, yld, pday, plot, day):
    """(N,10,10) の盤面から、各タイルの仕事符号を一括で決める。

    agent2.tile_jobs と同じ規則。継続作物(トマト/イチゴ)は水やりで収量が増えず
    枯死防止のみなので隔日で足りる、という違いを保つ。
    """
    N = kind.shape[0]
    job = np.zeros_like(kind, np.int32)
    arg = np.zeros_like(kind, np.int32)

    job = np.where(kind == K_WEED, J_DIG, job)

    empty = (kind == K_EMPTY)
    job = np.where(empty, J_PLANT, job)
    arg = np.where(empty, np.maximum(plot, 0), arg)

    pl = (kind == K_PLANT)
    c = np.where(kind == K_PLANT, what, plot)
    c = np.clip(c, 0, NC - 1)
    age = np.where(pday >= 0, day - pday, 0)
    myd = C_MYD[c]; mx = C_MAXY[c]; fyd = C_FYD[c]; w0 = C_W0[c]
    ong = C_ONG[c]; last = C_LAST[c]

    # --- 継続作物 ---
    o_harv = ong & pl & (yld > 0) & (age >= fyd)
    o_dig = ong & pl & (age > last) & ~o_harv
    o_wat = ong & pl & ~watd & (age % 2 == 0) & ~o_harv & ~o_dig
    # --- 通常作物 ---
    n = pl & ~ong
    n_dig = n & (age >= myd) & (yld <= 0)
    n_eh = n & ~n_dig & (age >= fyd) & (yld >= mx)
    n_late = n & ~n_dig & ~n_eh & (age >= myd)
    n_wat1 = n_late & ~watd & (yld < mx)
    n_harv = (n_late & ~n_wat1) | n_eh
    n_mid = n & ~n_dig & ~n_eh & ~n_late & ~watd
    n_wat2 = n_mid & ((age == 0) | ((age >= w0) & (age <= myd)) | (age % 2 == 0))

    job = np.where(o_harv | n_harv, J_HARV, job)
    job = np.where(o_dig | n_dig, J_DIG, job)
    job = np.where(o_wat | n_wat1 | n_wat2, J_WATER, job)
    job = np.where(pl & ~(o_harv | n_harv | o_dig | n_dig | o_wat | n_wat1 | n_wat2), J_NONE, job)
    return job, arg


def assign(job, arg, ux, uy, umask, nk, seed_left, center, zone=None, uzone=None):
    """ユニットを仕事へ貪欲に割り当てる。優先度が同じなら近い方を取る。

    ユニット方向にだけ短いループを回し(<=14回)、シード方向は配列のまま処理する。
    元実装は各ユニットが pool 全体を走査していて dict.get が2,290万回出ていた。
    """
    N, U = ux.shape
    G = BS * BS
    j = job.reshape(N, G); a = arg.reshape(N, G)
    yy, xx = np.divmod(np.arange(G), BS)
    pri = J_PRI[j]                                   # (N,G)
    avail = (j != J_NONE)
    # 種が無い作物の PLANT は出せない
    noseed = (j == J_PLANT) & (np.take_along_axis(seed_left, a, 1) <= 0)
    avail &= ~noseed
    # 中央の納屋 2x2 は対象外
    h = BS // 2
    blocked = ((yy >= h-1) & (yy <= h) & (xx >= h-1) & (xx <= h))
    avail &= ~blocked[None, :]

    ops = np.zeros((N, U, 3), np.int32)
    ty = np.full((N, U), -1, np.int32); tx = np.full((N, U), -1, np.int32)
    tj = np.zeros((N, U), np.int32)
    left = seed_left.copy()
    rows_all = np.arange(N)
    for u in range(U):
        d = np.abs(yy[None, :] - uy[:, u:u+1]) + np.abs(xx[None, :] - ux[:, u:u+1])
        base = pri * 1000 + d
        if zone is not None:
            # 担当区画の外は強い罰則。区画が無いと全ユニットが同じ高優先度タイルへ
            # 殺到して往復し、1作業あたりの歩数が 1.8 -> 4.3 に膨れる(実測)。
            base = base + (zone != uzone[:, u:u+1]) * 100000
        cost = np.where(avail, base, 1 << 30)
        best = cost.argmin(1)
        ok = (cost[rows_all, best] < (1 << 30)) & umask[:, u]
        ty[:, u] = np.where(ok, yy[best], -1)
        tx[:, u] = np.where(ok, xx[best], -1)
        tj[:, u] = np.where(ok, j[rows_all, best], J_NONE)
        ops[:, u, 1] = np.where(ok, a[rows_all, best], 0)
        avail[rows_all[ok], best[ok]] = False
        # PLANT を取ったら種を1つ減らす
        pj = ok & (tj[:, u] == J_PLANT)
        if pj.any():
            r2 = rows_all[pj]
            left[r2, ops[r2, u, 1]] -= 1
            gone = np.take_along_axis(left, a, 1) <= 0
            avail &= ~((j == J_PLANT) & gone)
    ops[:, :, 2] = 1
    return ops, ty, tx, tj


def move_or_do(ops, ty, tx, tj, ux, uy):
    """目標に居れば作業、居なければ1マス近づく。"""
    N, U, _ = ops.shape
    has = (ty >= 0)
    dy = ty - uy; dx = tx - ux
    at = has & (dy == 0) & (dx == 0)
    op = np.where(at, J_OP[tj], OP["PASS"]).astype(np.int32)
    mv = has & ~at
    step = np.where(dy != 0,
                    np.where(dy > 0, OP["S"], OP["N"]),
                    np.where(dx > 0, OP["E"], OP["W"])).astype(np.int32)
    op = np.where(mv, step, op)
    out = ops.copy()
    out[:, :, 0] = op
    out[:, :, 1] = np.where(at, ops[:, :, 1], 0)
    out[:, :, 2] = 1
    return out


# ---------------- 家畜側の仕事 ----------------
# 5=給餌 6=世話 7=肥料採取 8=畜産物の収穫 9=設置 10=柵を建てる 11=納屋で積む
J_FEED, J_CARE, J_COLL, J_AHARV, J_PLACE, J_BUILD, J_PICK = 5, 6, 7, 8, 9, 10, 11
JA_PRI = np.zeros(12, np.int32)
JA_PRI[J_NONE] = 9
JA_PRI[J_FEED] = 0        # 2日絶食で逃げるので最優先
JA_PRI[J_PLACE] = 1
JA_PRI[J_AHARV] = 2
JA_PRI[J_COLL] = 3        # 肥料は1手=1個、基準100で最高効率
JA_PRI[J_CARE] = 4        # 世話は次回産出 +1
JA_PRI[J_BUILD] = 5
JA_OP = np.zeros(12, np.int32)
JA_OP[J_FEED] = OP["FEED"]; JA_OP[J_CARE] = OP["CARE"]; JA_OP[J_COLL] = OP["COLLECT"]
JA_OP[J_AHARV] = OP["HARVEST"]; JA_OP[J_PLACE] = OP["PLACE"]
JA_OP[J_BUILD] = OP["BUILD_PASTURE"]; JA_OP[J_PICK] = OP["PICKUP"]


def anim_jobs_vec(kind, what, fedd, yld, fav, cared, carry_item, carry_n,
                  shed_anim, day, pen_radius=4):
    """家畜タイルの仕事を一括で決める。carry_* はユニットごとの携行品。

    給餌には小麦の携行が要る。携行していない場合は納屋で積む(J_PICK)ので、
    ここでは「餌が要るタイル」を示すだけにし、積むかどうかは割当側で決める。
    """
    N = kind.shape[0]
    pen = (kind == K_COOP) | (kind == K_PASTURE)
    has = pen & (what >= 9)
    job = np.zeros_like(kind, np.int32)
    job = np.where(has & ~fedd, J_FEED, job)
    job = np.where(has & (yld > 0) & (job == J_NONE), J_AHARV, job)
    job = np.where(has & fav & (job == J_NONE), J_COLL, job)
    job = np.where(has & ~cared & (job == J_NONE), J_CARE, job)
    # 空いた柵には買った動物を置く
    free_pen = pen & (what == 0)
    want_place = (shed_anim > 0)[:, None, None] & free_pen
    job = np.where(want_place, J_PLACE, job)
    # 柵が足りなければ中心付近の空きに建てる
    h = BS // 2
    yy, xx = np.meshgrid(np.arange(BS), np.arange(BS), indexing="ij")
    near = (np.abs(yy - h) + np.abs(xx - h)) <= pen_radius
    nofree = ~(free_pen.any((1, 2)))
    want_build = (shed_anim > 0) & nofree
    job = np.where(want_build[:, None, None] & (kind == K_EMPTY) & near[None], J_BUILD, job)
    return job


def assign_pri(job, arg, pri_tab, ux, uy, umask, taken=None):
    """優先度表 pri_tab を使った貪欲割当。assign と同じ機構だが仕事表を差し替えられる。"""
    N, U = ux.shape
    G = BS * BS
    j = job.reshape(N, G); a = arg.reshape(N, G)
    yy, xx = np.divmod(np.arange(G), BS)
    pri = pri_tab[j]
    avail = (j != J_NONE)
    if taken is not None: avail &= ~taken
    ops = np.zeros((N, U, 3), np.int32)
    ty = np.full((N, U), -1, np.int32); tx = np.full((N, U), -1, np.int32)
    tj = np.zeros((N, U), np.int32)
    rows_all = np.arange(N)
    for u in range(U):
        d = np.abs(yy[None, :] - uy[:, u:u+1]) + np.abs(xx[None, :] - ux[:, u:u+1])
        cost = np.where(avail, pri * 1000 + d, 1 << 30)
        best = cost.argmin(1)
        ok = (cost[rows_all, best] < (1 << 30)) & umask[:, u]
        ty[:, u] = np.where(ok, yy[best], -1)
        tx[:, u] = np.where(ok, xx[best], -1)
        tj[:, u] = np.where(ok, j[rows_all, best], J_NONE)
        ops[:, u, 1] = np.where(ok, a[rows_all, best], 0)
        avail[rows_all[ok], best[ok]] = False
    ops[:, :, 2] = 1
    return ops, ty, tx, tj, avail
