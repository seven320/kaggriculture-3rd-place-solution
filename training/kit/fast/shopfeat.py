"""店の抽選を観測に入れるための特徴。訓練経路と評価経路で必ず同じコードを使うこと。

店IDと買う品目の対応は **実測で確定** した(scratchpad/shopmap.py、256局×718ステップの最小二乗)。
出典が2つ食い違っていたため:
  - bc/extract_curve_cond.py の並び(PIZZA=1, YARN=3 …) は **誤り**
  - fast/carrot.py の CARROT_SHOPS={4,2} が **正しい**(PET_CAFE=4, FARMERS_MARKET=2)
1品目だけを買う店は係数が倍(0.50 対 0.25)。**メロンと肥料を買う店は一軒も無い。**

店は3〜4日ごとに1軒ずつ開き、day29 で8軒。day0 は0軒。
メロンは植えてから12日かかるので、作付けを決める day17 時点で分かるのは5軒程度＝部分情報。
"""
import os
import numpy as np

# SHOPFEAT=0 で店特徴を無効化(対照実験用)。既定は有効。
ENABLED = os.environ.get("SHOPFEAT", "1") == "1"

# 品目の並び: WHEAT CARROT TOMATO STRAW MELON EGG MILK WOOL FERT
NP_ = 9
NSHOP = 8
SHOP_NAME = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
             "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
# 実測値(1軒あたりの品目別消費増)。単品店は0.50、複数品目店は0.25。
BUY = np.zeros((NSHOP, NP_), np.float32)
for _s, _items, _w in [
        (0, [0, 5],          0.25),   # BAKERY         小麦 卵
        (1, [0, 3, 5],       0.25),   # BRUNCH_SPOT    小麦 イチゴ 卵
        (2, [0, 1, 2, 3],    0.25),   # FARMERS_MARKET 小麦 人参 トマト イチゴ
        (3, [0, 3, 6],       0.25),   # ICE_CREAM_SHOP 小麦 イチゴ 牛乳
        (4, [1],             0.50),   # PET_CAFE       人参(単品2倍)
        (5, [0, 2, 6],       0.25),   # PIZZA_SHOP     小麦 トマト 牛乳
        (6, [3, 6],          0.25),   # SMOOTHIE_SHOP  イチゴ 牛乳
        (7, [7],             0.50)]:  # YARN_STORE     羊毛(単品2倍)
    BUY[_s, _items] = _w

DIM = (NSHOP + NP_ + 1) if ENABLED else 0   # 店種別8 + 品目別需要9 + 総数1 = 18、無効時は0


def shop_feat(shops):
    """shops: env.shops() の (N, 16) int32、-1 は未開店。返り値 (N, DIM) float32。"""
    sh = np.asarray(shops)
    n = sh.shape[0]
    cnt = np.empty((n, NSHOP), np.float32)
    for v in range(NSHOP):
        cnt[:, v] = (sh == v).sum(1)
    dem = cnt @ BUY                      # 品目別の町の需要(軒数で重み付け)
    tot = cnt.sum(1, keepdims=True)
    return np.concatenate([cnt / 4.0, dem / 2.0, tot / 8.0], 1).astype(np.float32)
