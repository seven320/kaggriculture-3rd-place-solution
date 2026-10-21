# fasteval — league.py の速い版(同じ結果・同じ JSON)

## 使い方(`tools/materialize.py` で作った作業ツリーの `content/lab/fasteval` で)
```
AGENTS="a:bc:/content/lab/agents/X,b:bc:/content/lab/agents/Y" NSEED=192 SEED0=880001 \
  python3 fast_league.py /content/lab/xxx.json
```
- 出力は league.py と同じ鍵: `a|b`(局ごとの [A の所持金, B の所持金])、`a|b#shops`、`a|b#money300/500`、`a|b#seats`。ana_tom.py はそのまま読める。
- 種・席の決め方(SEED0 + 7919k、席 k%2)、貪欲、全組の総当たりも league.py と同じ。
- 環境変数: `FAST_GRAPH=1`(既定。CUDA graph で推論)、`FAST_INF=0`(既定。1 にすると市場デコーダを逐次キャッシュにする近似モード)。

## 作り
- エンジン: PPO の対局係と同じ C++ の `kagsim.VecMarket`(全局を1本に入れる = sp_fast.BatchSPPool)。特徴は BatchSPPool.state と同じ取り方で、まとめて取る。
- 合法手: 版0と版2を両方取り、各フォルダの obs_feat に合わせて組む(NMB_MARK があれば印 3、無ければその LEGAL_OPT の版)。
- 推論: エージェントは `league.BC` でそのまま読む(フォルダごとの model_ptr_infer・切り替え型の a.*/b.*)。計算は `rust_eval2.greedy_batch` と同じ演算を CUDA graph で再生。切り替え型は eager と同じく「全行 A → a だけ / 全行 B → b だけ / 混在 → 両方計算して行を選ぶ」を graph で分ける。
- 行動: 表引きでエンジン用コードに(fast_codes と同じ作りだが、数量の番兵 999/1000 まで表を広げた。下の注意1)。
- main.py の後処理: league.py と同じく `_pace`、`_land` をこの順で呼ぶ。番兵の行動で「何もしない」と分かった局は省き、変える局だけ本物の to_action の辞書に掛けて sanitize してから符号化(pr_rs05liq の `_liq` = 718手目に9品目 SELL 1000 もこれで一致)。
- 切り替え型の main.py が終了時にエージェントのフォルダへ書く `fire_*.json` は書かない(atexit を外す。結果には無関係)。

## 検証(09-29)
1. 恒等式 `check_lockstep.py`(league.py の打ち方で arena を進め、kagsim を同じ行動で並走): 4局×719手×両エージェントで
   特徴5種・合法手の印・行動コードの不一致 0、最終所持金 4/4 一致(g28 対 g24、pr_rs05liq 対 vpfn4g7 の両方)。
2. `compare.py`(同じ種・席で league.play と fast_league.play):
   | 組 | 局 | 所持金の完全一致 | 勝率(fast / league) | 時間 fast / league |
   |---|---|---|---|---|
   | pntom45g28 対 pntom45g24 | 192 | 192/192 | 0.573 / 0.573 | 156 s / 559 s(3.6倍、GPU は対局係と共有で 95%) |
   | pr_rs05liq 対 vpfn4g7 | 192 | 192/192 | 0.641 / 0.641 | 114 s / 582 s(5.1倍、GPU ほぼ空き) |
   店の一覧も 192/192 一致。別ホストで打ち直しても 192/192 同じ値(決定的)。
3. `FAST_INF=1`(近似): pr_rs05liq 対 vpfn4g7 192局で 89.5 s(厳密 113 s、同じ GPU)。所持金一致 152/192、勝敗一致 187/192、勝率 0.625(厳密 0.641)。
4. `leagueq_fast.py` を q_test の自前の依頼で試験: bc だけの依頼は fast(16局 30 s、league.py と 16/16 一致)、
   後処理が obs["farms"] を読む a79_land4 を含む依頼は league.py に回った。

## 時間の内訳(192局、GPU 空き)
特徴(CPU)25 s + 推論(GPU)88 s + エンジン 0.4 s。推論の大半はエンコーダ(B=192 で1回 約40 ms の GPU 時間、1手に2〜3回)。
league.py は 1観測ずつの featurize と JSON が主で、ここを消すと **GPU 律速**になる。GPU を対局係と共有すると推論はその分遅くなる。
30倍に届かないのはこのため(同じ値を出す限り、エンコーダの計算は減らせない)。

## 注意
1. **PPO の対局係の不一致(要確認)**: `kagg/bc/fast_codes.py` の表は数量クラスを MQ_MAX+2 / UQ_MAX+2 までしか持たず、
   番兵クラス(市場 130=999、131=1000)が 129 に切られて **128 個**になる(to_action は 999/1000)。check_lockstep で
   t=718 の SELL TOMATO 1000 → 128 として検出。apo.py は `P.to_codes = fast_codes.to_codes_fast` なので学習中の対局も同じ。
   売りは納屋が 100 までなので実害は小さいはずだが、買い(BUY_SEED 999 = 所持金が尽きるまで)は 128 で止まる。
2. league.py 自体も局数(バッチの大きさ)で fp16 の丸めが変わる(同じ種でも NSEED=8 と 16 で局2の所持金が違う)。
   fast_league は league.py と**同じバッチ**で同じ値を出すので、比べるときは同じ NSEED・SEED0 で。
3. 未対応 → league.py を使う: py 型、cfg に mkt(市場16特徴)、main.py の後処理が step/day/hour/player/private.shed 以外の
   観測の鍵を読むもの(例: exp290 の `_land` は obs["farms"])。fast_league.py はこれらで止まり、leagueq_fast.py は league.py に回す。
   logit アンサンブル(policy_a.pt …)は graph を使わず eager の greedy_batch_ens に回す(未検証)。
4. GPU の記憶: graph は作業領域を共有し、組ごとに捨てる。192局で確保 3.7 GB。

## leagueq.py からの呼び方(案。本番の待ち行列はまだ読ませていない)
- `leagueq_fast.py` は leagueq.py と同じ待ち行列の書式・取り合い(rename)・取り残しの戻し。既定の待ち行列は `lab/fasteval/q_test`。
- 本番の待ち行列を読ませるときは `cd /content/drive/MyDrive/kaggriculture/bc && LQF_Q=/content/lab/leagueq python3 /content/lab/fasteval/leagueq_fast.py`
  で起動する(leagueq.py のワーカーと同じ待ち行列を一緒に読める。依頼ごとに fast か league.py を自動で選ぶ)。
- 1本で CPU(特徴)と GPU(推論)を重ねるには `LQ_PAR=2`。
- 同じ依頼を league.py と fast で打った結果は局ごとに同じなので、既存の JSON に足しても混ざらない。

## ファイル
- `fast_league.py` 本体 / `leagueq_fast.py` 常駐ワーカー(案) / `check_lockstep.py` 恒等式 / `compare.py` 並べて比較 /
  `bench_parts.py` 推論の区間ごとの時間 / `results/` 検証の JSON とログ / `q_test/` 試験用の待ち行列
