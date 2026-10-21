"""MFU_FAST=1 と 0 で、方策の出力・勾配・学習の進み方が同じかを確かめる。重みも入力も同じものを使う。

前の版が落ちた理由と直した点(09-23):
  * Policy に forward は無い。呼ぶのは encode / unit_logits / unit_qty / market /
    market_qty / value。だから n0(*x) は必ず NotImplementedError になる。手本は ppo_tf.evaluate。
  * importlib.reload は **同じモジュール辞書を作り直す**ので、先に作ったネットの
    メソッド(__globals__ が同じ辞書)まで新しい FAST を見てしまう。順番を1つ
    間違えると「一致した」と嘘の合格が出る。そこで再読み込みはやめ、M.FAST を
    直に入れ替える(環境変数 MFU_FAST がやっているのと同じこと)。重みも入力も
    1組しか作らないので、差が出たらそれは FAST のせいだけ。
  * 合法マスクで埋めた -1e4 が最大値を支配して相対差が薄まる。合法な枠だけでも測る。
  * 出力だけでなく**勾配**も比べる。さらに **同じ設定で2回**走らせて
    「GPU 自身のばらつき(逆伝播の atomicAdd など)」を測り、FAST の差がその中に
    収まるかで判定する(bench_upd.py の base 対 base2 と同じ考え方)。
  * 乱数の入力では店の枠 g[:,47:63] が 8 を超えると Embedding(9) の範囲外で落ちる。

段階:
  1 順伝播(fp16 = 本番の autocast / fp32 = 丸めを外して式そのものを見る)
  2 本番と同じ PPO の損失の勾配(全パラメータ)。FAST0 対 FAST0 のばらつきと比べる
  3 Adam を STEPS 回まわしたあとの重み。FAST0 対 FAST0 のばらつきと比べる
  4 速さ(eager と torch.compile、MB 行の順伝播＋逆伝播)

使い方:
  python check_mfu.py <重み.pt> [対局係が書いた局.pt]
環境変数: B(行数、既定 1024)/ STEPS(既定 8)/ GRAD=0・TRAIN=0・BENCH=0 で段階を省く
"""
import contextlib
import glob
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_tf as M                                     # noqa: E402
import ppo_tf as P                                       # noqa: E402

DEV = "cuda"
W = sys.argv[1]
SHARD = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("SHARD", "")
if SHARD and ("*" in SHARD or os.path.isdir(SHARD)):
    pat = SHARD if "*" in SHARD else SHARD + "/*.pt"
    cand = sorted(glob.glob(pat), key=os.path.getmtime)
    SHARD = cand[-2] if len(cand) > 1 else (cand[0] if cand else "")   # 最新は書きかけのことがあるので1つ前
B = int(os.environ.get("B", "1024"))
STEPS = int(os.environ.get("STEPS", "8"))
DO_GRAD = os.environ.get("GRAD", "1") == "1"
DO_TRAIN = os.environ.get("TRAIN", "1") == "1"
DO_BENCH = os.environ.get("BENCH", "1") == "1"
BETA, ENT = 0.02, 1e-3                                   # BETA は本番と同じ。ENT は本番 0 だが経路を通すため入れる
NU, NC, NM, NS = M.NU, M.NC, M.NM, M.NS
CLIP = P.CLIP

torch.manual_seed(0)
print("torch %s / %s" % (torch.__version__, torch.cuda.get_device_name()), flush=True)
print("重み %s / 行 %d / model_tf の FAST の既定 %s" % (os.path.basename(W), B, M.FAST), flush=True)


# ---------------- 入力 ----------------
def synth(nb):
    """形と値の範囲だけ本物に合わせた乱数。TokenEnc が long に丸める列は整数にする。"""
    rg = torch.Generator().manual_seed(1234)
    tiles = torch.rand(nb, 2, 100, 13, generator=rg)
    ou = torch.rand(nb, 20, 15, generator=rg) * 100.0
    ou[..., 0] = (torch.rand(nb, 20, generator=rg) < 0.6).float()        # 生きているユニット
    ou[..., 1] = torch.randint(0, 10, (nb, 20), generator=rg).float()    # x
    ou[..., 2] = torch.randint(0, 10, (nb, 20), generator=rg).float()    # y
    pu = torch.zeros(nb, 20, 3)
    pu[..., 0] = (torch.rand(nb, 20, generator=rg) < 0.6).float()
    pu[..., 1] = torch.randint(0, 10, (nb, 20), generator=rg).float()
    pu[..., 2] = torch.randint(0, 10, (nb, 20), generator=rg).float()
    gg = torch.rand(nb, 75, generator=rg)
    gg[:, 34] = torch.randint(0, 720, (nb,), generator=rg).float() / 720.0        # step
    gg[:, 47:63] = torch.randint(-1, 8, (nb, 16), generator=rg).float()           # 店(-1 は空き枠、Embedding(9) の範囲内)
    lm = (torch.rand(nb, 20, NC, generator=rg) < 0.16).to(torch.uint8)
    lm[:, :, 0] = 1                                                      # どのユニットにも合法な手を1つは残す
    ul = lm.float().argmax(-1)
    uq = torch.randint(0, M.UQN, (nb, NU), generator=rg)
    ms = torch.randint(0, NM, (nb, NS), generator=rg)
    mq = torch.randint(0, M.MQN, (nb, NS), generator=rg)
    return dict(tiles=tiles.half(), ou=ou, pu=pu, g=gg, lm=lm, ul=ul, uq=uq, ms=ms, mq=mq)


def from_shard(path, nb):
    sh = torch.load(path, map_location="cpu", weights_only=False)
    D = sh["D"]
    n = D["tiles"].shape[0]
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))[:nb]
    out = {k: D[k][idx].clone() for k in ("tiles", "ou", "pu", "g", "lm", "ul", "uq", "ms", "mq")}
    del sh, D
    return out


x = from_shard(SHARD, B) if SHARD else synth(B)
print("入力 %s" % ("局 " + os.path.basename(SHARD) if SHARD else "乱数(形と範囲は本物に合わせた)"), flush=True)
x = {k: v.to(DEV) for k, v in x.items()}
KEYS = ("tiles", "ou", "pu", "g", "lm", "ul", "uq", "ms", "mq")
BATCH = [x[k] for k in KEYS]
LEGAL = x["lm"].bool()
# 手は保存された ul をそのまま使う(argmax にすると点のわずかな差で別の手に飛び、
# 「丸めの差」ではなく「別の手を選んだ差」になってしまう)
UL = x["ul"].clamp(min=0)


def mknet():
    torch.manual_seed(0)
    return M.load(W, DEV)


net = mknet()
print("方策 d=%d / パラメータ %.1fM" % (net.d, sum(p.numel() for p in net.parameters()) / 1e6), flush=True)


def cmp(a, b, name, mask=None, quiet=False):
    da, db = a.float(), b.float()
    if mask is not None:
        da, db = da[mask], db[mask]
    d = (da - db).abs()
    den = max(1e-9, da.abs().max().item())
    mx, rel = d.max().item(), d.max().item() / den
    if not quiet:
        print("  %-22s 形 %-20s 最大差 %.3e 相対 %.3e 平均差 %.3e"
              % (name, tuple(a.shape), mx, rel, d.mean().item()), flush=True)
    return rel


# ---------------- 1 順伝播 ----------------
def forward(fast, amp=True):
    M.FAST = fast                                        # 環境変数 MFU_FAST と同じ意味
    ctx = torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
    with ctx:
        h, pad = net.encode(x["tiles"], x["ou"], x["pu"], x["g"])
        s, q = net.unit_logits(h, x["ou"], x["lm"])
        uq = net.unit_qty(q, UL)
        prev = torch.cat([torch.full_like(x["ms"][:, :1], NM), x["ms"][:, :-1]], 1).clamp(max=NM)
        mo, y = net.market(h, pad, prev)
        mq = net.market_qty(y, x["ms"])
        v = net.value(h[:, -1]).squeeze(-1)
    return dict(埋め込み=h, 手の点=s, 質問=q, 数量=uq, 市場=mo, 市場の数量=mq, 価値=v)


print("\n== 1 順伝播: fp16(本番と同じ autocast)==", flush=True)
with torch.no_grad():
    o0 = forward(False)
    o1 = forward(True)
w16 = 0.0
for k in o0:
    w16 = max(w16, cmp(o0[k], o1[k], k))
w16 = max(w16, cmp(o0["手の点"], o1["手の点"], "手の点(合法枠だけ)", mask=LEGAL))
del o0, o1
torch.cuda.empty_cache()

print("\n== 1 順伝播: fp32(丸めを外して式そのものを見る)==", flush=True)
nb32 = min(B, 256)
_full, x_full, LEG_full, UL_full = BATCH, x, LEGAL, UL
x = {k: v[:nb32] for k, v in x.items()}; LEGAL = LEGAL[:nb32]; UL = UL[:nb32]
with torch.no_grad():
    f0 = forward(False, amp=False)
    f1 = forward(True, amp=False)
w32 = 0.0
for k in ("手の点", "数量"):
    w32 = max(w32, cmp(f0[k], f1[k], k + "(fp32)"))
w32 = max(w32, cmp(f0["手の点"], f1["手の点"], "手の点(合法枠・fp32)", mask=LEGAL))
del f0, f1
x, LEGAL, UL = x_full, LEG_full, UL_full
torch.cuda.empty_cache()

# ---------------- 本番と同じ損失 ----------------
torch.manual_seed(7)
ADV = torch.randn(B, device=DEV)                          # 優位度(固定の乱数。ここでは中身より経路が要る)
ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
ref = mknet()                                             # KL の基準。少しずらして KL の勾配を 0 でなくする
with torch.no_grad():
    gg = torch.Generator(device=DEV).manual_seed(11)
    for p in ref.parameters():
        p.add_(torch.randn(p.shape, generator=gg, device=DEV) * 0.002)
for p in ref.parameters():
    p.requires_grad_(False)
M.FAST = False
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    PER0 = P.evaluate(net, BATCH)[4].float().detach()     # 手を選んだときの対数確率(rr の分母)
    RD0 = tuple(t.float() if t.dtype.is_floating_point else t for t in P.evaluate(ref, BATCH)[3])
# 本番の PER0 は「少し前の版の対局係」が書いたものなので rr = exp(per-PER0) は 1 ではない。
# ここで PER0 = いまの値 のままだと rr ≡ 1 になり、min() の分かれ目ちょうどで
# 方策項が 0 に潰れて「相対差」が意味を失う。本番くらいのばらつきを足す。
PER0 = PER0 + 0.05 * torch.randn(PER0.shape, generator=torch.Generator(device=DEV).manual_seed(3), device=DEV)


def ppo_loss(n_):
    """apo.py の更新ループ(方策側)と同じ形の損失。"""
    with torch.autocast("cuda", dtype=torch.float16):
        lp, ent, v, dist, per, pmask = P.evaluate(n_, BATCH)
    ent, per = ent.float(), per.float()
    dist = tuple(t.float() if t.dtype.is_floating_point else t for t in dist)
    rr = torch.exp((per - PER0).clamp(-20, 20)); aa = ADV.unsqueeze(1)
    pl = -((torch.min(rr * aa, rr.clamp(1 - CLIP, 1 + CLIP) * aa) * pmask).sum(1) / pmask.sum(1).clamp(min=1)).mean()
    kl = P.kl_to_ref(dist, RD0).mean()
    return pl + BETA * kl - ENT * ent.mean(), pl, kl


# ---------------- 2 勾配 ----------------
if DO_GRAD:
    print("\n== 2 勾配(本番と同じ PPO の損失、全パラメータ)==", flush=True)

    def grads(fast):
        M.FAST = fast
        net.zero_grad(set_to_none=True)
        loss, pl, kl = ppo_loss(net)
        loss.backward()
        g = {n_: p.grad.detach().clone() for n_, p in net.named_parameters() if p.grad is not None}
        net.zero_grad(set_to_none=True)
        return g, float(loss), float(pl), float(kl)

    gA, lA, plA, klA = grads(False)
    gB, lB, plB, klB = grads(False)      # 同じ設定で2回 = GPU 自身のばらつき
    gF, lF, plF, klF = grads(True)
    print("  損失 FAST0 %.10f / FAST0 もう一度 %.10f / FAST1 %.10f" % (lA, lB, lF), flush=True)
    print("  方策項 %.10f / %.10f / %.10f   KL %.10f / %.10f / %.10f" % (plA, plB, plF, klA, klB, klF), flush=True)
    nz = [k for k in gA if gA[k].abs().max() > 0]
    # テンソルごとの「自分の最大値」で割ると、**数学的に 0 のパラメータ**で相対差が発散する。
    #   例 u_score.2.bias は Linear(d,1) の切片で、その勾配は log_softmax の性質から
    #   候補軸の和 = (Σp − 1) = 0 になり、計算値は丸めの残りかすしかない。
    # そこで「勾配全体でいちばん大きい値」でも割って、実害のある大きさかを見る。
    gmax = max(gA[k].abs().max().item() for k in nz)
    loc = lambda u, v: max(cmp(u[k], v[k], k, quiet=True) for k in nz)
    glb = lambda u, v: max((u[k] - v[k]).abs().max().item() for k in nz) / gmax
    kn = max(nz, key=lambda k: cmp(gA[k], gB[k], k, quiet=True))
    kf = max(nz, key=lambda k: cmp(gA[k], gF[k], k, quiet=True))
    print("  勾配 いちばん大きい値 %.3e" % gmax, flush=True)
    print("  テンソルごとの相対差 最大: FAST0 対 FAST0(ばらつき) %.3e(%s、その勾配の大きさ %.2e)"
          % (loc(gA, gB), kn, gA[kn].abs().max().item()), flush=True)
    print("  テンソルごとの相対差 最大: FAST0 対 FAST1          %.3e(%s、その勾配の大きさ %.2e)"
          % (loc(gA, gF), kf, gA[kf].abs().max().item()), flush=True)
    noise, fastd = glb(gA, gB), glb(gA, gF)
    print("  勾配全体の大きさで割った差 最大: FAST0 対 FAST0(ばらつき) %.3e / FAST0 対 FAST1 %.3e" % (noise, fastd), flush=True)
    print("  → %s" % ("ばらつきと同じ桁。式は変わっていない" if fastd <= max(noise * 20, 5e-3)
                     else "!!! ばらつきより明らかに大きい。中身を見ること"), flush=True)
    del gA, gB, gF
    torch.cuda.empty_cache()

# ---------------- 3 Adam を回したあとの重み ----------------
if DO_TRAIN:
    print("\n== 3 Adam %d 歩のあとの重み(本番と同じ GradScaler + 勾配の切り詰め)==" % STEPS, flush=True)

    def train(fast, tag):
        M.FAST = fast
        n_ = mknet()
        opt = torch.optim.Adam(n_.parameters(), lr=2.5e-5, eps=1e-5)
        scaler = torch.amp.GradScaler("cuda")
        hist = []
        for _ in range(STEPS):
            loss, pl, kl = ppo_loss(n_)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(n_.parameters(), 0.5)
            scaler.step(opt); scaler.update()
            hist.append(float(loss))
        sd = {k: v.detach().float().clone() for k, v in n_.state_dict().items()}
        print("  [%s] 損失 %s" % (tag, " ".join("%.8f" % h for h in hist)), flush=True)
        del n_, opt
        torch.cuda.empty_cache()
        return sd, hist

    sA, hA = train(False, "FAST0")
    sB, hB = train(False, "FAST0 もう一度")
    sF, hF = train(True, "FAST1")
    ks = [k for k in sA if sA[k].is_floating_point() and sA[k].abs().max() > 0]
    nAB = max(cmp(sA[k], sB[k], k, quiet=True) for k in ks)
    nAF = max(cmp(sA[k], sF[k], k, quiet=True) for k in ks)
    bitAB = all(torch.equal(sA[k], sB[k]) for k in sA)
    print("  重みの相対差 最大: FAST0 対 FAST0(ばらつき) %.3e(ビット一致 %s)" % (nAB, bitAB), flush=True)
    print("  重みの相対差 最大: FAST0 対 FAST1          %.3e" % nAF, flush=True)
    print("  最後の損失: %.8f / %.8f / %.8f" % (hA[-1], hB[-1], hF[-1]), flush=True)
    print("  → %s" % ("ばらつきと同じ桁。学習の進み方は変わらない" if nAF <= max(nAB * 20, 1e-4)
                     else "!!! ばらつきより明らかに大きい"), flush=True)
    del sA, sB, sF
    torch.cuda.empty_cache()

# ---------------- 4 速さ ----------------
if DO_BENCH:
    print("\n== 4 速さ(%d 行の 順伝播＋逆伝播。本番の1ミニバッチと同じ形)==" % B, flush=True)
    torch._dynamo.config.cache_size_limit = 32

    def bench(fn, fast, reps=12, warm=4):
        M.FAST = fast
        net.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        for _ in range(warm):
            fn().backward(); net.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(reps):
            fn().backward(); net.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return (time.time() - t0) / reps * 1e3, torch.cuda.max_memory_allocated() / 2 ** 30

    def eager():
        return ppo_loss(net)[0]

    t0, m0 = bench(eager, False)
    t1, m1 = bench(eager, True)
    print("  [eager]   MFU_FAST=0 %7.1f ms / ピーク %5.2f GB" % (t0, m0), flush=True)
    print("  [eager]   MFU_FAST=1 %7.1f ms / ピーク %5.2f GB  → %+.1f%%(速さ)、%+.1f%%(記憶)"
          % (t1, m1, (t0 / t1 - 1) * 100, (m1 / m0 - 1) * 100), flush=True)

    EV = torch.compile(P.evaluate, dynamic=False)          # 本番の COMPILE_UPD=1 と同じ場所

    def compiled():
        with torch.autocast("cuda", dtype=torch.float16):
            lp, ent, v, dist, per, pmask = EV(net, BATCH)
        ent, per = ent.float(), per.float()
        dist = tuple(t.float() if t.dtype.is_floating_point else t for t in dist)
        rr = torch.exp((per - PER0).clamp(-20, 20)); aa = ADV.unsqueeze(1)
        pl = -((torch.min(rr * aa, rr.clamp(1 - CLIP, 1 + CLIP) * aa) * pmask).sum(1) / pmask.sum(1).clamp(min=1)).mean()
        return pl + BETA * P.kl_to_ref(dist, RD0).mean() - ENT * ent.mean()

    c0, cm0 = bench(compiled, False, warm=6)
    c1, cm1 = bench(compiled, True, warm=6)
    print("  [compile] MFU_FAST=0 %7.1f ms / ピーク %5.2f GB" % (c0, cm0), flush=True)
    print("  [compile] MFU_FAST=1 %7.1f ms / ピーク %5.2f GB  → %+.1f%%(速さ)、%+.1f%%(記憶)"
          % (c1, cm1, (c0 / c1 - 1) * 100, (cm1 / cm0 - 1) * 100), flush=True)
    print("  (本番は COMPILE_UPD=1 = compile の行。1反復は 337 ミニバッチ相当)", flush=True)

print("\n結論: 順伝播の相対差 fp16 %.3e / fp32 %.3e → %s"
      % (w16, w32, "MFU_FAST は式を変えていない(fp16 の丸めの範囲)" if max(w16, w32) < 2e-3
         else "!!! 式が変わっている。使わないこと"), flush=True)
