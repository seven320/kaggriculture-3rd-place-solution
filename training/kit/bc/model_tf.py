"""実体をトークンにした transformer 方策(ポケカの ID 埋め込み→transformer→合法候補の採点 の考え方を移植)。

トークン(1手あたり 266):
  タイル 200(自分100・相手100): 種類ID・作物/家畜ID・x・y・自他 を埋め込み ＋ 数値11(水やり/給餌/収穫量/経過日数/寿命…)
  自分のユニット 20: ユニット番号・x・y を埋め込み ＋ 携行品12
  相手のユニット 20: x・y
  店 16枠: 店ID(空きはマスク)
  市場の品目 9: 品目ID ＋ 価格・市場在庫・自分の納屋
  大域 1: 時(24)・日(30)を埋め込み ＋ 所持金・区画・雇用・種・納屋の家畜…
ID を 1 つの数(what/12 など)で入れていた CNN は大小関係のない ID に大小を持たせていた。ここでは全部埋め込む。

出力:
  ユニット: そのユニットのトークン＋立っているマスのトークン から、合法な候補(44種、平均7)を採点。数量は候補ごとの頭
  市場: 候補23種(売る9・種5・小麦/肥料2・家畜3・雇用・土地・空・終了)を自己回帰で最大10個 → 順序つきの注文列
"""
import math
import os
import torch, torch.nn as nn, torch.nn.functional as F
NU, NC, NM, NS = 20, 44, 23, 11
UQN, MQN = 24, 132
# 09-23(MFU): 式は同じで速い書き方。カーネルの形と足す順が変わるので fp16 の丸め程度の差は出る
#   (exp298 の CRIT_LAST と同じ扱い)。既定は切。重みの形も読み込みも変わらない。
FAST = os.environ.get("MFU_FAST", "0") == "1"
# 09-23(マスク移行 mask0→mask2、denden さんの方法の移植): MASK_MIG=1 のとき、合法マスク lm の **bit1(値 3)** を
#   「新マスクでだけ増えた手(expanded = mask2 & ~mask0)」の印として使い、その手の logit から penalty を引く。
#     expanded = mask2_allowed & ~mask0_allowed ;  logits = raw_logits - expanded * penalty
#   元の重みは変えない。有限の penalty なので新しい手にも勾配が通り、logit が penalty を上回れば選べる。
#   **罰を lm に載せて運ぶ**ので、採取(ppo_tf.sample)も学習(ppo_tf.evaluate)も同じ net.unit_logits を通り、
#   「採取と学習で罰がずれる」事故が原理的に起こらない。印の付いていない席(相手・テープ)は従来どおり罰なし。
#   penalty は0次元テンソルに入れて in-place で書き換えるので、torch.compile を作り直さずに減衰できる。
#   MASK_MIG を設定しなければ下の分岐に一切入らず、既定は従来とビット単位で同じ。
MIG = os.environ.get("MASK_MIG") == "1"
# 09-23(保険): MIG_MIX=1 で「罰」ではなく**混合**にする。 π = (1-ε)·π_旧 + ε·π_新。
#   一定値の罰は局面によって新しい手の確率が 1 近くまで上がりうる(実測 最大 0.997)が、
#   混合なら**どの局面でも増えた手の確率は ε 以下**に厳密に抑えられる。係数(_MIG_V)は ε を表す。
MIG_MIX = os.environ.get("MIG_MIX") == "1"
_MIG_T = {}; _MIG_V = [0.0]


def mig_set(v):
    """今の penalty を設定する(既に作った0次元テンソルは in-place で書き換える)。"""
    _MIG_V[0] = float(v)
    for t in _MIG_T.values(): t.fill_(float(v))


def mig_get():
    return _MIG_V[0]


def _mig_t(dev):
    t = _MIG_T.get(dev)
    if t is None:
        t = torch.full((), float(_MIG_V[0]), dtype=torch.float32, device=dev); _MIG_T[dev] = t
    return t


def mig_apply(logits, legal):
    """マスク移行の罰。lm の bit1(値 3 = 新マスクでだけ増えた手)の logit から今の penalty を引く。
    **ここを unit_logits の中に置かないこと。** FASTINF(fastinfer.install)は Policy.unit_logits を
    丸ごと差し替えるので、中に書くと採取も学習も罰が消える(09-23 の g5r1m 崩壊の原因)。
    呼ぶのは ppo_tf.sample と ppo_tf.evaluate の2か所だけ(=採取と学習の入口)。"""
    l = legal.to(torch.uint8); xp = (l & 2) != 0
    if not MIG_MIX:
        return logits - _mig_t(logits.device).to(logits.dtype) * xp.to(logits.dtype)
    # 混合: 旧マスクだけの分布と新マスクの分布を ε で混ぜ、その**対数確率**を logit として返す
    e = _mig_t(logits.device).clamp(1e-8, 1.0 - 1e-8).float()
    f = logits.float()
    lp0 = torch.log_softmax(f.masked_fill(xp, -1e4), -1)      # 旧マスクだけ
    lp2 = torch.log_softmax(f.masked_fill(l == 0, -1e4), -1)  # 新マスク(増えた手も含む)
    return torch.logaddexp(lp0 + torch.log1p(-e), lp2 + torch.log(e)).to(logits.dtype)


def mig_warm(dev):
    """torch.compile より前に0次元テンソルを作っておく(compile の追跡中に作らせない = 値を焼き込ませない)。"""
    _mig_t(torch.device(dev) if not isinstance(dev, torch.device) else dev)


class TokenEnc(nn.Module):
    def __init__(self, d):
        super().__init__()
        E = lambda n: nn.Embedding(n, d)
        self.typ = E(6)                       # 0タイル 1自ユニット 2相手ユニット 3店 4品目 5大域
        self.kind, self.what, self.x, self.y, self.side = E(6), E(13), E(10), E(10), E(2)
        self.tnum = nn.Linear(11, d)
        self.uid, self.unum = E(NU), nn.Linear(12, d)
        self.shop = E(9)
        self.item, self.inum = E(9), nn.Linear(3, d)
        self.hour, self.day, self.gnum = E(24), E(30), nn.Linear(20, d)

    def forward(self, tiles, ou, pu, g):
        """tiles (B,2,100,13) / ou (B,20,15) / pu (B,20,3) / g (B,75) -> tokens (B,266,d), pad mask (B,266)"""
        B, dev = tiles.shape[0], tiles.device
        t = tiles.float()
        kind = (t[..., 0] * 5).round().long().clamp(0, 5)
        what = (t[..., 1] * 12).round().long().clamp(0, 11)
        has = (kind == 5) | (t[..., 6] > .5)
        what = torch.where(has, what, torch.full_like(what, 12))
        pos = torch.arange(100, device=dev)
        tx, ty = (pos % 10).view(1, 1, 100), (pos // 10).view(1, 1, 100)   # 09-22: 放送に任せる(結果はビット単位で同一)
        side = torch.arange(2, device=dev).view(1, 2, 1)
        tnum = torch.cat([t[..., 2:6], t[..., 6:13]], -1)          # 11
        tt = (self.kind(kind) + self.what(what) + self.x(tx) + self.y(ty) + self.side(side) + self.tnum(tnum) + self.typ.weight[0]).reshape(B, 200, -1)
        ex = ou[..., 0] > 0
        ux = ou[..., 1].long().clamp(0, 9); uy = ou[..., 2].long().clamp(0, 9)
        ut = self.uid(torch.arange(NU, device=dev)) + self.x(ux) + self.y(uy) + self.unum(torch.log1p(ou[..., 3:15])) + self.typ.weight[1]
        pex = pu[..., 0] > 0
        pt = self.x(pu[..., 1].long().clamp(0, 9)) + self.y(pu[..., 2].long().clamp(0, 9)) + self.typ.weight[2]
        sid = g[:, 47:63].long()
        st = self.shop(sid.clamp(min=-1) + 1) + self.typ.weight[3]            # 0 は空き枠
        sex = sid >= 0
        inv9 = g[:, 0:9]; pr9 = g[:, 9:18]; sh9 = g[:, 18:27]
        inum = torch.stack([pr9, (inv9 * 10000 - 10000) / 500, sh9], -1)
        it = self.item(torch.arange(9, device=dev)) + self.inum(inum) + self.typ.weight[4]
        step = (g[:, 34] * 720).round().long()
        # g の並び: obs_legal[0:37](在庫9 価格9 納屋9 種5 所持金自/相 step hour day) 自農場[37:42] 相手農場[42:47] 店[47:63] 納屋全12[63:75]
        gnum = torch.cat([g[:, 27:32], g[:, 32:34], g[:, 34:35], g[:, 37:42], g[:, 42:46], g[:, 72:75] / 10], 1)   # 20
        gt = (self.hour(step % 24) + self.day((step // 24).clamp(max=29)) + self.gnum(gnum) + self.typ.weight[5]).unsqueeze(1)
        tok = torch.cat([tt, ut, pt, st, it, gt], 1)
        pad = torch.cat([torch.zeros(B, 200, dtype=torch.bool, device=dev), ~ex, ~pex, ~sex,
                         torch.zeros(B, 10, dtype=torch.bool, device=dev)], 1)
        return tok, pad


class MLPEnc(nn.Module):
    """各層: h += MLP([LN(h), 有効トークンの平均])。トークン間のやりとりは平均だけ。"""
    def __init__(self, d, layers, ff):
        super().__init__()
        self.ln = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(2 * d, ff), nn.ReLU(), nn.Linear(ff, d)) for _ in range(layers))

    def forward(self, h, pad):
        w = (~pad).float().unsqueeze(-1)
        for ln, mlp in zip(self.ln, self.mlp):
            x = ln(h); m = (x * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1)
            h = h + mlp(torch.cat([x, m.expand_as(x)], -1))
        return h


class Policy(nn.Module):
    def __init__(self, d=128, layers=3, heads=4, ff=256, drop=0.1, enc="tf", dec_layers=2):
        super().__init__()
        self.d = d; self.enc_tok = TokenEnc(d); self.enc_kind = enc
        if enc == "tf":
            el = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(el, layers, enable_nested_tensor=False)
        else:   # 注意機構なし: トークンごとの MLP ＋ 全体の平均(DeepSets)。小さいモデルの比較用
            self.enc = MLPEnc(d, layers, ff)
        self.cand = nn.Embedding(NC, d)
        self.uq_in = nn.LayerNorm(2 * d)
        self.u_score = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.u_qty = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, UQN))
        self.mcand = nn.Embedding(NM + 1, d)            # +1 = 開始記号
        self.mpos = nn.Embedding(NS, d)
        dl = nn.TransformerDecoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dl, dec_layers)
        self.m_out = nn.Linear(d, NM)
        self.m_qty = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, MQN))
        self.value = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))   # 強化学習用

    def encode(self, tiles, ou, pu, g):
        tok, pad = self.enc_tok(tiles, ou, pu, g)
        if self.enc_kind == "tf": return self.enc(tok, src_key_padding_mask=pad), pad
        return self.enc(tok, pad), pad

    def unit_query(self, h, ou):
        B = h.shape[0]
        ux = ou[..., 1].long().clamp(0, 9); uy = ou[..., 2].long().clamp(0, 9)
        tile_idx = uy * 10 + ux                                              # 自分の農場のタイル(0..99)
        ht = torch.gather(h[:, :100], 1, tile_idx.unsqueeze(-1).expand(-1, -1, self.d))
        return torch.cat([h[:, 200:220], ht], -1)                            # (B,20,2d)

    def unit_logits(self, h, ou, legal):
        q = self.uq_in(self.unit_query(h, ou))                               # (B,20,2d)
        if FAST:
            # 09-23(MFU): 最初の Linear(3d→d) は [質問 2d | 候補 d] に分けられる。
            #   W = [Wq | Wc] とすると  cat([q,c]) @ W^T + b = (q@Wq^T + b) + c@Wc^T。
            #   候補ぶん c@Wc^T は盤面によらないので 1 回だけ計算して放送する。
            #   これで (B,20,44,3d) の実体化(MB=1024 で 1.7GB)と、そこにかかる 66 倍の乗算が消える。
            #   足す順が変わるだけで式は同じ。実測 MB=1024 で 1歩 576.8→562.7ms、ピーク 20.7→19.1GB。
            l0 = self.u_score[0]; l2 = self.u_score[2]
            a = F.linear(q, l0.weight[:, :2 * self.d], l0.bias)               # (B,20,d)
            c = F.linear(self.cand.weight, l0.weight[:, 2 * self.d:])         # (44,d)
            s = F.linear(F.relu(a.unsqueeze(2) + c.view(1, 1, NC, self.d)), l2.weight, l2.bias).squeeze(-1)
        else:
            B = q.shape[0]; ce = self.cand.weight.view(1, 1, NC, self.d).expand(B, NU, NC, self.d)
            s = self.u_score(torch.cat([q.unsqueeze(2).expand(-1, -1, NC, -1), ce], -1)).squeeze(-1)
        return s.masked_fill(~legal.bool(), -1e4), q   # fp16 で表せる値(-1e9 は溢れる)

    def unit_qty(self, q, cidx):
        return self.u_qty(torch.cat([q, self.cand(cidx.clamp(min=0))], -1))

    def market(self, h, pad, prev):
        """prev (B,11): 各位置の直前の候補(先頭は開始記号 NM)。教師強制で一度に計算。"""
        B = h.shape[0]; L = prev.shape[1]
        x = self.mcand(prev) + self.mpos(torch.arange(L, device=h.device)).unsqueeze(0)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=h.device), 1)
        y = self.dec(x, h, tgt_mask=causal, memory_key_padding_mask=pad)
        return self.m_out(y), y

    def market_qty(self, y, cidx):
        return self.m_qty(torch.cat([y, self.mcand(cidx.clamp(min=0, max=NM - 1))], -1))


def load(path, dev="cuda"):
    """重みと、横に置いた構成(<path>.cfg.json、無ければ既定の大きさ)から方策を作る。"""
    import json, os
    cfg = json.load(open(path + ".cfg.json")) if os.path.exists(path + ".cfg.json") else {}
    net = Policy(**cfg).to(dev); sd = torch.load(path, map_location=dev); net.load_state_dict(sd.get("net", sd) if isinstance(sd, dict) and "net" in sd else sd)
    return net.eval()
