"""exp280: 方策の計算を、結果を変えずに速くする(FASTINF=1 で ppo_tf / sp_lib から有効化)。
1) ユニットの頭の分解: Linear(cat[q, c]) = W_q·q + W_c·c + b。(B,20,44,3d) を作らずに同じ値を出す(学習にも効く)。
2) 市場デコーダの逐次キャッシュ: 記憶(266トークン)の K/V を1回だけ射影し、自己注意は過去の入力だけを保持して1枠ずつ進める(推論専用)。"""
import torch, torch.nn.functional as F
import model_tf as M


def unit_logits_fast(self, h, ou, legal):
    q = self.uq_in(self.unit_query(h, ou))                                   # (B,20,2d)
    lin1, act, lin2 = self.u_score[0], self.u_score[1], self.u_score[2]
    d2 = q.shape[-1]
    W = lin1.weight
    zq = F.linear(q, W[:, :d2], lin1.bias)                                  # (B,20,d)
    zc = F.linear(self.cand.weight, W[:, d2:])                               # (44,d)
    s = lin2(act(zq.unsqueeze(2) + zc.view(1, 1, M.NC, -1))).squeeze(-1)     # (B,20,44)
    return s.masked_fill(~legal.bool(), -1e4), q


def _proj(mha, x, which):
    d = mha.embed_dim; w, b = mha.in_proj_weight, mha.in_proj_bias
    i = {"q": 0, "k": 1, "v": 2}[which]
    return F.linear(x, w[i * d:(i + 1) * d], None if b is None else b[i * d:(i + 1) * d])


def _split(x, heads):
    B, L, d = x.shape
    return x.view(B, L, heads, d // heads).transpose(1, 2)                   # (B,h,L,dh)


class MarketCache:
    """net.market(h, pad, prev) を1枠ずつ進める版。step(tok) は m_out のロジット (B,NM) と y (B,1,d) を返す。"""
    def __init__(self, net, h, pad):
        self.net = net; self.layers = list(net.dec.layers); self.pos = 0
        self.heads = self.layers[0].self_attn.num_heads
        self.mem = []
        mask = None
        if pad is not None:
            mask = torch.zeros(pad.shape, dtype=h.dtype, device=h.device).masked_fill(pad, float("-inf"))[:, None, None, :]
        self.mem_mask = mask
        for L in self.layers:
            mh = L.multihead_attn
            self.mem.append((_split(_proj(mh, h, "k"), self.heads), _split(_proj(mh, h, "v"), self.heads)))
        self.past = [None] * len(self.layers)                                 # 各層の入力(正規化前)の履歴

    def step(self, tok):
        net = self.net
        x = net.mcand(tok.view(-1, 1)) + net.mpos.weight[self.pos].view(1, 1, -1)
        for li, L in enumerate(self.layers):
            hist = x if self.past[li] is None else torch.cat([self.past[li], x], 1)
            self.past[li] = hist
            sa = L.self_attn
            qn = L.norm1(x); kvn = L.norm1(hist)
            q = _split(_proj(sa, qn, "q"), self.heads); k = _split(_proj(sa, kvn, "k"), self.heads); v = _split(_proj(sa, kvn, "v"), self.heads)
            a = F.scaled_dot_product_attention(q, k, v)                       # 過去と自分だけ(因果は自動で満たす)
            a = a.transpose(1, 2).reshape(x.shape)
            x = x + L.dropout1(sa.out_proj(a))
            ca = L.multihead_attn
            q2 = _split(_proj(ca, L.norm2(x), "q"), self.heads)
            mk, mv = self.mem[li]
            a2 = F.scaled_dot_product_attention(q2, mk, mv, attn_mask=self.mem_mask)
            a2 = a2.transpose(1, 2).reshape(x.shape)
            x = x + L.dropout2(ca.out_proj(a2))
            x = x + L.dropout3(L.linear2(L.dropout(L.activation(L.linear1(L.norm3(x))))))
        self.pos += 1
        return net.m_out(x)[:, 0], x


def install():
    M.Policy.unit_logits = unit_logits_fast


def choose_capped(z, live, seeds, max_pass=8):
    """種の上限つきの行動選択を、20人の逐次ループなしで同じ結果にする。
    z: (B,20,44) の得点(貪欲ならロジット、標本なら ロジット+Gumbel)。live: (B,20)。seeds: (B,5)。
    元の実装は「ユニット順に、種が尽きた作物を隠して argmax」。ここでは隠す作物の集合を固定点まで更新する(順番の優先度は同じ)。"""
    B, NU, NC = z.shape
    masked = torch.zeros(B, NU, 5, dtype=torch.bool, device=z.device)
    neg = torch.finfo(z.dtype).min
    for _ in range(max_pass):
        zz = z.clone()
        zz[:, :, 15:20] = zz[:, :, 15:20].masked_fill(masked, neg)
        c = zz.argmax(-1)
        pick = (c.unsqueeze(-1) == (15 + torch.arange(5, device=z.device)).view(1, 1, 5)) & live.unsqueeze(-1)
        over = (pick.cumsum(1) > seeds.unsqueeze(1)) & pick          # その作物を選んだ順で上限を超えた分
        new = masked | over
        if bool((new == masked).all()): return c
        masked = new
    return c
