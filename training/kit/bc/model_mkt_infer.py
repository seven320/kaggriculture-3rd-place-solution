"""exp266: 推論用「市場の特徴つき(+ ポインタ頭)」方策。ID を焼き込んだ後の形。
学習は model_tf_mkt.PolicyMkt(+ model_tf_ptr のポインタ頭)。推論では ID 埋め込みは焼き込み済みなので、
残るのは (a) 品目トークンに足す Linear(16→d)、(b) 市場の頭のポインタ項。"""
import math
import torch, torch.nn as nn
import model_tf as M
ITEM0 = 256
CLS2ITEM = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 0, 8, None, None, None, None, None, None, None]
SCALE = [10., 50., 300., 50., 300., 10., 20., 10., 50., 20., 300., 20., 5., 5., 10., 20.]


class PolicyMktInfer(M.Policy):
    def __init__(self, nf=16, ptr=False, **kw):
        kw.pop("nid", None); super().__init__(**kw)
        self.nf = nf; self.use_ptr = ptr
        self.mnum = nn.Linear(nf, self.d)
        self.register_buffer("mscale", torch.tensor(SCALE[:nf]))
        if ptr:
            self.pq = nn.Linear(self.d, self.d)
            self.register_buffer("cls2item", torch.tensor([-1 if c is None else c for c in CLS2ITEM], dtype=torch.long))

    def encode(self, tiles, ou, pu, g, mkt=None):
        tok, pad = self.enc_tok(tiles, ou, pu, g)
        if mkt is not None:
            m = self.mnum(mkt.float() / self.mscale)
            tok = torch.cat([tok[:, :ITEM0], tok[:, ITEM0:ITEM0 + 9] + m.to(tok.dtype), tok[:, ITEM0 + 9:]], 1)
        return self.enc(tok, src_key_padding_mask=pad), pad

    def market(self, h, pad, prev):
        B, L = prev.shape
        x = self.mcand(prev) + self.mpos(torch.arange(L, device=h.device)).unsqueeze(0)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=h.device), 1)
        y = self.dec(x, h, tgt_mask=causal, memory_key_padding_mask=pad)
        out = self.m_out(y)
        if self.use_ptr:
            ptr = torch.einsum("bld,bkd->blk", self.pq(y), h[:, ITEM0:ITEM0 + 9]) / math.sqrt(self.d)
            out = out + ptr.index_select(-1, self.cls2item.clamp(min=0)) * (self.cls2item >= 0).to(ptr.dtype)
        return out, y
