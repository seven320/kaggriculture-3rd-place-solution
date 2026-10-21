"""exp264: 推論用のポインタ頭つき方策。学習は model_tf_ptr.PolicyPtr(ID 条件つき)だが、ID を焼き込んだあとは
普通の Policy + ポインタ頭だけが残る。リーグや提出ではこれを使う。"""
import math
import torch, torch.nn as nn
import model_tf as M
ITEM0 = 256
CLS2ITEM = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 0, 8, None, None, None, None, None, None, None]


class PolicyPtrInfer(M.Policy):
    def __init__(self, **kw):
        kw.pop("nid", None); kw.pop("nf", None); super().__init__(**kw)
        self.pq = nn.Linear(self.d, self.d)
        self.register_buffer("cls2item", torch.tensor([-1 if c is None else c for c in CLS2ITEM], dtype=torch.long))

    def market(self, h, pad, prev):
        B, L = prev.shape
        x = self.mcand(prev) + self.mpos(torch.arange(L, device=h.device)).unsqueeze(0)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=h.device), 1)
        y = self.dec(x, h, tgt_mask=causal, memory_key_padding_mask=pad)
        ptr = torch.einsum("bld,bkd->blk", self.pq(y), h[:, ITEM0:ITEM0 + 9]) / math.sqrt(self.d)
        m = self.cls2item.clamp(min=0)
        return self.m_out(y) + ptr.index_select(-1, m) * (self.cls2item >= 0).to(ptr.dtype), y
