"""exp233: ID 条件つき方策(model_tf_id.PolicyID)に、市場の特徴(品目ごと 8個: 相手の正味の売買 直前/24手/累計、自分の 24手/累計、
相手の盤面の作物数・収穫待ち・家畜数)を足す。品目トークン(系列の 256..264)に Linear(8→d) を加える(初期値0 = 足さない網と同じ出力から)。"""
import torch, torch.nn as nn
import model_tf_id as MI
SCALE = torch.tensor([10., 50., 300., 50., 300., 10., 20., 10.,      # exp262 の先読み8個
                      50., 20., 300., 20., 5., 5., 10., 20.])        # exp266: 相手の推定納屋・売り(24手/累計)・自分の売り・相手の盤面・値段と在庫の傾き
ITEM0 = 256          # 系列: タイル200 + 自ユニット20 + 相手ユニット20 + 店16 → 品目9(256..264)→ 大域(265)


class PolicyMkt(MI.PolicyID):
    def __init__(self, nid=16, nf=8, **kw):
        super().__init__(nid=nid, **kw)
        self.mnum = nn.Linear(nf, self.d); nn.init.zeros_(self.mnum.weight); nn.init.zeros_(self.mnum.bias)
        self.register_buffer("mscale", SCALE[:nf].clone())

    def encode(self, tiles, ou, pu, g, pid=None, mkt=None):
        tok, pad = self.enc_tok(tiles, ou, pu, g)
        if pid is not None:
            tok = torch.cat([tok[:, :-1], (tok[:, -1] + self.pid(pid)).unsqueeze(1)], 1)
        if mkt is not None:
            m = self.mnum(mkt.float() / self.mscale)
            tok = torch.cat([tok[:, :ITEM0], tok[:, ITEM0:ITEM0 + 9] + m.to(tok.dtype), tok[:, ITEM0 + 9:]], 1)
        if self.enc_kind == "tf": return self.enc(tok, src_key_padding_mask=pad), pad
        return self.enc(tok, pad), pad


class PolicyMktPtr(PolicyMkt):
    """exp266: 市場16特徴 + ポインタ頭(市場の候補を品目トークンとの内積でスコア)。MKT=1 PTR=1 のときはこれを使う。"""
    def __init__(self, nid=16, nf=8, **kw):
        import math, torch as _t
        super().__init__(nid=nid, nf=nf, **kw)
        import model_tf_ptr as _MP
        self.pq = nn.Linear(self.d, self.d); nn.init.zeros_(self.pq.weight); nn.init.zeros_(self.pq.bias)
        self.register_buffer("cls2item", _t.tensor([-1 if c is None else c for c in _MP.CLS2ITEM], dtype=_t.long))

    def market(self, h, pad, prev):
        import math
        B, L = prev.shape
        x = self.mcand(prev) + self.mpos(torch.arange(L, device=h.device)).unsqueeze(0)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=h.device), 1)
        y = self.dec(x, h, tgt_mask=causal, memory_key_padding_mask=pad)
        ptr = torch.einsum("bld,bkd->blk", self.pq(y), h[:, ITEM0:ITEM0 + 9]) / math.sqrt(self.d)
        return self.m_out(y) + ptr.index_select(-1, self.cls2item.clamp(min=0)) * (self.cls2item >= 0).to(ptr.dtype), y
