"""exp264(PTCG 上位10チーム全員の作り = ポインタ頭): 市場の頭を「固定の重み」から「品目トークンを指す」形に。
いまの市場の頭は Linear(d → 23クラス)。上位は「候補ごとにその実体のトークンとの内積でスコア」を使う。
ここでは 23クラスのうち品目に紐づく 0-8(売り)・9-13(種)・14,15(小麦・肥料の買い)を、系列の品目トークン(256..264)との内積で足す。
足す項は0で初期化するので、**最初は元の網と同じ出力**から始まる(公平な比較)。雇用・土地・空き・終了は元のまま。"""
import math
import torch, torch.nn as nn
import model_tf_id as MI
ITEM0 = 256
# クラス → 品目トークンの番号(0..8)。None は品目に紐づかない
CLS2ITEM = [0, 1, 2, 3, 4, 5, 6, 7, 8,      # 0-8 売り(9品目)
            0, 1, 2, 3, 4,                   # 9-13 種(小麦・ニンジン・トマト・イチゴ・メロン)
            0, 8,                            # 14 小麦の買い、15 肥料の買い
            None, None, None,                # 16-18 家畜
            None, None, None, None]          # 19 雇用、20 土地、21 空き、22 終了


class PolicyPtr(MI.PolicyID):
    def __init__(self, nid=16, **kw):
        super().__init__(nid=nid, **kw)
        self.pq = nn.Linear(self.d, self.d); nn.init.zeros_(self.pq.weight); nn.init.zeros_(self.pq.bias)
        idx = torch.tensor([-1 if c is None else c for c in CLS2ITEM], dtype=torch.long)
        self.register_buffer("cls2item", idx)

    def market(self, h, pad, prev):
        B, L = prev.shape
        x = self.mcand(prev) + self.mpos(torch.arange(L, device=h.device)).unsqueeze(0)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=h.device), 1)
        y = self.dec(x, h, tgt_mask=causal, memory_key_padding_mask=pad)
        base = self.m_out(y)                                              # (B,L,23) 元の頭
        q = self.pq(y)                                                    # (B,L,d) ポインタの問い合わせ
        toks = h[:, ITEM0:ITEM0 + 9]                                      # (B,9,d) 品目トークン
        ptr = torch.einsum("bld,bkd->blk", q, toks) / math.sqrt(self.d)    # (B,L,9)
        m = self.cls2item.clamp(min=0)
        add = ptr.index_select(-1, m) * (self.cls2item >= 0).to(ptr.dtype) # (B,L,23) 品目に紐づくクラスだけ
        return base + add, y
