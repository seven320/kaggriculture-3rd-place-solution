"""旧マスク外に下駄を足す方策(tools/mkagent_nmb.py が作った)。
重みは model_tf.Policy そのまま(state_dict の鍵も同じ)。unit_logits の後で、印 3(新マスクでだけ増えた手)の logit を変える:
  abs: l − pen                         (= model_tf.mig_apply、apo.py の MASK_MIG で MIG_PEN = pen)
  rel: min(l, 同じユニットの旧マスクの最大 logit) − pen   (= kagg/bc/nmb_lib.rel_apply、apo_nmbrel.py で MIG_PEN = pen)
pen >= 1e4 は -1e4 で塗る(= 旧マスクそのもの)。"""
import torch
import model_tf as M


class PolicyPtrInfer(M.Policy):
    def __init__(self, nmb_pen=6.0, nmb_mode="abs", **cfg):
        super().__init__(**cfg)
        self.nmb_pen = float(nmb_pen); self.nmb_mode = str(nmb_mode)
        assert self.nmb_mode in ("abs", "rel")

    def unit_logits(self, h, ou, legal):
        s, q = super().unit_logits(h, ou, legal)
        l = legal.to(torch.uint8); xp = (l & 2) != 0
        if self.nmb_pen >= 1e4: return s.masked_fill(xp, -1e4), q
        pen = torch.full((), self.nmb_pen, dtype=torch.float32, device=s.device).to(s.dtype)
        if self.nmb_mode == "abs": return s - pen * xp.to(s.dtype), q
        m0 = s.masked_fill(l != 1, -1e4).amax(-1, keepdim=True)
        return torch.where(xp, torch.minimum(s, m0) - pen, s), q
