"""09-24 nmb: 新マスク移行の「旧マスク外に下駄」方式の式(newmask-distill-plan.md 9節)。新しいファイル。走っているジョブは読まない。

lm の印(sp_fast.BatchSPPool._lm、MASK_MIG=1 MIG_V=2 MIG_BASE=0): 0 = 非合法 / 1 = 旧マスクでも合法 / 3 = 新マスクでだけ増えた手。
  abs(= model_tf.mig_apply そのもの): 増えた手の logit から pen を引く。
      gen21 では増えた手の生の logit が旧マスクの最大を大きく超える局面が多く、pen 6 で貪欲の一致 0.850・増えた手の質量 平均 0.164。使えない。
  rel(この式): 増えた手の logit を**同じユニットの旧マスクの最大 logit で頭打ち**にしてから pen を引く。
        l'_a = min(l_a, max_{b∈mask0} l_b) − pen   (a が増えた手)
        l'_a = l_a                                 (それ以外)
      増えた手1つの確率は旧マスクの最善手の e^{−pen} 倍以下 → 貪欲は元の方策と常に同じ(同点を除く)。
      増えた手の logit が旧の最大より下なら勾配はその手に流れる(PPO が使える手を上げられる。上限は旧の最善 − pen)。
      pen → ∞ で元の方策そのもの。
install(M) で model_tf.mig_apply をこの式に差し替える。ppo_tf.sample(採取・関門・ものさし)と ppo_tf.evaluate(学習・KL・ent)は
どちらも M.mig_apply を呼ぶので、同じ式が両方にかかる(FASTINF の unit_logits 差し替えの影響も受けない: mig_apply は unit_logits の外)。
"""
import torch


def rel_apply(logits, legal, pen):
    l = legal.to(torch.uint8); xp = (l & 2) != 0; old = l == 1
    m0 = logits.masked_fill(~old, -1e4).amax(-1, keepdim=True)      # 生きたユニットは待機が必ず旧マスクで合法
    if not torch.is_tensor(pen): pen = torch.tensor(float(pen), device=logits.device)
    return torch.where(xp, torch.minimum(logits, m0) - pen.to(logits.dtype), logits)


def install(M):
    """model_tf の mig_apply を rel 版に差し替える(罰の値は従来どおり M.mig_set / MIG_PEN で持つ)。"""
    assert M.MIG and not M.MIG_MIX, "nmb の rel は MASK_MIG=1(MIG_MIX なし)で使う"
    if getattr(M.mig_apply, "_nmb_rel", False): return
    def mig_apply_rel(logits, legal):
        return rel_apply(logits, legal, M._mig_t(logits.device))
    mig_apply_rel._nmb_rel = True
    M.mig_apply = mig_apply_rel
