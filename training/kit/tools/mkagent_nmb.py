"""「旧マスク外に下駄 B」方式(newmask-distill-plan.md 9節)のエージェント一式を作る。新しいファイル(mkagent*.py は変えない)。
方策の重みは**変えない**(gen21 / gen24 / その PPO の子をそのまま使う)。
  1. obs_feat.py = 正典(repo/common/bc/obs_feat.py)+ 末尾で legal_mask を差し替え、**旧マスク(版0)と新マスク(版2)を両方**作り
     sp_fast.BatchSPPool._lm と同じ印を返す: 0 = 非合法 / 1 = 旧マスクでも合法 / 3 = 新マスクでだけ増えた手。
     (lm != 0) = mask2 なので model_tf.unit_logits の masked_fill は mask2 で非合法を落とす。
  2. model_ptr_infer.py = PolicyPtrInfer(model_tf.Policy の子。state_dict の鍵は Policy と同じ)。unit_logits の後で
     印 3 の手の logit を変える。abs = l − NMB_PEN(model_tf.mig_apply)/ rel = min(l, 旧マスクの最大) − NMB_PEN(nmb_lib.rel_apply)。
     NMB_PEN ≥ 1e4 は「−∞」扱いで masked_fill(-1e4)(= 旧マスクそのもの。元の方策と手がビット単位で同じ)。
  3. cfg.json に "ptr": true と "nmb_pen": <値>。league.py(BC)は ptr でこの dir の PolicyPtrInfer を使う(league.py は変えない)。
     main.py は mkprobe_nm.py と同じ1行の差し替えで ptr を読む。
使い方: python3 tools/mkagent_nmb.py <重み.pt> <lab/agents の名前 or 絶対 dir> <NMB_PEN(例 6)> [方式 rel|abs = rel] [土台 dir = lab/agents/g5g4g21_nd]
  学習で使った入口と必ず同じ方式・同じ値にすること(rel = apo_nmbrel.py、abs = apo.py の MASK_MIG)。
"""
import json, os, shutil, sys, types
import torch
W = "/workspace/kaggriculture"
LINE = 'LEGAL_OPT = int(os.environ.get("LEGAL_OPT", "0"))'
OLD = '_NET = M.Policy(**_CFG); _NET.load_state_dict'
NEW = ('_P = _CFG.pop("ptr", False)\n'
       'if _P:\n    import model_ptr_infer as _MP\n'
       '_NET = (_MP.PolicyPtrInfer if _P else M.Policy)(**_CFG); _NET.load_state_dict')
FEAT_TAIL = '''

# ---- 09-24 nmb(newmask-distill-plan.md 9節、tools/mkagent_nmb.py が足した): 旧マスクと新マスクを両方作り、印を付けて返す ----
#   0 = 非合法 / 1 = 旧マスク(版0)でも合法 / 3 = 新マスク(版2)でだけ増えた手。sp_fast.BatchSPPool._lm(MASK_MIG=1 MIG_V=2 MIG_BASE=0)と同じ。
#   LEGAL_OPT は版を切り替えるためだけに使い、呼び出しの後で元に戻す。
NMB_MARK = True
_legal_mask_one = legal_mask


def legal_mask(raw, up, n, inv, shed, shed_total, seeds, day):
    global LEGAL_OPT
    o = LEGAL_OPT
    try:
        LEGAL_OPT = 0; L0 = _legal_mask_one(raw, up, n, inv, shed, shed_total, seeds, day)
        LEGAL_OPT = 2; L2 = _legal_mask_one(raw, up, n, inv, shed, shed_total, seeds, day)
    finally:
        LEGAL_OPT = o
    xp = (L2 != 0) & (L0 == 0)
    return ((L0 != 0) | xp).astype(np.uint8) | (xp.astype(np.uint8) * np.uint8(2))
'''
WRAP = '''"""旧マスク外に下駄を足す方策(tools/mkagent_nmb.py が作った。newmask-distill-plan.md 9節)。
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
'''


def sd_of(p):
    sd = torch.load(p, map_location="cpu")
    return sd.get("net", sd) if isinstance(sd, dict) and "net" in sd else sd


def main(ckpt, name, pen, mode="rel", base=W + "/lab/agents/g5g4g21_nd"):
    pen = float(pen); assert mode in ("abs", "rel")
    dst = name if os.path.isabs(name) else os.path.join(W, "lab/agents", name)
    if os.path.exists(dst): raise SystemExit("出力先が既にある(上書きしない): %s" % dst)
    shutil.copytree(base, dst, ignore=shutil.ignore_patterns("__pycache__"))
    assert not os.path.exists(dst + "/model_ptr_infer.py"), "土台が既に ptr 版"
    torch.save(sd_of(ckpt), dst + "/policy.pt")
    cfg = json.load(open(ckpt + ".cfg.json")) if os.path.exists(ckpt + ".cfg.json") else json.load(open(dst + "/cfg.json"))
    cfg = {k: v for k, v in cfg.items() if k not in ("ptr", "nmb_pen", "nmb_mode")}
    cfg["ptr"] = True; cfg["nmb_pen"] = pen; cfg["nmb_mode"] = mode; json.dump(cfg, open(dst + "/cfg.json", "w"))
    open(dst + "/model_ptr_infer.py", "w").write(WRAP)
    src = open(dst + "/main.py").read(); assert src.count(OLD) == 1, "main.py の読み込み行が想定と違う"
    open(dst + "/main.py", "w").write(src.replace(OLD, NEW, 1))
    s = open(W + "/repo/common/bc/obs_feat.py").read(); assert s.count(LINE) == 1
    open(dst + "/obs_feat.py", "w").write(s + FEAT_TAIL)
    # 検査: 環境変数を外して読み直し、印付きの版であること・重みが読めること
    lo = os.environ.pop("LEGAL_OPT", None)
    try:
        m = types.ModuleType("of_chk"); exec(compile(open(dst + "/obs_feat.py").read(), dst + "/obs_feat.py", "exec"), m.__dict__)
    finally:
        if lo is not None: os.environ["LEGAL_OPT"] = lo
    assert m.NMB_MARK and m.LEGAL_OPT == 0
    sys.path.insert(0, dst)
    import model_ptr_infer as MP
    c2 = dict(cfg); c2.pop("ptr")
    net = MP.PolicyPtrInfer(**c2); net.load_state_dict(torch.load(dst + "/policy.pt", map_location="cpu"))
    assert not [k for k in net.state_dict() if "pid" in k]
    print("%s: 重み %s / NMB_PEN %g(B = %g)方式 %s / 土台 %s / obs_feat は印付き(0/1/3)" % (dst, ckpt, pen, -pen, mode, base))
    return dst


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], *(a[3:5] or []))
