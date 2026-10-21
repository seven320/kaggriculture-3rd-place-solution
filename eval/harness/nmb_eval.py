"""v96/v97(cfg に ptr・nmb_pen・nmb_mode を持つ提出物)を evaluate_land.inference の網で打たせるための包み(2026-09-26)。
提出物の main.py は model_ptr_infer.PolicyPtrInfer(= model_tf.Policy + unit_logits の後で「新マスクでだけ増えた手」に下駄)を使い、
obs_feat.legal_mask は 0 = 非合法 / 1 = 旧マスク(版 0)で合法 / 3 = 新マスク(版 2)でだけ合法 の印を返す。ここではそれを再現する:
- 網: 重みは model_tf.Policy と同じ鍵なので、cfg から ptr・nmb_* を除いた構成で evaluate_land.inference(mask_version 0)に読ませる
  (nets/<name>/policy.pt は提出物の policy.pt の複製、policy.pt.cfg.json は構成キーだけ)。
- 合法手: 版 0 の legal(Rust の特徴、または versioned_features.featurize(obs, 0))と contracts.mask2(ou, g)(版 2)から
  印 = ((L0|xp) | xp*2)、xp = L2 & ~L0 を作る(obs_feat.legal_mask の nmb 版と同じ式)。
- unit_logits の後処理: PolicyPtrInfer.unit_logits と同じ式(rel: min(l, 同じユニットの旧マスクの最大 logit) − pen、abs: l − pen)。
- 後処理(_pace)は各提出物の main.py(pace_from)。"""
import json, shutil
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ARCH_DROP = ('ptr', 'nmb_pen', 'nmb_mode')


def stage(src_dir, name):
    """提出物の source dir から nets/<name>/ を作る(policy.pt の複製 + 構成キーだけの cfg)。返り値 (policy path, nmb 設定)。"""
    src = Path(src_dir); cfg = json.loads((src/'cfg.json').read_text())
    nmb = dict(ptr=cfg.get('ptr', False), pen=float(cfg.get('nmb_pen', 6.0)), mode=str(cfg.get('nmb_mode', 'abs')))
    assert nmb['ptr'] and nmb['mode'] in ('abs', 'rel'), nmb
    arch = {k: v for k, v in cfg.items() if k not in ARCH_DROP}
    d = HERE/'nets'/name; d.mkdir(parents=True, exist_ok=True)
    p = d/'policy.pt'
    if not p.exists(): shutil.copy2(src/'policy.pt', p)
    assert p.read_bytes() == (src/'policy.pt').read_bytes()
    (d/'policy.pt.cfg.json').write_text(json.dumps(arch))
    return p, nmb


def mark(legal0, ou, g):
    from scripts.kagg_bc_kit.contracts import mask2
    L0 = np.asarray(legal0) != 0; L2 = mask2(np.asarray(ou), np.asarray(g)) != 0
    xp = L2 & ~L0
    return (L0 | xp).astype(np.uint8) | (xp.astype(np.uint8) * np.uint8(2))


def patch_net(net, pen, mode):
    orig = net.unit_logits

    def unit_logits(h, ou, legal, free=None):
        s, q = orig(h, ou, legal) if free is None else orig(h, ou, legal, free)
        l = legal.to(torch.uint8); xp = (l & 2) != 0
        if pen >= 1e4: return s.masked_fill(xp, -1e4), q
        p = torch.full((), pen, dtype=torch.float32, device=s.device).to(s.dtype)
        if mode == 'abs': return s - p * xp.to(s.dtype), q
        m0 = s.masked_fill(l != 1, -1e4).amax(-1, keepdim=True)
        return torch.where(xp, torch.minimum(s, m0) - p, s), q
    net.unit_logits = unit_logits
    net._nmb = dict(pen=pen, mode=mode)
    return net


def _marked(base):
    class Marked(base):
        def feature_arrays(self, observations):
            arrays = list(super().feature_arrays(observations))
            arrays[4] = mark(arrays[4], arrays[1], arrays[3])
            return arrays
    Marked.__name__ = 'Marked' + base.__name__
    return Marked


def nmb_inference(src_dir, name, device='cpu', batch=64, native=True):
    """evaluate_land.inference と同じ網(mask_version 0)に nmb の印と下駄を足したもの。native=False は公式観測 dict 用(確認用)。"""
    from scripts.kagg_pfsp3.evaluate_land import inference
    from scripts.kagg_pfsp3.opponents import pace_from
    from scripts.kagg_ppo_eval_native.batch import NativeBatchAgent
    from scripts.kagg_bc_kit.optimized import BatchAgent
    p, nmb = stage(src_dir, name)
    main_py = Path(src_dir)/'main.py'
    inf = inference(p, device, batch, main_py)
    assert inf.mask_version == 0
    net = patch_net(inf._batch.net, nmb['pen'], nmb['mode'])
    inf._batch = _marked(NativeBatchAgent if native else BatchAgent)(net, pace_from(main_py), 0)
    return inf
