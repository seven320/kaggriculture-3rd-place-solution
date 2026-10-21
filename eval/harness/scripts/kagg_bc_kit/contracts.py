"""Explicit feature/mask contracts; missing metadata denotes the legacy line."""
import hashlib
import json
from pathlib import Path
import numpy as np

FEATURE_SCHEMA = 'kagg_75_v1'

def contract(mask_version=0):
    if type(mask_version) is not int or mask_version not in (0, 2):
        raise ValueError(f'unsupported mask version: {mask_version!r}')
    return dict(schema=1, features=FEATURE_SCHEMA, mask_version=mask_version)

def validate(value):
    expected = contract(value['mask_version'])
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError(f'unsupported feature contract: {value}')
    return expected

def check_nmb(architecture, feature_contract):
    """The nmb head (cfg ptr: true, submissions v96/v97) reads mask-0 vs mask-2 marks from a mask-2 legal input; under a
    mask-0 contract it would silently play mask 0 only. 2026-09-26."""
    if architecture.get('ptr') and feature_contract['mask_version'] != 2:
        raise ValueError(f'ptr/nmb policy needs a mask_version 2 contract (policy.pt.contract.json): {feature_contract}')

class PolicyConfig(dict):
    """Only architecture keys go to the model; metadata travels separately."""
    def __init__(self, architecture, feature_contract):
        super().__init__(architecture)
        self.feature_contract = validate(feature_contract)
        check_nmb(self, self.feature_contract)

def load_contract(weights):
    weights = Path(weights)
    p = Path(str(weights) + '.contract.json')
    if not p.exists():
        return contract(0)
    value = json.loads(p.read_text())
    result = validate(value)
    if value.get('weights_sha256') != hashlib.sha256(weights.read_bytes()).hexdigest():
        raise ValueError(f'weight/feature contract SHA mismatch: {weights}')
    return result

def save_contract(weights, cfg):
    weights = Path(weights)
    value = validate(getattr(cfg, 'feature_contract', contract(0)))
    value['weights_sha256'] = hashlib.sha256(weights.read_bytes()).hexdigest()
    if cfg.get('unit_adapter'): value['unit_adapter'] = dict(cfg['unit_adapter'])   # record only; load_contract ignores it
    path = Path(str(weights) + '.contract.json')
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)

def mask2(ou, g):
    """Kit v8 optimistic mask from public feature fields, vectorized by row.

    Seed demand across workers is still limited by the existing sampler.
    This is a policy candidate mask, not a guarantee of execution.
    """
    live = ou[..., 0] > 0
    x, y = ou[..., 1], ou[..., 2]
    out = np.zeros((*live.shape, 44), np.uint8)
    out[..., 0] = live
    for i, allowed in enumerate((y > 0, y < 9, x < 9, x > 0), 1):
        out[..., i] = live & allowed
    out[..., 5:15] = live[..., None]
    out[..., 20:44] = live[..., None]
    out[..., 15:20] = live[..., None] & (np.rint(g[..., None, 27:32] * 20) > 0)
    return out

def apply_masks(features, versions):
    versions = np.asarray(versions)
    if versions.shape != (len(features['ou']),) or not np.isin(versions, (0, 2)).all():
        raise ValueError('one supported mask version per observation required')
    rows = versions == 2
    if rows.any():
        features['legal'][rows] = mask2(features['ou'][rows], features['g'][rows])

def require_mask(pool, rows, version):
    actual = getattr(pool, 'mask_versions', None)
    if actual is None: actual=np.zeros(pool.batch,np.uint8)
    if np.any(actual[np.asarray(rows, dtype=np.int64)] != version):
        raise ValueError('pool/policy mask mismatch; configure player masks before collection')

def plan_masks(plan, learner, opponents):
    versions = np.full(len(plan)*2, learner.mask_version, np.uint8)
    for game, entry in enumerate(plan):
        if entry is not None:
            name, player = entry
            versions[2*game+1-player] = opponents[name].mask_version
    return versions
