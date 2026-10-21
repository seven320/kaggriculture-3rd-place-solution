"""Switch CPU/GPU inference without changing checkpoints or the agent API.

CLI input: one official observation dict, or a list of independent observations.
Example:
  python scripts/kagg_bc_kit/inference.py --device gpu --batch-size 32 \
      --input observations.json --output actions.json
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import torch
try:
    from .optimized import OUT, FactoredPolicy, BatchAgent, reference_agent
    from .sdpa_encoder import SdpaPolicy
    from .incremental_market import IncrementalSdpaPolicy
except ImportError:
    from optimized import OUT, FactoredPolicy, BatchAgent, reference_agent
    from sdpa_encoder import SdpaPolicy
    from incremental_market import IncrementalSdpaPolicy

class Inference:
    """device is explicit: an unavailable GPU is an error, never a silent fallback."""
    def __init__(self, device='cpu', batch_size=32, weights=None):
        if device not in ('cpu','gpu','cuda'):
            raise ValueError('device must be cpu or gpu (cuda is also accepted)')
        if not isinstance(batch_size,int) or isinstance(batch_size,bool) or batch_size<1:
            raise ValueError('batch_size must be a positive integer')
        self.device='cpu' if device=='cpu' else 'cuda'
        if self.device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('GPU requested, but CUDA is unavailable; select device=cpu explicitly')
        self.batch_size=batch_size
        self.weights=Path(weights) if weights else OUT/'bundle/policy.pt'
        cfg_path=Path(str(self.weights)+'.cfg.json')
        if not cfg_path.is_file():cfg_path=self.weights.parent/'cfg.json'
        if not cfg_path.is_file():raise FileNotFoundError(f'Model configuration missing: {cfg_path}')
        from scripts.kagg_bc_kit.contracts import load_contract
        self.contract=load_contract(self.weights); self.mask_version=self.contract["mask_version"]
        cfg=json.loads(cfg_path.read_text())
        from scripts.kagg_bc_kit.contracts import check_nmb
        check_nmb(cfg,self.contract)
        self.ref=reference_agent()
        if self.device=='cuda':torch.backends.cuda.matmul.allow_tf32=False
        policy_class=SdpaPolicy if self.device=='cpu' else IncrementalSdpaPolicy
        net=policy_class(**cfg).to(self.device)
        state=torch.load(self.weights,map_location='cpu',weights_only=True)
        net.load_state_dict(state,strict=True)
        self._batch=BatchAgent(net,self.ref._pace,self.mask_version)
        self.checkpoint_sha256=hashlib.sha256(self.weights.read_bytes()).hexdigest()

    def act_batch(self, observations):
        """Each observation can belong to a different independent live game."""
        return [a for i in range(0,len(observations),self.batch_size)
                for a in self._batch(observations[i:i+self.batch_size])]

    def agent(self, observation, configuration=None):
        return self._batch([observation])[0]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','gpu','cuda'),default='cpu')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--weights',type=Path,help='State dict; defaults to the exact v61 checkpoint')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    inputs=json.loads(args.input.read_text())
    single=isinstance(inputs,dict)
    observations=[inputs] if single else inputs
    if not isinstance(observations,list) or not all(isinstance(o,dict) for o in observations):
        parser.error('input must be an observation object or a list of observation objects')
    runtime=Inference(args.device,args.batch_size,args.weights)
    if runtime.device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    actions=runtime.act_batch(observations)
    if runtime.device=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(actions[0] if single else actions,ensure_ascii=False)+'\n')
    print(json.dumps(dict(device=runtime.device,batch_size=runtime.batch_size,
        decisions=len(actions),seconds=seconds,checkpoint_sha256=runtime.checkpoint_sha256,
        output=str(args.output)),ensure_ascii=False))

if __name__=='__main__':main()
