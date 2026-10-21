"""Validated PPO runtime optimizations; preserve game rules and categorical law."""
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from kaggriculture_agent.rust_simulation import RustEnvironment,AttrDict

def categorical_fast(logits,selected=None,mask=None,greedy=False):
    logits=logits.float()
    if mask is not None:logits=logits.masked_fill(~mask,-1e9)
    lp=F.log_softmax(logits,-1);prob=lp.exp()
    if selected is None:
        if greedy:selected=logits.argmax(-1)
        elif prob.is_cuda:
            # One-sample exponential race. Softmax already supplies positive
            # normalized weights; avoid multinomial's repeated host validation.
            flat=prob.reshape(-1,prob.shape[-1])
            noise=torch.empty_like(flat).exponential_(1)
            selected=torch.div(flat,noise).argmax(-1).reshape(prob.shape[:-1])
        else:selected=torch.multinomial(prob.reshape(-1,prob.shape[-1]),1).reshape(prob.shape[:-1])
    return selected,lp.gather(-1,selected.unsqueeze(-1)).squeeze(-1),-(prob*lp).sum(-1)

class FastRustEnvironment(RustEnvironment):
    """PPO only: plain observation dicts and a step count, no raw history copy.

    The collector has already stored features, actions, rewards and values.
    Generic RustEnvironment remains unchanged, including its full history API.
    """
    def _exchange(self,request):
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write(json.dumps(request,separators=(',',':'))+'\n');self._process.stdin.flush()
        line=self._process.stdout.readline()
        if not line:raise RuntimeError(f'Rust simulation server exited with {self._process.poll()}')
        payload=json.loads(line);actions=request.get('actions',[None,None])
        self.state=[AttrDict(observation=obs,reward=reward,status=status,action=action)
            for obs,reward,status,action in zip(payload['observations'],payload['rewards'],payload['statuses'],actions,strict=True)]
        self.done=bool(payload['done']);self.steps.append(None)
        if self.done and self._process.poll() is None:self._process.terminate();self._process.wait()
