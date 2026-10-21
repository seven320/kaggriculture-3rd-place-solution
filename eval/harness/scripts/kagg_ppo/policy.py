"""PPO probabilities for the kit's worker actions and autoregressive market."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.kagg_bc_kit.optimized import OUT,reference_agent
from scripts.kagg_bc_kit.incremental_market import IncrementalSdpaPolicy
from scripts.kagg_bc_kit.sdpa_encoder import SdpaPolicy
from scripts.kagg_bc_kit.versioned_features import featurize,to_action
from scripts.kagg_bc_kit.contracts import PolicyConfig,load_contract,require_mask
try:
    from .runtime import categorical_fast as categorical
except ImportError:
    from runtime import categorical_fast as categorical

FEATURES=('tiles','ou','pu','g','legal')
ACTIONS=('ul','uq','ms','mq')

def unit_masks(legal,ou,g,ul):
    mask=legal.bool().clone();live=ou[...,0]>0
    mask[...,0]|=~live
    planted=(ul>=15)&(ul<20)&live
    one=F.one_hot((ul-15).clamp(0,4),5)*planted.unsqueeze(-1)
    before=one.cumsum(1)-one
    seeds=(g[:,27:32]*20).round().long()
    mask[:,:,15:20]&=before<seeds[:,None,:]
    return mask,live

class Policy:
    def __init__(self,weights,device='cuda'):
        self.device=torch.device('cuda' if device=='gpu' else device)
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.backends.cuda.matmul.allow_tf32=False
        path=Path(weights);cfg_path=Path(str(path)+'.cfg.json')
        if not cfg_path.exists():cfg_path=path.parent/'cfg.json'
        self.contract=load_contract(path); self.mask_version=self.contract["mask_version"]
        self.cfg=PolicyConfig(json.loads(cfg_path.read_text()),self.contract)
        cls=IncrementalSdpaPolicy if self.device.type=='cuda' else SdpaPolicy
        self.net=cls(**self.cfg).to(self.device).eval()
        self.net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        self.ref=reference_agent()
        self.initial_weights=path

    def features(self,observations):
        if observations and '_native_pool' in observations[0]:
            pool=observations[0]['_native_pool']
            assert len(observations)==pool.batch
            assert all(o['_native_pool'] is pool and o['_native_index']==b for b,o in enumerate(observations))
            require_mask(pool,range(pool.batch),self.mask_version)
            return {k:torch.as_tensor(pool.features[k],device=self.device) for k in FEATURES}
        arrays=[np.stack(v) for v in zip(*(featurize(o,self.mask_version) for o in observations))]
        return {k:torch.as_tensor(a,device=self.device) for k,a in zip(FEATURES,arrays)}

    @torch.inference_mode()
    def sample(self,observations,greedy=False):
        self.net.eval()
        T=self.features(observations);B=len(observations)
        h,pad=self.net.encode(T['tiles'].float(),T['ou'],T['pu'],T['g'])
        from scripts.kagg_pfsp3 import m12
        live=T['ou'][...,0]>0
        logp=torch.zeros(B,device=self.device);entropy=torch.zeros_like(logp)
        decision_logp=torch.zeros(B,30,device=self.device)
        if m12.enabled(self.net):
            ul,lp_u,en_u,q,_=m12.sample_units(self.net,h,T,categorical,greedy=greedy)
            logp+=(lp_u*live).sum(1);entropy+=(en_u*live).sum(1);decision_logp[:,:20]=lp_u*live
        else:
            lg,q=self.net.unit_logits(h,T['ou'],T['legal'])
            seeds=(T['g'][:,27:32]*20).round().long();used=torch.zeros_like(seeds)
            ul=torch.zeros(B,20,dtype=torch.long,device=self.device)
        for u in range(0 if m12.enabled(self.net) else 20):
            mask=T['legal'][:,u].bool().clone();mask[:,0]|=~live[:,u]
            mask[:,15:20]&=used<seeds
            c,lp,en=categorical(lg[:,u],mask=mask,greedy=greedy)
            ul[:,u]=c;logp+=lp*live[:,u];entropy+=en*live[:,u]
            decision_logp[:,u]=lp*live[:,u]
            plant=(c>=15)&(c<20)&live[:,u]
            used+=F.one_hot((c-15).clamp(0,4),5)*plant[:,None]
        uq,lp,en=categorical(self.net.unit_qty(q,ul),greedy=greedy)
        qm=live&(ul>=20)
        logp+=(lp*qm).sum(1);entropy+=(en*qm).sum(1)
        decision_logp[:,:20]+=(lp*qm)
        prev=torch.full((B,1),23,dtype=torch.long,device=self.device)
        ms=torch.full((B,10),22,dtype=torch.long,device=self.device)
        mq=torch.zeros_like(ms);active=torch.ones(B,dtype=torch.bool,device=self.device)
        for j in range(10):
            logits,y=self.net.market(h,pad,prev)
            c,lp,en=categorical(logits[:,-1],greedy=greedy)
            c=c.masked_fill(~active,22);ms[:,j]=c
            logp+=lp*active;entropy+=en*active
            decision_logp[:,20+j]=lp*active
            qty,lp,en=categorical(self.net.market_qty(y[:,-1:],c[:,None])[:,0],greedy=greedy)
            mq[:,j]=qty
            qm=active&(c<19);logp+=lp*qm;entropy+=en*qm
            decision_logp[:,20+j]+=lp*qm
            active&=c!=22
            if not bool(active.any()):break
            prev=torch.cat([prev,c[:,None]],1)
        value=self.net.value(h[:,-1]).squeeze(-1)
        actions={'ul':ul,'uq':uq,'ms':ms,'mq':mq}
        decoded=[]
        a_cpu={k:v.cpu().numpy() for k,v in actions.items()};counts=live.sum(1).cpu().tolist()
        for b,obs in enumerate(observations):
            raw=to_action(a_cpu['ul'][b],a_cpu['uq'][b],a_cpu['ms'][b],a_cpu['mq'][b],counts[b])
            decoded.append(self.ref._pace(raw,obs))
        # Training must be able to save these tensors for backward, even on CPU.
        with torch.inference_mode(False):
            record={k:v.cpu().clone() for k,v in {**T,**actions,'old_logp':logp,'old_decision_logp':decision_logp,'old_value':value}.items()}
        return decoded,record,entropy.cpu()

    def score(self,batch):
        """Recompute the same joint probability from saved sampled commands."""
        self.net.eval()  # PPO ratios require identical dropout-free policies.
        T={k:v.to(self.device) for k,v in batch.items()}
        h,pad=self.net.encode(T['tiles'].float(),T['ou'],T['pu'],T['g'])
        ul,uq,ms,mq=(T[k].long() for k in ACTIONS)
        from scripts.kagg_pfsp3 import m12
        if m12.enabled(self.net): lg,q,legal,_=m12.batch_units(self.net,h,T,ul)
        else: lg,q=self.net.unit_logits(h,T['ou'],T['legal']);legal=T['legal']
        mask,live=unit_masks(legal,T['ou'],T['g'],ul)
        _,lp,en=categorical(lg,ul,mask)
        logp=(lp*live).sum(1);entropy=(en*live).sum(1)
        _,lp,en=categorical(self.net.unit_qty(q,ul),uq)
        qm=live&(ul>=20);logp+=(lp*qm).sum(1);entropy+=(en*qm).sum(1)
        prev=torch.cat([torch.full_like(ms[:,:1],23),ms[:,:-1]],1)
        logits,y=self.net.market(h,pad,prev)
        # Include the sampled STOP, exclude subsequent padding positions.
        stop=ms==22;valid=(stop.cumsum(1)-stop.long())==0
        _,lp,en=categorical(logits,ms)
        logp+=(lp*valid).sum(1);entropy+=(en*valid).sum(1)
        _,lp,en=categorical(self.net.market_qty(y,ms),mq)
        qm=valid&(ms<19);logp+=(lp*qm).sum(1);entropy+=(en*qm).sum(1)
        value=self.net.value(h[:,-1]).squeeze(-1)
        return logp,value,entropy
