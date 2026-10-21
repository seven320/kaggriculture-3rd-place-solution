"""Causal decoder: compute only the appended order and retain previous K/V."""
import torch
import torch.nn.functional as F
try:
    from .optimized import FactoredPolicy
    from .cached_market import MarketCache
    from .sdpa_encoder import SdpaPolicy
except ImportError:
    from optimized import FactoredPolicy
    from cached_market import MarketCache
    from sdpa_encoder import SdpaPolicy

class IncrementalMarket(MarketCache):
    def __init__(self,net,h,pad):
        super().__init__(net,h,pad)
        self.position=0
        self.self_kv=[]
        self.outputs=[]
        B,_,d=h.shape
        for layer in net.dec.layers:
            H=layer.self_attn.num_heads
            self.self_kv.append((h.new_empty(B,H,11,d//H),h.new_empty(B,H,11,d//H)))

    def append(self,prev):
        assert prev.shape[1]==self.position+1
        net=self.net;B=prev.shape[0];t=self.position;d=net.d
        x=net.mcand(prev[:,-1:])+net.mpos.weight[t][None,None]
        for layer,(memory_k,memory_v),(self_k,self_v) in zip(net.dec.layers,self.kv,self.self_kv):
            n=layer.norm1(x)
            attn=layer.self_attn;H=attn.num_heads
            qkv=F.linear(n,attn.in_proj_weight,attn.in_proj_bias)
            q,k,v=qkv.reshape(B,1,3,H,d//H).permute(2,0,3,1,4).unbind(0)
            self_k[:,:,t:t+1].copy_(k);self_v[:,:,t:t+1].copy_(v)
            a=F.scaled_dot_product_attention(q,self_k[:,:,:t+1],self_v[:,:,:t+1],dropout_p=0.)
            a=a.transpose(1,2).reshape(B,1,d)
            x=x+F.linear(a,attn.out_proj.weight,attn.out_proj.bias)
            n=layer.norm2(x);attn=layer.multihead_attn
            bias=attn.in_proj_bias[:d] if attn.in_proj_bias is not None else None
            q=F.linear(n,attn.in_proj_weight[:d],bias).view(B,1,H,d//H).transpose(1,2)
            a=F.scaled_dot_product_attention(q,memory_k,memory_v,attn_mask=self.mask,dropout_p=0.)
            a=a.transpose(1,2).reshape(B,1,d)
            x=x+F.linear(a,attn.out_proj.weight,attn.out_proj.bias)
            n=layer.norm3(x)
            x=x+layer.linear2(layer.activation(layer.linear1(n)))
        if net.dec.norm is not None:x=net.dec.norm(x)
        self.outputs.append(x)
        self.position+=1
        y=torch.cat(self.outputs,1)
        return net.m_out(y),y

class IncrementalPolicy(FactoredPolicy):
    """Optimization for the append-only, no-grad market loop in BatchAgent.

    Full teacher-forced or training calls retain the original decoder. Cache
    lifetime is one encoder output, so independent observations never share it.
    """
    def market(self,h,pad,prev):
        if self.training or torch.is_grad_enabled() or prev.shape[1]>10:
            self._incremental_cache=None
            return super().market(h,pad,prev)
        cache=getattr(self,'_incremental_cache',None)
        if cache is None or cache.memory is not h:
            if prev.shape[1]!=1:return super().market(h,pad,prev)
            cache=self._incremental_cache=IncrementalMarket(self,h,pad)
        if prev.shape[1]!=cache.position+1:
            self._incremental_cache=None
            return super().market(h,pad,prev)
        return cache.append(prev)

class IncrementalSdpaPolicy(SdpaPolicy,IncrementalPolicy):
    """Adopted GPU inference: SDPA encoder and incremental market decoder."""
    pass
