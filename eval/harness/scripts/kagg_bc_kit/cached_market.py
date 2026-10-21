"""Inference-only reuse of each decoder layer's projected board memory."""
import torch
import torch.nn.functional as F
try:
    from .optimized import FactoredPolicy
except ImportError:
    from optimized import FactoredPolicy

class MarketCache:
    def __init__(self,net,h,pad):
        self.net=net
        self.memory=h
        self.kv=[]
        B,S,d=h.shape
        for layer in net.dec.layers:
            attn=layer.multihead_attn
            heads=attn.num_heads
            bias=attn.in_proj_bias[d:] if attn.in_proj_bias is not None else None
            kv=F.linear(h,attn.in_proj_weight[d:],bias)
            k,v=kv.split(d,dim=-1)
            self.kv.append((k.view(B,S,heads,d//heads).transpose(1,2),
                            v.view(B,S,heads,d//heads).transpose(1,2)))
        self.mask=torch.zeros(B,1,1,S,device=h.device,dtype=h.dtype).masked_fill(pad[:,None,None,:],float('-inf'))

    def __call__(self,prev):
        net=self.net
        B,L=prev.shape
        x=net.mcand(prev)+net.mpos(torch.arange(L,device=prev.device)).unsqueeze(0)
        causal=torch.triu(torch.ones(L,L,dtype=torch.bool,device=x.device),1)
        for layer,(k,v) in zip(net.dec.layers,self.kv):
            assert layer.norm_first and not layer.training
            n=layer.norm1(x)
            x=x+layer.dropout1(layer.self_attn(n,n,n,attn_mask=causal,need_weights=False)[0])
            n=layer.norm2(x)
            attn=layer.multihead_attn
            d=n.shape[-1];heads=attn.num_heads
            bias=attn.in_proj_bias[:d] if attn.in_proj_bias is not None else None
            q=F.linear(n,attn.in_proj_weight[:d],bias).view(B,L,heads,d//heads).transpose(1,2)
            a=F.scaled_dot_product_attention(q,k,v,attn_mask=self.mask,dropout_p=0.)
            a=a.transpose(1,2).reshape(B,L,d)
            x=x+layer.dropout2(F.linear(a,attn.out_proj.weight,attn.out_proj.bias))
            n=layer.norm3(x)
            x=x+layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(n)))))
        if net.dec.norm is not None:x=net.dec.norm(x)
        return net.m_out(x),x

class CachedPolicy(FactoredPolicy):
    """Keep ordinary differentiable training; cache only no-grad/eval decoding."""
    def market(self,h,pad,prev):
        if self.training or torch.is_grad_enabled():
            self._market_cache=None
            return super().market(h,pad,prev)
        cache=getattr(self,'_market_cache',None)
        if cache is None or cache.memory is not h:
            cache=self._market_cache=MarketCache(self,h,pad)
        return cache(prev)
