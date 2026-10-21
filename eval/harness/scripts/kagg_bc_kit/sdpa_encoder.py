"""Same encoder weights and tokens, using SDPA without dense attention scores."""
import torch
import torch.nn.functional as F
try:
    from .optimized import FactoredPolicy
except ImportError:
    from optimized import FactoredPolicy

class SdpaPolicy(FactoredPolicy):
    def encode(self,tiles,ou,pu,g):
        if self.training or torch.is_grad_enabled():
            return super().encode(tiles,ou,pu,g)
        x,pad=self.enc_tok(tiles,ou,pu,g)
        B,T,d=x.shape
        mask=x.new_zeros(B,1,1,T).masked_fill(pad[:,None,None,:],float('-inf'))
        for layer in self.enc.layers:
            assert layer.norm_first and not layer.training
            n=layer.norm1(x)
            attn=layer.self_attn
            qkv=F.linear(n,attn.in_proj_weight,attn.in_proj_bias)
            q,k,v=qkv.reshape(B,T,3,attn.num_heads,d//attn.num_heads).permute(2,0,3,1,4).unbind(0)
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
            y=y.transpose(1,2).reshape(B,T,d)
            x=x+layer.dropout1(F.linear(y,attn.out_proj.weight,attn.out_proj.bias))
            n=layer.norm2(x)
            x=x+layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(n)))))
        if self.enc.norm is not None:x=self.enc.norm(x)
        return x,pad
