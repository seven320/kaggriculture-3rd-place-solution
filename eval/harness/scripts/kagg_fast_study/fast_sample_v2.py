"""CUDA-graph sampler v2 (capture outside inference mode): replay one captured graph per environment step instead of ~150 small kernel launches.

Same operations as scripts/kagg_ppo/policy.py:Policy.sample (unit loop, incremental market decode with a
fixed 10 positions, value head), so the action distribution is unchanged; only the random draws differ from an
un-captured run because the market loop no longer stops early when every row has finished (extra draws are masked).
Falls back to the ordinary path for greedy decoding or a batch size other than the captured one.
Autocast weight casts are captured with cache_enabled=False so replays re-cast the current parameters.
"""
import torch
import torch.nn.functional as F
from scripts.kagg_ppo_study.rollout import StudyPolicy
from scripts.kagg_ppo.policy import FEATURES,to_action
try:
    from scripts.kagg_ppo.runtime import categorical_fast as categorical
except ImportError:
    from scripts.kagg_ppo.policy import categorical


class GraphSamplePolicy(StudyPolicy):
    def __init__(self,weights,device='gpu',inference_batch=32,autocast_bf16=False,autocast_dtype=torch.bfloat16):
        super().__init__(weights,device,inference_batch)
        self.autocast_bf16 = autocast_bf16;self.autocast_dtype = autocast_dtype
        self._graph = None;self._static_in = None;self._static_out = None

    def _core(self,T):
        net = self.net;B = T['g'].shape[0]
        from scripts.kagg_pfsp3 import m12
        amp = lambda: torch.autocast('cuda',dtype=self.autocast_dtype,enabled=self.autocast_bf16,cache_enabled=False)
        if m12.enabled(net):  # m12 mask: sequential M1 exception, head with the free flag (scripts/kagg_pfsp3/m12.py)
            with amp(): h,pad = net.encode(T['tiles'].float(),T['ou'],T['pu'],T['g'])
            live = T['ou'][...,0] > 0
            ul,lp_u,en_u,q,_ = m12.sample_units(net,h,T,categorical,amp=amp)
            logp = (lp_u*live).sum(1);entropy = (en_u*live).sum(1)
            decision_logp=torch.zeros(B,30,device=self.device);decision_logp[:,:20]=lp_u*live
            return self._tail(T,h,pad,q,ul,live,logp,entropy,decision_logp)
        with torch.autocast('cuda',dtype=self.autocast_dtype,enabled=self.autocast_bf16,cache_enabled=False):
            h,pad = net.encode(T['tiles'].float(),T['ou'],T['pu'],T['g'])
            lg,q = net.unit_logits(h,T['ou'],T['legal'])
        lg = lg.float();live = T['ou'][...,0] > 0
        seeds = (T['g'][:,27:32]*20).round().long();used = torch.zeros_like(seeds)
        ul = torch.zeros(B,20,dtype=torch.long,device=self.device)
        logp = torch.zeros(B,device=self.device);entropy = torch.zeros_like(logp)
        decision_logp=torch.zeros(B,30,device=self.device)
        for u in range(20):
            mask = T['legal'][:,u].bool().clone();mask[:,0] |= ~live[:,u]
            mask[:,15:20] &= used < seeds
            c,lp,en = categorical(lg[:,u],mask=mask)
            ul[:,u] = c;logp += lp*live[:,u];entropy += en*live[:,u]
            decision_logp[:,u]=lp*live[:,u]
            plant = (c >= 15)&(c < 20)&live[:,u]
            used += F.one_hot((c-15).clamp(0,4),5)*plant[:,None]
        return self._tail(T,h,pad,q,ul,live,logp,entropy,decision_logp)

    def _tail(self,T,h,pad,q,ul,live,logp,entropy,decision_logp):
        net = self.net;B = T['g'].shape[0]
        with torch.autocast('cuda',dtype=self.autocast_dtype,enabled=self.autocast_bf16,cache_enabled=False):
            uq_logits = net.unit_qty(q,ul)
        uq,lp,en = categorical(uq_logits.float())
        qm = live&(ul >= 20);logp += (lp*qm).sum(1);entropy += (en*qm).sum(1)
        decision_logp[:,:20]+=(lp*qm)
        prev = torch.full((B,1),23,dtype=torch.long,device=self.device)
        ms = torch.full((B,10),22,dtype=torch.long,device=self.device)
        mq = torch.zeros_like(ms);active = torch.ones(B,dtype=torch.bool,device=self.device)
        net._incremental_cache = None
        for j in range(10):
            with torch.autocast('cuda',dtype=self.autocast_dtype,enabled=self.autocast_bf16,cache_enabled=False):
                logits,y = net.market(h,pad,prev)
            c,lp,en = categorical(logits[:,-1].float())
            c = c.masked_fill(~active,22);ms[:,j] = c
            logp += lp*active;entropy += en*active
            decision_logp[:,20+j]=lp*active
            with torch.autocast('cuda',dtype=self.autocast_dtype,enabled=self.autocast_bf16,cache_enabled=False):
                q_logits = net.market_qty(y[:,-1:],c[:,None])[:,0]
            qty,lp,en = categorical(q_logits.float())
            mq[:,j] = qty
            qm = active&(c < 19);logp += lp*qm;entropy += en*qm
            decision_logp[:,20+j]+=lp*qm
            active &= c != 22
            prev = torch.cat([prev,c[:,None]],1)
        with torch.autocast('cuda',dtype=self.autocast_dtype,enabled=self.autocast_bf16,cache_enabled=False):
            value = net.value(h[:,-1]).squeeze(-1).float()
        return dict(ul=ul,uq=uq,ms=ms,mq=mq,old_logp=logp,old_decision_logp=decision_logp,old_value=value,entropy=entropy,counts=live.sum(1))

    def _capture(self,T):
        # Capture outside inference mode: torch.cuda.graph registers the default CUDA generator's graph state, and
        # tensors created under inference mode there break a later cudagraph-trees capture (torch.compile
        # mode='reduce-overhead') with "Inplace update to inference tensor outside InferenceMode".
        self.net.eval()
        with torch.inference_mode(False),torch.no_grad():
            self._static_in = {k:v.clone() for k,v in T.items()}
            stream = torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._core(self._static_in)
            torch.cuda.current_stream().wait_stream(stream)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._static_out = self._core(self._static_in)

    @torch.inference_mode()
    def sample(self,observations,greedy=False):
        if greedy or len(observations) != self.inference_batch:
            return super().sample(observations,greedy)
        T = self.features(observations)
        if self._graph is None:
            self._capture(T)
        for k,v in T.items():
            self._static_in[k].copy_(v)
        self._graph.replay()
        out = self._static_out
        a_cpu = {k:out[k].cpu().numpy() for k in ('ul','uq','ms','mq')};counts = out['counts'].cpu().tolist()
        decoded = [self.ref._pace(to_action(a_cpu['ul'][b],a_cpu['uq'][b],a_cpu['ms'][b],a_cpu['mq'][b],counts[b]),obs) for b,obs in enumerate(observations)]
        with torch.inference_mode(False):
            record = {k:v.cpu().clone() for k,v in {**self._static_in,**{k:out[k] for k in ('ul','uq','ms','mq','old_logp','old_decision_logp','old_value')}}.items()}
        return decoded,record,out['entropy'].cpu().clone()
