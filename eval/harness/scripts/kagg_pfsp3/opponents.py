"""League opponents that may carry their own post-processing (_pace from a submission's main.py, e.g. stock-repair SLACK -3)."""
import hashlib
from scripts.kagg_bc_kit.isolated_agent import load_agent
from scripts.kagg_bc_kit.contracts import require_mask
from pathlib import Path
import numpy as np
import torch
from scripts.kagg_ppo.policy import FEATURES,to_action
from scripts.kagg_pfsp.collect import GreedyGraphOpponent,SampledGraphOpponent
from scripts.kagg_pfsp3.opponent_inference import validate_opponent_inference

_MODULES = {}
_REPAIRS = {}
def pace_from(main_py):
    """The _pace (repair) function of a submission's main.py, loaded once per file (its dir supplies model_tf/obs_feat/stock_target)."""
    key = str(Path(main_py).resolve())
    if key not in _MODULES:
        mod = load_agent(key)
        _MODULES[key] = mod
        if hasattr(mod,'_probe') and hasattr(mod,'TOM_PRICE'):
            from scripts.kagg_pfsp3.repair_observation import wrap_tomato_repair
            _REPAIRS[key]=wrap_tomato_repair(mod)
        else:
            _REPAIRS[key]=mod._pace
    return _REPAIRS[key]


class _Repair:
    """Mixin: decode with a custom repair function instead of the reference bundle's _pace, and draw from a private CUDA
    generator (its own CUDAGeneratorState, bound while the graph is captured) so the learner's own draws are unaffected by
    which opponents were played (user 2026-09-16: the same initial states should give the same learner games) and so the
    opponents' graphs can replay at the same time on their own streams without sharing the graph seed/offset tensors."""
    def set_repair(self,repair):
        self._repair = repair;return self

    def drain_pending(self):
        """decode せず待機する。event 未記録なら投入 stream 自体を待つ。"""
        if getattr(self, '_pending', False):
            if getattr(self, '_enqueue_complete', False): self._output_ready.synchronize()
            elif hasattr(self, '_stream'): self._stream.synchronize()
            # isolated collect の D2H が途中で失敗した場合も完了を確認する。
            torch.cuda.current_stream(self.device).synchronize()
            self._pending = False; self._pending_pool = None; self._pending_rows = None

    def set_stream_seed(self,seed):
        if not hasattr(self,'_gen'):
            torch.cuda.init();self._gen = torch.Generator(device='cuda')
        self._gen.manual_seed(int(seed));return self

    def _ensure_generator(self):
        if not hasattr(self,'_gen'):
            self.set_stream_seed(int.from_bytes(hashlib.sha256(str(self.initial_weights).encode()).digest()[:4],'little'))

    @property
    def _rng(self):
        self._ensure_generator();return self._gen.get_state()

    @_rng.setter
    def _rng(self,v):
        self._ensure_generator();self._gen.set_state(v)

    @torch.inference_mode()
    def enqueue(self,pool,rows):
        if getattr(self,'_pending',False): raise ValueError('previous enqueue must be collected')
        if len(rows) != self.inference_batch: raise ValueError('opponent row count mismatch')
        self._pending_pool = pool;self._pending_rows = tuple(rows)
        self._pending = True;self._enqueue_complete = False
        require_mask(pool,rows,self.mask_version)
        index = np.asarray(rows)
        gathered = {k:torch.from_numpy(pool.features[k][index]) for k in FEATURES}
        if not hasattr(self,'_host'):
            self._host = {k:torch.empty_like(v,pin_memory=True) for k,v in gathered.items()}
            self._stream = torch.cuda.Stream();self._input_ready = torch.cuda.Event();self._output_ready = torch.cuda.Event()
        for k,v in gathered.items(): self._host[k].copy_(v)
        if self._graph is None:
            T = {k:v.to(self.device) for k,v in self._host.items()}
            self._ensure_generator()
            default = torch.cuda.default_generators[torch.cuda.current_device()]
            outer = default.graphsafe_get_state();default.graphsafe_set_state(self._gen)
            try: self._capture(T)
            finally: default.graphsafe_set_state(outer)
        self._input_ready.record(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(self._input_ready)
            for k,v in self._host.items(): self._static_in[k].copy_(v,non_blocking=True)
            self._graph.replay()
        self._output_ready.record(self._stream);self._enqueue_complete = True

    @torch.inference_mode()
    def collect(self,pool,rows):
        if not getattr(self,'_pending',False): raise ValueError('no pending enqueue')
        if pool is not self._pending_pool or tuple(rows) != self._pending_rows: raise ValueError('collect must match enqueue')
        if not self._enqueue_complete: raise ValueError('enqueue did not complete')
        torch.cuda.current_stream().wait_event(self._output_ready)
        out = self._static_out;repair = getattr(self,'_repair',None) or self.ref._pace
        a = {k:out[k].cpu().numpy() for k in ('ul','uq','ms','mq')};counts = out['counts'].cpu().tolist()
        actions = [repair(to_action(a['ul'][b],a['uq'][b],a['ms'][b],a['mq'][b],counts[b]),pool.observations[r]) for b,r in enumerate(rows)]
        self._pending = False;self._pending_pool = None;self._pending_rows = None;return actions

    @torch.inference_mode()
    def act(self,pool,rows):
        self.enqueue(pool,rows);return self.collect(pool,rows)


class _PreparedRepair(_Repair):
    """新方式だけの準備。全相手の capture を済ませてから replay する。"""
    def prepare(self,pool,rows):
        if self._graph is not None: return
        if getattr(self,'_pending',False): raise ValueError('previous enqueue must be collected')
        if len(rows) != self.inference_batch: raise ValueError('opponent row count mismatch')
        require_mask(pool,rows,self.mask_version)
        index = np.asarray(rows)
        gathered = {k:torch.from_numpy(pool.features[k][index]) for k in FEATURES}
        if not hasattr(self,'_host'):
            self._host = {k:torch.empty_like(v,pin_memory=True) for k,v in gathered.items()}
            if not hasattr(self,'_stream'): self._stream = torch.cuda.Stream()
            self._input_ready = torch.cuda.Event();self._output_ready = torch.cuda.Event()
        for k,v in gathered.items(): self._host[k].copy_(v)
        T = {k:v.to(self.device) for k,v in self._host.items()}
        self._ensure_generator()
        default = torch.cuda.default_generators[torch.cuda.current_device()]
        outer = default.graphsafe_get_state();default.graphsafe_set_state(self._gen)
        try: self._capture(T)
        finally: default.graphsafe_set_state(outer)

    def _capture(self,T):
        # graph・static tensor・専用 stream の寿命を相手 object に揃える。
        self.net.eval()
        if not hasattr(self,'_stream'): self._stream = torch.cuda.Stream()
        current = torch.cuda.current_stream()
        with torch.inference_mode(False),torch.no_grad():
            self._static_in = {k:v.clone() for k,v in T.items()}
            self._stream.wait_stream(current)
            with torch.cuda.stream(self._stream):
                for _ in range(3): self._core(self._static_in)
            current.wait_stream(self._stream)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph,stream=self._stream):
                self._static_out = self._core(self._static_in)
            current.wait_stream(self._stream)


class _DeferredRepair(_PreparedRepair):
    """同じ stream に転送と replay を積み、最後の event 後に decode する。"""
    @torch.inference_mode()
    def enqueue(self,pool,rows):
        if getattr(self,'_pending',False): raise ValueError('previous enqueue must be collected')
        if len(rows) != self.inference_batch: raise ValueError('opponent row count mismatch')
        self.prepare(pool,rows)
        self._pending_pool = pool;self._pending_rows = tuple(rows)
        self._pending = True;self._enqueue_complete = False;self._deferred_ready = False
        require_mask(pool,rows,self.mask_version)
        index = np.asarray(rows)
        gathered = {k:torch.from_numpy(pool.features[k][index]) for k in FEATURES}
        for k,v in gathered.items(): self._host[k].copy_(v)
        if not hasattr(self,'_host_out'):
            self._host_out = {k:torch.empty_like(self._static_out[k],device='cpu',pin_memory=True)
                              for k in ('ul','uq','ms','mq','counts')}
        self._input_ready.record(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(self._input_ready)
            for k,v in self._host.items(): self._static_in[k].copy_(v,non_blocking=True)
            self._graph.replay()
            for k,v in self._host_out.items(): v.copy_(self._static_out[k],non_blocking=True)
            self._output_ready.record(self._stream)
        self._enqueue_complete = True

    @torch.inference_mode()
    def collect(self,pool,rows):
        if not getattr(self,'_pending',False): raise ValueError('no pending enqueue')
        if pool is not self._pending_pool or tuple(rows) != self._pending_rows: raise ValueError('collect must match enqueue')
        if not self._enqueue_complete or not self._deferred_ready: raise ValueError('deferred output must be awaited')
        out = self._host_out;repair = getattr(self,'_repair',None) or self.ref._pace
        a = {k:out[k].numpy() for k in ('ul','uq','ms','mq')};counts = out['counts'].tolist()
        actions = [repair(to_action(a['ul'][b],a['uq'][b],a['ms'][b],a['mq'][b],counts[b]),pool.observations[r]) for b,r in enumerate(rows)]
        self._pending = False;self._pending_pool = None;self._pending_rows = None;return actions

    def act(self,pool,rows):
        self.enqueue(pool,rows);self._output_ready.synchronize();self._deferred_ready = True
        return self.collect(pool,rows)


class IsolatedGreedyOpponent(_PreparedRepair,GreedyGraphOpponent): pass
class IsolatedSampledOpponent(_PreparedRepair,SampledGraphOpponent): pass
class DeferredGreedyOpponent(_DeferredRepair,GreedyGraphOpponent): pass
class DeferredSampledOpponent(_DeferredRepair,SampledGraphOpponent): pass


class GreedyOpponent(_Repair,GreedyGraphOpponent): pass
class SampledOpponent(_Repair,SampledGraphOpponent): pass


class _SimpleRepair:
    """Pre-stream implementation: private RNG state, synchronous copy/replay/decode.

    Keep the reference's seed and RNG save/restore operations verbatim; this
    path never uses _Repair's pinned buffers or per-opponent replay streams.
    """
    def set_repair(self,repair):
        self._repair = repair;return self

    def set_stream_seed(self,seed):
        g = torch.Generator(device='cuda');g.manual_seed(int(seed));self._rng = g.get_state();return self

    @torch.inference_mode()
    def act(self,pool,rows):
        assert len(rows) == self.inference_batch,(len(rows),self.inference_batch)
        require_mask(pool,rows,self.mask_version)
        index = np.asarray(rows)
        T = {k:torch.as_tensor(pool.features[k][index],device=self.device) for k in FEATURES}
        private = getattr(self,'_rng',None)
        if private is not None: outer = torch.cuda.get_rng_state();torch.cuda.set_rng_state(private)
        try:
            if self._graph is None:
                self._capture(T)
            for k,v in T.items():
                self._static_in[k].copy_(v)
            self._graph.replay()
        finally:
            if private is not None: self._rng = torch.cuda.get_rng_state();torch.cuda.set_rng_state(outer)
        out = self._static_out;repair = getattr(self,'_repair',None) or self.ref._pace
        a = {k:out[k].cpu().numpy() for k in ('ul','uq','ms','mq')};counts = out['counts'].cpu().tolist()
        return [repair(to_action(a['ul'][b],a['uq'][b],a['ms'][b],a['mq'][b],counts[b]),pool.observations[r]) for b,r in enumerate(rows)]


class SimpleGreedyOpponent(_SimpleRepair,GreedyGraphOpponent): pass
class SimpleSampledOpponent(_SimpleRepair,SampledGraphOpponent): pass


def build(entry,rows,applied,ac_dtype,*,opponent_inference='stream_parallel'):
    """entry: league opponent dict (path, kind, optional main_py)."""
    validate_opponent_inference(opponent_inference)
    if opponent_inference == 'stream_parallel_isolated':
        cls = IsolatedGreedyOpponent if entry['kind'] == 'greedy' else IsolatedSampledOpponent
    elif opponent_inference == 'single_stream_deferred':
        cls = DeferredGreedyOpponent if entry['kind'] == 'greedy' else DeferredSampledOpponent
    elif opponent_inference == 'simple':
        cls = SimpleGreedyOpponent if entry['kind'] == 'greedy' else SimpleSampledOpponent
    else:
        cls = GreedyOpponent if entry['kind'] == 'greedy' else SampledOpponent
    o = cls(Path(entry['path']),'gpu',rows,autocast_bf16=applied['autocast_bf16'],autocast_dtype=ac_dtype)
    if entry.get('main_py'): o.set_repair(pace_from(entry['main_py']))
    if entry['kind'] != 'greedy': o.set_stream_seed(int.from_bytes(hashlib.sha256(str(entry['path']).encode()).digest()[:4],'little'))
    return o
