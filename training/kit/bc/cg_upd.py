"""exp298: PPO の更新ループを CUDA グラフで回す(CUDAGRAPH=1、既定は切。apo.py・ppo_orbit2.py から使う)。
計算そのものは COMPILE_UPD=1 と同じ Inductor のカーネルで、それを CUDA グラフに録って再生するだけ(起動の手間を減らす)。
  - 更新ループの evaluate と批評家の encode だけを別の compile(mode="reduce-overhead")にする。
    対局の採取・PER0 の評価は従来の compile のまま(形がばらばらなので CUDA グラフにしない)。
  - 1歩ごとの記録(float(...) = GPU との同期 4回)をやめ、GPU 上に溜めて反復の最後に1回だけ取り出す。
値の一致は bench_upd.py で確認する。"""
import types
import torch
import model_tf as M


def make(raw_evaluate, crit):
    """(更新用の evaluate, 更新用の批評家の encode) を返す。どちらも元の関数を包み直してから compile する
    (従来の compile と同じコードに別の設定の compile を重ねないように)。"""
    def _ev_cg(net, b):
        return raw_evaluate(net, b)
    enc = type(crit).encode
    def _cenc_cg(*x):
        return enc(crit, *x)
    return (torch.compile(_ev_cg, dynamic=False, mode="reduce-overhead"),
            torch.compile(_cenc_cg, dynamic=False, mode="reduce-overhead"))


def _market_causal(self, h, pad, prev):
    """model_tf.Policy.market と同じ計算。違いは tgt_is_causal=True を明示するだけ。
    明示しないと TransformerDecoder が毎回 _detect_is_causal_mask で「因果マスクか」を GPU 上で比べて bool() で取り出す
    (= GPU との同期 + compile のグラフがそこで切れる)。このマスクは因果マスクそのものなので判定は必ず True で、
    その後の計算(自己注意に渡す attn_mask・is_causal)は判定したときとまったく同じ。"""
    B = h.shape[0]; L = prev.shape[1]
    x = self.mcand(prev) + self.mpos(torch.arange(L, device=h.device)).unsqueeze(0)
    causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=h.device), 1)
    y = self.dec(x, h, tgt_mask=causal, memory_key_padding_mask=pad, tgt_is_causal=True)
    return self.m_out(y), y


def hint_causal(*nets):
    """更新に使う方策(net・ref)だけ、市場デコーダの呼び出しを _market_causal に差し替える(インスタンスの属性なので他の方策には効かない)。
    FASTINF の採取は type(net).market を見て逐次キャッシュ版を使うので、採取の経路は変わらない。"""
    for n in nets:
        if n is not None and type(n).market is M.Policy.market:
            n.market = types.MethodType(_market_causal, n)


def step_begin():
    """ミニバッチの頭で呼ぶ。前のミニバッチの CUDA グラフの出力はもう使わない、と知らせる(記憶領域を使い回せる)。"""
    torch.compiler.cudagraph_mark_step_begin()


def stat4(pl, kl, ent, rr, pmask, clip):
    """ログ用の4つ(方策の損失・KL・エントロピー・切られた割合)を GPU 上のまま返す。
    切られた割合は x[mask].mean() を sum/count で(真偽の選び出しは要素数を知るために同期するので)。記録だけの値で、学習には効かない。"""
    with torch.no_grad():
        m = pmask.bool()
        cf = (((rr - 1).abs() > clip) & m).float().sum() / m.sum()
        return torch.stack([pl.detach().float(), kl.detach().float(), ent.detach().mean().float(), cf.float()])


def fetch(stats):
    """stat4 の並びを1回で取り出して、従来と同じ [(pl, kl, ent, 切), ...] の形に。"""
    if not stats: return []
    return [tuple(r) for r in torch.stack(stats).double().cpu().tolist()]
