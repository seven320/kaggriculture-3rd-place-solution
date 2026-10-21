"""exp298: 批評家の計算を減らす(CRIT_LAST=1、既定は切。apo.py・ppo_orbit2.py から使う)。
批評家は encode の出力のうち最後のトークン(大域トークン h[:, -1])しか読まない。
最後の層は、注意の K/V には全トークンが要るが、Q・注意の出力・FFN は最後のトークンの分だけでよい。
→ 最後の層だけ 266 行ではなく 1 行で計算し、encode は (B, 1, d) を返す(読む側の h[:, -1] はそのまま動く)。
式は同じ。カーネルの形が変わるので fp16 の丸め程度の差は出る。"""
import types
import torch
import model_tf as M


def _encode_last(self, tiles, ou, pu, g):
    tok, pad = self.enc_tok(tiles, ou, pu, g)
    layers = self.enc.layers
    x = tok
    for layer in layers[:-1]:                      # 最後の層の手前までは従来どおり全トークン
        x = layer(x, src_key_padding_mask=pad)
    L = layers[-1]
    assert L.norm_first
    n = L.norm1(x)                                  # K/V は全トークンの正規化後
    a = L.self_attn(n[:, -1:], n, n, key_padding_mask=pad, need_weights=False)[0]
    y = x[:, -1:] + L.dropout1(a)
    y = y + L._ff_block(L.norm2(y))
    if self.enc.norm is not None: y = self.enc.norm(y)
    return y, pad


def install_last(crit):
    """批評家のインスタンスの encode を差し替える(ほかの方策には効かない)。compile より前に呼ぶ。"""
    assert type(crit).encode is M.Policy.encode and crit.enc_kind == "tf", "CRIT_LAST は素の Policy(tf)の批評家だけ"
    crit.encode = types.MethodType(_encode_last, crit)
    return crit
