"""exp227: プレイヤー ID 条件つきの方策(ポケカ10位の identity head の移植)。
model_tf.Policy に「ID の埋め込み」を1つ足し、大域トークン(系列の最後)に加える。
大域トークンは typ.weight[5] を足して作るので、学習後に typ.weight[5] += pid.weight[k] と焼き込めば
ID k に固定した普通の model_tf.Policy になる(bake)。推論側(エージェント・リーグ・realcf)は何も変えずに使える。"""
import torch, torch.nn as nn
import model_tf as M


class PolicyID(M.Policy):
    def __init__(self, nid=16, **kw):
        super().__init__(**kw)
        self.pid = nn.Embedding(nid, self.d)
        nn.init.zeros_(self.pid.weight)          # 最初は ID なしの網と同じ出力

    def encode(self, tiles, ou, pu, g, pid=None):
        tok, pad = self.enc_tok(tiles, ou, pu, g)
        if pid is not None:
            tok = torch.cat([tok[:, :-1], (tok[:, -1] + self.pid(pid)).unsqueeze(1)], 1)
        if self.enc_kind == "tf": return self.enc(tok, src_key_padding_mask=pad), pad
        return self.enc(tok, pad), pad


def bake(state, k):
    """PolicyID の state_dict → ID k に固定した model_tf.Policy の state_dict。"""
    sd = {n: v.clone() for n, v in state.items() if n != "pid.weight"}
    sd["enc_tok.typ.weight"][5] += state["pid.weight"][k].to(sd["enc_tok.typ.weight"].dtype)
    return sd
