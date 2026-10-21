"""Full-game self-play collection, GAE and clipped PPO updates."""
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
try:
    from .policy import ROOT,FEATURES,ACTIONS
except ImportError:
    from policy import ROOT,FEATURES,ACTIONS
sys.path.insert(0,str(ROOT/'src'))
try:
    from .runtime import FastRustEnvironment as RustEnvironment
    from .native import NativePool
except ImportError:
    from runtime import FastRustEnvironment as RustEnvironment
    from native import NativePool
from scripts.kagg_bc_kit.rust_adapter import compatible_actions

def gae(rewards,values,dones,gamma=1.,lam=1.,bootstrap=None):
    if bootstrap is None:bootstrap=torch.zeros_like(values[-1])
    advantages=torch.zeros_like(rewards);last=torch.zeros_like(bootstrap)
    following=bootstrap
    for t in reversed(range(len(rewards))):
        keep=(~dones[t]).to(rewards.dtype)
        delta=rewards[t]+gamma*following*keep-values[t]
        last=delta+gamma*lam*keep*last;advantages[t]=last;following=values[t]
    return advantages,advantages+values

def collect(policy,num_envs,episode_steps,seed_start,log=print,timings=None):
    """Both players use the same frozen policy for each complete iteration."""
    if num_envs<1 or not 2<=episode_steps<=720:raise ValueError('invalid rollout size')
    start=time.perf_counter()
    pool=NativePool(num_envs,episode_steps,seed_start)
    init_seconds=time.perf_counter()-start;sample_seconds=0.;env_seconds=0.
    records=[];rows=[]
    try:
        for t in range(episode_steps-1):
            tick=time.perf_counter()
            actions,record,_=policy.sample(pool.observations)
            sample_seconds+=time.perf_counter()-tick
            tick=time.perf_counter()
            pool.step(actions)
            rewards=torch.zeros(2*num_envs);dones=torch.from_numpy(pool.done.copy()).bool()
            for i in range(num_envs):
                if dones[2*i]:
                    money=pool.meta[2*i,4:6].tolist()
                    reward=math.log1p(max(0,money[0]))-math.log1p(max(0,money[1]))
                    rewards[2*i:2*i+2]=torch.tensor([reward,-reward]);dones[2*i:2*i+2]=True
                    rows.append(dict(seed=seed_start+i,money=money,reward_player0=reward))
            env_seconds+=time.perf_counter()-tick
            record['reward']=rewards;record['done']=dones;records.append(record)
            if (t+1)%120==0:log(f'rollout {t+1}/{episode_steps-1}, games={num_envs}')
        assert pool.done.all()
    finally:
        pool.close()
    tick=time.perf_counter()
    data={k:torch.stack([r[k] for r in records]) for k in records[0]}
    advantages,returns=gae(data['reward'],data['old_value'],data['done'])
    data['advantage']=advantages;data['return']=returns
    # time, environment/player -> transition; never cross those axes in GAE.
    flat={k:v.flatten(0,1) for k,v in data.items()}
    if timings is not None:timings.update(environment_init_seconds=init_seconds,sampling_seconds=sample_seconds,
        environment_step_seconds=env_seconds,buffer_and_gae_seconds=time.perf_counter()-tick)
    return flat,rows

def critic_warmup(policy,data,steps=32,batch_size=64,lr=3e-4):
    opt=torch.optim.Adam(policy.net.value.parameters(),lr=lr)
    losses=[]
    for _ in range(steps):
        idx=torch.randint(len(data['return']),(min(batch_size,len(data['return'])),))
        b={k:data[k][idx].to(policy.device) for k in FEATURES}
        # Clone outside inference_mode so the value head can save its input.
        with torch.no_grad():h,_=policy.net.encode(b['tiles'].float(),b['ou'],b['pu'],b['g'])
        value=policy.net.value(h[:,-1].detach().clone()).squeeze(-1)
        loss=(value-data['return'][idx].to(policy.device)).square().mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.net.value.parameters(),1.);opt.step()
        losses.append(float(loss.detach()))
    return losses

def ppo_loss(logp,old_logp,adv,value,returns,entropy,clip=.2,value_coef=.5,entropy_coef=.001):
    log_ratio=logp-old_logp;ratio=log_ratio.exp()
    actor=-torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
    critic=.5*(value-returns).square().mean()
    total=actor+value_coef*critic-entropy_coef*entropy.mean()
    stats=dict(actor=float(actor.detach()),critic=float(critic.detach()),entropy=float(entropy.mean().detach()),
        approx_kl=float(((ratio-1)-log_ratio).mean().detach()),clip_fraction=float(((ratio-1).abs()>clip).float().mean().detach()))
    return total,stats

def update(policy,data,optimizer,epochs=4,batch_size=64,target_kl=.02):
    n=len(data['return']);adv=data['advantage']
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    stats=[]
    for ep in range(epochs):
        for idx in torch.randperm(n).split(batch_size):
            b={k:data[k][idx] for k in (*FEATURES,*ACTIONS)}
            logp,value,entropy=policy.score(b)
            loss,row=ppo_loss(logp,data['old_logp'][idx].to(policy.device),adv[idx].to(policy.device),
                value,data['return'][idx].to(policy.device),entropy)
            assert torch.isfinite(loss),'non-finite PPO loss'
            if row['approx_kl']>target_kl:
                return stats,dict(stopped_for_kl=True,kl=row['approx_kl'],epoch=ep+1)
            optimizer.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(policy.net.parameters(),.5)
            assert torch.isfinite(norm),'non-finite PPO gradient'
            optimizer.step();row.update(epoch=ep+1,loss=float(loss.detach()),grad_norm=float(norm));stats.append(row)
    return stats,dict(stopped_for_kl=False)
