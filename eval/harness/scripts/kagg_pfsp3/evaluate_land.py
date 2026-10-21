"""Uncorrected greedy/fp32 evaluation, seed-cluster paired CI and final selection."""
from contextlib import ExitStack
from pathlib import Path
import argparse
import hashlib
import json, math
import os
import subprocess
import sys
import numpy as np
import torch
from scripts.kagg_ppo.native import NativePool, LIBRARY, ITEMS
from scripts.kagg_ppo_eval_native.evaluate import NativeInference
from scripts.kagg_ppo_eval_native.batch import NativeBatchAgent
from scripts.kagg_pfsp3.opponents import pace_from
from scripts.kagg_pfsp3.land import LandController, ratio, se_metrics, _unit_phase
from scripts.kagg_pfsp3.gate import summarize

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def _finite(x):
    """Non-finite floats become None: the fp16 GradScaler's skipped steps report grad_norm=inf by design (existing GPU trainers write them as
    Infinity); JSON must stay strict for everything else."""
    if isinstance(x,float): return x if math.isfinite(x) else None
    if isinstance(x,dict): return {k:_finite(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [_finite(v) for v in x]
    return x


def write_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp'); tmp.write_text(json.dumps(_finite(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def write_jsonl(path,rows):
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def cfg_path(weights):
    p=Path(str(weights)+'.cfg.json')
    return p if p.exists() else Path(weights).parent/'cfg.json'


def identity(weights, main_py=None):
    from scripts.kagg_bc_kit.optimized import OUT
    main=Path(main_py) if main_py else OUT/'bundle/main.py'
    # Content identity, independent of path or torch.save container identity.
    state=torch.load(weights,map_location='cpu',weights_only=True)
    content=hashlib.sha256()
    for key,value in sorted(state.items()):
        a=value.detach().cpu().contiguous()
        content.update(key.encode()); content.update(str(a.dtype).encode()); content.update(str(tuple(a.shape)).encode())
        content.update(a.numpy().tobytes())
    pace={p.name:sha(p) for p in sorted(main.parent.iterdir()) if p.is_file() and p.suffix in ('.py','.json') and not p.name.endswith('.pt.cfg.json')}
    from scripts.kagg_bc_kit.contracts import load_contract
    return dict(feature_contract=load_contract(weights),policy_sha256=sha(weights),policy_content_sha256=content.hexdigest(),cfg_sha256=sha(cfg_path(weights)),pace=pace)


def inference(weights,device='cpu',batch=64,main_py=None):
    inf=NativeInference(device,batch,Path(weights))
    if main_py: inf._batch=NativeBatchAgent(inf._batch.net,pace_from(main_py),inf.mask_version)
    if device!='cpu':
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    return inf


def _feature_tiles(obs):
    """Decode only the fields used by SE counts and the first-harvest audit.

    Read the original float32 tiles, not the float16 inference copy. Integer
    fields and day offsets are rounded back using the native FFI's scales.
    """
    pool=obs['_native_pool']; index=obs['_native_index']; day=obs['day']
    tiles=[]
    for f in pool.tiles[index,0]:
        kind=round(float(f[0])*5)
        if kind in (0,1,2):
            tiles.append((None,'LOCKED','WEED')[kind]); continue
        tile=dict(yield_units=round(float(f[4])*6))
        if kind==5:
            tile.update(crop=ITEMS[round(float(f[1])*12)],watered_today=bool(f[2]),
                planted_day=day-round(float(f[10])*30),
                fertilized_until_day=-1 if f[12]==-1 else day+round(float(f[12])*30))
        else:
            tile['kind']='COOP' if kind==3 else 'PASTURE'
            if f[6]: tile['animal']=ITEMS[round(float(f[1])*12)]
        tiles.append(tile)
    return [tiles[y*10:(y+1)*10] for y in range(10)]


def _feature_se_metrics(obs):
    return se_metrics(dict(player=0,farms=[dict(tiles=_feature_tiles(obs))]))


def _first_se_harvest(obs, action):
    """Keep the old sequential unit audit, but decode a single farm on demand."""
    pool=obs['_native_pool']; index=obs['_native_index']
    units=pool.arrays['ou'][index]; units=units[units[:,0]>0]
    actions=[action.get('farmer',['PASS']),*action.get('hands',[])]
    if not any(a and a[0]=='HARVEST' and u[1]>=5 and u[2]>=5 for u,a in zip(units,actions)):
        return False
    positions=[[int(u[1]),int(u[2])] for u in units]
    farm=dict(tiles=_feature_tiles(obs),farmer=positions[0],hands=positions[1:])
    g=pool.arrays['g'][index]
    private=dict(shed=obs['private']['shed'],seeds={name:round(float(n)*20) for name,n in zip(ITEMS[:5],g[27:32])},
                 inventories=[{name:int(n) for name,n in zip(ITEMS,u[3:]) if n} for u in units])
    # Inventory insertion order affects receipts, not whether a unit harvested.
    orders=[list(inv) for inv in private['inventories']]
    _,_,harvest=_unit_phase(dict(player=0,day=obs['day'],farms=[farm],private=private),action,orders)
    return any(x>=5 and y>=5 for x,y,_,_ in harvest)


class _EvaluationLandController(LandController):
    """Disabled intervention path; collect evaluation fields from native arrays."""
    def _sync_orders(self, observations):
        # Evaluation never buys supplies on the policy's behalf or audits receipts.
        pass

    def before_step(self, observations, actions, **kwargs):
        if self.enabled or self.choice['assigned'] or self.b or self.supply_enabled or self.forced_steps:
            raise AssertionError('evaluation intervention')
        lands=[float(o['_native_pool'].arrays['g'][o['_native_index'],39])*4 for o in observations]
        self.pre_land=lands[self.learner_player]
        executed,forced=super().before_step(observations,actions,lands=lands)
        if any(forced) or executed!=actions: raise AssertionError('evaluation intervention')
        row=self.rows[self.learner_player]
        self.harvest=(self.pre_land==4 and row['first_se_harvest_step'] is None
                      and _first_se_harvest(observations[self.learner_player],executed[self.learner_player]))
        return executed,forced

    def after_step(self, observations):
        p=self.learner_player; obs=observations[p]; row=self.rows[p]; pre=self.pre[p]
        land=float(obs['_native_pool'].arrays['g'][obs['_native_index'],39])*4
        bought=any(a and a[0]=='BUY_LAND' for a in self.executed[p].get('market',[]))
        if bought and self.pre_land<4 and land==4 and not row['land4_success']:
            row.update(land4_success=True,purchase_step=pre['step'],purchase_day=pre['day'])
        if row['purchase_step'] is not None:
            dt=obs['step']-(row['purchase_step']+1)
            if dt in (24,72): row.update({f'{k}_{dt}':v for k,v in _feature_se_metrics(obs).items()})
            if self.harvest: row['first_se_harvest_step']=pre['step']
        return [0.,0.]


def play_land(learner,frozen,seeds,*,opponent='start',episode_steps=720,progress=None,trace=None):
    return play_land_multi(learner,{opponent:frozen},seeds,episode_steps=episode_steps,progress=progress,
                           trace=None if trace is None else lambda key,t,a:trace(key[1:],t,a))


def play_land_multi(learner,opponents,seeds,*,episode_steps=720,progress=None,trace=None):
    """Aggregate learner inference across BC opponents, retaining seed/player order."""
    seeds=list(seeds)
    if not seeds or len(set(seeds))!=len(seeds): raise ValueError('distinct nonempty seeds required')
    if not opponents: raise ValueError('nonempty opponents required')
    if episode_steps<2: raise ValueError('episode_steps must include an action and terminal observation')
    if not LIBRARY.is_file(): raise FileNotFoundError(LIBRARY)
    groups=[]
    for seed in seeds:
        if groups and seed==groups[-1][-1]+1: groups[-1].append(seed)
        else: groups.append([seed])
    with ExitStack() as stack:
        pools=[]; mapping={}
        for name in opponents:
            for group in groups:
                for p in (0,1):
                    pool=stack.enter_context(NativePool(len(group),episode_steps,group[0],land_observation_format='buffer',mask_versions=[learner.mask_version if player==p else opponents[name].mask_version for _ in group for player in (0,1)])); pools.append(pool)
                    for i,seed in enumerate(group): mapping[name,seed,p]=(pool,i,p)
        keys=[(name,seed,p) for name in opponents for seed in seeds for p in (0,1)]
        cases=[mapping[key] for key in keys]
        controllers={key:_EvaluationLandController(key[1],1,key[2],enabled=False) for key in keys}
        opponent_indices={name:[j for j,key in enumerate(keys) if key[0]==name] for name in opponents}
        for step in range(episode_steps-1):
            own=learner.act_batch([pool.observations[2*i+p] for pool,i,p in cases])
            other=[None]*len(cases)
            for name,frozen in opponents.items():
                indices=opponent_indices[name]
                actions=frozen.act_batch([cases[j][0].observations[2*cases[j][1]+1-cases[j][2]] for j in indices])
                if len(actions)!=len(indices): raise ValueError('opponent action count mismatch')
                for j,action in zip(indices,actions): other[j]=action
            if len(own)!=len(cases): raise ValueError('learner action count mismatch')
            executed={pool:[None]*pool.batch for pool in pools}
            for j,(key,(pool,i,p)) in enumerate(zip(controllers,cases)):
                pair=[None,None]; pair[p]=own[j]; pair[1-p]=other[j]
                a,forced=controllers[key].before_step(pool.observations[2*i:2*i+2],pair)
                if any(forced) or a!=pair: raise AssertionError('evaluation intervention')
                executed[pool][2*i:2*i+2]=a
                if trace is not None: trace(key,step,a)
            for pool in pools: pool.step(executed[pool])
            for key,(pool,i,p) in mapping.items(): controllers[key].after_step(pool.observations[2*i:2*i+2])
            if (step+1)%24==0 or step==episode_steps-2:
                snapshots={pool:[g['observations'] for g in pool.debug()] for pool in pools}
                for pool,i,p in cases:
                    obs=pool.observations[2*i+p]; full=snapshots[pool][i][p]
                    if (len(full['farms'][p]['unlocked_quadrants'])!=round(float(pool.arrays['g'][2*i+p,39])*4)
                            or se_metrics(full)!=_feature_se_metrics(obs)):
                        raise AssertionError('evaluation native land/SE feature mismatch')
            if progress and (step+1)%120==0: progress(step+1)
        if not all(pool.done.all() for pool in pools): raise RuntimeError('incomplete evaluation')
        rows=[]
        for key in keys:
            pool,i,p=mapping[key]
            rows.append(controllers[key].finish(snapshots[pool][i],key[0])['learner_outcomes'][0])
        return rows


def evaluation_summary(rows):
    if rows and all('margin' in r for r in rows):
        stats=summarize(rows)
    else:
        w=sum(r['result']=='win' for r in rows); d=sum(r['result']=='draw' for r in rows)
        stats=dict(n=len(rows),wins=w,draws=d,losses=len(rows)-w-d,win_rate=ratio(w+.5*d,len(rows)),margin_mean=None)
    measured=all('land4_success' in r and 'se_plants_72' in r and 'se_animals_72' in r for r in rows)
    bought=[r for r in rows if r.get('land4_success')]
    observable=[r for r in bought if r.get('se_plants_72') is not None and r.get('se_animals_72') is not None]
    used=[r for r in observable if r['se_plants_72']+r['se_animals_72']>0]
    return dict(stats,purchase_measurements_complete=measured,purchase_games=len(bought) if measured else None,
        land4_rate=ratio(len(bought),len(rows)) if measured else None,observable_72_games=len(observable) if measured else None,
        se_used_games=len(used) if measured else None,
        se_use_total=sum(r['se_plants_72']+r['se_animals_72'] for r in observable) if measured else None,
        final_land_wdl={str(n):{kind:sum(r.get('final_land_count')==n and r['result']==kind for r in rows) for kind in ('win','draw','loss')} for n in (1,2,3,4)})


def _evaluation_manifest(weights, config, private):
    ev=config['evaluation']; base=config['base']; ranges=config['seed_ranges']
    group='private' if private else 'public'
    opponents={'start':dict(policy=base['initial_bc'],main_py=base.get('repair_main_py')),**ev['submissions'],
               **{name:dict(policy=path,main_py=None) for name,path in ev['bc_pool'].items()}}
    manifest=dict(policy=identity(weights,base.get('repair_main_py')),engine_sha256=sha(LIBRARY),group=group,
        opponents={n:identity(v['policy'],v.get('main_py')) for n,v in opponents.items()},
        fixed_seeds=list(range(*ranges[group+'_fixed'])),bc_seeds=list(range(*ranges[group+'_bc'])),
        corrections=False,greedy=True,precision='fp32')
    return opponents,manifest


def _shard_manifest(manifest, index, count):
    if type(count) is not int or count<1 or type(index) is not int or not 0<=index<count:
        raise ValueError('invalid evaluation shard index/count')
    result=dict(manifest,shard=dict(index=index,count=count))
    for key in ('fixed_seeds','bc_seeds'):
        seeds=manifest[key]
        if len(seeds)%count: raise ValueError(f'{key} count must be divisible by shards')
        result[key]=seeds[index*len(seeds)//count:(index+1)*len(seeds)//count]
    return result


def evaluate(weights, config, out, *, device='cpu', private=False, progress=print, shard=None):
    out=Path(out); ev=config['evaluation']
    opponents,manifest=_evaluation_manifest(weights,config,private); group=manifest['group']
    if shard is not None: manifest=_shard_manifest(manifest,*shard)
    expected={(n,s,p) for n in opponents for s in manifest['bc_seeds' if n in ev['bc_pool'] else 'fixed_seeds'] for p in (0,1)}
    if (out/'result.json').exists():
        saved=json.loads((out/'result.json').read_text())
        rows=[json.loads(line) for line in (out/'outcomes.jsonl').read_text().splitlines()]
        if saved['manifest']!=manifest or saved['outcomes_sha256']!=sha(out/'outcomes.jsonl') or {_key(r) for r in rows}!=expected:
            raise ValueError('evaluation reuse SHA/seed mismatch')
        if not saved['summary']['purchase_measurements_complete']: raise ValueError('purchase measurements missing')
        return rows,saved
    out.mkdir(parents=True,exist_ok=True); all_rows=[]
    batch=int(ev.get('inference_batch',64))
    learner=inference(weights,device,batch,config['base'].get('repair_main_py'))
    for name,entry in opponents.items():
        if name in ev['bc_pool']: continue
        frozen=inference(entry['policy'],device,batch,entry.get('main_py'))
        all_rows.extend(play_land(learner,frozen,manifest['fixed_seeds'],opponent=name,progress=lambda t:progress(f'{group} {name} step={t}')))
        del frozen
    if ev['bc_pool']:
        # One learner forward for the 352-game BC panel; each opponent retains
        # its own net and pace. Fixed opponents keep their original batch size.
        learner.batch_size=int(ev.get('bc_inference_batch',512))
        frozen_pool={name:inference(path,device,batch) for name,path in ev['bc_pool'].items()}
        all_rows.extend(play_land_multi(learner,frozen_pool,manifest['bc_seeds'],progress=lambda t:progress(f'{group} bc_pool step={t}')))
        del frozen_pool
    del learner
    if device!='cpu':
        import gc
        gc.collect(); torch.cuda.empty_cache()
    if {_key(r) for r in all_rows}!=expected or len(all_rows)!=len(expected): raise ValueError('evaluation key mismatch')
    write_jsonl(out/'outcomes.jsonl',all_rows)
    result=dict(manifest=manifest,outcomes_sha256=sha(out/'outcomes.jsonl'),summary=evaluation_summary(all_rows),
                per_opponent={n:evaluation_summary([r for r in all_rows if r['opponent']==n]) for n in opponents})
    write_json(out/'result.json',result)
    return all_rows,result


def _merge_evaluation(rows, manifest, config, out):
    """Restore the single-process order and recompute all summaries from rows."""
    if 'shard' in manifest: raise ValueError('merge requires a full evaluation manifest')
    order={name:i for i,name in enumerate(manifest['opponents'])}
    expected={(n,s,p) for n in order
              for s in manifest['bc_seeds' if n in config['evaluation']['bc_pool'] else 'fixed_seeds'] for p in (0,1)}
    if len(rows)!=len(expected) or {_key(r) for r in rows}!=expected:
        raise ValueError('evaluation key mismatch')
    rows=sorted(rows,key=lambda r:(order[r['opponent']],r['seed'],r['learner_player']))
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    write_jsonl(out/'outcomes.jsonl',rows)
    result=dict(manifest=manifest,outcomes_sha256=sha(out/'outcomes.jsonl'),summary=evaluation_summary(rows),
                per_opponent={n:evaluation_summary([r for r in rows if r['opponent']==n]) for n in order})
    write_json(out/'result.json',result)
    return rows,result


def evaluate_sharded(weights, config, out, *, shards=1, device='cpu', private=False, progress=print):
    if type(shards) is not int or shards<1: raise ValueError('evaluation shards must be positive')
    if shards==1: return evaluate(weights,config,out,device=device,private=private,progress=progress)
    out=Path(out).resolve()
    if (out/'result.json').exists():
        return evaluate(weights,config,out,device=device,private=private,progress=progress)
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('sharded evaluation requires a parent outside torchrun')
    if device=='cpu': raise ValueError('sharded evaluation requires CUDA')
    _,manifest=_evaluation_manifest(weights,config,private)
    shard_manifests=[_shard_manifest(manifest,i,shards) for i in range(shards)]
    out.mkdir(parents=True,exist_ok=True)
    config_path=out/'shard_config.json'; write_json(config_path,config)
    children=[]
    try:
        for i in range(shards):
            command=[sys.executable,'-B','-m','scripts.kagg_pfsp3.evaluate_land',
                     '--config',str(config_path),'--weights',str(Path(weights).resolve()),'--out',str(out/f'shard{i}'),
                     '--shard',str(i),'--shards',str(shards),'--device',device]
            if private: command.append('--private')
            if progress: progress(f'{manifest["group"]} shard {i}/{shards}: starting')
            children.append(subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(i)}))
    finally:
        codes=[child.wait() for child in children]
    if any(codes): raise RuntimeError(f'evaluation shard processes failed: {codes}')
    rows=[]
    for i,shard_manifest in enumerate(shard_manifests):
        shard_out=out/f'shard{i}'
        saved=json.loads((shard_out/'result.json').read_text())
        part=[json.loads(line) for line in (shard_out/'outcomes.jsonl').read_text().splitlines()]
        expected={(n,s,p) for n in manifest['opponents']
                  for s in shard_manifest['bc_seeds' if n in config['evaluation']['bc_pool'] else 'fixed_seeds'] for p in (0,1)}
        if (saved['manifest']!=shard_manifest or saved['outcomes_sha256']!=sha(shard_out/'outcomes.jsonl')
                or len(part)!=len(expected) or {_key(r) for r in part}!=expected):
            raise ValueError(f'evaluation shard {i} SHA/seed mismatch')
        if not saved['summary']['purchase_measurements_complete']: raise ValueError('purchase measurements missing')
        rows.extend(part)
    return _merge_evaluation(rows,manifest,config,out)


def _key(row):
    return row['opponent'],row['seed'],row['learner_player']


def _score(row): return {'win':1.,'draw':.5,'loss':0.}[row['result']]


def cluster_ci(values):
    """One row per seed, columns kept together across players and opponents."""
    values=np.asarray(values,dtype=np.float64)
    if values.ndim==1: values=values[:,None]
    if not len(values) or not np.isfinite(values).all(): return None
    rng=np.random.Generator(np.random.PCG64(2026091808))
    draws=rng.integers(0,len(values),size=(10000,len(values)))
    means=values.mean(axis=1)[draws].mean(axis=1)
    return np.percentile(means,[2.5,97.5],method='linear').tolist()


def paired_summary(candidate,start,opponents=None,*,direct=False):
    names=sorted(opponents or {r['opponent'] for r in candidate})
    a={_key(r):r for r in candidate if r['opponent'] in names}
    b={_key(r):r for r in start if r['opponent'] in names}
    if len(a)!=sum(r['opponent'] in names for r in candidate) or set(a)!=set(b) or not a:
        return dict(status='undetermined',reason='missing_or_duplicate_keys')
    seeds=sorted({key[1] for key in a})
    expected={(n,s,p) for n in names for s in seeds for p in (0,1)}
    if set(a)!=expected: return dict(status='undetermined',reason='incomplete_seed_cluster')
    deltas=[[_score(a[n,s,p])-(.5 if direct else _score(b[n,s,p])) for n in names for p in (0,1)] for s in seeds]
    plus=sum(_score(a[k])>_score(b[k]) for k in a); minus=sum(_score(a[k])<_score(b[k]) for k in a)
    return dict(status='complete',n=len(a),seeds=len(seeds),candidate=evaluation_summary(list(a.values())),
        baseline=evaluation_summary(list(b.values())),difference=float(np.mean(deltas)),ci95=cluster_ci(deltas),
        improved=plus,worsened=minus,unchanged=len(a)-plus-minus,
        became_win=sum(a[k]['result']=='win' and b[k]['result']!='win' for k in a),
        lost_win=sum(a[k]['result']!='win' and b[k]['result']=='win' for k in a),
        cluster='seed: all players and included opponents',bootstrap_samples=10000,bootstrap_seed=2026091808,
        per_opponent={n:paired_summary([r for r in a.values() if r['opponent']==n],[r for r in b.values() if r['opponent']==n],[n],direct=direct)
                      for n in names} if len(names)>1 else {})


def select_candidate(public, policy_paths, settings, selection_path, *, private=False, evaluation_iterations=(90,100)):
    if private: raise ValueError('private reselection prohibited')
    final_iterations=list(evaluation_iterations)[-2:]
    if len(final_iterations)!=2 or len(set(final_iterations))!=2 or set(public)!=set(final_iterations):
        raise ValueError('selection requires the last two public evaluations only')
    fixed_names=tuple(settings.get('pfsp',{}).get('submission_opponents',{})) or ('v66','v62','v68')
    table=[]
    for n in final_iterations:
        rows=public[n]; fixed=[r for r in rows if r['opponent'] in fixed_names]; start=[r for r in rows if r['opponent']=='start']
        if len(fixed)!=384*len(fixed_names) or len(start)!=384 or len({_key(r) for r in rows})!=len(rows): raise ValueError('incomplete selection evaluation')
        table.append(dict(iteration=n,fixed_score=sum(map(_score,fixed))/len(fixed),start_score=sum(map(_score,start))/len(start),
                          policy=str(policy_paths[n]),policy_sha256=sha(policy_paths[n]),cfg_sha256=sha(cfg_path(policy_paths[n]))))
    selected=max(table,key=lambda r:(r['fixed_score'],r['start_score'],r['iteration']))
    record=dict(selected,table=table,settings=settings,source='public',private_reselection=False)
    path=Path(selection_path)
    if path.exists():
        if json.loads(path.read_text())!=record: raise ValueError('selection already fixed')
    else: write_json(path,record)
    return record


def adoption(iteration,block_rates,public90,public100,private_candidate,private_start,*,integrity=True,fixed_names=('v66','v62','v68'),expected_iterations=100,expected_bc_games=352):
    evaluations=[public90,public100,private_candidate]
    if iteration!=expected_iterations or not integrity or any(not x for x in evaluations) or not private_start:
        return dict(decision='未判定',reason='completion_or_integrity_missing')
    summaries=[evaluation_summary([r for r in rows if r['opponent'] in ('start',*fixed_names)]) for rows in evaluations]
    fixed=paired_summary(private_candidate,private_start,list(fixed_names))
    direct=paired_summary(private_candidate,private_start,['start'],direct=True)
    bc_names=sorted({r['opponent'] for r in private_candidate}-{'start',*fixed_names})
    bc=paired_summary(private_candidate,private_start,bc_names) if bc_names else dict(status='undetermined')
    if any(s['n']!=384*(1+len(fixed_names)) or not s['purchase_measurements_complete'] for s in summaries) or any(x['status']!='complete' for x in (fixed,direct,bc)) or bc.get('n')!=expected_bc_games or len(block_rates)!=2 or any(r is None for r in block_rates):
        return dict(decision='未判定',reason='measurements_or_pairing_missing')
    sustain=all(r>=.30 for r in block_rates) and all(s['land4_rate']>0 for s in summaries)
    used=all(s['observable_72_games']>=1 and s['se_use_total']>0 for s in summaries)
    win_delta=fixed['candidate']['wins']-fixed['baseline']['wins']
    accepted=sustain and used and direct['ci95'][0]>0 and win_delta>0
    rejected=any(r==0 for r in block_rates) or win_delta<0
    return dict(decision='採用' if accepted else '却下' if rejected else '保留',sustained=sustain,se_used=used,
                block_rates=block_rates,purchase_summaries=summaries,fixed=fixed,start_direct=direct,bc=bc,
                reason=None if accepted else 'predeclared_conditions_not_met')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--shard',type=int,required=True)
    parser.add_argument('--shards',type=int,required=True)
    parser.add_argument('--device',choices=('cpu','cuda','gpu'),default='cpu')
    parser.add_argument('--private',action='store_true')
    args=parser.parse_args(argv)
    evaluate(args.weights,json.loads(args.config.read_text()),args.out,device=args.device,
             private=args.private,shard=(args.shard,args.shards))


if __name__=='__main__': main()
