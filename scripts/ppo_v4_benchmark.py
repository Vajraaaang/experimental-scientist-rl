"""Complete cached-simulator/batched-PPO CPU versus native MPS timing."""
from pathlib import Path
import argparse
import json
import tempfile
import time
import numpy as np
import torch
from ppo_followup import setup,task_batch
from experimentlab.private_cache import CachedScientistEnv
from experimentlab.learning import ActorCritic,save_checkpoint,load_checkpoint
from experimentlab.ppo_v4 import collect_batch,batch_episodes,update_ppo
from experimentlab.compute import synchronize,doctor,peak_memory
from experimentlab.cli import write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=128)
    p.add_argument('--accelerated',action='store_true');args=p.parse_args()
    if args.batch<1:raise SystemExit('positive batch required')
    torch.set_num_threads(1);context=setup();config,pool,val,cache,bank,startup=context
    result={'hardware':doctor(),'cache_setup_seconds':startup,'batch_episodes':args.batch,'accelerated':args.accelerated,'devices':{}}
    for device in ('cpu','mps'):
        if device=='mps' and not torch.backends.mps.is_available():
            result['devices'][device]={'status':'unavailable'};continue
        torch.manual_seed(10);model=ActorCritic(kind='recurrent',hidden=64).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=.0003,eps=1e-5);measurements=[]
        try:
            for rep in range(4):
                tasks=task_batch(pool,config,10,rep*args.batch,args.batch,namespace=9)
                synchronize(device);tick=time.perf_counter()
                data,metrics=collect_batch(model,[CachedScientistEnv(t,config,cache) for t in tasks],device)
                prepared={}
                if args.accelerated:
                    from experimentlab.tensor_batch import pack_episodes,whole_episode_advantages
                    batch=pack_episodes(data,device);a,t=whole_episode_advantages(batch,1.)
                    prepared={'prepared_advantages':a,'prepared_targets':t}
                else:batch=batch_episodes(data,device)
                update_ppo(model,optimizer,batch,epochs=2,entropy_coef=.03,normalize_advantage=False,**prepared)
                collect_batch(model,[CachedScientistEnv(t,config,cache) for t in val[:16]],device,deterministic=True)
                with tempfile.TemporaryDirectory(prefix='ppo-v4-',dir=tempfile.gettempdir()) as folder:
                    checkpoint=Path(folder).resolve()/'checkpoint.pt'
                    save_checkpoint(checkpoint,model,optimizer,{'transitions':metrics['transitions']})
                    load_checkpoint(checkpoint,model,optimizer)
                synchronize(device);elapsed=time.perf_counter()-tick
                if rep:measurements.append({'seconds':elapsed,'transitions':metrics['transitions'],
                                            'transitions_per_second':metrics['transitions']/elapsed})
            result['devices'][device]={'status':'ok','measurements':measurements,
                                      'median_seconds':float(np.median([m['seconds'] for m in measurements])),
                                      'median_transitions_per_second':float(np.median([m['transitions_per_second'] for m in measurements])),
                                      'peak_process_rss_bytes':peak_memory()}
        except (RuntimeError,TypeError,NotImplementedError) as error:
            result['devices'][device]={'status':'unsupported','error':str(error)}
    selected='cpu'
    if result['devices'].get('mps',{}).get('status')=='ok':
        if result['devices']['mps']['median_transitions_per_second']>1.05*result['devices']['cpu']['median_transitions_per_second']:selected='mps'
    result['selected_device']=selected
    result['forecast_15m_transitions_seconds']=15000000/result['devices'][selected]['median_transitions_per_second']
    result['scope']='complete batched rollout/update/evaluation/checkpoint workload; startup separately measured'
    name=f'device-benchmark-accelerated-{args.batch}.json' if args.accelerated else 'device-benchmark.json'
    write_json(ROOT/'reports/ppo-v4'/name,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
