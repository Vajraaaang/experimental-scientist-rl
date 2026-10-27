"""Native complete calibrated rollout/update benchmark, no silent MPS fallback."""
from pathlib import Path
import os
import json
import time
import tempfile
import numpy as np
import torch
from ppo_efficiency_v6 import setup,OUT,CALIBRATION,TemperaturePolicy
from ppo_followup import task_batch
from ppo_cost_study import load_model,model_path
from experimentlab.tasks import make_task
from experimentlab.private_cache import CachedScientistEnv
from experimentlab.cost_policy import PublicCostPolicy
from experimentlab.ppo_v4 import collect_batch,update_ppo
from experimentlab.tensor_batch import pack_episodes
from experimentlab.paired_returns import paired_trajectory_advantages
from experimentlab.learning import save_checkpoint,load_checkpoint
from experimentlab.cli import write_json


def main():
    torch.set_num_threads(1);context=setup();config,pool,tune,confirm,cache,bank,setup_seconds=context
    parents=task_batch(pool,config,20,0,64,79)
    tasks=[make_task(t.mechanism,t.seed*8+j,config,t.target) for t in parents for j in range(4)]
    results={}
    for device in ('cpu','mps'):
        if device=='mps' and not torch.backends.mps.is_available():
            results[device]={'available':False};continue
        base=load_model(model_path('trained',20)).to(device);model=PublicCostPolicy(base,**CALIBRATION).to(device)
        training=TemperaturePolicy(model,2.)
        optimizer=torch.optim.Adam(base.parameters(),lr=.0005,eps=1e-5);measurements=[]
        sync=lambda:torch.mps.synchronize() if device=='mps' else None
        for repeat in range(3):
            torch.manual_seed(9070+repeat);sync();tick=time.perf_counter()
            records,metrics=collect_batch(training,[CachedScientistEnv(t,config,cache) for t in tasks],device)
            batch=pack_episodes(records,device);a,t=paired_trajectory_advantages(batch,4)
            update=update_ppo(training,optimizer,batch,epochs=4,clip=.1,entropy_coef=.001,value_coef=.1,
                              prepared_advantages=a,prepared_targets=t,normalize_advantage=False,center_advantage=False,target_kl=.01)
            collect_batch(model,[CachedScientistEnv(t,config,cache) for t in tune[:32]],device,deterministic=True)
            with tempfile.TemporaryDirectory(prefix='ppo-v6-') as folder:
                checkpoint=Path(folder).resolve()/'checkpoint.pt';save_checkpoint(checkpoint,base,optimizer,{'repeat':repeat})
                load_checkpoint(checkpoint,base,optimizer)
            sync();elapsed=time.perf_counter()-tick
            measurements.append({'seconds':elapsed,'transitions':metrics['transitions'],'transitions_per_second':metrics['transitions']/elapsed,'initial_logp_error':update['initial_logp_error']})
        results[device]={'available':True,'measurements':measurements,'median_transitions_per_second':float(np.median([m['transitions_per_second'] for m in measurements]))}
    data={'scope':'complete256-episode cached public-calibrated PPO rollout,host packing,paired returns,4 update epochs,32-episode greedy evaluation,checkpoint roundtrip','private_cache_setup_seconds':setup_seconds,'concurrent_cpu_training':True,'mps_fallback_environment':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','unset'),'devices':results}
    write_json(OUT/'device-benchmark.json',data);print(json.dumps(data,indent=2),flush=True)


if __name__=='__main__':main()
