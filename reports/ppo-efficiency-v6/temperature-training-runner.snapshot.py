"""Development-only calibrated PPO continuation; old locked outcomes are never read."""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from research_stage import UnfilteredBank,quadrature_family,unfiltered_tasks
from ppo_followup import task_batch,evaluate
from ppo_cost_study import model_path,load_model
from experimentlab.envs import PilotConfig
from experimentlab.tasks import continuous_family,make_task,mechanism_hash
from experimentlab.private_cache import PrivateDynamicsCache,CachedScientistEnv
from experimentlab.cost_policy import PublicCostPolicy
from experimentlab.paired_returns import paired_trajectory_advantages
from experimentlab.ppo_v4 import collect_batch,update_ppo
from experimentlab.tensor_batch import pack_episodes
from experimentlab.learning import save_checkpoint
from experimentlab.cli import write_json,code_hash

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/ppo-efficiency-v6'
CALIBRATION={'duration_penalty':4.,'dose_penalty':4.,'stop_bonus':.5}
SEEDS=list(range(20,25))
BASE={'batch':256,'epochs':2,'clip':.1,'entropy':.0001,'learning_rate':.0001,
      'normalize_advantage':False,'center_advantage':False,'value_coef':.1,'target_kl':.01,
      'evaluation_every':100000,'episode_namespace':71,'paired_rollouts':4}
CONTROLS={
 'calibrated_restart':dict(BASE),
 'strong_updates':{**BASE,'epochs':4,'learning_rate':.0005},
 'strong_exploration':{**BASE,'epochs':4,'learning_rate':.0005,'entropy':.01,'entropy_final':.001,'entropy_anneal_fraction':.5},
 'gae_exploration':{**BASE,'epochs':4,'learning_rate':.0005,'entropy':.01,'entropy_final':.001,'entropy_anneal_fraction':.5,'paired_rollouts':1,'center_advantage':True},
}


class TemperaturePolicy(torch.nn.Module):
    """Public training schedule; values/rewards and deployment are unchanged."""
    def __init__(self,base,temperature=1.):
        super().__init__()
        if not np.isfinite(temperature) or temperature<=0:raise ValueError('positive finite temperature required')
        self.base=base;self.horizon=base.horizon;self.temperature=float(temperature)

    def forward(self,tokens):
        logits,values=self.base(tokens)
        return logits/self.temperature,values


def digest(path):return hashlib.sha256(Path(path).resolve().read_bytes()).hexdigest()


def protect_previous(output=OUT):
    path=output/'preserved-studies.json'
    if path.exists():return json.loads(path.read_text())
    records={}
    for folder in ('feasibility-v1','research-v2','unfiltered-v3','ppo-v4','ppo-cost-v5'):
        for p in sorted((ROOT/'reports'/folder).rglob('*')):
            if p.is_file():records[str(p.relative_to(ROOT))]=digest(p)
    write_json(path,records);return records


def setup(output=OUT):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    protect_previous(output)
    config=PilotConfig();pool=continuous_family(7401,4096)
    dev=continuous_family(8401,4096)[2048:4096]
    start=time.perf_counter();cache=PrivateDynamicsCache(pool+dev,config)
    bank=UnfilteredBank(quadrature_family(2),config)
    cases=unfiltered_tasks(dev,bank,1024,95441);tuning,confirmation=cases[:512],cases[512:]
    manifest={'train':[mechanism_hash(m) for m in pool],
              'tuning':[mechanism_hash(t.mechanism) for t in tuning],
              'confirmation':[mechanism_hash(t.mechanism) for t in confirmation],
              'train_pool_seed':7401,'development_generator_seed':8401,'development_parent_indices':[2048,4096],
              'development_episode_seed':95441,'reserved_test_pool_seed':97401,'reserved_test_size':4096,
              'reserved_test_episode_seed':105442,'config':asdict(config),'prior':'unfiltered',
              'fresh_test_generated':False}
    assert not(set(manifest['train'])&set(manifest['tuning']))
    assert not(set(manifest['tuning'])&set(manifest['confirmation']))
    path=output/'development-manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=manifest:raise ValueError('existing v6 development manifest differs')
    else:write_json(path,manifest)
    protocol={'controls':CONTROLS,'pilot_seeds':[20,21],'pilot_budget_per_run':500000,
              'main_seeds':SEEDS,'main_budget_per_seed':3000000,'calibration':CALIBRATION,
              'selection':'mean original tuning return; all five seeds retained; confirmation remains development',
              'closure_margins':{'cost_increase_upper':.10,'mse_increase_upper':.0001,'return_improvement_lower':-.005},
              'no_imitation_labels_in_ppo':True,'source_hash_at_start':code_hash()}
    if not (output/'protocol.json').exists():write_json(output/'protocol.json',protocol)
    return config,pool,tuning,confirmation,cache,bank,time.perf_counter()-start


def train(name,settings,seed,budget,context,output,initial,calibration):
    config,pool,validation,cache,bank,_=context
    output=Path(output).resolve();destination=output/'development'/name/f'seed{seed}'
    if destination.exists() and any(destination.iterdir()):raise FileExistsError('existing training output; refusing overwrite')
    destination.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(970000+seed)
    base=load_model(Path(initial).resolve());base.train()
    model=PublicCostPolicy(base,**calibration)
    optimizer=torch.optim.Adam(base.parameters(),lr=settings.get('learning_rate',.0005),eps=1e-5)
    batch_size=settings.get('batch',256);group_size=settings.get('paired_rollouts',4)
    if batch_size%group_size or group_size<1 or group_size>8:raise ValueError('invalid rollout grouping')
    count=episodes=updates=0;curve=[];evaluations=[];maximum_error=0.;tick=time.perf_counter()
    best=evaluate(model,validation,config,cache);selected_count=0
    initial_sha=digest(initial)
    metadata={'manifest_path':str(output/'development-manifest.json'),'prior':'unfiltered',
              'scope':'ppo_efficiency_v6_development','settings':settings,'seed':seed,
              'resolved_config':asdict(config),'calibration':calibration,'initial_checkpoint_sha256':initial_sha}
    def save(path):
        save_checkpoint(path,base,optimizer,{'transitions':count,'episodes':episodes,'updates':updates},metadata)
    save(destination/'selected.pt')
    evaluations.append({'transitions':0,'greedy':best})
    while count<budget:
        parents=task_batch(pool,config,seed,episodes//group_size,batch_size//group_size,settings.get('episode_namespace',71))
        tasks=[make_task(t.mechanism,t.seed*8+j,config,t.target) for t in parents for j in range(group_size)] if group_size>1 else parents
        fraction=count/max(1,budget)
        temperature=1.+(settings.get('temperature_start',1.)-1.)*max(0.,1.-fraction/settings.get('temperature_end_fraction',.7))
        training_policy=TemperaturePolicy(model,temperature)
        records,rollout=collect_batch(training_policy,[CachedScientistEnv(t,config,cache) for t in tasks],'cpu')
        count+=rollout['transitions'];episodes+=batch_size;updates+=1
        batch=pack_episodes(records,'cpu');prepared={}
        if group_size>1:
            advantages,targets=paired_trajectory_advantages(batch,group_size)
            prepared={'prepared_advantages':advantages,'prepared_targets':targets}
        progress=count/max(1,budget);start_entropy=settings.get('entropy',.0001)
        end_entropy=settings.get('entropy_final',start_entropy)
        entropy=start_entropy+(end_entropy-start_entropy)*min(1.,progress/settings.get('entropy_anneal_fraction',1.))
        metrics=update_ppo(training_policy,optimizer,batch,epochs=settings.get('epochs',4),clip=settings.get('clip',.1),
                           entropy_coef=entropy,value_coef=settings.get('value_coef',.1),gae_lambda=1.,
                           target_kl=settings.get('target_kl',.01),
                           normalize_advantage=settings.get('normalize_advantage',False),
                           center_advantage=settings.get('center_advantage',group_size==1),**prepared)
        maximum_error=max(maximum_error,metrics['initial_logp_error'])
        if not curve or count//25000>curve[-1]['transitions']//25000:
            curve.append({'transitions':count,'rollout':rollout,'update':metrics,'entropy_coef':entropy,'training_temperature':temperature})
        every=settings.get('evaluation_every',100000)
        if count//every>evaluations[-1]['transitions']//every or count>=budget:
            free=evaluate(model,validation,config,cache)
            evaluations.append({'transitions':count,'greedy':free})
            if free['mean_return']>best['mean_return']:
                best=free;selected_count=count;save(destination/'selected.pt')
            write_json(destination/'progress.json',{'settings':settings,'calibration':calibration,'curve':curve,'evaluations':evaluations})
            print(json.dumps({'run':name,'seed':seed,'transitions':count,'return':free['mean_return'],
                              'cost':free['mean_variable_cost'],'rmse':free['aggregate_rmse'],'best_at':selected_count,
                              'entropy':metrics['entropy'],'kl':metrics['post_kl']}),flush=True)
    save(destination/'checkpoint.pt')
    result={'name':name,'seed':seed,'settings':settings,'calibration':calibration,
            'transitions':count,'policy_decisions':count,'episodes':episodes,'updates':updates,
            'curve':curve,'evaluations':evaluations,'selected_checkpoint':str(destination/'selected.pt'),
            'selected_at_transitions':selected_count,'selected_validation':best,
            'checkpoint':str(destination/'checkpoint.pt'),'initial_checkpoint':str(Path(initial).resolve()),
            'initial_checkpoint_sha256':initial_sha,'maximum_initial_logp_error':maximum_error,
            'training_and_validation_seconds':time.perf_counter()-tick,'scope':'development_only',
            'policy_parameters':sum(p.numel() for p in base.parameters()),'device':'cpu'}
    write_json(destination/'result.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('setup','train'),default='setup')
    parser.add_argument('--name',default='strong_exploration');parser.add_argument('--seed',type=int,default=20)
    parser.add_argument('--transitions',type=int,default=500000);parser.add_argument('--settings')
    parser.add_argument('--initial');parser.add_argument('--output',default=str(OUT))
    args=parser.parse_args();torch.set_num_threads(1);output=Path(args.output).resolve()
    context=setup(output)
    if args.stage=='train':
        settings=json.loads(Path(args.settings).resolve().read_text()) if args.settings else CONTROLS[args.name]
        initial=Path(args.initial).resolve() if args.initial else model_path('trained',args.seed)
        mapped=(context[0],context[1],context[2],context[4],context[5],context[6])
        train(args.name,settings,args.seed,args.transitions,mapped,output,initial,CALIBRATION)
    else:print('Reserved new development and final-test protocol; final test not generated.')


if __name__=='__main__':main()
