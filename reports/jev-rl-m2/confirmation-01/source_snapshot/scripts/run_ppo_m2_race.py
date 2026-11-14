"""Execute the frozen finite M2 CPU race; preserve every job and budget failure."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'.venv/bin/python'
CLI=ROOT/'scripts/train_ppo_m2.py'
PROTOCOL=ROOT/'docs/jev_rl_m2_protocol.md'
TRAIN_CAP=60_000_000
VALIDATION_CAP=2_000_000
WALL_CAP=3600.
RECIPES=((128,.0003,2),(256,.0003,2),(512,.0003,2),(1024,.0001,4))


def base_job(name,seed,kind='recurrent',hidden=64):
    return {'name':name,'seed':seed,'kind':kind,'hidden':hidden,'backend':'cpu','threads':1,
            'batch':256,'selector_interactions':1_000_000,'joint_interactions':1_000_000,
            'joint_hard_interactions':1_250_000,'wall_snapshot_seconds':240.,'hard_wall_seconds':900.,
            'eval_every_interactions':100_000,'joint_lr':.0003,'joint_epochs':2,
            'entropy_start':.03,'entropy_end':.001,'initialization':'scratch'}


def architecture_jobs():
    jobs=[]
    for ascending,descending in zip((64,128,256,512),(512,256,128,64)):
        for kind,width,seed in [('history',ascending,20),('recurrent',descending,21),
                                ('recurrent',ascending,20),('history',descending,21)]:
            jobs.append(base_job(f'arch-{kind}-h{width}',seed,kind,width))
    return jobs


def primary_eligible(row):
    return bool(row['status'] in ('completed','incomplete_budget') and row.get('correct') and row.get('best')
                and row.get('primary_complete',row['status']=='completed'))


def select_group(rows,seeds=(20,21)):
    scores=[]
    for name in sorted({r['name'] for r in rows}):
        group=[r for r in rows if r['name']==name]
        if (len(group)!=len(seeds) or {r['seed'] for r in group}!=set(seeds)
            or any(not primary_eligible(r) for r in group)):
            continue
        n=len(group)
        scores.append((-sum(r['best']['mean_return'] for r in group)/n,
                       sum(r['best']['mean_mse'] for r in group)/n,
                       sum(r['best']['mean_variable_cost'] for r in group)/n,name))
    if not scores:raise ValueError('no complete correct cohort with every required seed')
    return min(scores)[-1]


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)


def read_complete_json(path):
    try:
        value=json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):
        return None
    return value if isinstance(value,dict) else None


class Race:
    def __init__(self,output):
        self.output=Path(output).expanduser().resolve()
        self.output.mkdir(parents=True,exist_ok=False)
        (self.output/'inputs').mkdir();(self.output/'jobs').mkdir()
        self.start=time.perf_counter();self.lock=threading.Lock();self.rows=[];self.groups={}
        self.reserved_train=self.reserved_validation=0
        self.charged_train=self.charged_validation=0
        self.environment={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
        self.source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__).resolve(),ROOT/'integrations/ppo_m2_train.py',CLI,PROTOCOL)}
        write(self.output/'plan.json',{'protocol_sha256':self.source_hashes[str(PROTOCOL.relative_to(ROOT))],
            'sources':self.source_hashes,'wall_cap_seconds':WALL_CAP,'training_transition_cap':TRAIN_CAP,
            'validation_transition_cap':VALIDATION_CAP,'maximum_concurrency':4,
            'architecture_jobs':architecture_jobs(),'recipes':RECIPES,'all_seeds':[20,21,22,23,24],
            'selection':'all screen seeds retained; mean original tuning return, MSE, cost, identifier',
            'future_confirmation_and_final_opened':False,'hosted_calls':0})

    def save_ledger(self):
        write(self.output/'ledger.json',{'rows':self.rows,'groups':self.groups,
            'elapsed_seconds':time.perf_counter()-self.start,
            'charged_training_upper_bound':self.charged_train,'reserved_training':self.reserved_train,
            'charged_validation_upper_bound':self.charged_validation,'reserved_validation':self.reserved_validation,
            'training_transition_cap':TRAIN_CAP,'validation_transition_cap':VALIDATION_CAP,
            'wall_cap_seconds':WALL_CAP})

    def job(self,config):
        config=dict(config);job_id=f"{config['name']}-seed{config['seed']}"
        reserve=config['selector_interactions']+config['joint_hard_interactions']
        val_reserve=((reserve+config['eval_every_interactions']-1)//config['eval_every_interactions']+8)*128*5+80
        with self.lock:
            remaining=WALL_CAP-(time.perf_counter()-self.start)
            if (remaining<30 or self.charged_train+self.reserved_train+reserve>TRAIN_CAP
                or self.charged_validation+self.reserved_validation+val_reserve>VALIDATION_CAP):
                row={'job_id':job_id,'name':config['name'],'seed':config['seed'],'config':config,
                     'status':'not_started_budget','correct':False,'best':None}
                self.rows.append(row);self.save_ledger();return row
            config['hard_wall_seconds']=min(config['hard_wall_seconds'],max(1.,remaining-15))
            self.reserved_train+=reserve;self.reserved_validation+=val_reserve
            config_path=self.output/'inputs'/(job_id+'.json')
            if config_path.exists():raise FileExistsError(job_id)
            write(config_path,config);self.save_ledger()
        destination=self.output/'jobs'/job_id;tick=time.perf_counter();timed_out=False
        log_path=self.output/(job_id+'.log')
        with log_path.open('x') as log:
            process=subprocess.Popen([str(PYTHON),str(CLI),'--config',str(config_path),'--output',str(destination)],
                                     cwd=ROOT,env=self.environment,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if time.perf_counter()-self.start>=WALL_CAP or time.perf_counter()-tick>config['hard_wall_seconds']+12:
                    timed_out=True;process.terminate();break
                time.sleep(.3)
            try:exit_code=process.wait(timeout=3)
            except subprocess.TimeoutExpired:process.kill();exit_code=process.wait()
        path=destination/'result.json';failure=destination/'failure.json'
        result=read_complete_json(path)
        known_failure=read_complete_json(failure)
        if result is not None and not {'totals','correctness','best_primary','status'}<=result.keys():
            result=None
        best=None;checkpoint=None;correct=False;primary_complete=False;wall_complete=False
        if result is not None:
            actual_train=result['totals']['training_transitions'];actual_val=result['totals']['validation_transitions']
            charged_train,charged_val=actual_train,actual_val
            proof=result['correctness'];native=proof.get('cache_reference') or {}
            correct=bool(exit_code==0 and proof['source_hashes_unchanged'] and proof['initial_checkpoint_unchanged']
                         and native.get('passed') and proof['max_initial_logp_error']<=proof['initial_logp_tolerance'])
            if result['best_primary'] is not None:
                metrics=result['best_primary']['metrics'];checkpoint=result['best_primary']['checkpoint']
                best={key:metrics[key] for key in ('mean_return','mean_mse','mean_variable_cost')}
            status=result['status'] if exit_code==0 else 'failed_after_result'
            primary_complete=result['snapshots']['equal_interaction'] is not None
            wall_complete=result['snapshots']['equal_wall'] is not None or config['wall_snapshot_seconds']==0
        else:
            # A terminated collector/update may have unlogged work. Charge its
            # entire reservation conservatively; retain the observed lower bound.
            counts={} if known_failure is None else known_failure['counts']
            actual_train=counts.get('training_transitions',0);actual_val=counts.get('validation_transitions',0)
            charged_train,charged_val=reserve,val_reserve
            status='terminated_budget' if timed_out else 'failed'
        row={'job_id':job_id,'name':config['name'],'seed':config['seed'],'config':config,'status':status,
             'exit_code':exit_code,'correct':correct,'best':best,'checkpoint':checkpoint,
             'primary_complete':primary_complete,'wall_complete':wall_complete,
             'result_path':str(path) if path.exists() else None,'log_path':str(log_path),
             'wall_seconds':time.perf_counter()-tick,'observed_training_transitions':actual_train,
             'observed_validation_transitions':actual_val,'charged_training_upper_bound':charged_train,
             'charged_validation_upper_bound':charged_val}
        with self.lock:
            self.reserved_train-=reserve;self.reserved_validation-=val_reserve
            self.charged_train+=charged_train;self.charged_validation+=charged_val
            self.rows.append(row);self.save_ledger()
        print(json.dumps({k:row[k] for k in ('job_id','status','best','wall_seconds','observed_training_transitions')},allow_nan=False),flush=True)
        return row

    def run_jobs(self,name,jobs):
        with ThreadPoolExecutor(max_workers=4) as executor:rows=list(executor.map(self.job,jobs))
        with self.lock:self.groups[name]=[r['job_id'] for r in rows];self.save_ledger()
        return rows

    def initial_record(self,row):
        if not row.get('correct') or not row.get('checkpoint'):raise ValueError('invalid parent checkpoint')
        path=self.output/'inputs'/(row['job_id']+'-initial.json')
        if not path.exists():write(path,row['checkpoint'])
        return str(path)

    def continuation(self,name,seed,base,recipe,initialization='scratch',parent=None):
        batch,lr,epochs=recipe
        config=base_job(name,seed,base['kind'],base['hidden'])
        config.update(initialization=initialization,selector_interactions=0,joint_interactions=500_000,
                      joint_hard_interactions=750_000,wall_snapshot_seconds=70.,hard_wall_seconds=360.,
                      batch=batch,joint_lr=lr,joint_epochs=epochs,entropy_start=.01,entropy_end=.001)
        if parent is not None:config['initial_checkpoint_record']=self.initial_record(parent)
        return config

    def run(self):
        architecture=self.run_jobs('architecture',architecture_jobs());selection={}
        try:
            name=select_group(architecture);parents=[r for r in architecture if r['name']==name]
            base=parents[0]['config'];selection['architecture']=name
            jobs=[self.continuation(f'update-b{b}-lr{lr:g}-e{e}',p['seed'],base,(b,lr,e),parent=p)
                  for b,lr,e in RECIPES for p in parents]
            updates=self.run_jobs('updates',jobs);update_name=select_group(updates)
            winners=[r for r in updates if r['name']==update_name];selection['update_recipe']=update_name
            recipe=tuple(winners[0]['config'][k] for k in ('batch','joint_lr','joint_epochs'))
            extra=self.run_jobs('scratch_extra_base',[base_job('extra-'+name,s,base['kind'],base['hidden']) for s in (22,23,24)])
            extra_jobs=[self.continuation('extra-'+update_name,p['seed'],base,recipe,parent=p)
                        for p in extra if primary_eligible(p) and p.get('checkpoint')]
            finished=self.run_jobs('scratch_extra_updates',extra_jobs)
            selection['scratch_selected_records']={str(r['seed']):r['checkpoint'] for r in winners+finished
                                                    if primary_eligible(r)}
            selection['scratch_all_five']=set(selection['scratch_selected_records'])=={'20','21','22','23','24'}
            selection['scratch_wall_all_five']=len(winners+finished)==5 and all(r['wall_complete'] for r in winners+finished)
        except ValueError as error:
            selection['scratch_not_selected']=str(error)
        warm_jobs=[self.continuation(f'v6-lr{lr:g}-e{epochs}',seed,{'kind':'recurrent','hidden':64},(512,lr,epochs),'v6')
                   for lr,epochs in ((.0001,2),(.0003,4)) for seed in (20,21)]
        warm=self.run_jobs('v6_screen',warm_jobs)
        try:
            warm_name=select_group(warm);winning=[r for r in warm if r['name']==warm_name]
            config=winning[0]['config'];recipe=tuple(config[k] for k in ('batch','joint_lr','joint_epochs'))
            additional=self.run_jobs('v6_extra',[self.continuation(warm_name,s,config,recipe,'v6') for s in (22,23,24)])
            selection['v6_recipe']=warm_name
            selection['v6_selected_records']={str(r['seed']):r['checkpoint'] for r in winning+additional
                                               if primary_eligible(r)}
            selection['v6_all_five']=set(selection['v6_selected_records'])=={'20','21','22','23','24'}
        except ValueError as error:selection['v6_not_selected']=str(error)
        imitation=[]
        for seed in range(20,25):
            config=self.continuation('historical-imitation-ppo-continuation',seed,{'kind':'recurrent','hidden':64},(512,.0001,2),'imitation_ppo')
            config.update(joint_hard_interactions=500_000,wall_snapshot_seconds=0.,hard_wall_seconds=300.,entropy_start=.001)
            imitation.append(config)
        im_rows=self.run_jobs('imitation_ppo_continuation',imitation)
        selection['imitation_ppo_records']={str(r['seed']):r['checkpoint'] for r in im_rows
                                             if primary_eligible(r)}
        for name,sha in self.source_hashes.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('race source/protocol changed')
        selection.update(elapsed_seconds=time.perf_counter()-self.start,
                         charged_training_upper_bound=self.charged_train,
                         charged_validation_upper_bound=self.charged_validation,
                         all_observed_training_transitions=sum(r.get('observed_training_transitions',0) for r in self.rows),
                         all_observed_validation_transitions=sum(r.get('observed_validation_transitions',0) for r in self.rows),
                         completed_jobs=sum(r['status']=='completed' for r in self.rows),total_jobs=len(self.rows),
                         confirmation_opened=False,final_opened=False,hosted_calls=0,
                         scope='development selection only; comparisons require frozen confirmation')
        write(self.output/'selection.json',selection)
        print(json.dumps(selection,indent=2),flush=True)
        return selection


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    Race(args.output).run()


if __name__=='__main__':main()
