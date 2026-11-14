"""Freeze, run and replay bounded OFFLINE M2 development confirmation.

prepare reads completed tuning receipts only. run is the only entry point which
constructs the registered confirmation pool. There is deliberately no final mode.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]

import numpy as np
import torch
from experimentlab.cost_policy import PublicCostPolicy
from experimentlab.envs import PilotConfig, ScientistEnv
from experimentlab.learning import ActorCritic
from experimentlab.ppo_v4 import collect_batch
from experimentlab.tasks import continuous_family, make_task, mechanism_hash, PrivateTask
from experimentlab.worlds import Mechanism, integrate, equilibrium
from research_stage import UnfilteredBank, quadrature_family
from scripts.jev_rl_study import model_records, previous_hashes
from scripts.run_ppo_m2_race import primary_eligible

SEEDS = (20, 21, 22, 23, 24)
SCHEMA = 'm2-ppo-confirmation-v1'
EVALUATION = {'generator_seed':991321, 'count':128, 'episode_seed':991322000}
LIMITS = {'max_episodes':10_000, 'max_step_calls':50_000, 'wall_seconds':600.}
BASELINES = ('prior', 'fixed', 'fixed_quality', 'additive', 'random', 'bo', 'eig', 'voi')
ORIGINALS = ('ppo', 'imitation', 'calibrated_imitation', 'imitation_ppo')
EXTRAS = {'scratch_selected':'scratch_selected_records', 'v6_continued':'v6_selected_records',
          'imitation_ppo_continued':'imitation_ppo_records'}
PRIMARY = (('scratch_selected','ppo'), ('v6_continued','ppo'), ('imitation_ppo_continued','imitation_ppo'))
METRICS = ('final_mse', 'variable_cost', 'total_assay_cost', 'return_', 'effort', 'assays', 'immediate_stop')
BOOTSTRAP = {'draws':10000,'rng_seed':991499,'resampling':'paired independent seed and mechanism indices',
             'descriptive_level':.95,'primary_level':1-.05/3,'families':3,'mse_margin':.0001}


def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    # Unmeasured entries of public readouts are NaN, serialized explicitly null.
    if isinstance(value, float) and math.isnan(value): return None
    return value


def canonical(value):
    return json.dumps(clean(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def resolve_file(root, value):
    root = Path(root).resolve()
    path = (root/value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('artifact must be an existing repository file: '+str(value))
    return path


def verify_hashes(root, hashes):
    for name, sha in hashes.items():
        if file_hash(resolve_file(root, name)) != sha:
            raise ValueError('source or artifact hash mismatch: '+str(name))


def source_hashes(root):
    """Runtime sources and public protocol only; never snapshot the original spec."""
    files = set((root/'src').rglob('*.py'))
    files.update((root/'integrations').glob('*.py'))
    files.update(root/name for name in ('scripts/ppo_m2_confirm.py','scripts/jev_rl_study.py',
        'scripts/run_ppo_m2_race.py','scripts/research_stage.py','scripts/ppo_followup.py',
        'scripts/train_ppo_m2.py','requirements.lock','pyproject.toml',
        'docs/jev_rl_protocol.md','docs/jev_rl_m2_protocol.md'))
    return {str(p.relative_to(root)):file_hash(p) for p in sorted(files) if p.is_file()}


def prior_metadata(root):
    identifiers, hashes = previous_hashes(root)
    # M1/M2 manifest metadata only: no historical outcomes or learning curves.
    paths = []
    for folder in ('jev-rl-m1','jev-rl-m2'):
        paths.extend((root/'reports'/folder).rglob('*manifest*.json'))
        paths.extend((root/'reports'/folder).rglob('*manifest*.json.gz'))
    def visit(value):
        if isinstance(value, dict):
            for item in value.values(): visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
        elif isinstance(value, str) and re.fullmatch('[0-9a-f]{64}',value): identifiers.add(value)
    for path in sorted(set(paths)):
        visit(read_gzip(path) if path.suffix=='.gz' else json.loads(path.read_text()))
        hashes[str(path.relative_to(root))] = file_hash(path)
    return identifiers, hashes


def normalize_record(root, record, seed):
    record = dict(record)
    path = resolve_file(root, record['path'])
    if file_hash(path) != record['sha256']:
        raise ValueError('checkpoint hash mismatch: '+str(path))
    if 'seed' in record and record['seed'] != seed: raise ValueError('checkpoint training seed mismatch')
    calibration = record['calibration']
    if set(calibration) != {'duration_penalty','dose_penalty','stop_bonus'} or any(
            type(x) not in (int,float) or not math.isfinite(x) or x < 0 for x in calibration.values()):
        raise ValueError('invalid frozen public cost calibration')
    record.update(path=str(path.relative_to(root)), seed=seed)
    return record


def correct_result(result):
    proof = result.get('correctness', {})
    return (proof.get('source_hashes_unchanged') is True and proof.get('initial_checkpoint_unchanged') is True
        and (proof.get('cache_reference') or {}).get('passed') is True
        and 0 <= proof.get('max_initial_logp_error', math.inf) <= proof.get('initial_logp_tolerance', -1))


def tuning_score(metrics):
    values = [float(metrics[k]) for k in ('mean_return','mean_mse','mean_variable_cost')]
    if not all(math.isfinite(x) for x in values): raise ValueError('nonfinite tuning score')
    return dict(zip(('mean_return','mean_mse','mean_variable_cost'), values))


def aggregate_tuning(scores):
    if set(scores) != set(map(str,SEEDS)): return None
    return {key:float(np.mean([v[key] for v in scores.values()])) for key in next(iter(scores.values()))}


def prepare(race, output, *, root=ROOT):
    """Seal all models and tuning decisions; do not generate or score any tasks."""
    root, race, output = Path(root).resolve(), Path(race).resolve(), Path(output).resolve()
    selected_path = output.parent/'selected_ppo_records.json'
    if output.exists() or selected_path.exists(): raise FileExistsError('refusing to overwrite freeze or selected records')
    try:
        selection, ledger, plan = [json.loads((race/name).read_text()) for name in ('selection.json','ledger.json','plan.json')]
    except (OSError,json.JSONDecodeError) as error:
        raise ValueError('requires completed race selection, ledger and plan') from error
    if (ledger.get('reserved_training') != 0 or ledger.get('reserved_validation') != 0
            or selection.get('confirmation_opened') is not False or selection.get('final_opened') is not False
            or selection.get('hosted_calls') != 0 or plan.get('all_seeds') != list(SEEDS)):
        raise ValueError('requires completed race with unopened confirmation/final and all five seeds')
    verify_hashes(root, plan['sources'])
    evidence = {str(resolve_file(root, str(race/name)).relative_to(root)):file_hash(race/name)
                for name in ('selection.json','ledger.json','plan.json')}
    originals = model_records(root,SEEDS)
    cohorts, models = {}, {}
    for method in ORIGINALS:
        records = {str(s):normalize_record(root, originals[f'{method}-seed{s}'], s) for s in SEEDS}
        cohorts[method] = {'status':'ready','records':records,'required_seeds':list(SEEDS),'missing_seeds':[],
                           'source':'fixed_v6_selection_registry','tuning':None}
        models.update({f'{method}-seed{s}':r for s,r in records.items()})
    results = {}
    for row in ledger['rows']:
        if row.get('result_path'):
            path = resolve_file(root,row['result_path'])
            # These are completed training aggregate tuning receipts, never per-case confirmation outcomes.
            results[row['job_id']] = json.loads(path.read_text())
            evidence[str(path.relative_to(root))] = file_hash(path)
    for method, key in EXTRAS.items():
        initialization={'scratch_selected':'scratch','v6_continued':'v6',
                        'imitation_ppo_continued':'imitation_ppo'}[method]
        selected = selection.get(key,{})
        if not isinstance(selected,dict) or set(selected)-set(map(str,SEEDS)):
            raise ValueError('invalid selected cohort seed mapping')
        records, scores, invalid = {}, {}, []
        for seed in SEEDS:
            if str(seed) not in selected: continue
            record = normalize_record(root,selected[str(seed)],seed)
            records[str(seed)] = record
            matching = [r for r in ledger['rows'] if r['seed']==seed and r.get('checkpoint')
                        and r['checkpoint']['sha256']==record['sha256']
                        and resolve_file(root,r['checkpoint']['path'])==resolve_file(root,record['path'])]
            valid = []
            for row in matching:
                result = results.get(row['job_id'],{})
                best = result.get('best_primary') or {}
                if (primary_eligible(row) and correct_result(result) and result.get('seed')==seed
                        and row.get('config',{}).get('initialization')==initialization
                        and result.get('config',{}).get('initialization')==initialization
                        and result.get('snapshots',{}).get('equal_interaction') is not None
                        and best.get('selection_eligible') is True and best.get('training_biases_zero') is True
                        and best.get('checkpoint',{}).get('sha256')==record['sha256']):
                    valid.append((row,result,best))
            if len(valid)!=1:
                invalid.append(seed)
            else:
                row,result,best = valid[0]
                scores[str(seed)] = tuning_score(best['metrics'])
                record['m2_provenance'] = {'job_id':row['job_id'], 'result_path':str(resolve_file(root,row['result_path']).relative_to(root)),
                    'result_sha256':file_hash(resolve_file(root,row['result_path'])),
                    'config':result['config'],'inherited_pretraining':result.get('inherited_pretraining')}
        missing = [s for s in SEEDS if str(s) not in records]
        ready = not missing and not invalid
        cohorts[method] = {'status':'ready' if ready else 'blocked','records':records,
            'required_seeds':list(SEEDS),'missing_seeds':missing,'invalid_seeds':invalid,
            'tuning':aggregate_tuning(scores) if ready else None,
            'tuning_by_seed':scores,'source':'completed_m2_primary_selection'}
        if ready: models.update({f'{method}-seed{s}':r for s,r in records.items()})
    incumbent_scores = {}
    for row in ledger['rows']:
        result = results.get(row['job_id'],{})
        if row.get('config',{}).get('initialization')!='v6' or not row.get('correct') or not correct_result(result): continue
        for event in result.get('evaluations',[]):
            if event.get('reason')!='initial_incumbent' or event.get('selection_eligible') is not True: continue
            seed = str(row['seed'])
            score = tuning_score(event['metrics'])
            if seed in incumbent_scores and score != incumbent_scores[seed]:
                raise ValueError('inconsistent original v6 tuning receipts')
            incumbent_scores[seed] = score
    cohorts['ppo']['tuning_by_seed'] = incumbent_scores
    cohorts['ppo']['tuning'] = aggregate_tuning(incumbent_scores)
    cohorts['ppo']['missing_tuning_seeds'] = [s for s in SEEDS if str(s) not in incumbent_scores]
    cohorts['ppo']['selection_candidate_status'] = 'ready' if cohorts['ppo']['tuning'] else 'blocked_missing_tuning_seeds'
    candidates = []
    for method in ('scratch_selected','v6_continued','ppo'):
        score = cohorts[method]['tuning']
        if cohorts[method]['status']=='ready' and score is not None:
            candidates.append((-score['mean_return'],score['mean_mse'],score['mean_variable_cost'],method))
    if not candidates: raise ValueError('no complete five-seed pure PPO tuning candidate; selection blocked')
    selected = min(candidates)[-1]
    old_freeze_path = root/'reports/ppo-efficiency-v6/fresh-locked/freeze.json'
    old_freeze = json.loads(old_freeze_path.read_text())
    config = asdict(PilotConfig(final_cultures=128))
    if old_freeze['config'] != config: raise ValueError('original simulator configuration changed')
    identifiers, metadata_hashes = prior_metadata(root)
    fixed_inputs = {'reports/ppo-efficiency-v6/fresh-locked/freeze.json':file_hash(old_freeze_path),
                    'reports/ppo-efficiency-v6/selection.json':file_hash(root/'reports/ppo-efficiency-v6/selection.json')}
    freeze = {'schema':SCHEMA,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'development confirmation; selection-affected, not a new final test',
        'evaluation':dict(EVALUATION),'training_seeds':list(SEEDS),'config':config,'limits':dict(LIMITS),
        'cohorts':cohorts,'models':models,'baselines':list(BASELINES)+['reference'],
        'reference_scope':'privileged nominal mechanism reference with zero assay cost; not an attainable public controller',
        'schedules':{'fixed':old_freeze['fixed'],'fixed_quality':old_freeze['fixed_quality'],'additive':old_freeze['additive_schedule']},
        'planner_draws':12,'planner_seed':991323000,'belief_bank':'unfiltered quadrature_family(order=2)',
        'selected_ppo':{'cohort':selected,'kind':'registered_v6' if selected=='ppo' else 'm2',
            'selection_basis':'all-five original tuning return, lower MSE, lower variable cost, identifier; no confirmation',
            'tuning':cohorts[selected]['tuning']},
        'race_inputs':evidence,'race_ledger':ledger,'race_selection':selection,'race_plan':plan,
        'original_registry_inputs':fixed_inputs,'historical_manifest_hashes':metadata_hashes,
        'historical_identifiers':sorted(identifiers),'sources':source_hashes(root),
        'bootstrap':dict(BOOTSTRAP),
        'selected_ppo_records_content_sha256':digest(cohorts[selected]['records']),
        'primary_comparisons':[list(p) for p in PRIMARY],
        'final_test_opened':False,'confirmation_constructed':False,'hosted_calls':0}
    spec = root/'autonomous_scientist_build_spec.md'
    if spec.is_file(): freeze['original_spec_sha256'] = file_hash(spec)
    freeze['seal_sha256'] = digest(freeze)
    write_new(output,freeze)
    write_new(selected_path,cohorts[selected]['records'])
    return freeze


def bootstrap_difference(a, b, *, draws=10000, seed=991499, budget=None):
    """Crossed two-way bootstrap: preserve each method pair at every sampled cell."""
    a,b = np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=2 or not a.size or not np.isfinite(a-b).all():
        raise ValueError('paired finite seed by mechanism matrices required')
    rng=np.random.default_rng(seed)
    samples=np.empty(draws)
    delta=a-b
    for start in range(0,draws,256):
        if budget is not None: budget.check()
        count=min(256,draws-start)
        si=rng.integers(delta.shape[0],size=(count,delta.shape[0]))
        mi=rng.integers(delta.shape[1],size=(count,delta.shape[1]))
        samples[start:start+count]=delta[si[:,:,None],mi[:,None,:]].mean(axis=(1,2))
    return {'difference':float(delta.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'ci98_333333':np.quantile(samples,[.05/6,1-.05/6]).tolist()}


def summarize_rows(rows, expected, count, *, comparisons=None, draws=10000, budget=None):
    """No incomplete or selectively retained cohort enters an inferential summary."""
    index = {}
    for row in rows:
        key=(row['method'],row['training_seed'],row['task_index'])
        if key in index: raise ValueError('duplicate episode row')
        if row['method'] not in expected or row['training_seed'] not in expected[row['method']] or not 0<=row['task_index']<count:
            raise ValueError('row outside frozen design')
        index[key]=row
    cohorts={}
    for method,seeds in expected.items():
        cells=[index.get((method,s,i)) for s in seeds for i in range(count)]
        present=[r for r in cells if r is not None]
        complete=[r for r in present if r['status']=='completed']
        summary={'status':'complete' if len(complete)==len(cells) else 'incomplete',
                 'expected_count':len(cells),'attempted_count':len(present),'completed_count':len(complete),
                 'failed_count':len(present)-len(complete),'missing_count':len(cells)-len(present),
                 'all_attempt_variable_cost':sum(r['metrics']['variable_cost'] for r in present),
                 'all_attempt_latency_seconds':sum(r['metrics']['latency_seconds'] for r in present)}
        if summary['status']=='complete':
            summary.update({f'mean_{k}':float(np.mean([r['metrics'][k] for r in complete])) for k in METRICS})
            summary['aggregate_rmse']=math.sqrt(summary['mean_final_mse'])
        cohorts[method]=summary
    pairs={}
    comparisons=PRIMARY if comparisons is None else comparisons
    for left,right in comparisons:
        name=f'{left}-minus-{right}'
        if any(method not in cohorts or cohorts[method]['status']!='complete' for method in (left,right)):
            pairs[name]={'status':'blocked','reason':'requires every declared seed and mechanism in both cohorts'}
            continue
        ls,rs=expected[left],expected[right]
        if ls!=rs and ls!=[None] and rs!=[None]: raise ValueError('training seeds are not paired')
        seeds=rs if ls==[None] else ls
        def matrix(method, metric):
            return np.array([[index[(method,None if expected[method]==[None] else s,i)]['metrics'][metric]
                              for i in range(count)] for s in seeds])
        metrics={k:bootstrap_difference(matrix(left,k),matrix(right,k),draws=draws,budget=budget) for k in METRICS}
        tiled=ls!=rs and (ls==[None] or rs==[None])
        pairs[name]={'status':'complete','direction':f'{left} minus {right}','metrics':metrics,
            'paired_cells':len(seeds)*count,'baseline_tiled_across_training_seeds':tiled,
            'independent_baseline_replicates':1 if tiled else None,
            'scientific_return_gain':metrics['return_']['ci98_333333'][0]>0 and metrics['final_mse']['ci98_333333'][1]<=.0001,
            'inference_scope':'development confirmation; descriptive, selection-affected; conservative intersection-union gate'}
    return {'cohorts':cohorts,'comparisons':pairs,'raw_row_count':len(rows),
            'failures':dict(Counter(r.get('failure') or 'unspecified' for r in rows if r['status']!='completed')),
            'bootstrap_draws':draws,'bootstrap_rng_seed':991499}


class LimitReached(RuntimeError):
    pass


class Budget:
    def __init__(self, max_episodes=10000, max_step_calls=50000, wall_seconds=600.):
        self.max_episodes,self.max_step_calls,self.wall_seconds=max_episodes,max_step_calls,wall_seconds
        self.deadline_seconds=wall_seconds
        self.episodes=self.step_calls=0
        self.started=time.perf_counter()

    def check(self):
        if time.perf_counter()-self.started>=self.deadline_seconds: raise LimitReached('CPU wall cap')

    def episode(self):
        self.check()
        if self.episodes>=self.max_episodes: raise LimitReached('episode cap')
        self.episodes+=1

    def step(self):
        self.check()
        if self.step_calls>=self.max_step_calls: raise LimitReached('step-call cap')
        self.step_calls+=1


class RecordedEnv(ScientistEnv):
    """Original uncached simulator with bounded action/reward receipts only."""
    def __init__(self,task,config,budget):
        super().__init__(task,config)
        self.account=budget
        self.started=False
        self.reset_error=None

    def reset(self):
        self.account.episode()
        try:
            obs,info=super().reset()
        except Exception as error:
            self.reset_error=type(error).__name__+': '+str(error)
            raise
        self.recorded_actions=[];self.recorded_rewards=[];self.recorded_steps=[]
        self.step_errors=[]
        self.started=True
        return obs,info

    def step(self,action):
        self.account.step()
        try:
            obs,reward,terminated,truncated,info=super().step(action)
        except Exception as error:
            self.step_errors.append({'action':int(action),'after_actions':len(self.recorded_actions),
                                     'error':type(error).__name__+': '+str(error)})
            raise
        self.recorded_actions.append(int(action));self.recorded_rewards.append(float(reward))
        self.recorded_steps.append({'budget_ticks':self.budget_ticks,'assays':self.count,
            'terminated':bool(terminated),'truncated':bool(truncated),'info':info})
        return obs,reward,terminated,truncated,info


def episode_row(env,method,seed,index,*,latency_seconds,failure=None):
    completed=env.private_result is not None and env.done and not env.step_errors
    variable=(env.config.budget*20-env.budget_ticks)/20
    metrics={'final_mse':None,'effort':None,'return_':None,'variable_cost':variable,'setup_cost':3.,
        'total_assay_cost':3.+variable,'assays':env.count,'immediate_stop':completed and env.count==0,
        'observed_return':float(sum(env.recorded_rewards)),'latency_seconds':float(latency_seconds)}
    if completed:
        metrics.update(env.private_result)
        metrics['return_']=float(sum(env.recorded_rewards))
    return clean({'method':method,'training_seed':seed,'task_index':index,
        'status':'completed' if completed else 'failed','failure':None if completed else failure or 'incomplete episode',
        'actions':env.recorded_actions,'rewards':env.recorded_rewards,'steps':env.recorded_steps,'step_errors':env.step_errors,
        'public_record':env.public_record(),'metrics':metrics})


def replay_row(row,task,config):
    env=RecordedEnv(task,config,Budget(max_episodes=1,max_step_calls=config.decision_cap,wall_seconds=600.))
    env.reset()
    errors=list(row.get('step_errors',[]))
    for index in range(len(row['actions'])+1):
        for failed in [x for x in errors if x['after_actions']==index]:
            try: env.step(failed['action'])
            except Exception as error:
                if type(error).__name__+': '+str(error)!=failed['error']: raise ValueError('environment error mismatch') from error
            else: raise ValueError('recorded environment error did not recur')
        if index<len(row['actions']):
            action=row['actions'][index]
            if type(action) is not int: raise ValueError('non-integer recorded action')
            env.step(action)
    replayed=episode_row(env,row['method'],row['training_seed'],row['task_index'],
                         latency_seconds=row['metrics']['latency_seconds'],failure=row.get('failure'))
    for field in ('rewards','steps','step_errors','public_record','metrics','status'):
        if canonical(replayed[field])!=canonical(row[field]): raise ValueError('replayed '+field+' mismatch')
    return {'passed':True,'step_calls':len(row['actions'])+len(errors),'completed':replayed['status']=='completed'}


def load_policy(root,record):
    verify_hashes(root,{record['path']:record['sha256']})
    # The hash must match the frozen repository artifact before deserialization.
    state=torch.load(resolve_file(root,record['path']),map_location='cpu',weights_only=False)
    model=ActorCritic(**state['architecture'])
    model.load_state_dict(state['model'])
    return PublicCostPolicy(model,**record['calibration']).cpu().eval()


def make_bank(config):
    return UnfilteredBank(quadrature_family(2),config)


def run_local(env,method,bank,freeze,index):
    from experimentlab.baselines import DecisionPlanner
    from experimentlab.bo import DiscreteBO
    from integrations.jev_research import belief_from_public, local_choice
    obs,_=env.reset()
    planner=DecisionPlanner(bank,method,freeze['planner_draws']) if method in ('voi','eig') else None
    if method=='bo': planner=DiscreteBO(bank,belief_from_public(obs,bank))
    rng=np.random.default_rng(freeze['planner_seed']+index)
    while not env.done:
        env.account.check()
        action=local_choice(method,obs,bank,freeze['schedules'].get(method,()),planner,rng)
        obs,_,_,_,_=env.step(int(action))


def run_reference(env,task):
    env.reset()
    us=np.array([a.intervention for a in env.actions[36:]])
    y=integrate(task.mechanism,np.broadcast_to(equilibrium(task.mechanism),(9,8)),us,6.,env.config.dt)[:,6:8]
    score=100*np.mean((y-task.target)**2,axis=1)+env.config.effort_weight*us.sum(axis=1)
    env.step(36+int(np.argmin(score)))


def evaluate_cases(freeze,tasks,budget,*,root=ROOT,sink=None):
    """Score declared cases; this helper never constructs assessment mechanisms."""
    config=PilotConfig(**freeze['config'])
    rows,failures=[],[]
    def record(env,method,seed,index,elapsed,error=None,scope='single episode decision and simulator wall seconds'):
        if env.reset_error:
            failures.append({'method':method,'training_seed':seed,'task_index':index,
                             'reset_failed':True,'error':env.reset_error})
        if not env.started: return
        row=episode_row(env,method,seed,index,latency_seconds=elapsed,failure=error)
        row['latency_scope']=scope
        rows.append(row)
        if sink is not None: sink(row)
    for method,cohort in freeze['cohorts'].items():
        if cohort['status']!='ready': continue
        for seed,model_record in cohort['records'].items():
            envs=[];started=time.perf_counter();error=None
            try:
                budget.check()
                policy=load_policy(root,model_record)
                envs=[RecordedEnv(task,config,budget) for task in tasks]
                collect_batch(policy,envs,'cpu',deterministic=True)
            except Exception as exc:
                error=type(exc).__name__+': '+str(exc)
                failures.append({'method':method,'training_seed':int(seed),'error':error})
            finally:
                started_count=sum(env.started for env in envs)
                elapsed=time.perf_counter()-started
                for index,env in enumerate(envs):
                    record(env,method,int(seed),index,elapsed/max(1,started_count),error,
                        'one CPU neural batch wall including checkpoint load and simulator, divided by started episodes; not independent per-episode latency')
            if error and error.startswith('LimitReached:'): return rows,failures
    bank=None
    try:
        budget.check()
        if any(m!='reference' for m in freeze['baselines']): bank=make_bank(config)
    except Exception as exc:
        failures.append({'method':'belief_bank_setup','error':type(exc).__name__+': '+str(exc)})
        return rows,failures
    for method in freeze['baselines']:
        for index,task in enumerate(tasks):
            env=RecordedEnv(task,config,budget);started=time.perf_counter();error=None
            try:
                budget.check()
                if method=='reference': run_reference(env,task)
                else: run_local(env,method,bank,freeze,index)
            except Exception as exc:
                error=type(exc).__name__+': '+str(exc)
                failures.append({'method':method,'task_index':index,'error':error})
            finally:
                record(env,method,None,index,time.perf_counter()-started,error)
            if error and error.startswith('LimitReached:'): return rows,failures
    return rows,failures


def architecture_comparison_limit(freeze):
    capacities=set()
    for record in freeze['cohorts'].get('scratch_selected',{}).get('records',{}).values():
        config=record.get('m2_provenance',{}).get('config',{})
        if config.get('kind') in ('recurrent','history') and 'hidden' in config:
            capacities.add(('GRU' if config['kind']=='recurrent' else 'history MLP')+str(config['hidden']))
    scratch=', '.join(sorted(capacities)) or 'capacity recorded in checkpoint provenance'
    return (f'Scratch cohort: {scratch}; preserved imitation: GRU64. This is a comparison of complete '
            'systems whose network capacities and training histories are not held equal across all cohorts. '
            'It does not isolate PPO algorithmic or training value; matched-capacity imitation and '
            'matched lifetime-compute comparisons remain necessary for that claim.')


def confirmation_summary(rows,freeze,count,*,budget=None,intervals=True):
    expected={method:cohort['required_seeds'] for method,cohort in freeze['cohorts'].items()}
    expected.update({method:[None] for method in freeze['baselines']})
    draws=freeze['bootstrap']['draws']
    result=summarize_rows(rows,expected,count,draws=draws,budget=budget,
                          comparisons=None if intervals else [])
    comparisons=[(method,other) for method in ('scratch_selected','v6_continued','ppo','imitation_ppo_continued')
                 for other in ('imitation','calibrated_imitation','fixed','fixed_quality','voi')]
    descriptive=summarize_rows(rows,expected,count,comparisons=comparisons if intervals else [],
                               draws=draws,budget=budget)['comparisons']
    if not intervals:
        blocked={'status':'blocked','reason':'CPU deadline reached; bootstrap intervals were not computed'}
        result['comparisons']={f'{a}-minus-{b}':dict(blocked) for a,b in PRIMARY}
        descriptive={f'{a}-minus-{b}':dict(blocked) for a,b in comparisons}
    for pair in descriptive.values():
        pair.pop('scientific_return_gain',None)
        pair['inference_scope']='descriptive 95% development comparison; no multiplicity-adjusted gain claim'
        for metric in pair.get('metrics',{}).values(): metric.pop('ci98_333333',None)
    result.update(descriptive_comparisons=descriptive,final_test_opened=False,intervals_computed=intervals,
        limitations=['All outcomes are development confirmation after tuning; intervals are selection-affected.',
            architecture_comparison_limit(freeze),
            'Five training seeds and common mechanisms are crossed resampling units; baseline copies are not independent replicates.',
            'Aggregate RMSE is sqrt(mean MSE), never mean per-case RMSE.',
            'Parameter holdout within familiar motifs does not establish structural or biological generalization.',
            'Warm starts inherit prior compute outside the new-training budget; equal lifetime compute is not claimed.',
            'Latency is scoped CPU runtime, with allocated neural batch latency; no temperature or profiling claim.'])
    return result


def read_freeze(path,*,root=None):
    freeze=json.loads(Path(path).read_text())
    seal=freeze.get('seal_sha256')
    if seal!=digest({k:v for k,v in freeze.items() if k!='seal_sha256'}): raise ValueError('freeze seal mismatch')
    if (freeze.get('schema')!=SCHEMA or freeze.get('evaluation')!=EVALUATION or freeze.get('limits')!=LIMITS
            or freeze.get('config')!=asdict(PilotConfig(final_cultures=128))
            or freeze.get('training_seeds')!=list(SEEDS) or freeze.get('final_test_opened') is not False
            or freeze.get('bootstrap')!=BOOTSTRAP or freeze.get('primary_comparisons')!=[list(p) for p in PRIMARY]):
        raise ValueError('freeze does not match registered confirmation design')
    if set(freeze['cohorts'])!=set(ORIGINALS)|set(EXTRAS) or freeze['baselines']!=list(BASELINES)+['reference']:
        raise ValueError('freeze is missing declared cohorts or baselines')
    models={}
    for method,cohort in freeze['cohorts'].items():
        if cohort['required_seeds']!=list(SEEDS): raise ValueError('cohort does not retain all five seeds')
        if cohort['status']=='ready':
            if set(cohort['records'])!=set(map(str,SEEDS)): raise ValueError('ready cohort lacks a required seed')
            models.update({f'{method}-seed{s}':r for s,r in cohort['records'].items()})
    if models!=freeze['models']: raise ValueError('frozen model index differs from complete cohorts')
    candidates=[]
    for method in ('scratch_selected','v6_continued','ppo'):
        cohort=freeze['cohorts'][method]
        score=cohort.get('tuning')
        if cohort['status']=='ready' and score is not None:
            candidates.append((-score['mean_return'],score['mean_mse'],score['mean_variable_cost'],method))
    if not candidates or freeze['selected_ppo']['cohort']!=min(candidates)[-1]:
        raise ValueError('pure PPO slot differs from all-five tuning selection')
    chosen=freeze['selected_ppo']['cohort']
    if (freeze['selected_ppo']['kind']!=('registered_v6' if chosen=='ppo' else 'm2')
            or digest(freeze['cohorts'][chosen]['records'])!=freeze['selected_ppo_records_content_sha256']):
        raise ValueError('selected PPO records differ from frozen tuning selection')
    if root is not None:
        for group in ('sources','race_inputs','original_registry_inputs','historical_manifest_hashes'):
            verify_hashes(root,freeze[group])
        for cohort in freeze['cohorts'].values():
            for seed,record in cohort['records'].items(): normalize_record(root,record,int(seed))
    return freeze


def write_gzip_new(path,value):
    with gzip.open(path,'xt',encoding='utf-8') as stream: stream.write(canonical(value)+'\n')


def read_gzip(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream: return json.load(stream)


def task_record(task):
    return {'episode_seed':task.seed,'target':clean(task.target),'mechanism_hash':mechanism_hash(task.mechanism),
            'mechanism':{k:clean(getattr(task.mechanism,k)) for k in ('W','b','gamma')}}


def restore_tasks(manifest,config):
    tasks=[]
    for record in manifest['tasks']:
        mechanism=Mechanism(**record['mechanism'])
        if mechanism_hash(mechanism)!=record['mechanism_hash']: raise ValueError('mechanism hash mismatch')
        task=PrivateTask(mechanism,tuple(record['target']),record['episode_seed'])
        if list(make_task(mechanism,task.seed,config).target)!=list(task.target): raise ValueError('task target mismatch')
        tasks.append(task)
    return tasks


def report_text(freeze,summary,result):
    lines=['# Offline M2 development confirmation','',
        'This is development confirmation after tuning. The new final pool remains unopened. No hosted calls were made.','',
        f"Pure PPO slot frozen from all-five tuning: **{freeze['selected_ppo']['cohort']}** ({freeze['selected_ppo']['kind']}). Confirmation does not change this selection.",'',
        '| Cohort | Completed / expected | MSE | Aggregate RMSE | Variable cost | Original return |',
        '|---|---:|---:|---:|---:|---:|']
    for method,row in summary['cohorts'].items():
        values=[f"{row[k]:.7g}" if k in row else 'incomplete' for k in
                ('mean_final_mse','aggregate_rmse','mean_variable_cost','mean_return_')]
        lines.append(f"| {method} | {row['completed_count']} / {row['expected_count']} | "+' | '.join(values)+' |')
    lines+=['','All five training seeds remain required. Incomplete cohorts have no inferential summary. Baselines are evaluated once per mechanism and tiled only for paired resampling; those copies are not independent replicates.',
        '', '## Predeclared signed comparisons','',
        f"Differences are first method minus comparator. The registered paired two-way bootstrap over training seed and mechanism uses {freeze['bootstrap']['draws']:,} draws and RNG 991499. Intervals computed: {summary['intervals_computed']}. Each completed primary comparison reports 95% descriptive and 98.333333% Bonferroni intervals for the three families. A conservative joint gate requires adjusted return lower bound > 0 and adjusted MSE-increase upper bound <= 0.0001.",'']
    for name,pair in summary['comparisons'].items():
        if pair['status']!='complete':
            lines.append(f"- {name}: blocked; {pair['reason']}.")
            continue
        lines.append(f"- {name}: development joint gate {'met' if pair['scientific_return_gain'] else 'not met'}.")
        for metric in ('final_mse','variable_cost','total_assay_cost','return_'):
            value=pair['metrics'][metric]
            lines.append(f"  - {metric}: {value['difference']:.7g}; 95% {value['ci95']}; 98.333333% {value['ci98_333333']}.")
    lines+=['','Descriptive comparisons against both imitation controllers, fixed/fixed-quality designs and VOI are in summary.json. The privileged reference uses the true nominal mechanism and zero assay cost; it is not a public controller.',
        '',f"Recorded episodes: {result['accounting']['episodes']}; step calls: {result['accounting']['step_calls']}; failures: {len(result['errors'])}. The episode and step-call caps are 10,000 and 50,000; CPU work has a 600-second deadline. Partial traces and missing-case counts remain in the raw records.",
        '', 'Raw actions, rewards, public traces and per-case metrics are compressed JSONL. The mechanism manifest and source snapshot support credentials-free and weights-free replay. No binary checkpoint is copied into this report.',
        '', '## Limits','']
    lines += ['- '+x for x in summary['limitations']]
    return '\n'.join(lines)+'\n'


def run(freeze_path,output,*,root=ROOT):
    root,freeze_path,output=Path(root).resolve(),Path(freeze_path).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError('refusing to overwrite confirmation directory')
    freeze=read_freeze(freeze_path,root=root)  # Must precede construction of any confirmation mechanism.
    budget=Budget(**freeze['limits'])
    torch.set_num_threads(1)
    config=PilotConfig(**freeze['config'])
    # The caller has already sealed the models, recipes, sources and selection.
    mechanisms=continuous_family(EVALUATION['generator_seed'],EVALUATION['count'])
    tasks=[make_task(m,EVALUATION['episode_seed']+i,config) for i,m in enumerate(mechanisms)]
    hashes=[mechanism_hash(m) for m in mechanisms]
    forbidden=set(freeze['historical_identifiers'])
    train_hashes={mechanism_hash(m) for m in continuous_family(7401,4096)}
    tuning_hashes={mechanism_hash(m) for m in continuous_family(991311,128)}
    forbidden.update(train_hashes|tuning_hashes)
    if len(set(hashes))!=EVALUATION['count'] or set(hashes)&forbidden:
        raise ValueError('confirmation mechanisms overlap or duplicate prior data')
    budget.check()
    output.mkdir(parents=True)
    shutil.copyfile(freeze_path,output/'freeze.json')
    for name in freeze['sources']:
        target=output/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(resolve_file(root,name),target)
    task_manifest={'schema':SCHEMA,'evaluation':dict(EVALUATION),'tasks':[task_record(t) for t in tasks],
        'disjointness':{'training_seed':7401,'training_count':4096,'training_hashes_sha256':digest(sorted(train_hashes)),
                       'tuning_seed':991311,'tuning_count':128,'tuning_hashes_sha256':digest(sorted(tuning_hashes)),
                       'training_hashes':sorted(train_hashes),'tuning_hashes':sorted(tuning_hashes),
                       'historical_identifiers_checked':len(freeze['historical_identifiers']),'overlap':False},
        'final_test_opened':False}
    write_gzip_new(output/'task-manifest.json.gz',task_manifest)
    manifest={'schema':SCHEMA,'freeze_sha256':file_hash(output/'freeze.json'),
        'task_manifest_sha256':file_hash(output/'task-manifest.json.gz'),'sources':freeze['sources'],
        'software':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__},
        'device':'cpu','torch_threads':1,'simulator':'original uncached ScientistEnv','started_at_utc':datetime.now(timezone.utc).isoformat(),
        'case_count_per_training_seed':EVALUATION['count'],'limits':freeze['limits'],'hosted_calls':0}
    write_new(output/'run-manifest.json',manifest)  # Tasks and hashes are saved before the first score.
    # Reserve finalization time within the original 600-second CPU work cap.
    # Simulator operations finish before a deadline check; any overshoot is measured.
    budget.deadline_seconds=max(0.,budget.wall_seconds-30.)
    with gzip.open(output/'episodes.jsonl.gz','xt',encoding='utf-8') as raw, gzip.open(output/'metrics.jsonl.gz','xt',encoding='utf-8') as metrics:
        def sink(row):
            raw.write(canonical(row)+'\n');raw.flush()
            metrics.write(canonical({k:row[k] for k in ('method','training_seed','task_index','status','failure','metrics','latency_scope')})+'\n');metrics.flush()
        rows,errors=evaluate_cases(freeze,tasks,budget,root=root,sink=sink)
    scoring_elapsed=time.perf_counter()-budget.started
    budget.deadline_seconds=budget.wall_seconds
    try:
        summary=confirmation_summary(rows,freeze,len(tasks),budget=budget)
    except LimitReached as error:
        errors.append({'method':'summary','error':str(error)})
        summary=confirmation_summary(rows,freeze,len(tasks),intervals=False)
    write_new(output/'summary.json',summary)
    result={'schema':SCHEMA,'status':'completed' if all(x['status']=='complete' for x in summary['cohorts'].values()) and not errors else 'partial',
        'run_manifest_sha256':file_hash(output/'run-manifest.json'),
        'artifacts':{name:file_hash(output/name) for name in ('episodes.jsonl.gz','metrics.jsonl.gz','summary.json')},
        'accounting':{'episodes':budget.episodes,'step_calls':budget.step_calls,
            'scoring_wall_seconds':scoring_elapsed,'total_wall_seconds':time.perf_counter()-budget.started,
            'wall_cap_seconds':budget.wall_seconds,'scoring_wall_overrun_seconds':max(0.,scoring_elapsed-budget.wall_seconds),
            'scoring_deadline_seconds':max(0.,budget.wall_seconds-30.),'finalization_reserve_seconds':30.,
            'unrecorded_reset_failures':sum(bool(e.get('reset_failed')) for e in errors),
            'summary_seconds':time.perf_counter()-budget.started-scoring_elapsed},
        'errors':errors,'final_test_opened':False,'hosted_calls':0}
    try:
        for group in ('sources','race_inputs','original_registry_inputs','historical_manifest_hashes'):
            verify_hashes(root,freeze[group])
        for record in freeze['models'].values(): verify_hashes(root,{record['path']:record['sha256']})
        result['source_and_model_hashes_unchanged']=True
    except (OSError,ValueError) as error:
        errors.append({'method':'postrun_integrity','error':type(error).__name__+': '+str(error)})
        result['status']='partial'
        result['source_and_model_hashes_unchanged']=False
    report=report_text(freeze,summary,result)
    with (output/'REPORT.md').open('x') as stream:stream.write(report)
    result['artifacts']['REPORT.md']=file_hash(output/'REPORT.md')
    result['accounting']['total_wall_seconds']=time.perf_counter()-budget.started
    result['accounting']['total_wall_overrun_seconds']=max(0.,result['accounting']['total_wall_seconds']-budget.wall_seconds)
    write_new(output/'results.json',result)
    return {'output':str(output),'status':result['status'],'accounting':result['accounting'],'errors':len(errors)}


def replay(output,*,runtime_root=ROOT):
    """Reexecute every recorded action without models, credentials, or pool generation."""
    output=Path(output).resolve()
    result=json.loads((output/'results.json').read_text())
    manifest=json.loads((output/'run-manifest.json').read_text())
    if file_hash(output/'run-manifest.json')!=result['run_manifest_sha256']: raise ValueError('run manifest hash mismatch')
    verify_hashes(output,result['artifacts'])
    if file_hash(output/'freeze.json')!=manifest['freeze_sha256']: raise ValueError('freeze file hash mismatch')
    freeze=read_freeze(output/'freeze.json')
    verify_hashes(output/'source_snapshot',manifest['sources'])
    # Invoke the snapshot's script if the working checkout has subsequently changed.
    verify_hashes(Path(runtime_root),manifest['sources'])
    if file_hash(output/'task-manifest.json.gz')!=manifest['task_manifest_sha256']: raise ValueError('task manifest hash mismatch')
    task_manifest=read_gzip(output/'task-manifest.json.gz')
    if task_manifest['evaluation']!=EVALUATION: raise ValueError('task generator declaration mismatch')
    config=PilotConfig(**freeze['config'])
    tasks=restore_tasks(task_manifest,config)
    if (len(tasks)!=EVALUATION['count'] or [t.seed for t in tasks]!=
            list(range(EVALUATION['episode_seed'],EVALUATION['episode_seed']+EVALUATION['count']))):
        raise ValueError('task count or episode seeds mismatch')
    hashes=[mechanism_hash(t.mechanism) for t in tasks]
    disjointness=task_manifest['disjointness']
    prior=set(freeze['historical_identifiers'])
    for family in ('training','tuning'):
        values=disjointness[family+'_hashes']
        if digest(sorted(values))!=disjointness[family+'_hashes_sha256']:
            raise ValueError('prior mechanism hash index mismatch')
        prior.update(values)
    if len(set(hashes))!=EVALUATION['count'] or set(hashes)&prior: raise ValueError('task leakage')
    with gzip.open(output/'episodes.jsonl.gz','rt',encoding='utf-8') as stream:
        rows=[json.loads(line) for line in stream]
    with gzip.open(output/'metrics.jsonl.gz','rt',encoding='utf-8') as stream:
        metrics=[json.loads(line) for line in stream]
    expected_metrics=[{k:r[k] for k in ('method','training_seed','task_index','status','failure','metrics','latency_scope')} for r in rows]
    if canonical(metrics)!=canonical(expected_metrics): raise ValueError('raw metric index mismatch')
    verified=[replay_row(row,tasks[row['task_index']],config) for row in rows]
    steps=sum(v['step_calls'] for v in verified)
    reset_failures=sum(bool(e.get('reset_failed')) for e in result['errors'])
    if (result['accounting']['episodes']!=len(rows)+reset_failures or result['accounting']['step_calls']!=steps
            or result['accounting']['unrecorded_reset_failures']!=reset_failures):
        raise ValueError('episode or step-call accounting mismatch')
    if result['accounting']['episodes']>LIMITS['max_episodes'] or steps>LIMITS['max_step_calls']:
        raise ValueError('recorded workload exceeds caps')
    recorded_summary=json.loads((output/'summary.json').read_text())
    summary=confirmation_summary(rows,freeze,len(tasks),intervals=recorded_summary['intervals_computed'])
    if canonical(summary)!=canonical(recorded_summary): raise ValueError('computed summary or intervals mismatch')
    if report_text(freeze,summary,result)!=(output/'REPORT.md').read_text(): raise ValueError('computed report mismatch')
    return {'passed':True,'episodes_replayed':len(verified),'step_calls':steps,
            'summary_verified':True,'intervals_verified':summary['intervals_computed'],'credentials_free':True,'weights_free':True,
            'status':result['status'],'final_test_opened':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prepare_parser=commands.add_parser('prepare');prepare_parser.add_argument('--race',type=Path,required=True)
    prepare_parser.add_argument('--output',type=Path,required=True)
    run_parser=commands.add_parser('run');run_parser.add_argument('--freeze',type=Path,required=True)
    run_parser.add_argument('--output',type=Path,required=True)
    replay_parser=commands.add_parser('replay');replay_parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':
        freeze=prepare(args.race,args.output)
        answer={'freeze':str(args.output),'seal_sha256':freeze['seal_sha256'],'selected_ppo':freeze['selected_ppo'],
                'blocked_cohorts':[k for k,v in freeze['cohorts'].items() if v['status']=='blocked']}
    elif args.command=='run': answer=run(args.freeze,args.output)
    else: answer=replay(args.output)
    print(json.dumps(answer,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
