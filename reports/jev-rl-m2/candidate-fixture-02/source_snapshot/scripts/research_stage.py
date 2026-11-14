"""Additional-compute development stage, with explicit scientific gates.

Run this script with the project's native Python. It never reads locked test
outcomes. All outputs resolve under the selected absolute output directory.
"""
from pathlib import Path
from dataclasses import replace
from itertools import product
import argparse
import hashlib
import json
import time
import numpy as np
import torch

from experimentlab.cli import write_json, code_hash
from experimentlab.envs import PilotConfig, ScientistEnv
from experimentlab.tasks import finite_family, continuous_family, make_task, mechanism_hash
from experimentlab.worlds import matched_mechanism, integrate, equilibrium
from experimentlab.inference import PredictionBank, Belief
from experimentlab.baselines import DecisionPlanner
from experimentlab.evaluation import tune_fixed, run_strategy, reference_result, summarize, paired_interval
from experimentlab.learning import ActorCritic, collect_episode, episode_batch, imitation_update, ppo_update, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class UnfilteredBank(PredictionBank):
    def admitted(self,target,tolerance=.04):
        return np.ones(len(self.mechanisms),dtype=bool)


def unfiltered_tasks(pool,bank,n,seed):
    rng=np.random.default_rng(seed)
    if n>len(pool): raise ValueError('distinct mechanism pool exhausted')
    chosen=rng.permutation(len(pool))[:n]
    return [make_task(pool[int(j)],seed*100000+i,bank.config,.5+rng.choice([-.08,.08],2)) for i,j in enumerate(chosen)]


def quadrature_family(order=2):
    """Complete discrete sign support times independent uniform-parameter quadrature.

    No true episode parameter is inserted. Two-point Gauss-Legendre gives equal
    weights, compatible with the uniform discrete belief support implementation.
    """
    if order not in (1,2): raise ValueError('only equal-weight order1/2 supported')
    nodes = np.polynomial.legendre.leggauss(order)[0]
    gains = .975+.075*nodes
    rates = 1+.1*nodes
    result = []
    for m, gain, rate in product(finite_family(), gains, rates):
        W = m.W.copy(); W[:6] *= gain
        result.append(matched_mechanism(W,m.gamma*rate))
    return result


def tasks_from_pool(pool, bank, n, seed, distinct=True):
    """Public targets first; admission on true nominal dynamics, same as bank prior."""
    rng = np.random.default_rng(seed)
    us = np.array([a.intervention for a in bank.actions[36:]])
    nominal = np.stack([integrate(m,np.broadcast_to(equilibrium(m),(9,8)),us,6,bank.config.dt)[:,6:8] for m in pool])
    used = set(); tasks = []
    for i in range(n):
        target = .5+rng.choice([-.08,.08],2)
        loss = np.mean((nominal-target)**2,axis=-1)
        admitted = np.flatnonzero((loss.min(axis=1)<=.04**2)&(loss.min(axis=1)<loss[:,0]-.0001))
        if distinct: admitted = np.array([j for j in admitted if int(j) not in used])
        if not len(admitted): raise ValueError('declared independent pool exhausted')
        j=int(rng.choice(admitted)); used.add(j)
        tasks.append(make_task(pool[j],seed*100000+i,bank.config,target))
    return tasks


def evaluate_methods(bank,tasks,schedules,draws=12):
    rows={name:[] for name in schedules}; rows['reference']=[]
    for i, task in enumerate(tasks):
        for name,schedule in schedules.items():
            method='fixed' if name.startswith('fixed') else name
            r,_=run_strategy(task,bank,method,schedule,draws,seed=i+440)
            rows[name].append(r)
        rows['reference'].append(reference_result(task,bank))
        if (i+1)%16==0: print(f'assessment {i+1}/{len(tasks)}',flush=True)
    return rows


def assess(output,calibration=128,evaluation=128,unfiltered=False):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    config=PilotConfig(readout_sd=.02 if unfiltered else .08,budget=8)
    protocol={'stage':'unfiltered_prior_development_revision' if unfiltered else 'additional_compute_development','noise':config.readout_sd,'budget':8,'offset_sd':0.,
              'cost_weight':.01,'calibration_mechanisms':calibration,'validation_mechanisms':evaluation,
              'fixed_beam_width':24,'quadrature_particles':128,'draws_per_hypothesis':12,
              'gate_C':'positive mechanism-paired utility and error CI plus >=.10 raw gap recovery against strongest fixed design',
              'locked_test_opened':False,'source_hash':code_hash(),
              'script_hash':hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()}
    protocol['prior']='unfiltered' if unfiltered else 'feasibility_conditioned'
    if unfiltered:
        protocol['revision_rationale']='Feasibility admission makes an unmeasured sibling control inferable when its partner is ineffective. Remove this implicit information by retaining unreachable targets and all mechanisms. Dynamics, costs, actions, baselines, noise and targets stay at the original primary setting.'
    write_json(output/'protocol.json',protocol)
    import shutil
    shutil.copyfile(Path(__file__).resolve(),output/'research_stage.snapshot.py')
    Bank=UnfilteredBank if unfiltered else PredictionBank
    bank=Bank(quadrature_family(2),config)
    train_pool=continuous_family(7401,1024); val_pool=continuous_family(8401,1024)
    sample_tasks=unfiltered_tasks if unfiltered else tasks_from_pool
    calibration_tasks=sample_tasks(train_pool,bank,calibration,2041)
    tasks=sample_tasks(val_pool,bank,evaluation,2042)
    # Complete support versus mid-point quadrature measured before broader use.
    timing=[]
    for order in (1,2):
        candidate=Bank(quadrature_family(order),config)
        start=time.perf_counter(); selected=[]
        for i,t in enumerate(tasks[:8]):
            obs,_=ScientistEnv(t,config).reset(); belief=Belief(candidate,t.target,admitted=True); belief.update(obs['history'][0])
            selected.append(DecisionPlanner(candidate,'voi',12).choose(obs,belief,np.random.default_rng(i)))
        timing.append({'particles':len(candidate.mechanisms),'cache_seconds':candidate.build_seconds,
                       'forward_culture_steps':candidate.forward_steps,'eight_first_decisions_seconds':time.perf_counter()-start,
                       'actions':selected})
    write_json(output/'planning-profile.json',timing)
    print('Tuning fixed utility designs on independent calibration mechanisms...',flush=True)
    schedule,tuning=tune_fixed(bank,calibration_tasks,beam_width=24)
    quality_bank=Bank(quadrature_family(2),replace(config,cost_weight=0.))
    quality_schedule,quality_tuning=tune_fixed(quality_bank,calibration_tasks,beam_width=24)
    print(f'Fixed utility {schedule}; fixed quality {quality_schedule}',flush=True)
    rows=evaluate_methods(bank,tasks,{'prior':(),'fixed':schedule,'fixed_quality':quality_schedule,'voi':()})
    groups=[mechanism_hash(t.mechanism) for t in tasks]
    comparison={}
    for name in ('fixed','fixed_quality'):
        utility=paired_interval([r['return_'] for r in rows['voi']],[r['return_'] for r in rows[name]],groups=groups)
        error=paired_interval([r['final_mse'] for r in rows[name]],[r['final_mse'] for r in rows['voi']],groups=groups)
        prior=np.mean([r['final_mse'] for r in rows['prior']]); reference=np.mean([r['final_mse'] for r in rows['reference']])
        recovery=error['mean']/(prior-reference)
        comparison[name]={'utility':utility,'error':error,'raw_gap_recovery':recovery,
                          'gate_C':bool(utility['ci95'][0]>0 and error['ci95'][0]>0 and recovery>=.1)}
    result={'protocol':protocol,'config':config.__dict__,'tuning':tuning,'quality_tuning':quality_tuning,
            'summary':{k:summarize(v) for k,v in rows.items()},'comparison':comparison,
            'gate_C':all(c['gate_C'] for c in comparison.values()),'rows':rows,
            'median_ESS':float(np.median([r['ess_min'] for r in rows['voi']])),
            'validation_hashes':groups,'calibration_hashes':[mechanism_hash(t.mechanism) for t in calibration_tasks],
            'bank_hashes':[mechanism_hash(m) for m in bank.mechanisms]}
    write_json(output/'assessment.json',result)
    print(json.dumps({'summary':result['summary'],'comparison':comparison,'gate_C':result['gate_C']},indent=2),flush=True)
    return result


def learned_evaluation(model,bank,tasks):
    rows=[]
    for task in tasks:
        env=ScientistEnv(task,bank.config); tick=time.perf_counter()
        episode=collect_episode(model,env,'cpu',deterministic=True)
        row=dict(env.private_result)
        row.update(return_=sum(episode['rewards']),assays=env.count,online_seconds=time.perf_counter()-tick)
        rows.append(row)
    return summarize(rows)


def train(output,assessment,transitions=100000,seeds=(0,1,2),demonstrations=512):
    """Larger runs require the scientific assessment, never a reward-curve gate."""
    if not assessment['gate_C']:
        result={'status':'scientific_gate_not_met','requested_transition_budget_per_seed':transitions,
                'requested_seeds':list(seeds),'reason':'Stronger fixed comparisons do not establish meaningful adaptive advantage; substantial scientific PPO remains gated.'}
        write_json(Path(output).resolve()/'training-decision.json',result)
        return result
    output=Path(output).resolve()
    torch.set_num_threads(1)
    config=PilotConfig(**assessment['config'])
    unfiltered=assessment['protocol'].get('prior')=='unfiltered'
    bank=(UnfilteredBank if unfiltered else PredictionBank)(quadrature_family(2),config)
    training_pool=continuous_family(7401,4096); validation_pool=continuous_family(8401,1024)
    training=unfiltered_tasks(training_pool,bank,4096,3041) if unfiltered else tasks_from_pool(training_pool,bank,4096,3041,distinct=False)
    validation=(unfiltered_tasks if unfiltered else tasks_from_pool)(validation_pool,bank,128,3042)
    write_json(output/'learning-manifest.json',{'train':[mechanism_hash(t.mechanism) for t in training],
                'validation':[mechanism_hash(t.mechanism) for t in validation],'test_opened':False})
    teacher=[]; tick=time.perf_counter(); model=ActorCritic(kind='recurrent',hidden=64)
    for i,task in enumerate(training[:demonstrations]):
        belief=Belief(bank,task.target,admitted=True); planner=DecisionPlanner(bank,'voi',12); seen=[0]; rng=np.random.default_rng(i)
        def choose(obs):
            while seen[0]<len(obs['history']): belief.update(obs['history'][seen[0]]); seen[0]+=1
            return planner.choose(obs,belief,rng)
        teacher.append(collect_episode(model,ScientistEnv(task,config),'cpu',teacher=choose))
        if (i+1)%32==0: print(f'teacher {i+1}/{demonstrations}',flush=True)
    teacher_seconds=time.perf_counter()-tick
    torch.save(teacher,output/'public-history-demonstrations.pt')
    batch=episode_batch(teacher,'cpu')
    results={}
    for seed in seeds:
        for kind,mode in [('history','ppo'),('recurrent','imitation'),('recurrent','ppo'),('recurrent','imitation_ppo')]:
            torch.manual_seed(seed)
            model=ActorCritic(kind=kind,hidden=64); opt=torch.optim.Adam(model.parameters(),lr=3e-4)
            tick=time.perf_counter(); logs=[]; count=0; episodes=0
            if mode.startswith('imitation'):
                for _ in range(500): imitation_update(model,opt,batch)
            while mode!='imitation' and count<transitions:
                rollouts=[]
                for j in range(32):
                    task=training[(episodes+j)%len(training)]
                    # Fresh experiment/evaluation randomness on every training deployment.
                    task=make_task(task.mechanism,seed*10000000+episodes+j+500000,config,task.target)
                    rollouts.append(collect_episode(model,ScientistEnv(task,config),'cpu'))
                count+=sum(len(r['actions']) for r in rollouts); episodes+=len(rollouts)
                metrics=ppo_update(model,opt,episode_batch(rollouts,'cpu'),epochs=4)
                if len(logs)==0 or count//10000>logs[-1]['transitions']//10000:
                    logs.append({'transitions':count,'seconds':time.perf_counter()-tick,'metrics':metrics})
                    print(f'{seed} {kind} {mode} {count}/{transitions}',flush=True)
            elapsed=time.perf_counter()-tick
            name=f'{kind}-{mode}-seed{seed}'
            result={'validation':learned_evaluation(model,bank,validation),'training_seconds':elapsed,
                    'on_policy_transitions':count,'teacher_seconds_shared':teacher_seconds if mode.startswith('imitation') else 0.,
                    'teacher_transitions':sum(len(r['actions']) for r in teacher) if mode.startswith('imitation') else 0,
                    'seed':seed,'architecture':model.architecture,'curve':logs}
            save_checkpoint(output/f'{name}.pt',model,opt,{'episodes':episodes,'transitions':count},
                            {'resolved_config':config.__dict__,'scope':'development_multiseed','seed':seed})
            results[name]=result; write_json(output/'learned-results.json',results)
    return results


def amortize(output,demonstrations=512,epochs=1000,seeds=(0,1,2)):
    """Larger supervised planning check, permitted without an RL-necessity claim.

    This never bypasses the scientific PPO gate. It tests the separate useful
    outcome: can a compact policy reproduce the public-history planner?
    """
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    config=PilotConfig(readout_sd=.08,budget=8)
    bank=PredictionBank(quadrature_family(2),config)
    training=tasks_from_pool(continuous_family(7401,4096),bank,demonstrations,3041,distinct=True)
    validation=tasks_from_pool(continuous_family(8401,1024),bank,128,2042,distinct=True)
    write_json(output/'protocol.json',{'scope':'amortized_planner_learning_not_RL_specific',
               'demonstrations':demonstrations,'epochs':epochs,'seeds':list(seeds),
               'validation_mechanisms':128,'no_validation_hyperparameter_selection':True,
               'source_hash':code_hash(),'script_hash':hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
               'train_hashes':[mechanism_hash(t.mechanism) for t in training],
               'validation_hashes':[mechanism_hash(t.mechanism) for t in validation],
               'bank_hashes':[mechanism_hash(m) for m in bank.mechanisms],'locked_test_opened':False})
    teacher=[]; teacher_results=[]; tick=time.perf_counter(); shell=ActorCritic(kind='recurrent',hidden=64)
    for i,task in enumerate(training):
        belief=Belief(bank,task.target,admitted=True); planner=DecisionPlanner(bank,'voi',12); seen=[0]; rng=np.random.default_rng(i)
        def choose(obs):
            while seen[0]<len(obs['history']): belief.update(obs['history'][seen[0]]); seen[0]+=1
            return planner.choose(obs,belief,rng)
        env=ScientistEnv(task,config)
        teacher.append(collect_episode(shell,env,'cpu',teacher=choose))
        teacher_results.append(dict(env.private_result))
        if (i+1)%32==0: print(f'amortization teacher {i+1}/{demonstrations}; elapsed {time.perf_counter()-tick:.1f}s',flush=True)
    teacher_seconds=time.perf_counter()-tick
    torch.save(teacher,output/'public-history-demonstrations.pt')
    batch=episode_batch(teacher,'cpu'); results={}
    for kind in ('history','recurrent'):
        for seed in seeds:
            torch.manual_seed(seed); model=ActorCritic(kind=kind,hidden=64)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            tick=time.perf_counter(); curve=[]
            for epoch in range(epochs):
                metrics=imitation_update(model,optimizer,batch)
                if (epoch+1)%250==0:
                    curve.append({'epoch':epoch+1,**metrics})
                    print(f'{kind} imitation seed{seed} epoch{epoch+1}: {metrics}',flush=True)
            train_seconds=time.perf_counter()-tick
            summary=learned_evaluation(model,bank,validation)
            name=f'{kind}-imitation-seed{seed}'
            save_checkpoint(output/f'{name}.pt',model,optimizer,{'demonstrations':demonstrations,'epochs':epochs},
                            {'resolved_config':config.__dict__,'seed':seed,'scope':'amortized_planner_development'})
            results[name]={'validation':summary,'training_seconds':train_seconds,'teacher_seconds_shared':teacher_seconds,
                           'teacher_transitions':sum(len(t['actions']) for t in teacher),'curve':curve,
                           'architecture':model.architecture}
            write_json(output/'learning-results.json',results)
    final={'scope':'amortized_planner_development_not_RL_specific','models':results,
           'teacher_seconds':teacher_seconds,'teacher_transitions':sum(len(t['actions']) for t in teacher),
           'demonstrations':demonstrations,'epochs':epochs,'seed_count':len(seeds),
           'limitations':['Scientific PPO gate remains unmet.','Fixed designs remain a strong and computationally cheaper comparison.',
                          'Validation only; no locked-test or structural generalization claim.']}
    write_json(output/'result.json',final)
    return final


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('assess','train','all','amortize','revise'),default='all')
    p.add_argument('--output',default=str(ROOT/'reports/research-v2'))
    p.add_argument('--transitions',type=int,default=100000)
    args=p.parse_args(); output=Path(args.output).resolve()
    if args.stage=='amortize':
        print(json.dumps(amortize(output),indent=2),flush=True); return
    if args.stage=='revise':
        assessment=assess(output,unfiltered=True)
        print(json.dumps(train(output,assessment,args.transitions),indent=2),flush=True); return
    if args.stage in ('assess','all'): assessment=assess(output)
    else: assessment=json.loads((output/'assessment.json').read_text())
    if args.stage in ('train','all'): print(json.dumps(train(output,assessment,args.transitions),indent=2),flush=True)


if __name__=='__main__': main()
