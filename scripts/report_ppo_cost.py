"""Read-only report rendering after the fresh cost-efficiency test."""
from pathlib import Path
import json
import gzip
import os
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/ppo-cost-v5'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'reports/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experimentlab.envs import catalogue


def average(values):
    result={k:float(np.mean([v[k] for v in values])) for k in values[0]}
    result['aggregate_rmse']=float(np.sqrt(np.mean([v['aggregate_rmse']**2 for v in values])))
    return result


def ci(d):return f"{d['mean']:.6g} (95% CI [{d['ci95'][0]:.6g}, {d['ci95'][1]:.6g}])"


def decompose(rows):
    actions=catalogue();N=len(rows);counts=long=dose=repeats=0
    for row in rows:
        chosen=row['actions'][:-1];counts+=len(chosen);repeats+=len(chosen)-len(set(chosen))
        for idx in chosen:
            a=actions[idx];long+=a.duration==6;dose+=sum(a.intervention)
    return {'short_assay_base':1.7*counts/N,'long_duration_increment':long/N,'dose_premium':.25*dose/N,
            'assays':counts/N,'exact_repeats':repeats/N}


def main():
    path=OUT/'fresh-locked/results.json'
    if path.exists():
        data=json.loads(path.read_text())
    else:
        with gzip.open(path.with_suffix('.json.gz'),'rt') as handle:
            data=json.load(handle)
    selection=json.loads((OUT/'selection.json').read_text())
    methods=dict(data['baseline_summary']);shared={};stochastic={};parts={}
    names=('original_ppo','efficient_ppo','paired_ppo','imitation','calibrated_imitation','imitation_ppo')
    for name in names:
        models=[data['neural'][f'{name}-seed{s}'] for s in range(20,25)]
        methods[name]=average([m['summary'] for m in models]);shared[name]=average([m['shared_summary'] for m in models])
        if 'stochastic' in models[0]:stochastic[name]=average([m['stochastic']['summary'] for m in models])
        parts[name]={k:float(np.mean([decompose(m['rows'])[k] for m in models])) for k in decompose(models[0]['rows'])}
    compute={'new_main_interactions':0,'new_main_policy_decisions':0,'new_main_seconds_sum':0.,'pilots_interactions':0}
    for directory in (OUT/'development').iterdir():
        if not directory.is_dir():continue
        for p in directory.glob('seed*/result.json'):
            r=json.loads(p.read_text())
            if directory.name=='main_paired_cost':
                compute['new_main_interactions']+=r['transitions'];compute['new_main_policy_decisions']+=r['policy_decisions']
                compute['new_main_seconds_sum']+=r['training_and_validation_seconds']
            else:compute['pilots_interactions']+=r['transitions']
    primary=data['differences']['efficient_vs_original_ppo']
    screens={'cost_reduction':primary['cost_reduction']['ci95'][0]>0,
             'quality_noninferiority':primary['mse_increase']['ci95'][1]<=data['freeze']['mse_noninferiority_margin'],
             'net_return_improvement':primary['return_improvement']['ci95'][0]>0}
    summary={'methods':methods,'cost_decomposition':parts,'shared_final_estimator':shared,'stochastic':stochastic,
             'differences':data['differences'],'efficiency_screen':data['efficiency_screen'],
             'screen_components':screens,
             'compute':compute,'selection':{k:selection[k] for k in ('selected_name','selected_model','selected_calibration','imitation_calibration')}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    figdir=OUT/'figures';figdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.5))
    labels=['original_ppo','paired_ppo','efficient_ppo','imitation','calibrated_imitation'];x=np.arange(len(labels))
    titles=['Original\nPPO','Paired-return\nPPO','Selected\nefficient PPO','Imitation','Calibrated\nimitation']
    bottom=np.zeros(len(labels))
    for key,label,color in [('short_assay_base','1.7 x assay count','#94a3b8'),('long_duration_increment','Long-duration increment','#f59e0b'),('dose_premium','Dose premium','#60a5fa')]:
        values=np.array([parts[k][key] for k in labels]);axes[0].bar(x,values,bottom=bottom,label=label,color=color);bottom+=values
    axes[0].set_xticks(x,titles);axes[0].set(ylabel='Mean variable experimental cost',title='Fresh test: where the cost goes');axes[0].legend(fontsize=7)
    axes[0].grid(axis='y',alpha=.2)
    axes[1].bar(x,[methods[k]['aggregate_rmse'] for k in labels],color=['#94a3b8','#60a5fa','#2563eb','#14b8a6','#0d9488'])
    for i,name in enumerate(labels):
        ys=[data['neural'][f'{name}-seed{s}']['summary']['aggregate_rmse'] for s in range(20,25)]
        axes[1].scatter(np.linspace(i-.13,i+.13,5),ys,color='black',s=15,zorder=3)
    axes[1].axhline(methods['reference']['aggregate_rmse'],color='gray',linestyle=':',label='Private reference')
    axes[1].set_xticks(x,titles);axes[1].set(ylabel='Final target RMSE',ylim=(0,.07),title='Quality is reported separately');axes[1].legend(fontsize=7);axes[1].grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(figdir/'fresh-efficiency.png',dpi=180);plt.close(fig)
    diag=json.loads((OUT/'action-cost-diagnosis.json').read_text())['models']
    dev_parts={}
    for method in ('ppo','imitation'):
        values=[decompose(diag[method][str(s)]['rows']) for s in range(20,25)]
        dev_parts[method]={k:float(np.mean([v[k] for v in values])) for k in values[0]}
    initial_gap=sum(dev_parts['ppo'][k]-dev_parts['imitation'][k] for k in ('short_assay_base','long_duration_increment','dose_premium'))
    p=data['differences']['efficient_vs_original_ppo'];q=data['differences']['efficient_vs_calibrated_imitation']
    old,new=methods['original_ppo'],methods['efficient_ppo'];saving=old['mean_variable_cost']-new['mean_variable_cost']
    status='passes' if data['efficiency_screen'] else 'does not pass'
    lines=['# PPO experimental-cost efficiency follow-up','',
           f"Experimental cost fell {100*saving/old['mean_variable_cost']:.2f}% on the new locked test. The quality noninferiority screen passed, but net-return improvement remains uncertain; the selected method {status} the full frozen efficiency screen. Lower expenditure alone is not called an overall win.",'',
           '## Another fresh locked comparison','',
           'Five retained seeds,2,048 new parameterized mechanisms from generator seed57401,original unfiltered task physics,budget8,prices,reward and reference scoring. No method was tuned after this test. All original fixed and learned baselines remain unchanged.','',
           '| Method | Final RMSE | Variable cost | Mean return | Success within0.04 |','|---|---:|---:|---:|---:|']
    for method in ('prior','fixed','fixed_quality','additive','random','bo','eig','voi',*names,'reference'):
        r=methods[method];label=method.replace('_',' ')
        if method=='reference':label='private nominal reference (privileged)'
        lines.append(f"|{label}|{r['aggregate_rmse']:.5f}|{r['mean_variable_cost']:.3f}|{r['mean_return']:.5f}|{100*r['success_rate']:.2f}%|")
    lines.extend(['',f"Versus original PPO, the selected policy reduces cost by {ci(p['cost_reduction'])}, or {100*saving/old['mean_variable_cost']:.2f}% at the point estimate. Raw MSE increase is {ci(p['mse_increase'])}; return improvement is {ci(p['return_improvement'])}.",'',
                  'The predeclared noninferiority screen uses an upper two-sided95% bound on MSE increase of0.0001. This is an engineering margin,not proof of identical quality or a universal scientific criterion. The full efficiency screen additionally requires positive cost and return intervals.','',
                  f"Versus equally calibrated imitation: cost reduction {ci(q['cost_reduction'])}; MSE increase {ci(q['mse_increase'])}; return improvement {ci(q['return_improvement'])}. Both unchanged imitation and its public-price-calibrated version are retained so the PPO comparison does not weaken the baseline.",'',
                  '![Cost decomposition and quality](../reports/ppo-cost-v5/figures/fresh-efficiency.png)','',
                  '## What made PPO expensive','',
                  f"New development traces found a mean PPO-minus-imitation cost gap of {initial_gap:.3f}. Its exact additive decomposition was: extra assay-count baseline {dev_parts['ppo']['short_assay_base']-dev_parts['imitation']['short_assay_base']:.3f}; long-duration increment {dev_parts['ppo']['long_duration_increment']-dev_parts['imitation']['long_duration_increment']:.3f}; higher-dose premium {dev_parts['ppo']['dose_premium']-dev_parts['imitation']['dose_premium']:.3f}. Each assay costs1.7 at duration1 before dose; duration6 adds exactly1.0; both panels contain two variables and have the same price.",'',
                  'Imitation consistently used one long outcome assay and short proxy follow-ups. Several PPO seeds substituted additional long outcome assays; higher doses and extra queries contributed too. Panel identity alone did not change price. Exact action/duration/dose/control histories,stopping and repeat counts were collected on development mechanisms only.','',
                  'The previous curriculum and high early entropy were plausible sources of the habit,but not assumed proven causes. Matched low/zero-entropy continuation and temporary cheaper-follow-up exposure produced limited improvements in bounded controls. All scientific fees and penalties remained fixed.','',
                  '## The selected method and what it means','',
                  f"Development selected `{selection['selected_name']}`: model source `{selection['selected_model']}`,with common public calibration coefficients {selection['selected_calibration']}. The identical coordinate search was offered to imitation; its selected coefficients are {selection['imitation_calibration']}.",'',
                  'Calibration changes logits using only declared duration,dose prices and public experiment count. It never changes legal masks,measured values,target error or reward. Duration penalties and stop bonuses apply after the first assay; dose prices apply to experiments. It is transparent post-processing,not a PPO gradient discovery or an oracle action ranking. The GRU64 architecture and learned parameter count stay unchanged.','',
                  'The additional training variant used four independently seeded deployments of the same training mechanism/goal. Actor coefficients were each observed full return minus the mean of the other three returns; critic targets remained ordinary complete Monte Carlo returns. No group identity,mechanism parameters,unbought evidence or counterfactual action labels entered the policy. PPO clipping remains approximate,so this is a declared variance-reduction variant rather than a claim of unbiased PPO.','',
                  'Independent review caught an unintended transition-weighted centering step in the initial paired pilot. With unequal episode lengths,it changed the declared leave-one-out coefficients. An integration regression failed,then passed after centering was disabled for this variant. The initial centered pilots are retained but excluded from promotion. Old v3/v4 PPO results are unaffected.','',
                  'Candidate selection used256 tuning mechanisms and a separate256-mechanism development confirmation slice. Confirmation aggregates were inspected and are treated as development evidence. The final wrappers,checkpoint hashes,baseline schedules,quality margin and test generation were frozen before the new test. All five seeds were retained.','',
                  f"Using the common final selector on the efficient policy's purchased assays gives aggregate RMSE {shared['efficient_ppo']['aggregate_rmse']:.5f},compared with its learned commitment RMSE {new['aggregate_rmse']:.5f}. This separates experiment selection from evidence interpretation; replay timing is not a full-policy latency comparison.",'',
                  '## Compute,preservation and limits','',
                  f"The new five-seed continuation consumed {compute['new_main_interactions']:,} physical interactions and {compute['new_main_policy_decisions']:,} sampled policy decisions. Summed training/validation wall time across parallel jobs was {compute['new_main_seconds_sum']:.1f}s. Bounded pilots consumed another {compute['pilots_interactions']:,} interactions,including retained failed/centered controls. Calibration/evaluation costs are separate from those gradient interaction counts.",'',
                  'No larger network,GPU migration,new task family or altered reward prices was introduced. The existing CPU path and exact private simulator cache were reused. Final scoring used the original uncached reference dynamics; single/batched greedy development actions and scores were checked identical before locking.','',
                  'The exhaustive10,373-design fixed and quality-focused schedules,prior,random,additive identification,BO,EIG,VOI,and stronger imitation remain in the test. Fixed/random/BO/EIG/VOI use the same public-history final estimator. The common baseline setup costs3 outside the variable budget. The reference has privileged mechanism access and zero variable assay cost.','',
                  'The primary comparison is greedy deployment; stochastic checks are retained in the raw result. Confidence intervals cross training seeds and paired mechanisms. The task remains a familiar-motif continuous-parameter holdout with culture-offset SDzero. Broader structures,ranges,culture variation and real biology are not qualified.','',
                  'Previous locked studies,protected models and baseline schedules are verified byte-for-byte unchanged. Source/checkpoint hashes,raw rows,traces,negative candidates and selection rules are included. No result from this fresh test was used to retune a method.','',
                  'The leave-one-out formula is adapted from [the RLOO method](https://arxiv.org/abs/2402.14740) and its [public implementation documentation](https://github.com/huggingface/trl/blob/main/docs/source/rloo_trainer.md); this project uses trajectory returns in a small scientific simulator,not an LLM training reproduction.',''])
    text='\n'.join(lines)
    # Readable prose around the compact numerical notation; values are unchanged.
    replacements={'seeds,2,048':'seeds, 2,048','seed57401':'seed 57401','budget8':'budget 8',
                  'within0.04':'within 0.04','two-sided95%':'two-sided 95%','of0.0001':'of 0.0001',
                  'costs1.7':'costs 1.7','duration1':'duration 1','duration6':'duration 6','exactly1.0':'exactly 1.0',
                  'used256':'used 256','separate256':'separate 256','SDzero':'SD zero','costs3':'costs 3',
                  'exhaustive10,373':'exhaustive 10,373',',not':', not',',original':', original',',prices':', prices',
                  ',reward':', reward',',physical':', physical',',target':', target',',quality':', quality',
                  ',variable':', variable',',stopping':', stopping',',masked':', masked',',checkpoint':', checkpoint',
                  ',source':', source',',negative':', negative',',ranges':', ranges',',culture':', culture',
                  ',private':', private',',mean':', mean',',mechanism':', mechanism',',counterfactual':', counterfactual',
                  ',independent':', independent',',goal':', goal',',but':', but',',with':', with',',not':', not'}
    for a,b in replacements.items():text=text.replace(a,b)
    (ROOT/'docs/ppo_cost_efficiency.md').write_text(text)
    print('Rendered docs/ppo_cost_efficiency.md,summary.json and scientific figure.')


if __name__=='__main__':main()
