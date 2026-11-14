"""Small, local fixtures only: never construct the M2 confirmation/final pools."""
import copy
import importlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def module():
    assert importlib.util.find_spec('scripts.ppo_m2_confirm') is not None, 'confirmation helper missing'
    return importlib.import_module('scripts.ppo_m2_confirm')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def race_fixture(tmp_path, monkeypatch, missing=None):
    m = module()
    root = tmp_path/'repo'
    root.mkdir()
    records = {}
    for method in ('ppo', 'imitation', 'calibrated_imitation', 'imitation_ppo'):
        for seed in m.SEEDS:
            p = root/f'models/{method}-{seed}.pt'
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(f'{method}-{seed}'.encode())
            records[f'{method}-seed{seed}'] = {'seed':seed, 'path':str(p.relative_to(root)),
                'sha256':m.file_hash(p), 'calibration':{'duration_penalty':0., 'dose_penalty':0., 'stop_bonus':0.}}
    monkeypatch.setattr(m, 'model_records', lambda root, seeds:copy.deepcopy(records))
    monkeypatch.setattr(m, 'source_hashes', lambda root:{})
    monkeypatch.setattr(m, 'prior_metadata', lambda root:(set(), {}))
    monkeypatch.setattr(m, 'continuous_family', lambda *a, **kw:pytest.fail('prepare must not generate mechanisms'))
    write(root/'reports/ppo-efficiency-v6/fresh-locked/freeze.json',
          {'config':asdict(m.PilotConfig(final_cultures=128)), 'fixed':[4,12,31],
           'fixed_quality':[4,14,31], 'additive_schedule':[11,35]})
    write(root/'reports/ppo-efficiency-v6/selection.json', {'registry':{}})
    race = root/'race'
    selection = {'confirmation_opened':False, 'final_opened':False, 'hosted_calls':0}
    rows = []
    for method, key, initialization, ret in [
        ('scratch_selected','scratch_selected_records','scratch',-.3),
        ('v6_continued','v6_selected_records','v6',-.2),
        ('imitation_ppo_continued','imitation_ppo_records','imitation_ppo',100.)]:
        selection[key] = {}
        for seed in m.SEEDS:
            if missing == (method, seed):
                continue
            p = root/f'models/{method}-{seed}.pt'
            p.write_bytes(f'{method}-{seed}'.encode())
            rec = {'path':str(p), 'sha256':m.file_hash(p), 'calibration':records[f'ppo-seed{seed}']['calibration']}
            selection[key][str(seed)] = rec
            metrics = {'mean_return':ret, 'mean_mse':.001, 'mean_variable_cost':2.}
            result_path = race/f'jobs/{method}-{seed}/result.json'
            result = {'schema':'m2-ppo-train-v1','seed':seed,'status':'completed',
                'config':{'initialization':initialization},
                'correctness':{'source_hashes_unchanged':True,'initial_checkpoint_unchanged':True,
                    'cache_reference':{'passed':True},'max_initial_logp_error':0.,'initial_logp_tolerance':5e-5},
                'snapshots':{'equal_interaction':{}},
                'best_primary':{'checkpoint':rec,'metrics':metrics,'selection_eligible':True,'training_biases_zero':True},
                'evaluations':[], 'inherited_pretraining':{'included_in_new_training_budget':False}}
            if initialization == 'v6':
                result['evaluations'] = [{'reason':'initial_incumbent','selection_eligible':True,
                    'metrics':{'mean_return':-.4,'mean_mse':.002,'mean_variable_cost':2.}}]
            write(result_path,result)
            rows.append({'job_id':f'{method}-{seed}','name':method,'seed':seed,'config':{'initialization':initialization},
                'status':'completed','correct':True,'primary_complete':True,'best':metrics,
                'checkpoint':rec,'result_path':str(result_path)})
    write(race/'plan.json',{'all_seeds':list(m.SEEDS),'sources':{},'future_confirmation_and_final_opened':False})
    write(race/'selection.json',selection)
    write(race/'ledger.json',{'rows':rows,'reserved_training':0,'reserved_validation':0,
        'charged_training_upper_bound':123,'charged_validation_upper_bound':45})
    return m, root, race


def test_prepare_is_pure_freezes_all_seeds_and_excludes_imitation_from_ppo(tmp_path, monkeypatch):
    m, root, race = race_fixture(tmp_path, monkeypatch)
    output = root/'frozen/freeze.json'
    freeze = m.prepare(race, output, root=root)
    assert freeze['evaluation'] == {'generator_seed':991321,'count':128,'episode_seed':991322000}
    assert freeze['selected_ppo']['cohort'] == 'v6_continued'
    assert freeze['selected_ppo']['kind'] == 'm2'
    assert all(set(c['records']) == set(map(str,m.SEEDS)) for c in freeze['cohorts'].values())
    selected = json.loads((output.parent/'selected_ppo_records.json').read_text())
    assert set(selected) == set(map(str,m.SEEDS))
    assert all(r['sha256'] for r in selected.values())
    assert freeze['config']['final_cultures'] == 128
    assert freeze['final_test_opened'] is False
    with pytest.raises(FileExistsError):m.prepare(race, output, root=root)


def test_incomplete_cohort_is_blocked_and_not_silently_reduced(tmp_path, monkeypatch):
    m, root, race = race_fixture(tmp_path, monkeypatch, ('scratch_selected',24))
    freeze = m.prepare(race, root/'freeze.json', root=root)
    cohort = freeze['cohorts']['scratch_selected']
    assert cohort['status'] == 'blocked'
    assert cohort['missing_seeds'] == [24]
    assert not any(k.startswith('scratch_selected-seed') for k in freeze['models'])
    assert set(cohort['records']) == {'20','21','22','23'}


def test_prepare_rejects_unfinished_race_and_tampered_checkpoint(tmp_path, monkeypatch):
    m, root, race = race_fixture(tmp_path, monkeypatch)
    ledger = json.loads((race/'ledger.json').read_text())
    ledger['reserved_training'] = 1
    write(race/'ledger.json', ledger)
    with pytest.raises(ValueError,match='completed race'):m.prepare(race,root/'freeze.json',root=root)
    ledger['reserved_training'] = 0
    write(race/'ledger.json',ledger)
    (root/'models/v6_continued-20.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='hash'):m.prepare(race,root/'freeze.json',root=root)


def metric_row(method, seed, index, mse, ret):
    return {'method':method,'training_seed':seed,'task_index':index,'status':'completed',
        'metrics':{'final_mse':mse,'variable_cost':2.,'total_assay_cost':5.,'return_':ret,
                   'effort':.4,'assays':1,'immediate_stop':False,'observed_return':ret,
                   'latency_seconds':0.001}}


def test_summary_uses_aggregate_rmse_and_seed_mechanism_pairing():
    m = module()
    rows = [metric_row('a',s,i,mse,2+s+i) for s in (20,21) for i,mse in enumerate((.01,.09))]
    rows += [metric_row('b',s,i,mse,1+s+i) for s in (20,21) for i,mse in enumerate((.01,.09))]
    summary = m.summarize_rows(rows, {'a':[20,21],'b':[20,21]}, 2, comparisons=[('a','b')], draws=100)
    assert summary['cohorts']['a']['aggregate_rmse'] == pytest.approx(np.sqrt(.05))
    pair = summary['comparisons']['a-minus-b']
    assert pair['metrics']['return_']['difference'] == 1
    assert pair['metrics']['return_']['ci95'] == [1.,1.]
    assert pair['metrics']['return_']['ci98_333333'] == [1.,1.]
    assert pair['scientific_return_gain'] is True
    incomplete = m.summarize_rows(rows[:-1], {'a':[20,21],'b':[20,21]},2,comparisons=[('a','b')],draws=100)
    assert incomplete['cohorts']['b']['status'] == 'incomplete'
    assert incomplete['cohorts']['b']['missing_count'] == 1
    assert incomplete['comparisons']['a-minus-b']['status'] == 'blocked'


def test_deterministic_baseline_is_tiled_without_replication_claim():
    m=module()
    rows=[metric_row('a',s,i,.01,1.) for s in (20,21) for i in range(2)]
    rows += [metric_row('fixed',None,i,.01,0.) for i in range(2)]
    result=m.summarize_rows(rows,{'a':[20,21],'fixed':[None]},2,comparisons=[('a','fixed')],draws=100)
    pair=result['comparisons']['a-minus-fixed']
    assert pair['baseline_tiled_across_training_seeds'] is True
    assert pair['independent_baseline_replicates'] == 1
    assert pair['paired_cells'] == 4


def test_real_finite_fixture_action_replay_detects_cost_and_trace_tampering():
    m = module()
    from experimentlab.tasks import finite_family, make_task
    config=m.PilotConfig(final_cultures=2)
    task=make_task(finite_family()[0], 12345, config)
    env=m.RecordedEnv(task, config, m.Budget(max_episodes=3,max_step_calls=15,wall_seconds=10))
    env.reset(); env.step(4); env.step(36)
    row=m.episode_row(env,'fixture',20,0,latency_seconds=.001)
    assert m.replay_row(row,task,config)['passed'] is True
    bad=copy.deepcopy(row);bad['metrics']['variable_cost']+=1
    with pytest.raises(ValueError,match='metrics'):m.replay_row(bad,task,config)
    bad=copy.deepcopy(row);bad['public_record']['target'][0]+=.01
    with pytest.raises(ValueError,match='public'):m.replay_row(bad,task,config)
    bad=copy.deepcopy(row);bad['rewards'][0]+=.01
    with pytest.raises(ValueError,match='reward'):m.replay_row(bad,task,config)


def test_budget_stops_without_fabricating_completion():
    m=module()
    from experimentlab.tasks import finite_family, make_task
    config=m.PilotConfig(final_cultures=2)
    task=make_task(finite_family()[0],54321,config)
    env=m.RecordedEnv(task,config,m.Budget(max_episodes=1,max_step_calls=1,wall_seconds=10))
    env.reset();env.step(4)
    with pytest.raises(m.LimitReached):env.step(36)
    row=m.episode_row(env,'fixture',20,0,latency_seconds=.001,failure='step cap')
    assert row['status']=='failed' and row['metrics']['final_mse'] is None
    assert len(row['actions'])==1
    assert m.replay_row(row,task,config)['passed'] is True


def test_original_v6_can_win_without_any_confirmation_selection(tmp_path, monkeypatch):
    m,root,race=race_fixture(tmp_path,monkeypatch)
    for path in (race/'jobs').glob('v6_continued-*/result.json'):
        value=json.loads(path.read_text())
        value['evaluations'][0]['metrics']['mean_return']=1.
        write(path,value)
    freeze=m.prepare(race,root/'freeze.json',root=root)
    assert freeze['selected_ppo']['cohort']=='ppo'
    assert freeze['selected_ppo']['kind']=='registered_v6'


def test_freeze_validation_precedes_pool_construction(tmp_path,monkeypatch):
    m,root,race=race_fixture(tmp_path,monkeypatch)
    output=root/'freeze.json'
    m.prepare(race,output,root=root)
    frozen=json.loads(output.read_text());frozen['config']['cost_weight']=0
    write(output,frozen)
    assert hasattr(m,'run'), 'bounded confirmation runner missing'
    with pytest.raises(ValueError,match='seal'):m.run(output,root/'run',root=root)
    assert not (root/'run').exists()


def test_finite_fixture_batch_runtime_preserves_native_final_actions(tmp_path,monkeypatch):
    m=module()
    from experimentlab.tasks import finite_family, make_task
    assert hasattr(m,'evaluate_cases'), 'offline case evaluator missing'
    config=m.PilotConfig(final_cultures=2)
    tasks=[make_task(x,12300+i,config) for i,x in enumerate(finite_family()[:2])]
    policy=m.ActorCritic(kind='recurrent',hidden=64,horizon=5)
    with m.torch.no_grad():
        for parameter in policy.parameters():parameter.zero_()
        policy.actor.bias[36]=10
    monkeypatch.setattr(m,'load_policy',lambda *args:policy)
    monkeypatch.setattr(m,'make_bank',lambda config:None)
    freeze={'config':asdict(config),'cohorts':{'fixture':{'status':'ready','records':{'20':{}}}},
            'baselines':[],'schedules':{},'planner_draws':12,'planner_seed':55}
    rows,failures=m.evaluate_cases(freeze,tasks,m.Budget(wall_seconds=10),root=tmp_path)
    assert not failures and len(rows)==2
    assert all(r['actions']==[36] and r['status']=='completed' for r in rows)
    assert all(m.replay_row(r,tasks[r['task_index']],config)['passed'] for r in rows)


def test_report_scope_and_descriptive_comparisons_do_not_claim_final_inference():
    m=module()
    assert hasattr(m,'confirmation_summary'), 'confirmation report summary missing'
    rows=[]
    for method in ('ppo','scratch_selected','imitation','calibrated_imitation','imitation_ppo','v6_continued','imitation_ppo_continued'):
        rows.extend(metric_row(method,s,i,.01,1. if method=='scratch_selected' else 0.) for s in m.SEEDS for i in range(2))
    rows.extend(metric_row(method,None,i,.01,0.) for method in m.BASELINES+('reference',) for i in range(2))
    freeze={'cohorts':{method:{'required_seeds':list(m.SEEDS)} for method in m.ORIGINALS+tuple(m.EXTRAS)},
            'baselines':list(m.BASELINES)+['reference'],'bootstrap':{'draws':100}}
    summary=m.confirmation_summary(rows,freeze,2)
    assert summary['comparisons']['scratch_selected-minus-ppo']['scientific_return_gain'] is True
    pair=summary['descriptive_comparisons']['scratch_selected-minus-calibrated_imitation']
    assert 'scientific_return_gain' not in pair
    assert 'ci98_333333' not in pair['metrics']['return_']
    assert summary['final_test_opened'] is False


def test_complete_finite_mock_publication_and_weights_free_replay(tmp_path,monkeypatch):
    m,root,race=race_fixture(tmp_path,monkeypatch)
    from experimentlab.tasks import finite_family
    finite=finite_family()
    # All generators are replaced with finite fixtures; neither real assessment
    # generator is called. Production CLI has no configurable generator/count.
    monkeypatch.setattr(m,'EVALUATION',{'generator_seed':991321,'count':2,'episode_seed':991322000})
    monkeypatch.setattr(m,'BOOTSTRAP',{**m.BOOTSTRAP,'draws':50})
    def fixture_family(seed,count):
        return finite[:2] if seed==991321 else [finite[2 if seed==7401 else 3]]
    monkeypatch.setattr(m,'continuous_family',fixture_family)
    policy=m.ActorCritic(kind='recurrent',hidden=64,horizon=5)
    with m.torch.no_grad():
        for parameter in policy.parameters():parameter.zero_()
        policy.actor.bias[36]=10
    monkeypatch.setattr(m,'load_policy',lambda *args:policy)
    monkeypatch.setattr(m,'make_bank',lambda config:None)
    def commit(env,*args):env.reset();env.step(36)
    monkeypatch.setattr(m,'run_local',commit)
    freeze_path=root/'frozen/freeze.json'
    m.prepare(race,freeze_path,root=root)
    output=root/'confirmation-fixture'
    result=m.run(freeze_path,output,root=root)
    assert result['status']=='completed'
    assert result['accounting']['episodes']==88
    monkeypatch.setattr(m,'load_policy',lambda *args:pytest.fail('replay must not load weights'))
    monkeypatch.setattr(m,'continuous_family',lambda *args:pytest.fail('replay must not generate mechanisms'))
    verified=m.replay(output,runtime_root=root)
    assert verified['passed'] and verified['episodes_replayed']==88 and verified['weights_free']
    assert not list(output.rglob('*.pt'))
    with pytest.raises(FileExistsError):m.run(freeze_path,output,root=root)
    summary_path=output/'summary.json'
    tampered=json.loads(summary_path.read_text());tampered['raw_row_count']+=1
    write(summary_path,tampered)
    with pytest.raises(ValueError,match='hash'):m.replay(output,runtime_root=root)


def test_invalid_primary_certificate_blocks_the_entire_cohort(tmp_path,monkeypatch):
    m,root,race=race_fixture(tmp_path,monkeypatch)
    path=race/'jobs/scratch_selected-24/result.json'
    receipt=json.loads(path.read_text());receipt['best_primary']['selection_eligible']=False
    write(path,receipt)
    freeze=m.prepare(race,root/'freeze.json',root=root)
    assert freeze['cohorts']['scratch_selected']['status']=='blocked'
    assert freeze['cohorts']['scratch_selected']['invalid_seeds']==[24]
    assert not any(k.startswith('scratch_selected-seed') for k in freeze['models'])


def test_step_exception_is_recorded_and_replayed_without_losing_a_charge():
    m=module()
    from experimentlab.tasks import finite_family, make_task
    config=m.PilotConfig(final_cultures=2)
    task=make_task(finite_family()[0],321,config)
    budget=m.Budget(wall_seconds=10)
    env=m.RecordedEnv(task,config,budget)
    env.reset();env.step(36)
    with pytest.raises(RuntimeError,match='episode ended'):env.step(36)
    row=m.episode_row(env,'fixture',20,0,latency_seconds=.001)
    assert row.get('step_errors'), 'failed environment step must be preserved'
    assert budget.step_calls==2
    assert m.replay_row(row,task,config)['step_calls']==2


def test_bootstrap_can_stop_at_the_cpu_deadline_without_inventing_intervals():
    m=module()
    budget=m.Budget(wall_seconds=0.)
    assert 'budget' in __import__('inspect').signature(m.bootstrap_difference).parameters
    with pytest.raises(m.LimitReached):
        m.bootstrap_difference([[1.,2.]],[[0.,1.]],draws=100,budget=budget)


def test_source_snapshot_excludes_the_original_spec(tmp_path):
    m=module()
    (tmp_path/'src').mkdir();(tmp_path/'src/example.py').write_text('value = 1\n')
    (tmp_path/'autonomous_scientist_build_spec.md').write_text('private location\n')
    hashes=m.source_hashes(tmp_path)
    assert 'src/example.py' in hashes
    assert 'autonomous_scientist_build_spec.md' not in hashes


def test_imitation_initialization_cannot_be_mislabeled_as_pure_scratch(tmp_path,monkeypatch):
    m,root,race=race_fixture(tmp_path,monkeypatch)
    path=race/'ledger.json'
    ledger=json.loads(path.read_text())
    for row in ledger['rows']:
        if row['name']=='scratch_selected':row['config']['initialization']='imitation_ppo'
    write(path,ledger)
    freeze=m.prepare(race,root/'freeze.json',root=root)
    assert freeze['cohorts']['scratch_selected']['status']=='blocked'
    assert freeze['cohorts']['scratch_selected']['invalid_seeds']==list(m.SEEDS)


def test_prior_metadata_reads_compressed_manifests_without_outcomes(tmp_path,monkeypatch):
    m=module()
    monkeypatch.setattr(m,'previous_hashes',lambda root:(set(),{}))
    path=tmp_path/'reports/jev-rl-m2/old/task-manifest.json.gz'
    path.parent.mkdir(parents=True)
    m.write_gzip_new(path,{'tasks':[{'mechanism_hash':'a'*64}]})
    (path.parent/'results.json').write_text('must never be parsed')
    identifiers,hashes=m.prior_metadata(tmp_path)
    assert 'a'*64 in identifiers
    assert str(path.relative_to(tmp_path)) in hashes
