import importlib
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def module():
    assert importlib.util.find_spec('scripts.run_ppo_m2_race') is not None,'race orchestrator missing'
    return importlib.import_module('scripts.run_ppo_m2_race')


def test_architecture_plan_is_balanced_and_bounded():
    m=module();jobs=m.architecture_jobs()
    assert len(jobs)==16
    assert {(j['kind'],j['hidden'],j['seed']) for j in jobs}=={
        (kind,width,seed) for kind in ('history','recurrent') for width in (64,128,256,512) for seed in (20,21)}
    assert sum(j['selector_interactions']+j['joint_hard_interactions'] for j in jobs)==36_000_000
    assert all(j['backend']=='cpu' and j['threads']==1 for j in jobs)


def test_selection_never_drops_bad_or_missing_seed():
    m=module()
    def row(name,seed,ret):
        return {'name':name,'seed':seed,'status':'completed','correct':True,
                'best':{'mean_return':ret,'mean_mse':.002,'mean_variable_cost':4.}}
    rows=[row('lucky',20,100),row('retained',20,-.3),row('retained',21,-.5)]
    assert m.select_group(rows)=='retained'
    rows.append(row('lucky',21,-1000))
    assert m.select_group(rows)=='retained'
    rows[-1]['status']='incomplete_budget'
    assert m.select_group(rows)=='retained'
    with pytest.raises(ValueError):m.select_group([row('only',20,1)])


def test_selection_ties_use_quality_then_cost_then_name():
    m=module();rows=[]
    for name,mse,cost in [('z',.003,1.),('b',.002,3.),('a',.002,3.)]:
        for seed in (20,21):
            rows.append({'name':name,'seed':seed,'status':'completed','correct':True,
                         'best':{'mean_return':-.3,'mean_mse':mse,'mean_variable_cost':cost}})
    assert m.select_group(rows)=='a'


def test_interrupted_json_is_a_failed_receipt_not_a_controller_crash(tmp_path):
    m=module();path=tmp_path/'result.json';path.write_text('{"totals":')
    assert hasattr(m,'read_complete_json'),'incomplete-result reader missing'
    assert m.read_complete_json(path) is None
    path.write_text('{"complete":true}')
    assert m.read_complete_json(path)=={'complete':True}


def test_primary_selection_does_not_censor_fast_missing_wall_snapshot():
    m=module();rows=[]
    for name,status,ret in [('fast','incomplete_budget',-.2),('slow','completed',-.3)]:
        for seed in (20,21):
            rows.append({'name':name,'seed':seed,'status':status,'correct':True,
                         'primary_complete':True,'wall_complete':status=='completed',
                         'best':{'mean_return':ret,'mean_mse':.002,'mean_variable_cost':4.}})
    assert m.select_group(rows)=='fast'
    rows[0]['primary_complete']=False
    assert m.select_group(rows)=='slow'
