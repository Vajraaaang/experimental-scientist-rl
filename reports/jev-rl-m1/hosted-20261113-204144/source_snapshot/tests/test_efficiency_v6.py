from pathlib import Path
import runpy
import json
import torch
import pytest
from experimentlab.cli import ROOT
from experimentlab.learning import ActorCritic,save_checkpoint
from experimentlab.envs import PilotConfig
from experimentlab.tasks import continuous_family,make_task
from experimentlab.private_cache import PrivateDynamicsCache


def api(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    return runpy.run_path(str(ROOT/'scripts/ppo_efficiency_v6.py'))


@pytest.mark.parametrize('self_selector',[False,True])
def test_calibrated_training_keeps_behavior_ratios_and_checkpoint_contract(tmp_path,monkeypatch,self_selector):
    functions=api(monkeypatch);torch.set_num_threads(1)
    config=PilotConfig();pool=continuous_family(7401,4)
    tasks=[make_task(m,77000+i,config,[.42,.58]) for i,m in enumerate(pool)]
    cache=PrivateDynamicsCache(pool,config);output=tmp_path.resolve()
    (output/'development-manifest.json').write_text('{}')
    initial=output/'initial.pt';model=ActorCritic(kind='recurrent',hidden=64)
    save_checkpoint(initial,model,torch.optim.Adam(model.parameters()),{})
    settings={'batch':4,'epochs':2,'paired_rollouts':2,'learning_rate':.0005,'entropy':.005,
              'entropy_final':.001,'value_coef':.1,'evaluation_every':20,'episode_namespace':71,'self_selector':self_selector}
    calibration={'duration_penalty':4.,'dose_penalty':4.,'stop_bonus':.5}
    result=functions['train']('smoke',settings,20,20,(config,pool,tasks,cache,None,0.),output,initial,calibration)
    assert result['transitions']>=20
    if self_selector:assert result['policy_decisions']==result['episodes']
    assert result['maximum_initial_logp_error']<2e-5
    state=torch.load(result['selected_checkpoint'],map_location='cpu',weights_only=False)
    assert state['architecture']==model.architecture
    assert result['calibration']==calibration
    assert 'base.encoder.0.weight' not in state['model']
    assert 'encoder.0.weight' in state['model']
    assert result['initial_checkpoint_sha256']
    with pytest.raises(FileExistsError):
        functions['train']('smoke',settings,20,20,(config,pool,tasks,cache,None,0.),output,initial,calibration)


def test_tempered_exploration_changes_only_public_distribution(monkeypatch):
    functions=api(monkeypatch)
    temperature_policy=functions.get('TemperaturePolicy')
    assert temperature_policy is not None,'missing temporary temperature exploration wrapper'
    base=ActorCritic(kind='recurrent',hidden=64)
    tokens=torch.randn(2,3,75)
    logits,values=base(tokens)
    cold,cold_values=temperature_policy(base,1.)(tokens)
    hot,hot_values=temperature_policy(base,4.)(tokens)
    assert torch.equal(cold,logits) and torch.equal(cold_values,values)
    assert torch.allclose(hot,logits/4.) and torch.equal(hot_values,values)
    assert sum(p.numel() for p in temperature_policy(base,4.).parameters())==sum(p.numel() for p in base.parameters())
    with pytest.raises(ValueError):temperature_policy(base,0.)


def test_repeated_setup_does_not_rewrite_shared_manifest(tmp_path,monkeypatch):
    functions=api(monkeypatch)
    functions['setup'](tmp_path.resolve())
    path=tmp_path.resolve()/'development-manifest.json'
    original=path.read_bytes();timestamp=path.stat().st_mtime_ns
    functions['setup'](tmp_path.resolve())
    assert path.read_bytes()==original
    assert path.stat().st_mtime_ns==timestamp,'parallel workers must read the initialized manifest without truncating it'


def test_cheap_public_schedules_are_affordable_grouped_and_low_dose(monkeypatch):
    functions=api(monkeypatch)
    schedules=functions.get('cheap_schedules')
    assert schedules is not None,'missing public cheap-assay curriculum'
    from experimentlab.envs import catalogue
    actions=catalogue();designs=schedules(range(64),72)
    assert designs==schedules(range(64),72)
    assert len({d for d in designs})>20
    for design in designs:
        assert 1<=len(design)<=3
        assert actions[design[0]].duration==6 and actions[design[0]].panel==(6,7)
        assert sum(actions[a].cost_ticks for a in design)<=160
        assert all(sum(actions[a].intervention)==pytest.approx(.4) for a in design)
        assert all(actions[a].duration==1 and actions[a].panel==(4,5) for a in design[1:])


def test_gap_closure_uses_interval_bounds_in_the_declared_directions(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    functions=runpy.run_path(str(ROOT/'scripts/ppo_v6_locked.py'))
    margins={'cost_increase_upper':.10,'mse_increase_upper':.0001,'return_improvement_lower':-.005}
    passing={'cost_increase':{'ci95':[-.02,.09]},'mse_increase':{'ci95':[-.0001,.00009]},'return_improvement':{'ci95':[-.004,.01]}}
    assert all(functions['closure_components'](passing,margins).values())
    failing={**passing,'cost_increase':{'mean':-.02,'ci95':[-.10,.11]},'return_improvement':{'mean':.003,'ci95':[-.006,.01]}}
    assert functions['closure_components'](failing,margins)=={'cost':False,'quality':True,'return':False}


def test_commitment_restart_preserves_assay_head_and_common_logit_component(monkeypatch):
    functions=api(monkeypatch)
    reset=functions.get('reset_commitment_rankings')
    assert reset is not None,'missing controlled terminal-ranking restart'
    torch.manual_seed(123);model=ActorCritic(kind='recurrent',hidden=64)
    weights=model.actor.weight.detach().clone();biases=model.actor.bias.detach().clone()
    reset(model)
    assert torch.equal(model.actor.weight[:36],weights[:36])
    assert torch.equal(model.actor.bias[:36],biases[:36])
    assert torch.allclose(model.actor.weight[36:].mean(0),weights[36:].mean(0),atol=1e-7)
    assert torch.allclose(model.actor.bias[36:].mean(),biases[36:].mean(),atol=1e-7)
    assert model.actor.weight[36:].std(0).mean()<weights[36:].std(0).mean()/2


def test_atomic_compressed_packets_roundtrip_and_remove_failed_temporary_files(tmp_path,monkeypatch):
    import numpy as np
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    functions=runpy.run_path(str(ROOT/'scripts/ppo_v6_locked.py'))
    path=tmp_path.resolve()/'packet.json.gz'
    functions['write_packet'](path,{'cost':np.float64(.3),'actions':np.array([1,2]),'valid':np.bool_(True)})
    assert functions['read_packet'](path)=={'cost':.3,'actions':[1,2],'valid':True}
    broken=tmp_path.resolve()/'broken.json.gz'
    with pytest.raises(TypeError):functions['write_packet'](broken,{'unsupported':{1,2}})
    assert not broken.exists() and not broken.with_suffix('.gz.tmp').exists()
