import runpy
import numpy as np
from experimentlab.cli import ROOT
from experimentlab.envs import PilotConfig
from experimentlab.inference import PredictionBank
from experimentlab.tasks import continuous_family, mechanism_hash


def stage():
    return runpy.run_path(str(ROOT/'scripts/research_stage.py'))


def test_stratified_bank_covers_all_modes_without_true_parameter_insertion():
    api=stage()
    support=api['quadrature_family'](2)
    assert len(support)==128
    assert len({tuple(np.sign(m.W).ravel()) for m in support})==32
    bank=PredictionBank(support,PilotConfig(readout_sd=.08))
    tasks=api['tasks_from_pool'](continuous_family(8401,64),bank,8,2042)
    assert len({mechanism_hash(t.mechanism) for t in tasks})==8
    assert not({mechanism_hash(t.mechanism) for t in tasks}&{mechanism_hash(m) for m in support})


def test_more_compute_does_not_bypass_scientific_training_gate(tmp_path):
    api=stage()
    result=api['train'](tmp_path.resolve(),{'gate_C':False},transitions=100000)
    assert result['status']=='scientific_gate_not_met'
    assert not list(tmp_path.glob('*.pt'))
    assert (tmp_path.resolve()/'training-decision.json').is_file()
