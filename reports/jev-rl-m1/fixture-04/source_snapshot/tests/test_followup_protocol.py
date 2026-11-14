import runpy
import numpy as np
import pytest
from experimentlab.cli import ROOT


def test_crossed_bootstrap_reports_quality_reduction_in_expected_direction(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    api=runpy.run_path(str(ROOT/'scripts/ppo_v4_locked.py'))
    prior=np.full((5,8),.01);improved=np.full((5,8),.008)
    result=api['hierarchical_difference'](prior,improved,samples=200)
    assert result['mean']==pytest.approx(.002)
    assert result['ci95']==pytest.approx([.002,.002])
    assert result['training_seeds']==5 and result['mechanisms']==8


def test_crossed_bootstrap_requires_matched_seed_mechanism_arrays(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    api=runpy.run_path(str(ROOT/'scripts/ppo_v4_locked.py'))
    with pytest.raises(ValueError,match='matched'):
        api['hierarchical_difference'](np.zeros((5,8)),np.zeros((3,8)))
