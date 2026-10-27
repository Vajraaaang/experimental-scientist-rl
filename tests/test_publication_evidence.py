"""Publication packaging must preserve data and locked-test overwrite guards."""
from pathlib import Path
import gzip
import importlib.util
import json
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_compressed_cost_evidence_matches_published_summary():
    with gzip.open(ROOT/'reports/ppo-cost-v5/fresh-locked/results.json.gz','rt') as handle:
        data=json.load(handle)
    summary=json.loads((ROOT/'reports/ppo-cost-v5/summary.json').read_text())
    assert data['freeze']['mechanisms']==2048
    assert data['differences']==summary['differences']
    assert data['efficiency_screen'] is False
    assert summary['screen_components']=={'cost_reduction':True,'quality_noninferiority':True,'net_return_improvement':False}


def test_locked_runner_refuses_compressed_completed_results(tmp_path,monkeypatch):
    scripts=str(ROOT/'scripts')
    monkeypatch.syspath_prepend(scripts)
    spec=importlib.util.spec_from_file_location('publication_cost_locked',ROOT/'scripts/ppo_cost_locked.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output=tmp_path.resolve()/'fresh-locked';output.mkdir()
    (output/'results.json.gz').write_bytes(gzip.compress(b'{}'))
    monkeypatch.setattr(module,'OUT',tmp_path.resolve())
    with pytest.raises(SystemExit,match='refusing rerun'):
        module.main()
