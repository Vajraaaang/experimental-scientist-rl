import json
import numpy as np
import pytest
from experimentlab.tasks import split_manifests, continuous_family
from experimentlab.cli import main, load_config, ROOT


def test_continuous_split_mechanisms_are_valid_and_disjoint():
    manifests = split_manifests()
    hashes = [set(t['mechanism_hash'] for t in manifests[s]) for s in ('train','validation','test')]
    assert not (hashes[0]&hashes[1] or hashes[0]&hashes[2] or hashes[1]&hashes[2])
    assert all(len(h) == 128 for h in hashes)
    assert max(np.abs(m.W).sum(axis=1).max() for m in continuous_family(123, 100)) <= 3.2


def test_cli_diagnostic_and_strict_json(tmp_path, capsys):
    assert main(['validate-science', '--output', str(tmp_path.resolve())]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['branching']['adaptive_accuracy'] == 1
    assert (tmp_path.resolve()/'diagnostics.json').is_file()


def test_locked_test_evaluation_cannot_run_without_freeze(capsys):
    with pytest.raises(SystemExit) as exc:
        main(['evaluate', '--checkpoint', '/private/tmp/unqualified.pt', '--suite', 'test-id'])
    assert exc.value.code == 2
    assert 'not qualified' in capsys.readouterr().err


def test_config_rejects_unknown_or_inconsistent_fields(tmp_path):
    import yaml
    data = load_config(ROOT/'configs/pilot.yaml')
    data['unrecognized_option'] = True
    path = tmp_path.resolve()/'bad.yaml'; path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='keys differ'): load_config(path)
    del data['unrecognized_option']; data['planner_draws'] = 0
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='planner_draws'): load_config(path)
