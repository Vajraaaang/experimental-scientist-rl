from experimentlab.compute import _command


def test_optional_machine_inspection_handles_missing_platform_commands(tmp_path):
    result = _command([str(tmp_path.resolve() / 'unavailable-platform-command')])
    assert isinstance(result, dict)
    assert 'unavailable' in result
