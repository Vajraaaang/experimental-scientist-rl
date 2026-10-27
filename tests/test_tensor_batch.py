"""Whole-episode packing and tensor GAE must preserve the scalar PPO contract."""
import copy

import numpy as np
import pytest
import torch

from experimentlab.learning import episode_batch, returns_gae
from experimentlab.tensor_batch import pack_episodes, whole_episode_advantages


def episodes():
    rng = np.random.default_rng(807)
    records = []
    for index, n in enumerate((1, 5, 2, 4, 3)):
        actions = rng.integers(0, 45, size=n)
        masks = rng.random((n, 45)) > .4
        masks[np.arange(n), actions] = True
        record = {
            'tokens': rng.normal(size=(n, 75)).astype(np.float32),
            'masks': masks, 'actions': actions.tolist(),
            'rewards': rng.normal(0, .3, size=n).tolist(),
            'old_logp': rng.normal(-2, .2, size=n).tolist(),
            'old_values': rng.normal(0, .2, size=n).tolist(),
        }
        if index % 2:
            record.update(terminated=[False] * (n - 1) + [True], truncated=[False] * n,
                          biases=rng.normal(size=(n, 45)).astype(np.float32),
                          policy_steps=[False] * (n - 1) + [True])
        records.append(record)
    return records


def legacy_batch(records):
    # Keep the pre-tensorization packer as an independent CPU reference even
    # after ppo_v4.batch_episodes delegates to the new module.
    batch = episode_batch(records, 'cpu')
    batch['biases'] = torch.zeros((*batch['actions'].shape, 45))
    batch['policy_valid'] = batch['valid'].clone()
    for i, record in enumerate(records):
        n = len(record['actions'])
        if 'biases' in record:
            batch['biases'][i, :n] = torch.as_tensor(np.asarray(record['biases']), dtype=torch.float32)
        if 'policy_steps' in record:
            batch['policy_valid'][i, :n] = torch.as_tensor(record['policy_steps'], dtype=torch.bool)
    batch['policy_valid'] &= batch['valid']
    return batch


DEVICES = ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='native MPS unavailable in this process'))]


@pytest.mark.parametrize('device', DEVICES)
def test_every_packed_field_matches_legacy_with_mixed_lengths_and_defaults(device):
    data = episodes()
    expected, actual = legacy_batch(data), pack_episodes(data, device)
    assert expected.keys() == actual.keys()
    for key in expected:
        assert actual[key].dtype == expected[key].dtype, key
        assert actual[key].device.type == device, key
        assert torch.equal(actual[key].cpu(), expected[key]), key
    assert not actual['policy_valid'][1, :-1].any()
    assert actual['valid'][1].all()  # forced prefix still trains the critic


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('gae_lambda', [0., .95, 1.])
def test_tensor_gae_matches_complete_episode_scalar_returns(device, gae_lambda):
    data = episodes()
    batch = pack_episodes(data, device)
    expected = legacy_batch(data)
    advantages, targets = torch.zeros_like(expected['rewards']), torch.zeros_like(expected['rewards'])
    for i, record in enumerate(data):
        n = len(record['actions'])
        advantages[i, :n], targets[i, :n] = returns_gae(
            expected['rewards'][i, :n], expected['old_values'][i, :n], gae_lambda=gae_lambda)
    # PPO may replace old_values with a model forward, including padded slots.
    batch['old_values'] = batch['old_values'].clone().requires_grad_()
    with torch.no_grad():
        batch['old_values'][~batch['valid']] = 123.
    actual_a, actual_t = whole_episode_advantages(batch, gae_lambda)
    torch.testing.assert_close(actual_a.cpu(), advantages, atol=1e-6, rtol=0)
    torch.testing.assert_close(actual_t.cpu(), targets, atol=1e-6, rtol=0)
    assert not actual_a.requires_grad and not actual_t.requires_grad
    assert actual_a[~batch['valid']].eq(0).all()
    assert actual_t[~batch['valid']].eq(0).all()


@pytest.mark.parametrize('case', ['empty', 'empty_episode', 'truncated', 'unterminated',
                                  'all_forced', 'shape', 'nonfinite', 'illegal_action', 'horizon'])
def test_invalid_input_is_rejected_on_host_before_device_access(case):
    data = copy.deepcopy(episodes())
    if case == 'empty':
        data = []
    elif case == 'empty_episode':
        data[0]['actions'] = []
    elif case == 'truncated':
        data[0]['truncated'] = [True]
    elif case == 'unterminated':
        data[0]['terminated'] = [False]
    elif case == 'all_forced':
        for record in data:
            record['policy_steps'] = [False] * len(record['actions'])
    elif case == 'shape':
        data[0]['biases'] = np.zeros((1, 44))
    elif case == 'nonfinite':
        data[0]['rewards'] = [float('nan')]
    elif case == 'illegal_action':
        data[0]['masks'][0, data[0]['actions'][0]] = False
    elif case == 'horizon':
        data[0]['actions'] = [0] * 6
    # A device that cannot possibly be resolved proves validation runs first.
    with pytest.raises(ValueError):
        pack_episodes(data, 'not-a-device')
