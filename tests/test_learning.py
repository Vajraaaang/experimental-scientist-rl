import copy
import numpy as np
import torch
import pytest
from experimentlab.learning import ActorCritic, masked_distribution, episode_batch, ppo_update, returns_gae, save_checkpoint, load_checkpoint


def example():
    rng = np.random.default_rng(7)
    return [{'tokens': rng.normal(size=(n, 75)).astype(np.float32),
             'masks': np.ones((n, 45), dtype=bool), 'actions': np.zeros(n, dtype=np.int64),
             'rewards': np.array([-.01]*(n-1)+[-.2], dtype=np.float32),
             'old_logp': np.full(n, -np.log(45), dtype=np.float32),
             'old_values': np.zeros(n, dtype=np.float32)} for n in (2, 5)]


def test_padding_masks_do_not_change_valid_recurrent_outputs():
    torch.manual_seed(8)
    m = ActorCritic(75, 45, hidden=32, kind='recurrent')
    b = episode_batch(example(), 'cpu')
    logits, values = m(b['tokens'])
    solo, _ = m(b['tokens'][:1, :2])
    assert torch.allclose(logits[:1, :2], solo, atol=1e-6)
    masks = b['masks'].clone(); masks[0, 2:] = False
    dist = masked_distribution(logits, masks, b['valid'])
    assert torch.isfinite(dist.log_prob(b['actions'])).all()
    assert dist.probs[0, 2:, 0].eq(1).all()
    assert m(b['tokens'])[0].equal(logits)  # no memory crosses calls


def test_action_masks_and_finite_whole_episode_update():
    b = episode_batch(example(), 'cpu')
    b['masks'][:, :, 3:] = False
    for kind in ('history', 'recurrent'):
        m = ActorCritic(75, 45, hidden=32, kind=kind)
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
        with torch.no_grad():
            logits, v = m(b['tokens'])
            d = masked_distribution(logits, b['masks'], b['valid'])
            b['old_logp'] = d.log_prob(b['actions'])
            b['old_values'] = v
        metrics = ppo_update(m, optimizer, b)
        assert all(np.isfinite(v) for v in metrics.values())
        assert metrics['initial_logp_error'] < 1e-6
        assert all((d.sample() < 3)[b['valid']])


def test_true_terminal_and_external_truncation_have_different_bootstrap():
    r = torch.tensor([1.])
    v = torch.tensor([2.])
    assert returns_gae(r, v, terminated=True, last_value=5.)[1].item() == 1.
    assert returns_gae(r, v, terminated=False, last_value=5.)[1].item() == 6.


def test_collector_records_terminal_flags_and_batch_rejects_external_truncation():
    from experimentlab.learning import collect_episode
    from experimentlab.envs import PilotConfig,ScientistEnv
    from experimentlab.tasks import finite_family,make_task
    config=PilotConfig(budget=0.)
    env=ScientistEnv(make_task(finite_family()[0],941,config),config)
    records=collect_episode(ActorCritic(),env,'cpu')
    assert records['terminated']==[True] and records['truncated']==[False]
    bad=example()[0]
    bad['terminated']=[False,False];bad['truncated']=[False,True]
    with pytest.raises(ValueError,match='complete'):
        episode_batch([bad],'cpu')


def test_checkpoint_cpu_random_and_optimizer_continuation(tmp_path):
    torch.manual_seed(5)
    model = ActorCritic(75, 45, hidden=32, kind='recurrent')
    opt = torch.optim.Adam(model.parameters())
    b = episode_batch(example(), 'cpu'); ppo_update(model, opt, b)
    filename = tmp_path.resolve() / 'checkpoint.pt'
    save_checkpoint(filename, model, opt, {'episodes': 7})
    a = torch.rand(4)
    metrics = ppo_update(model, opt, b)
    expected = copy.deepcopy(model.state_dict())
    other = ActorCritic(75, 45, hidden=32, kind='recurrent')
    other_opt = torch.optim.Adam(other.parameters())
    state = load_checkpoint(filename, other, other_opt)
    assert state['episodes'] == 7 and torch.equal(torch.rand(4), a)
    ppo_update(other, other_opt, b)
    assert all(torch.equal(other.state_dict()[k], v) for k, v in expected.items())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='native MPS unavailable in this process')
def test_python_float_lists_transfer_directly_to_mps_float32():
    episodes = example()
    for e in episodes:
        e['old_values'] = [0.] * len(e['actions'])
        e['old_logp'] = [-float(np.log(45))] * len(e['actions'])
    batch = episode_batch(episodes, 'mps')
    assert batch['old_values'].dtype == torch.float32
    model = ActorCritic(hidden=32, kind='recurrent').to('mps')
    metrics = ppo_update(model, torch.optim.Adam(model.parameters()), batch)
    assert np.isfinite(metrics['loss'])
