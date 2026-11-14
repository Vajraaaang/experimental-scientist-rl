"""Host-padded rollout packing and device-batched complete-episode GAE.

Only data movement and tensor bookkeeping live here. Scientific rewards,
features, legal actions, and the sampled/forced policy distinction are retained.
"""
import numpy as np
import torch

from .learning import TOKEN_SIZE


def pack_episodes(episodes, device):
    """Validate/pad on the host, then transfer each field once to ``device``.

    Legacy demonstrations may omit terminal flags, behavior biases, and the
    policy-step mask. Their defaults are complete termination, zero bias, and
    all valid decisions sampled. External truncation requires a different
    bootstrap contract and is rejected before touching the device.
    """
    if not episodes:
        raise ValueError('batch requires at least one complete terminal episode')
    lengths = [len(episode['actions']) for episode in episodes]
    if min(lengths) < 1 or max(lengths) > 5:
        raise ValueError('complete episodes must contain between one and five decisions')
    B, T = len(episodes), max(lengths)
    shapes = {'tokens': (B, T, TOKEN_SIZE), 'masks': (B, T, 45),
              'actions': (B, T), 'rewards': (B, T), 'old_logp': (B, T),
              'old_values': (B, T), 'terminated': (B, T), 'truncated': (B, T),
              'valid': (B, T), 'biases': (B, T, 45), 'policy_valid': (B, T)}
    bool_fields = {'masks', 'terminated', 'truncated', 'valid', 'policy_valid'}
    host = {key: np.zeros(shape, dtype=np.bool_ if key in bool_fields else
                          np.int64 if key == 'actions' else np.float32)
            for key, shape in shapes.items()}
    for i, (episode, n) in enumerate(zip(episodes, lengths)):
        defaults = {'terminated': [False] * (n - 1) + [True],
                    'truncated': [False] * n, 'biases': np.zeros((n, 45), np.float32),
                    'policy_valid': [True] * n}
        for key in shapes:
            if key == 'valid':
                host[key][i, :n] = True
                continue
            source = 'policy_steps' if key == 'policy_valid' else key
            values = episode.get(source, defaults.get(key))
            if values is None:
                raise ValueError(f'missing episode field: {source}')
            values = np.asarray(values, dtype=host[key].dtype)
            if values.shape != host[key][i, :n].shape:
                raise ValueError(f'invalid episode field shape: {source}')
            if values.dtype.kind == 'f' and not np.isfinite(values).all():
                raise ValueError(f'nonfinite episode field: {source}')
            host[key][i, :n] = values
        if host['truncated'][i, :n].any() or not host['terminated'][i, n - 1]:
            raise ValueError('PPO batches require complete terminal episodes; external truncation needs explicit bootstrapping')
        actions = host['actions'][i, :n]
        if ((actions < 0) | (actions >= 45)).any() or not host['masks'][i, np.arange(n), actions].all():
            raise ValueError('episode action is outside its legal mask')
    if not host['policy_valid'].any():
        raise ValueError('batch has no sampled policy decision')
    return {key: torch.from_numpy(values).to(device) for key, values in host.items()}


def whole_episode_advantages(batch, gae_lambda=1.):
    """Return detached gamma=1 advantages/targets for complete padded episodes.

    The only loop is over at most five decision indices; every operation spans
    all episodes. No episode lengths, flags, or scalar values leave the device.
    Host validation in ``pack_episodes`` establishes complete-terminal episodes
    and contiguous valid prefixes. Forced decisions still receive value targets.
    """
    rewards, values, valid = batch['rewards'], batch['old_values'], batch['valid']
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])
    nxt = torch.zeros_like(running)
    for t in reversed(range(rewards.shape[1])):
        delta = rewards[:, t] + nxt - values[:, t]
        running = torch.where(valid[:, t], delta + gae_lambda * running, 0.)
        advantages[:, t] = running
        nxt = torch.where(valid[:, t], values[:, t], 0.)
    targets = torch.where(valid, advantages + values, 0.)
    return advantages.detach(), targets.detach()
