"""Small float32 full-history and recurrent policies, whole-episode PPO.

No transition shuffling. Complete episodes only; checkpoints are at episode
boundaries. Recurrent state is rebuilt from its zero initial state per episode.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import platform
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

TOKEN_SIZE = 75


def token(obs):
    packet = obs['history'][-1]
    mask = packet['measured_mask']
    readouts = packet['readouts']
    mean = np.zeros(8); sd = np.zeros(8)
    mean[mask] = (readouts[:, mask].mean(axis=0) - .5) / .1
    sd[mask] = readouts[:, mask].std(axis=0) / .1
    action = np.zeros(46); action[packet['action_id'] + 1] = 1
    return np.concatenate([(obs['target'] - .5) / .08,
                           [obs['budget_ticks'] / 320, obs['experiment_count'] / 4, float(obs['reset'])],
                           mean, sd, mask.astype(float), action]).astype(np.float32)


class ActorCritic(nn.Module):
    def __init__(self, input_size=TOKEN_SIZE, actions=45, hidden=64, kind='history', horizon=5):
        super().__init__()
        if kind not in ('history', 'recurrent', 'latest'): raise ValueError('unknown model kind')
        self.kind, self.horizon = kind, horizon
        self.architecture = dict(input_size=input_size, actions=actions, hidden=hidden, kind=kind, horizon=horizon)
        if kind == 'history':
            self.encoder = nn.Sequential(nn.Linear(horizon * input_size + horizon, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        else:
            self.encoder = nn.Sequential(nn.Linear(input_size, hidden), nn.Tanh())
            if kind == 'recurrent': self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.actor = nn.Linear(hidden, actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, tokens):
        B, T, D = tokens.shape
        if T > self.horizon: raise ValueError('sequence exceeds configured horizon')
        if self.kind == 'history':
            # At decision t, include only tokens <=t. Explicit history-padding mask.
            padded = torch.nn.functional.pad(tokens, (0, 0, 0, self.horizon - T))
            causal = torch.arange(self.horizon, device=tokens.device)[None] <= torch.arange(T, device=tokens.device)[:, None]
            histories = padded[:, None] * causal[None, :, :, None]
            mask = causal[None].expand(B, -1, -1).to(tokens.dtype)
            encoded = self.encoder(torch.cat([histories.flatten(2), mask], dim=-1))
        else:
            encoded = self.encoder(tokens)
            if self.kind == 'recurrent': encoded, _ = self.gru(encoded)  # hidden=None resets each sequence
        return self.actor(encoded), self.critic(encoded).squeeze(-1)


def masked_distribution(logits, action_masks, valid=None):
    masks = action_masks.clone()
    if valid is not None:
        masks[~valid] = False
        masks[..., 0] |= ~valid  # defined dummy before constructing distribution
    if not masks.any(dim=-1).all(): raise ValueError('valid decision has no available actions')
    return Categorical(logits=logits.masked_fill(~masks, -1e9))


def returns_gae(rewards, values, terminated=True, last_value=0., gae_lambda=.95):
    advantage = torch.zeros_like(rewards)
    running = torch.zeros((), device=rewards.device)
    nxt = torch.zeros((), device=rewards.device) if terminated else torch.as_tensor(last_value, device=rewards.device)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + nxt - values[t]  # gamma=1
        running = delta + gae_lambda * running
        advantage[t] = running
        nxt = values[t]
    return advantage, advantage + values


def episode_batch(episodes, device):
    B, T = len(episodes), max(len(e['actions']) for e in episodes)
    shapes = {'tokens': (B, T, TOKEN_SIZE), 'masks': (B, T, 45), 'actions': (B, T),
              'rewards': (B, T), 'old_logp': (B, T), 'old_values': (B, T)}
    result = {k: torch.zeros(shape, dtype=torch.bool if k == 'masks' else torch.long if k == 'actions' else torch.float32, device=device)
              for k, shape in shapes.items()}
    result['valid'] = torch.zeros((B, T), dtype=torch.bool, device=device)
    for i, e in enumerate(episodes):
        n = len(e['actions']); result['valid'][i, :n] = True
        for k in shapes:
            result[k][i, :n] = torch.as_tensor(np.asarray(e[k]), dtype=result[k].dtype, device=device)
    return result


def ppo_update(model, optimizer, batch, epochs=2, clip=.2):
    valid = batch['valid']
    advantage, targets = torch.zeros_like(batch['rewards']), torch.zeros_like(batch['rewards'])
    for b in range(len(valid)):
        n = int(valid[b].sum())
        a, ret = returns_gae(batch['rewards'][b, :n], batch['old_values'][b, :n])
        advantage[b, :n], targets[b, :n] = a, ret
    advantage = advantage.detach(); targets = targets.detach()
    mean, sd = advantage[valid].mean(), advantage[valid].std(unbiased=False)
    advantage = (advantage - mean) / (sd + 1e-8)
    initial_error = None
    for _ in range(epochs):
        logits, values = model(batch['tokens'])
        dist = masked_distribution(logits, batch['masks'], valid)
        logp = dist.log_prob(batch['actions'])
        if initial_error is None: initial_error = float((logp[valid] - batch['old_logp'][valid]).abs().max().detach().cpu())
        ratio = torch.exp(logp - batch['old_logp'])
        surrogate = torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage)
        value_loss = ((values - targets)**2)[valid].mean()
        entropy = dist.entropy()[valid].mean()
        loss = -surrogate[valid].mean() + .5 * value_loss - .01 * entropy
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite PPO loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()
    return {'loss': float(loss.detach().cpu()), 'value_error': float(value_loss.detach().cpu()),
            'entropy': float(entropy.detach().cpu()), 'gradient_norm': float(grad.detach().cpu()),
            'kl': float((batch['old_logp'] - logp)[valid].mean().detach().cpu()),
            'initial_logp_error': initial_error}


def collect_episode(model, env, device, deterministic=False, teacher=None):
    obs, _ = env.reset()
    records = {k: [] for k in ('tokens', 'masks', 'actions', 'rewards', 'old_logp', 'old_values')}
    history = []
    while True:
        current = token(obs); history.append(current)
        with torch.no_grad():
            logits, values = model(torch.tensor(np.array(history)[None], dtype=torch.float32, device=device))
            mask = torch.as_tensor(obs['action_mask'][None, None], device=device)
            dist = masked_distribution(logits[:, -1:], mask)
            if teacher is not None: action = int(teacher(obs))
            else: action = int(dist.probs.argmax(-1).item() if deterministic else dist.sample().item())
            logp = dist.log_prob(torch.tensor([[action]], device=device)).item()
        records['tokens'].append(current)
        records['masks'].append(obs['action_mask'])
        records['actions'].append(action)
        records['old_logp'].append(logp)
        records['old_values'].append(values[0, -1].item())
        obs, reward, terminated, truncated, _ = env.step(action)
        records['rewards'].append(reward)
        if truncated: raise RuntimeError('external truncation: collector requires complete episodes')
        if terminated: break
    return records


def imitation_update(model, optimizer, batch, epochs=1):
    valid = batch['valid']
    for _ in range(epochs):
        logits, _ = model(batch['tokens'])
        dist = masked_distribution(logits, batch['masks'], valid)
        loss = -dist.log_prob(batch['actions'])[valid].mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()
    with torch.no_grad():
        logits, _ = model(batch['tokens'])
        dist = masked_distribution(logits, batch['masks'], valid)
        accuracy = (dist.probs.argmax(-1) == batch['actions'])[valid].float().mean()
    return {'cross_entropy': float(loss.detach().cpu()), 'training_action_accuracy': float(accuracy.cpu())}


def save_checkpoint(filename, model, optimizer, counters, provenance=None):
    filename = Path(filename).expanduser().resolve()
    filename.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for source in sorted((root/'src').rglob('*.py')):
        digest.update(source.name.encode()); digest.update(source.read_bytes())
    from .envs import catalogue
    metadata = {'code_hash': digest.hexdigest(), 'torch_version': torch.__version__,
                'python_version': platform.python_version(), 'architecture': platform.machine(),
                'device': str(next(model.parameters()).device),
                'simulator_version': 'exp-euler-v1',
                'action_catalogue_hash': hashlib.sha256(json.dumps([asdict(a) for a in catalogue()]).encode()).hexdigest(),
                'normalization': {'type':'fixed_public_constants','state_center':.5,'state_scale':.1,'target_scale':.08},
                **(provenance or {})}
    manifest_path=Path(metadata.get('manifest_path',root/'manifests/continuous-splits.json')).expanduser().resolve()
    metadata['manifest_path']=str(manifest_path)
    for name, path in [('dependency_lock_hash',root/'requirements.lock'), ('manifest_hash',manifest_path)]:
        metadata[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'architecture': model.architecture, 'counters': counters,
                'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
                'mps_rng': torch.mps.get_rng_state() if next(model.parameters()).device.type == 'mps' else None,
                'provenance': metadata, 'resume_boundary': 'complete episodes only'}, filename)


def load_checkpoint(filename, model, optimizer=None):
    state = torch.load(Path(filename).expanduser().resolve(), map_location=next(model.parameters()).device, weights_only=False)
    model.load_state_dict(state['model'])
    if optimizer is not None: optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'].cpu()); np.random.set_state(state['numpy_rng'])
    if state['mps_rng'] is not None and next(model.parameters()).device.type == 'mps': torch.mps.set_rng_state(state['mps_rng'].cpu())
    return state['counters']


def delayed_memory_check(device='cpu', updates=220):
    """No action/external-memory channel: recall first bit after distractors."""
    torch.manual_seed(913)
    model = ActorCritic(hidden=32, kind='recurrent').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.006)
    rng = np.random.default_rng(104)
    def data(n):
        x = rng.normal(0, .1, (n, 5, TOKEN_SIZE)).astype(np.float32)
        bit = rng.integers(0, 2, n)
        x[:, :, 0] = 0.; x[:, 0, 0] = bit * 2 - 1
        return torch.tensor(x, device=device), torch.tensor(bit, device=device)
    start = time.perf_counter()
    for _ in range(updates):
        x, y = data(64)
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(logits[:, -1, :2], y)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    x, y = data(512)
    with torch.no_grad(): accuracy = (model(x)[0][:, -1, :2].argmax(-1) == y).float().mean().item()
    return {'accuracy': accuracy, 'updates': updates, 'seconds': time.perf_counter() - start,
            'scope': 'Delayed-evidence implementation check, not scientific policy performance.'}
