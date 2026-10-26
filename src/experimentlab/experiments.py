"""Bounded learning execution checks; not a research-scale training claim."""
from pathlib import Path
import time
import numpy as np
import torch
from .envs import PilotConfig, ScientistEnv
from .tasks import continuous_family, mechanism_hash
from .inference import PredictionBank, Belief
from .baselines import DecisionPlanner
from .evaluation import independent_continuous_tasks, summarize
from .learning import (ActorCritic, collect_episode, episode_batch, imitation_update,
                       ppo_update, save_checkpoint, load_checkpoint, delayed_memory_check)


def learning_smoke(output, device='cpu', episodes=32, teacher_episodes=32, imitation_epochs=20):
    torch.set_num_threads(1)
    config = PilotConfig()
    bank = PredictionBank(continuous_family(6101, 64), config)
    train = independent_continuous_tasks(bank, 'train', max(episodes, teacher_episodes), 2311)
    validation = independent_continuous_tasks(bank, 'validation', 32, 2312)
    results = {}
    # Full-history baseline is executed before recurrent learning.
    for kind, mode in [('history', 'ppo'), ('recurrent', 'imitation'), ('recurrent', 'ppo'), ('recurrent', 'imitation_ppo')]:
        torch.manual_seed(19)
        model = ActorCritic(kind=kind, hidden=64).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        start = time.perf_counter(); teacher_seconds = 0.; teacher_transitions = 0
        transitions = 0; log = []
        if mode in ('imitation', 'imitation_ppo'):
            demos = []; teacher_start = time.perf_counter()
            for task in train[:teacher_episodes]:
                planner = DecisionPlanner(bank, 'voi', draws=8)
                belief = Belief(bank, task.target, admitted=True)
                seen = [0]
                rng = np.random.default_rng(task.seed)
                def teacher(obs):
                    while seen[0] < len(obs['history']):
                        belief.update(obs['history'][seen[0]]); seen[0] += 1
                    return planner.choose(obs, belief, rng)
                demos.append(collect_episode(model, ScientistEnv(task, config), device, teacher=teacher))
            teacher_seconds = time.perf_counter()-teacher_start
            teacher_transitions = sum(len(d['actions']) for d in demos)
            batch = episode_batch(demos, device)
            log.append(imitation_update(model, opt, batch, epochs=imitation_epochs))
        if mode in ('ppo', 'imitation_ppo'):
            for offset in range(0, episodes, 8):
                rollouts = [collect_episode(model, ScientistEnv(task, config), device) for task in train[offset:offset+8]]
                transitions += sum(len(d['actions']) for d in rollouts)
                log.append(ppo_update(model, opt, episode_batch(rollouts, device)))
        training_seconds = time.perf_counter()-start
        rows = []
        for task in validation:
            env = ScientistEnv(task, config)
            tick = time.perf_counter()
            record = collect_episode(model, env, device, deterministic=True)
            row = dict(env.private_result)
            row.update(return_=sum(record['rewards']), assays=env.count, online_seconds=time.perf_counter()-tick)
            rows.append(row)
        name = f'{kind}_{mode}'
        checkpoint = Path(output).resolve() / f'{name}.pt'
        save_checkpoint(checkpoint, model, opt, {'episodes': episodes, 'transitions': transitions},
                        {'scope': 'execution_smoke_only', 'seed': 19, 'resolved_config':config.__dict__,
                         'bank_seed':6101,'train_pool_seed':7101,'validation_pool_seed':8101})
        results[name] = {'architecture': model.architecture, 'teacher_seconds': teacher_seconds,
                         'teacher_transitions': teacher_transitions, 'on_policy_transitions': transitions,
                         'training_seconds_including_teacher': training_seconds, 'updates': log,
                         'validation_summary': summarize(rows), 'checkpoint': str(checkpoint)}
    return {'scope': 'single_seed_tiny_execution_smoke_not_Gate_D_or_E_evidence', 'device': device,
            'mechanism_disjoint_train_validation': True, 'truth_inserted_in_bank': False,
            'train_mechanism_hashes': [mechanism_hash(t.mechanism) for t in train],
            'validation_mechanism_hashes': [mechanism_hash(t.mechanism) for t in validation],
            'bank_mechanism_hashes': [mechanism_hash(m) for m in bank.mechanisms], 'models': results,
            'delayed_memory': delayed_memory_check(device),
            'claims': {'useful_learned_scientific_strategy': 'not established', 'PPO_added_value': 'not established'}}


def evaluate_checkpoint(checkpoint, n=32):
    """CPU reference validation, including shared-estimator interpretation ablation."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('provenance',{}).get('scope') not in (None,'execution_smoke_only'):
        raise ValueError('Use the matched research evaluation script for this checkpoint to preserve its actual manifest and prior.')
    model = ActorCritic(**state['architecture'])
    load_checkpoint(checkpoint, model)
    model.eval(); torch.set_num_threads(1)
    config = PilotConfig()
    bank = PredictionBank(continuous_family(6101, 64), config)
    tasks = independent_continuous_tasks(bank, 'validation', n, 2312)
    learned_rows, estimator_rows = [], []
    for task in tasks:
        env = ScientistEnv(task, config); start = time.perf_counter()
        record = collect_episode(model, env, 'cpu', deterministic=True)
        row = dict(env.private_result)
        row.update(return_=sum(record['rewards']), assays=env.count, online_seconds=time.perf_counter()-start)
        learned_rows.append(row)
        replay = ScientistEnv(task, config); obs, _ = replay.reset()
        belief = Belief(bank, obs['target'], admitted=True); belief.update(obs['history'][0])
        total = 0.; start = time.perf_counter()
        for a in record['actions'][:-1]:
            obs, reward, _, _, _ = replay.step(a); total += reward
            belief.update(obs['history'][-1])
        _, reward, _, _, _ = replay.step(belief.commitment()); total += reward
        row = dict(replay.private_result)
        row.update(return_=total, assays=replay.count, online_seconds=time.perf_counter()-start)
        estimator_rows.append(row)
    return {'scope': 'mechanism_disjoint_development_validation_not_locked_test', 'checkpoint': str(checkpoint),
            'mechanisms': n, 'learned_commitment': summarize(learned_rows),
            'same_assays_shared_estimator_commitment': summarize(estimator_rows)}
