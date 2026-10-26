"""Native CLI. All filesystem destinations resolve to absolute paths."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)): return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, data):
    path = Path(path).expanduser().resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False)+'\n')


def code_hash():
    digest = hashlib.sha256()
    for file in sorted((ROOT / 'src').rglob('*.py')):
        digest.update(file.name.encode()); digest.update(file.read_bytes())
    return digest.hexdigest()


def load_config(path):
    data = yaml.safe_load(Path(path).expanduser().resolve().read_text())
    if not isinstance(data, dict) or data.get('schema_version') != 1: raise ValueError('unsupported configuration schema')
    if 'budgets' in data:
        expected = {'schema_version','seed','budgets','noise_grid','offset_grid','cost_weights','primary_noise',
                    'primary_cost_weight','max_experiments','cultures','final_cultures','dt','planner_draws',
                    'fixed_tuning_tasks','validation_tasks','bootstrap_samples','fixed_beam_width','adaptive_gap_threshold','scope'}
        from .envs import PilotConfig
        if set(data) != expected: raise ValueError(f'pilot configuration keys differ: {set(data)^expected}')
        for key in ('planner_draws','fixed_tuning_tasks','validation_tasks','bootstrap_samples','fixed_beam_width'):
            if not isinstance(data[key], int) or isinstance(data[key], bool) or data[key] < 1: raise ValueError(f'invalid {key}')
        if data['max_experiments'] > 4: raise ValueError('bounded pilot allows at most four assays')
        if not 0 < data['adaptive_gap_threshold'] <= 1: raise ValueError('invalid adaptive threshold')
        if data['primary_noise'] not in data['noise_grid'] or data['primary_cost_weight'] not in data['cost_weights']:
            raise ValueError('primary setting must be in declared grids')
        for budget in data['budgets']:
            for noise in data['noise_grid']:
                for offset in data['offset_grid']:
                    for cost in data['cost_weights']:
                        PilotConfig(budget=budget, readout_sd=noise, offset_sd=offset, cost_weight=cost,
                                    max_experiments=data['max_experiments'], cultures=data['cultures'],
                                    final_cultures=data['final_cultures'], dt=data['dt'])
    else:
        expected = {'schema_version','seed','device','hidden_size','benchmark_episodes','benchmark_repeats',
                    'smoke_training_episodes','smoke_teacher_episodes','smoke_imitation_epochs','max_experiments','initial_transition_cap'}
        if set(data) != expected: raise ValueError(f'development configuration keys differ: {set(data)^expected}')
        if data['device'] not in ('auto','cpu','mps'): raise ValueError('invalid device')
        for key in expected-{'schema_version','seed','device'}:
            if not isinstance(data[key],int) or isinstance(data[key],bool) or data[key]<1: raise ValueError(f'invalid {key}')
        if data['max_experiments'] != 4 or data['hidden_size'] != 64:
            raise ValueError('first measured learning/compute release uses four assays and hidden size64')
        if data['smoke_training_episodes'] > 128 or data['smoke_teacher_episodes'] > 128:
            raise ValueError('smoke exceeds distinct development mechanism pool')
    return data


def validate_science():
    from .diagnostics import numerical_audit, branching_audit
    from .worlds import fixture, integrate, ode_endpoint
    m = fixture(1); x = np.full(4, .5); u = np.array([.8,0,0,0])
    exact = ode_endpoint(m, x, u, 6)
    return {'diagnostic_pair': numerical_audit(), 'branching': branching_audit(),
            'integration': {'method': 'exponential Euler', 'dtype': 'float64',
                            'dt': .05, 'max_error_vs_DOP853': float(np.max(abs(integrate(m,x,u,6)-exact))),
                            'dt_half_max_error': float(np.max(abs(integrate(m,x,u,6,.025)-exact)))}}


def feasibility(config, output, quick=False):
    from .envs import PilotConfig
    from .tasks import finite_family, split_manifests
    from .inference import PredictionBank
    from .evaluation import run_comparison
    from .compute import peak_memory
    from .audits import generous_audit
    start = time.perf_counter(); output = Path(output).resolve()
    source_hash = hashlib.sha256((ROOT/'autonomous_scientist_build_spec.md').read_bytes()).hexdigest()
    write_json(output/'protocol.json', {'config': config, 'code_hash': code_hash(), 'source_hash': source_hash,
                                      'test_opened': False, 'mode': 'quick_debug' if quick else 'development_feasibility'})
    write_json(ROOT/'manifests/continuous-splits.json', split_manifests())
    science = validate_science(); write_json(output/'diagnostics.json', science)
    if not quick:
        print('Reconstructing generous-budget audit...', flush=True)
        audit = generous_audit(); write_json(output/'generous-audit.json', audit)
    scenarios = []
    for budget in config['budgets']:
        for noise in config['noise_grid']:
            scenarios.append((budget, noise, 0., config['primary_cost_weight']))
        if not quick:
            for offset in config['offset_grid']:
                if offset != 0: scenarios.append((budget, config['primary_noise'], offset, config['primary_cost_weight']))
            for cost in config['cost_weights']:
                if cost != config['primary_cost_weight']: scenarios.append((budget, config['primary_noise'], 0., cost))
    results = []
    for budget, noise, offset, cost in scenarios:
        name = f'budget{budget}-noise{noise}-offset{offset}-cost{cost}'
        print(f'Running {name}...', flush=True)
        settings = PilotConfig(budget=budget, readout_sd=noise, offset_sd=offset, cost_weight=cost,
                               max_experiments=config['max_experiments'], cultures=config['cultures'],
                               final_cultures=config['final_cultures'], dt=config['dt'])
        bank = PredictionBank(finite_family(), settings)
        result = run_comparison(bank, 8 if quick else config['fixed_tuning_tasks'],
                                16 if quick else config['validation_tasks'], seed=config['seed'],
                                draws=4 if quick else config['planner_draws'], beam=config['fixed_beam_width'],
                                bootstrap_samples=config['bootstrap_samples'], gap_threshold=config['adaptive_gap_threshold'])
        result['name'] = name
        write_json(output/f'{name}.json', result)
        write_json(output/f'{name}-replay.json', result['example'])
        results.append(result)
        print(json.dumps({'scenario': name, 'gate_B': result['gate_B'], 'gate_C_screen': result['gate_C_screen'],
                          'fixed': result['summary']['fixed'], 'voi': result['summary']['voi']}), flush=True)
    final = {'config': config, 'quick_debug': quick, 'code_hash': code_hash(), 'source_hash': source_hash,
             'results': results, 'seconds': time.perf_counter()-start, 'peak_process_rss_bytes': peak_memory(),
             'test_opened': False}
    write_json(output/'feasibility.json', final)
    generate_report(output)
    return {'output': str(output), 'scenarios': len(results), 'seconds': final['seconds'],
            'scientific_adaptive_advantage_established': False,
            'reason': 'finite-support screening only; continuous mechanism holdouts and inference validation remain gated'}


def generate_report(output):
    output = Path(output).resolve()
    data = json.loads((output/'feasibility.json').read_text())
    lines = ['# Measured feasibility report', '',
             'This is a bounded synthetic development pilot. Locked test results have not been opened.', '',
             'The two-mechanism diagnostic and branching toy establish implementation controls, not adaptive RL value.', '',
             '| Budget | Noise SD | Offset SD | Cost weight | Method | RMSE | Variable cost | Return | Online s/episode |',
             '|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for result in data['results']:
        c = result['config']
        for method, row in result['summary'].items():
            lines.append(f"|{c['budget']}|{c['readout_sd']}|{c['offset_sd']}|{c['cost_weight']}|{method}|{row['aggregate_rmse']:.5f}|{row['mean_variable_cost']:.3f}|{row['mean_return']:.5f}|{row['mean_online_seconds']:.5f}|")
    lines.extend(['', '## Gate evidence', '',
                  'Finite-support development episodes share possible hidden mechanisms. Mechanism-group bootstrap intervals quantify this support only; they do not establish continuous or structural generalization.', ''])
    for r in data['results']:
        lines.append(f"- {r['name']}: information gate B={r['gate_B']}; finite-support C screen={r['gate_C_screen']}; raw gap recovery={r['gap_recovery']:.3f}; adaptive-minus-fixed return {r['adaptive_utility_difference']}; error reduction {r['adaptive_error_reduction']}.")
    lines.extend(['', '## Interpretation and limits', '',
                  'The no-experiment method is prior-informed. Fixed, random, EIG and VOI use the same final posterior estimator. Additive identification changes the estimator and is shown separately. All methods share setup cost3 outside variable budget. The private nominal reference has privileged mechanism access and zero variable cost; it is not an attainable baseline.', '',
                  'The finite prior is explicitly conditioned on the same nominal admission rule used for tasks. Nonzero culture variation uses an approximate Gaussian likelihood, not an exact posterior. A failed one-step planner is not evidence against all adaptive experimental design.', '',
                  'Imitation/PPO execution checks and complete CPU/MPS benchmarks are recorded separately. Tiny smoke results cannot establish useful learned scientific behavior, PPO added value, or multiseed significance. Larger scientific training is withheld until development gates justify it.', '',
                  'Strong fixed schedules are preserved, including full-dose outcome identification. A documented development revision must address a demonstrated scientific limitation without weakening these baselines. Continuous-parameter, topology and range holdouts remain future gates; no locked-test outcome may guide revisions.', '',
                  f"Total feasibility wall time: {data['seconds']:.2f}s. Peak process RSS: {data['peak_process_rss_bytes']/2**30:.3f} GiB.",
                  f"Code content hash: {data['code_hash']}", f"Specification SHA256: {data['source_hash']}", ''])
    (output/'feasibility-report.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='experimentlab')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('doctor', 'validate-science', 'benchmark', 'feasibility', 'baselines', 'train', 'evaluate', 'replay', 'report'):
        p = sub.add_parser(name)
        p.add_argument('--output', default=str(ROOT/'reports/latest'))
        if name in ('benchmark','feasibility','baselines','train'): p.add_argument('--config', default=str(ROOT/'configs'/('pilot.yaml' if name in ('feasibility','baselines') else 'development.yaml')))
        if name in ('feasibility','baselines'): p.add_argument('--quick', action='store_true')
        if name in ('validate-science','baselines','evaluate'): p.add_argument('--suite', default='diagnostic')
        if name == 'train':
            p.add_argument('--device', choices=('cpu','mps','auto'), default='cpu')
            p.add_argument('--smoke', action='store_true', help='explicit bounded learning execution check')
        if name == 'evaluate': p.add_argument('--checkpoint', required=True)
        if name == 'replay': p.add_argument('--episode', required=True)
        if name == 'report': p.add_argument('--run-directory', required=True)
    args = parser.parse_args(argv); output = Path(args.output).expanduser().resolve()
    if args.command == 'doctor':
        from .compute import doctor
        result = doctor(); write_json(output/'hardware.json', result)
    elif args.command == 'validate-science':
        if args.suite == 'diagnostic':
            result = validate_science(); write_json(output/'diagnostics.json', result)
        elif args.suite == 'inference':
            from .checks import predictive_check, planner_draw_check
            result = {'predictive': [predictive_check(0.), predictive_check(.03)], 'planner_draws': planner_draw_check()}
            write_json(output/'inference-checks.json', result)
        else: parser.error('available science suites: diagnostic, inference')
    elif args.command == 'benchmark':
        from .compute import benchmark
        config = load_config(args.config)
        result = benchmark(config['benchmark_episodes'], config['benchmark_repeats']); write_json(output/'benchmark.json', result)
    elif args.command in ('feasibility','baselines'):
        if args.command == 'baselines' and args.suite == 'continuous-validation':
            from .checks import continuous_revision
            result = continuous_revision(); write_json(output/'continuous-development.json', result)
        else:
            if args.command == 'baselines' and args.suite not in ('diagnostic','validation'): parser.error('feasibility does not open locked test')
            result = feasibility(load_config(args.config), output, args.quick)
    elif args.command == 'train':
        from .experiments import learning_smoke
        from .compute import benchmark
        config = load_config(args.config)
        if not args.smoke: parser.error('scientific training is gated; use --smoke for the bounded execution check')
        device = args.device
        if device == 'auto': device = benchmark(config['benchmark_episodes'], config['benchmark_repeats'])['selected_device']
        result = learning_smoke(output, device, config['smoke_training_episodes'], config['smoke_teacher_episodes'], config['smoke_imitation_epochs'])
        write_json(output/'learning-smoke.json', result)
    elif args.command == 'evaluate':
        if args.suite.startswith('test'): parser.error('locked evaluation not qualified: freeze and development gates required')
        if args.suite not in ('diagnostic','validation'): parser.error('only development validation is available')
        from .experiments import evaluate_checkpoint
        result = evaluate_checkpoint(args.checkpoint)
        write_json(output/'checkpoint-validation.json', result)
    elif args.command == 'replay':
        result = json.loads(Path(args.episode).expanduser().resolve().read_text())
    else:
        generate_report(args.run_directory); result = {'report': str(Path(args.run_directory).expanduser().resolve()/'feasibility-report.md')}
    print(json.dumps(clean(result), indent=2, allow_nan=False))
    return 0
