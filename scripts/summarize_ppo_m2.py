"""Regenerate descriptive M2 tables and a figure from existing public evidence.

No training, task generation, checkpoint access, model selection or hosted calls.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('refusing to replace a report directory')
    inputs = {}

    def read(relative):
        path = root / relative
        inputs[relative] = sha256(path)
        return json.loads(path.read_text())

    base = 'reports/jev-rl-m2/'
    ledger = read(base + 'race-01/ledger.json')
    selection = read(base + 'race-01/selection.json')
    hardware = read(base + 'hardware-sweep/summary.json')
    confirmation = read(base + 'confirmation-01/summary.json')
    screens = {}
    for group in ('architecture', 'updates', 'v6_screen'):
        cohorts = defaultdict(list)
        for job in ledger['groups'][group]:
            result = read(base + f'race-01/jobs/{job}/result.json')
            if result['status'] != 'completed':
                raise ValueError('incomplete race evidence')
            cohorts[result['name']].append(result)
        screens[group] = {}
        for name, rows in sorted(cohorts.items()):
            if sorted(r['seed'] for r in rows) != [20, 21]:
                raise ValueError('expected both prespecified screen seeds')
            snapshots = {}
            for kind in ('equal_interaction', 'equal_wall'):
                values = [r['snapshots'][kind] for r in rows]
                if any(v is None for v in values):
                    raise ValueError('missing prespecified snapshot')
                snapshots[kind] = {
                    'mean_return': statistics.fmean(v['metrics']['mean_return'] for v in values),
                    'mean_mse': statistics.fmean(v['metrics']['mean_mse'] for v in values),
                    'mean_variable_cost': statistics.fmean(v['metrics']['mean_variable_cost'] for v in values),
                    'mean_elapsed_seconds': statistics.fmean(v['elapsed_seconds'] for v in values),
                    'mean_new_training_transitions': statistics.fmean(v['training_transitions'] for v in values),
                    'phases': [v['phase'] for v in values],
                    'seeds': [r['seed'] for r in rows],
                }
            screens[group][name] = snapshots
    sustained = {r['config']['backend']: r['totals'] for r in hardware['runs']
                 if r['name'].startswith('sustained-')}
    if set(sustained) != {'cpu', 'mps', 'split'}:
        raise ValueError('missing hardware comparator')
    data = {'schema': 'ppo-m2-descriptive-report-v1', 'source_sha256': sha256(__file__),
            'inputs': inputs, 'screens': screens, 'sustained_hardware': sustained,
            'race': {k: selection[k] for k in ('completed_jobs', 'total_jobs',
                'all_observed_training_transitions', 'all_observed_validation_transitions',
                'elapsed_seconds', 'architecture', 'update_recipe', 'v6_recipe')},
            'confirmation': confirmation,
            'scope': 'Existing development evidence only; two-seed screen means have no uncertainty claim; no new selection or final evaluation.'}
    lines = ['# M2 measured comparisons', '', data['scope'], '',
             'All return values use the original reward; higher is better. Raw MSE and variable assay cost are lower-is-better. API dollars are a separate unit.', '',
             '## Sustained full-loop hardware', '',
             '| Backend | Training transitions / cycle second | Training transitions / whole-run second | Wall seconds |',
             '|---|---:|---:|---:|']
    for name, row in sustained.items():
        lines.append(f"| {name} | {row['training_transitions_per_cycle_second']:.2f} | {row['training_transitions_per_wall_second']:.2f} | {row['wall_seconds']:.2f} |")
    for group, cohorts in screens.items():
        lines += ['', '## ' + group.replace('_', ' ').title(), '',
                  'Means over seeds 20 and 21 on the tuning split. Endpoint snapshots, not the best intermediate checkpoint. Continuation transitions exclude inherited pretraining.', '',
                  '| Configuration | Snapshot | Return | Raw MSE | Assay cost | New transitions | Elapsed seconds | Phase(s) |',
                  '|---|---|---:|---:|---:|---:|---:|---|']
        for name, snapshots in cohorts.items():
            for kind, row in snapshots.items():
                lines.append(f"| {name} | {kind} | {row['mean_return']:.6f} | {row['mean_mse']:.8f} | {row['mean_variable_cost']:.4f} | {row['mean_new_training_transitions']:.1f} | {row['mean_elapsed_seconds']:.3f} | {', '.join(row['phases'])} |")
    lines += ['', 'The 240-second architecture comparison includes the selector curriculum. GRU512 was still in that phase at both wall snapshots and its unforced policy stopped immediately. GRU128 had the best two-seed return at this shorter wall budget. GRU512 had the best two-seed endpoint return at approximately two million transitions. This does not make one configuration best for every compute budget.', '',
              'The update screen uses a shared trained GRU512 parent for each seed. Its 70-second comparison excludes inherited training time; it is not a scratch-from-zero comparison. Selection itself follows the frozen race rules and may use an eligible intermediate checkpoint.', '',
              '## Five-seed development confirmation', '',
              '| Method | Episodes | Return | Raw MSE | Assay cost |', '|---|---:|---:|---:|---:|']
    for name, row in confirmation['cohorts'].items():
        lines.append(f"| {name} | {row['completed_count']}/{row['expected_count']} | {row['mean_return_']:.6f} | {row['mean_final_mse']:.8f} | {row['mean_variable_cost']:.4f} |")
    lines += ['', '| Prespecified comparison | Return delta | 95% interval | 98.333% interval | Gain screen |', '|---|---:|---|---|---|']
    for name, row in confirmation['comparisons'].items():
        metric = row['metrics']['return_']
        fmt = lambda values: '[' + ', '.join(f'{v:.6f}' for v in values) + ']'
        lines.append(f"| {name} | {metric['difference']:.6f} | {fmt(metric['ci95'])} | {fmt(metric['ci98_333333'])} | {row['scientific_return_gain']} |")
    lines += ['', 'All three prespecified gain screens failed. The selected GRU512 and historical GRU64 imitation systems differ in capacity and training history; this is not an isolated PPO-versus-imitation algorithm comparison. Confirmation uses 128 familiar-family development mechanisms and five learned seeds, not the unopened final split.', '',
              '![Measured M2 comparisons](m2-comparison.png)', '']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), layout='constrained')
    names = ['cpu', 'split', 'mps']
    axes[0].barh(names, [sustained[n]['training_transitions_per_wall_second'] for n in names], color=['#217a6b', '#8b9bad', '#8b9bad'])
    axes[0].set(title='Full-loop hardware: GRU64', xlabel='Training transitions / whole-run second')
    labels = [f'GRU{h}' for h in (64, 128, 256, 512)]
    for key, label, color in [('equal_interaction', '~2M transitions', '#217a6b'), ('equal_wall', '~240 seconds', '#be6b3b')]:
        values = [screens['architecture'][f'arch-recurrent-h{h}'][key]['mean_return'] for h in (64, 128, 256, 512)]
        axes[1].plot(labels, values, 'o-', label=label, color=color)
    axes[1].set(title='Architecture screen: two-seed means', ylabel='Original return (higher is better)')
    axes[1].legend(frameon=False)
    pairs = ['scratch_selected-minus-ppo', 'v6_continued-minus-ppo', 'imitation_ppo_continued-minus-imitation_ppo']
    labels = ['Selected PPO − v6', 'Continued v6 − v6', 'Continued IM→PPO − IM→PPO']
    for y, name in enumerate(pairs):
        metric = confirmation['comparisons'][name]['metrics']['return_']
        axes[2].plot(metric['ci98_333333'], [y, y], color='#8b9bad', lw=2)
        axes[2].plot(metric['ci95'], [y, y], color='#217a6b', lw=5)
        axes[2].plot(metric['difference'], y, 'o', color='#102d3f', ms=5)
    axes[2].axvline(0, color='#6c7780', ls='--', lw=1)
    axes[2].set(yticks=range(3), yticklabels=labels, xlabel='Paired return difference', title='Confirmation: all gain screens fail')
    axes[2].invert_yaxis()
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('M2 development evidence • CPU favored • compute budget changes the ranking', fontsize=14)
    output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (output / 'COMPARISONS.md').write_text('\n'.join(lines))
    fig.savefig(output / 'm2-comparison.png', dpi=180)
    plt.close(fig)
    return {'output': str(output), 'input_files': len(inputs), 'new_training_steps': 0, 'new_hosted_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output), indent=2))
