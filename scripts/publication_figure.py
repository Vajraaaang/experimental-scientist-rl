"""Render the recorded paired development comparison; no new experiments."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runs' / 'matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    study = ROOT / 'reports' / 'unfiltered-v3'
    baselines = json.loads((study / 'matched-baselines.json').read_text())['summary']
    initial = json.loads((study / 'initial-learning-completed.json').read_text())['records']
    imitation = json.loads((study / 'converged-imitation' / 'results.json').read_text())
    fine_tune = json.loads((study / 'strong-finetune' / 'results.json').read_text())
    groups = [
        ('Prior', [baselines['prior']['aggregate_rmse']]),
        ('Fixed', [baselines['fixed']['aggregate_rmse']]),
        ('VOI planner', [baselines['voi']['aggregate_rmse']]),
        ('Full-history PPO', [v['validation']['aggregate_rmse'] for k, v in initial.items() if k.startswith('history-ppo')]),
        ('GRU PPO', [v['validation']['aggregate_rmse'] for k, v in initial.items() if k.startswith('recurrent-ppo')]),
        ('GRU imitation', [v['validation']['aggregate_rmse'] for k, v in imitation.items() if k.startswith('recurrent')]),
        ('GRU imitation + PPO', [v['validation']['aggregate_rmse'] for v in fine_tune.values()]),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = ['#94a3b8', '#64748b', '#0f766e', '#c2410c', '#ea580c', '#2563eb', '#7c3aed']
    for x, ((label, values), color) in enumerate(zip(groups, colors)):
        ax.bar(x, sum(values) / len(values), color=color, width=.65, alpha=.8)
        if len(values) > 1:
            ax.scatter([x - .1, x, x + .1][:len(values)], values, s=24, color='#0f172a', zorder=3)
    ax.set_xticks(range(len(groups)), [label for label, _ in groups], rotation=18, ha='right')
    ax.set_ylabel('Final target RMSE (lower is better)')
    ax.set_ylim(0, .09)
    ax.set_title('Unfiltered synthetic task: paired development validation')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.01, .01, '128 mechanisms; three training seeds shown as dots. Fixed is the matched-slice tuned design. Locked test not evaluated.', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    destination = ROOT / 'docs' / 'figures'
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / 'rl-comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
