"""Render the published hosted-development results from their bound artifacts."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import fmean

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = Path('reports/jev-rl-m2')
INK = '#edf4ff'
MUTED = '#a6b7cc'
CYAN = '#74e5f2'
LILAC = '#c4acf9'
AMBER = '#ffbc8d'
SURFACE = '#131e30'
GRID = '#2a3c53'
ZERO = '#8c9fb7'
MONO = 'DejaVu Sans Mono'


def signed(value, digits):
    return f'{value:+.{digits}f}'.replace('-', '−')


def load_evidence(root):
    paths = {
        'summary': BASE / 'hosted-analysis-01/summary.json',
        'manifest': BASE / 'hosted-study-01/manifest.json',
        'results': BASE / 'hosted-study-01/results.json',
    }
    inputs, data = {}, {}
    for key, relative in paths.items():
        raw = (root / relative).read_bytes()
        inputs[relative.as_posix()] = sha256(raw).hexdigest()
        data[key] = json.loads(raw)
    summary, manifest, results = (data[key] for key in ('summary', 'manifest', 'results'))
    for key in ('manifest', 'results'):
        if summary['provenance'][f'{key}_sha256'] != inputs[paths[key].as_posix()]:
            raise ValueError(f'public {key} does not match the hosted analysis')
    if results['manifest_sha256'] != inputs[paths['manifest'].as_posix()]:
        raise ValueError('public results do not match the manifest')
    if manifest['mode'] != 'hosted' or summary['fixture_mode']:
        raise ValueError('plots require the hosted development evaluation')

    comparisons = []
    for name in ('arbitration-minus-ppo', 'rerank-minus-arbitration', 'probe-jev-minus-posterior'):
        pair = summary['primary_comparisons'][name]
        if pair['status'] != 'complete' or type(pair['benefit_gate_met']) is not bool:
            raise ValueError(f'incomplete registered comparison: {name}')
        fields = ('brier',) if name.startswith('probe-') else ('return_', 'final_mse')
        comparisons.append({
            'id': name, 'benefit_gate_met': pair['benefit_gate_met'],
            'paired_cells': pair.get('paired_cells'),
            'metrics': {field: pair['metrics'][field] for field in fields},
        })

    methods = []
    for name, label in [
        ('ppo', 'Standalone PPO'), ('imitation', 'Strong imitation'), ('voi', 'Bayesian VOI'),
        ('forecast_only', 'Forecast-only Jev'), ('rerank', 'Jev rerank'),
        ('arbitration', 'Full Jev arbitration'),
    ]:
        method = summary['methods'][name]
        count = method['completed_count']
        cost = method['api']['estimated_usd']
        if not count or count != method['expected_count'] or cost is None:
            raise ValueError(f'missing outcomes or API usage for {name}')
        methods.append({
            'id': name, 'label': label, 'episodes': count, 'mean_return': method['mean_return_'],
            'api_usd': cost, 'api_usd_per_episode': cost / count,
        })
    if methods[-2]['episodes'] != methods[-1]['episodes']:
        raise ValueError('rerank and arbitration must have equal episode counts')

    probe = summary['common_probe']
    schedule = manifest['schedules']['probe']
    case_by_id = {row['case_id']: row for row in results['cases']}
    case_rows = sorted((case for case in manifest['cases'] if case['method'] == 'probe'),
                       key=lambda case: case['task_index'])
    events = []
    columns = None
    for case in case_rows:
        forecasts = case_by_id[case['case_id']]['result']['forecasts']
        identity = [(event['action_id'], event['node']) for event in forecasts]
        if [action for action, _ in identity] != [action for action in schedule for _ in range(2)]:
            raise ValueError('fixed-probe events do not match the assay schedule')
        if columns is not None and identity != columns:
            raise ValueError('fixed-probe event columns differ by mechanism')
        columns = identity
        losses = [event['brier']['jev'] - event['brier']['posterior'] for event in forecasts]
        events.append({'task_index': case['task_index'], 'deltas': losses, 'mean_delta': fmean(losses)})
    if not probe['complete_paired_coverage'] or sum(len(row['deltas']) for row in events) != probe['paired_valid_events']:
        raise ValueError('plots require the complete fixed-probe cohort')
    for row in events:
        expected = probe['per_mechanism'][row['task_index']]['paired_mean_delta_brier']
        if not math.isclose(row['mean_delta'], expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError('event losses differ from the published mechanism mean')
    if not math.isclose(fmean(row['mean_delta'] for row in events),
                        probe['conditional_mean_delta_brier'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('event losses differ from the published paired mean')

    return {
        'scope': 'Hosted development on synthetic mechanisms',
        'model': manifest['model'],
        'mechanisms': len(manifest['tasks']), 'seeds': manifest['training_seeds'],
        'completed_cases': sum(row['result']['status'] == 'completed' for row in results['cases']),
        'mse_margin': summary['analysis_contract']['bootstrap']['mse_margin'],
        'input_sha256': inputs, 'comparisons': comparisons, 'methods': methods,
        'probe': {
            'paired_events': probe['paired_valid_events'], 'mean_delta': probe['conditional_mean_delta_brier'],
            'jev_brier': probe['paired_mean_jev_brier'], 'posterior_brier': probe['paired_mean_posterior_brier'],
            'columns': columns, 'rows': events,
        },
    }


def canvas(title, subtitle, height):
    fig = plt.figure(figsize=(11.8, height), facecolor='#0b1220')
    background = fig.add_axes((0, 0, 1, 1), zorder=-3)
    background.imshow(np.linspace(1, 0, 256)[:, None], extent=(0, 1, 0, 1), aspect='auto',
                      cmap=LinearSegmentedColormap.from_list('canvas', ['#0a1220', '#14233c']),
                      interpolation='bicubic')
    background.set_axis_off()
    fig.text(.055, .952, 'EXPERIMENTAL SCIENTIST RL  /  HOSTED DEVELOPMENT', color=CYAN,
             fontsize=11.5, fontfamily=MONO, weight='bold', va='center')
    fig.add_artist(Line2D((.055, .945), (.925, .925), transform=fig.transFigure, color=GRID, lw=.9))
    fig.text(.055, .884, title, color=INK, fontsize=28, weight='bold', va='center')
    fig.text(.055, .832, subtitle, color=MUTED, fontsize=12.5, va='center')
    return fig


def panel(fig, left, bottom, width, height):
    fig.add_artist(Rectangle((left, bottom), width, height, transform=fig.transFigure,
                             facecolor=SURFACE, edgecolor=GRID, linewidth=.9, zorder=-2))


def xaxis(ax, ticks, formatter):
    ax.set_facecolor('none')
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(formatter))
    ax.tick_params(axis='x', colors=MUTED, length=0, pad=6, labelsize=11)
    ax.set_yticks([])
    ax.set_axisbelow(True)
    ax.grid(axis='x', color=GRID, linewidth=.65)
    for spine in ax.spines.values():
        spine.set_visible(False)


def effect_band(fig, bottom, height, title, detail, entries, ticks, limits, *, digits, scale=1, margin=None):
    panel(fig, .055, bottom, .89, height)
    top = bottom + height
    fig.text(.075, top - .027, title, color=INK, fontsize=14, weight='bold', va='center')
    fig.text(.925, top - .027, detail, color=MUTED, fontsize=10.7, ha='right', va='center')
    axis_bottom = bottom + .045
    axis_height = height - .095
    ax = fig.add_axes((.298, axis_bottom, .435, axis_height))
    ax.set_xlim(limits)
    ax.set_ylim(-.5, len(entries) - .5)
    xaxis(ax, ticks, lambda value, _: f'{value:g}'.replace('-', '−'))
    ax.axvline(0, color=ZERO, linewidth=1, linestyle='--')
    if margin is not None:
        ax.axvline(margin, color=AMBER, linewidth=1.15, linestyle=':')
    for i, (label, metric, color) in enumerate(entries):
        y = len(entries) - i - 1
        yfig = axis_bottom + (y + .5) / len(entries) * axis_height
        point = metric['difference'] * scale
        bounds = [number * scale for number in metric['ci98_333333']]
        inner = [number * scale for number in metric['ci95']]
        ax.hlines(y, *bounds, color=color, lw=1.5)
        ax.vlines(bounds, y - .09, y + .09, color=color, lw=1.2)
        ax.hlines(y, *inner, color=color, lw=5)
        ax.scatter(point, y, s=47, marker='D', c=INK, edgecolors=SURFACE, linewidths=.9, zorder=4)
        fig.text(.075, yfig, label, color=INK, fontsize=12, va='center')
        fig.text(.755, yfig + .010, signed(point, digits), color=color, fontsize=13,
                 fontfamily=MONO, weight='bold', va='center')
        fig.text(.755, yfig - .013, f'[{signed(bounds[0], digits)}, {signed(bounds[1], digits)}]',
                 color=MUTED, fontsize=10.2, fontfamily=MONO, va='center')


def plot_effects(data):
    effects = data['comparisons']
    headline = (f'{data["mechanisms"]} mechanisms  ·  {len(data["seeds"])} PPO seeds  ·  '
                f'{data["completed_cases"]} completed cases')
    fig = canvas('Jev policy and forecast effects', headline, 8.7)
    main = [('Arbitration − PPO', effects[0]['metrics'], LILAC),
            ('Rerank − arbitration', effects[1]['metrics'], CYAN)]
    cells = effects[0]['paired_cells']
    margin = data['mse_margin'] * 1000
    effect_band(fig, .558, .219, 'Return difference', f'Higher is better · {cells} matched cells per comparison',
                [(label, metrics['return_'], color) for label, metrics, color in main],
                [-.15, -.1, -.05, 0, .05, .1, .15], (-.174, .174), digits=5)
    effect_band(fig, .32, .219, 'Target MSE difference (× 1,000)', f'Lower is better · dotted line: +{margin:.2f} margin',
                [(label, metrics['final_mse'], color) for label, metrics, color in main],
                [-2, -1, 0, 1, 2], (-2.14, 2.14), digits=3, scale=1000, margin=margin)
    effect_band(fig, .112, .189, 'Brier error difference', f'Lower is better · {data["probe"]["paired_events"]} matched events',
                [('Jev − Bayesian posterior', effects[2]['metrics']['brier'], AMBER)],
                [-.08, -.04, 0, .04, .08], (-.092, .092), digits=5)
    passed = sum(item['benefit_gate_met'] for item in effects)
    fig.text(.055, .063, f'{passed} of {len(effects)} registered benefit criteria passed.', color=INK,
             fontsize=12.5, weight='bold')
    fig.text(.055, .034, 'Diamond: paired estimate   /   Thick: 95% CI   /   Thin and bracketed: adjusted 98.33% CI',
             color=MUTED, fontsize=10.7)
    return fig


def plot_spend(data):
    methods = data['methods']
    fig = canvas('Policy return and API spend',
                 'Eight mechanisms  ·  Virtual assay costs are included in return', 6.65)
    panel(fig, .055, .158, .89, .603)
    ybottom, height = .222, .399
    returns = fig.add_axes((.327, ybottom, .205, height))
    spending = fig.add_axes((.638, ybottom, .19, height))
    xaxis(returns, [-.36, -.34, -.32, -.30], lambda x, _: f'{x:.2f}'.replace('-', '−'))
    xaxis(spending, [0, .001, .002], lambda x, _: f'{x:.3f}')
    returns.set_xlim(-.368, -.293)
    spending.set_xlim(0, .0023)
    fig.text(.075, .707, 'Policy', fontsize=13.5, color=INK, weight='bold')
    fig.text(.264, .707, 'Cases', fontsize=11, color=MUTED, ha='center')
    fig.text(.327, .707, 'Mean return ↑', fontsize=13.5, color=INK, weight='bold')
    fig.text(.638, .707, 'Estimated API spend / episode ↓', fontsize=13, color=INK, weight='bold')
    fig.text(.638, .670, 'USD', fontsize=10.6, color=MUTED, fontfamily=MONO)
    for ax in (returns, spending):
        ax.set_ylim(-.5, len(methods) - .5)
    for i, method in enumerate(methods):
        y = len(methods) - i - 1
        yfig = ybottom + (y + .5) / len(methods) * height
        hosted = method['api_usd'] > 0
        color = CYAN if hosted else LILAC
        fig.text(.075, yfig, method['label'], color=INK, fontsize=12, va='center')
        fig.text(.264, yfig, str(method['episodes']), color=MUTED, fontsize=11.5,
                 ha='center', va='center', fontfamily=MONO)
        returns.scatter(method['mean_return'], y, s=48, color=color,
                        marker='D' if hosted else 'o', zorder=3)
        ret = f'{method["mean_return"]:.4f}'.replace('-', '−')
        fig.text(.548, yfig, ret, color=color, fontsize=11.5, fontfamily=MONO, va='center')
        if hosted:
            spending.barh(y, method['api_usd_per_episode'], height=.34, color=color)
        else:
            spending.plot(0, y, marker='o', ms=4, color=color, clip_on=False)
        fig.text(.846, yfig, f'${method["api_usd_per_episode"]:.6f}' if hosted else '$0',
                 color=color, fontsize=11.4, fontfamily=MONO, va='center')
    separation = ybottom + height / 2
    fig.add_artist(Line2D((.075, .925), (separation, separation), transform=fig.transFigure, lw=.7,
                          linestyle=':', color=ZERO))
    rerank, arbitration = methods[-2:]
    reduction = 100 * (1 - rerank['api_usd'] / arbitration['api_usd'])
    fig.text(.055, .096, f'Reranking used {reduction:.1f}% less API spend than full arbitration.',
             color=CYAN, fontsize=13, weight='bold')
    fig.text(.055, .049, 'Mean return stayed below standalone PPO. The paired-effects figure shows uncertainty.',
             color=MUTED, fontsize=11)
    return fig


def plot_probe(data):
    probe = data['probe']
    fig = canvas('Forecast error by mechanism',
                 f'{probe["paired_events"]} matched events  ·  {data["mechanisms"]} mechanisms × three assays × two variables', 8.15)
    panel(fig, .055, .171, .89, .593)
    columns = probe['columns']
    rows = probe['rows']
    positions = [0, 1, 2.18, 3.18, 4.36, 5.36, 6.82]
    ax = fig.add_axes((.205, .209, .714, .431), facecolor='none')
    ax.set_xlim(-.53, 7.35)
    ax.set_ylim(len(rows) - .46, -.54)
    ax.set_axis_off()
    maximum = max(abs(delta) for row in rows for delta in row['deltas'])
    limit = max(.1, math.ceil(maximum * 10) / 10)
    colors = LinearSegmentedColormap.from_list('brier', ['#72e7e1', '#243247', '#f6a987'])
    norm = Normalize(-limit, limit)

    def figure_point(x, y):
        return fig.transFigure.inverted().transform(ax.transData.transform((x, y)))

    for row_index, row in enumerate(rows):
        _, fy = figure_point(0, row_index)
        fig.text(.075, fy, f'{row["task_index"]:03}', color=INK, fontsize=12, fontfamily=MONO, va='center')
        values = [*row['deltas'], row['mean_delta']]
        for i, (x, number) in enumerate(zip(positions, values, strict=True)):
            fill = colors(norm(number))
            luminance = .213 * fill[0] + .715 * fill[1] + .072 * fill[2]
            ax.add_patch(Rectangle((x - .471, row_index - .442), .942, .884,
                                   facecolor=fill, linewidth=.85 if i == 6 else 0, edgecolor=ZERO))
            cell = signed(number, 3) if abs(number) >= .0005 else '≈0'
            ax.text(x, row_index, cell, color='#0a1321' if luminance > .51 else INK,
                    fontfamily=MONO, fontsize=11, weight='bold' if i == 6 else 'normal',
                    ha='center', va='center')
    fig.text(.075, .669, 'Mechanism', color=MUTED, fontsize=10.8)
    for i, (action, node) in enumerate(columns):
        x, _ = figure_point(positions[i], 0)
        fig.text(x, .669, f'X{node}', color=MUTED, fontsize=11, ha='center', fontfamily=MONO)
        if i % 2 == 0:
            center, _ = figure_point((positions[i] + positions[i + 1]) / 2, 0)
            fig.text(center, .714, f'Assay {action:02}', color=INK, fontsize=13, weight='bold', ha='center')
    mean_x, _ = figure_point(positions[-1], 0)
    fig.text(mean_x, .714, 'Mechanism mean', color=INK, fontsize=11.5, weight='bold', ha='center')
    bar = fig.add_axes((.055, .075, .442, .018))
    bar.imshow(np.linspace(0, 1, 256)[None, :], cmap=colors, extent=(-limit, limit, 0, 1), aspect='auto')
    bar.set_axis_off()
    fig.text(.055, .118, 'Jev − posterior Brier loss', color=INK, fontsize=11.2, weight='bold')
    fig.text(.055, .042, f'−{limit:.2f} · Jev lower', color=MUTED, fontsize=10.5)
    fig.text(.276, .042, '0', color=MUTED, fontsize=10.5, ha='center')
    fig.text(.497, .042, f'+{limit:.2f} · posterior lower', color=MUTED, fontsize=10.5, ha='right')
    fig.text(.945, .111, f'Mean difference {signed(probe["mean_delta"], 5)}', color=AMBER,
             fontsize=13.5, weight='bold', ha='right')
    fig.text(.945, .062, f'Jev {probe["jev_brier"]:.5f}  /  posterior {probe["posterior_brier"]:.5f}',
             color=MUTED, fontsize=11.6, ha='right')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data = load_evidence(args.root.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'figure-data.json').write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    with mpl.rc_context({'font.family': 'DejaVu Sans', 'figure.dpi': 168}):
        for name, plot in [
            ('paired-effects.png', plot_effects),
            ('return-and-spend.png', plot_spend),
            ('forecast-error.png', plot_probe),
        ]:
            figure = plot(data)
            figure.savefig(output / name, metadata={'Title': name.removesuffix('.png').replace('-', ' ')})
            plt.close(figure)


if __name__ == '__main__':
    main()
