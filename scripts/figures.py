"""Generate standalone scientific figures from recorded measurements."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'reports/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    output = ROOT/'reports/feasibility-v1'
    data = json.loads((output/'feasibility.json').read_text())
    fig, axes = plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax, noise in zip(axes,(.02,.08)):
        result = next(r for r in data['results'] if r['config']['budget']==8 and r['config']['readout_sd']==noise and r['config']['offset_sd']==0 and r['config']['cost_weight']==.01)
        for method in ('prior','fixed','identification','additive','random','eig','voi'):
            row = result['summary'][method]
            ax.scatter(row['mean_variable_cost'],row['aggregate_rmse'],s=45)
            ax.annotate(method,(row['mean_variable_cost'],row['aggregate_rmse']),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.axhline(.04,color='gray',linestyle=':',label='declared success threshold')
        ax.set(title=f'Budget 8; readout SD {noise}',xlabel='Mean purchased assay cost')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Aggregate final target RMSE')
    fig.suptitle('Finite-support development pilot: cost and final quality')
    fig.tight_layout()
    fig.savefig(output/'cost-quality.png',dpi=180)
    fig.savefig(output/'cost-quality.pdf')
    plt.close(fig)


if __name__ == '__main__': main()
