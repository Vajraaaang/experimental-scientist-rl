"""Profile one bounded local PPO configuration; subprocesses compose seed sweeps."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'src', ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from integrations.m3_ppo_profile import ProfileConfig, profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cpu', 'mps', 'split'), default='cpu')
    parser.add_argument('--kind', choices=('history', 'recurrent'), default='recurrent')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--seconds', '--seconds-cap', dest='seconds_cap', type=float, default=30.)
    parser.add_argument('--transition-cap', type=int, default=3_000_000)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--workload', choices=('selector', 'joint'), default='selector')
    parser.add_argument('--checkpoint-every', type=int, default=1)
    parser.add_argument('--initialization', choices=('scratch', 'v6'), default='scratch')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--gae-lambda', type=float, default=.95)
    parser.add_argument('--output', type=Path, required=True, help='new exclusive output directory; relative paths resolve from current directory')
    args = vars(parser.parse_args(argv))
    output = args.pop('output').expanduser().resolve()
    result = profile(ProfileConfig(**args), output)
    print(json.dumps({'output': str(output), 'status': result['status'],
                      'stop_reason': result['stop_reason'], 'totals': result['totals'],
                      'max_initial_logp_error': result['correctness']['max_initial_logp_error']}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
