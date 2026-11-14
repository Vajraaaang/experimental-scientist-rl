"""Run one bounded local M2 PPO candidate in a new output directory."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'src', ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from integrations.ppo_m2_train import TrainConfig, train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON object containing TrainConfig fields')
    parser.add_argument('--output', required=True, type=Path, help='new exclusive output directory')
    for field in ('name', 'kind', 'backend', 'initialization', 'initial_checkpoint_record'):
        parser.add_argument('--' + field.replace('_', '-'))
    for field in ('seed', 'hidden', 'batch', 'threads', 'selector_interactions', 'joint_interactions',
                  'joint_hard_interactions', 'eval_every_interactions', 'joint_epochs', 'train_pool_size', 'tuning_count'):
        parser.add_argument('--' + field.replace('_', '-'), type=int)
    for field in ('wall_snapshot_seconds', 'hard_wall_seconds', 'joint_lr', 'entropy_start', 'entropy_end'):
        parser.add_argument('--' + field.replace('_', '-'), type=float)
    parser.add_argument('--smoke', action='store_true', default=None,
                        help='label reduced pools as smoke checks, never study evidence')
    arguments = vars(parser.parse_args(argv))
    output = arguments.pop('output').expanduser().resolve()
    config_path = arguments.pop('config')
    values = {} if config_path is None else json.loads(config_path.expanduser().resolve().read_text())
    if not isinstance(values, dict):
        parser.error('--config must contain a JSON object')
    values.update({key: value for key, value in arguments.items() if value is not None})
    result = train(TrainConfig(**values), output)
    print(json.dumps({'output': str(output), 'status': result['status'], 'stop_reason': result['stop_reason'],
                      'totals': result['totals'], 'selected_checkpoint': result['selected_checkpoint']}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
