import argparse
import subprocess
from pathlib import Path
import sys

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Dict, List

from tools.experiment_registry import (
    get_experiments,
    infer_detection_weights,
    infer_run_root,
    infer_tracking_detector_weights,
    infer_tracking_eval_dir,
    infer_tracking_output_dir,
    infer_tracking_results_path,
)


def _quote(value):
    value = str(value)
    if any(char.isspace() for char in value):
        return f'"{value}"'
    return value


def _placeholder(value, fallback):
    return value if value else fallback


def build_experiment_suite(
    mode='plan',
    subset='all',
    experiments_root='outputs/experiments',
    detector_weight_run='full_project',
    source_rgb='',
    source_ir='',
    tracking_gt='',
    image_root='',
    track_eval_config='configs/main/tracking_eval.yaml',
):
    experiments = get_experiments(subset=subset)
    suite = []
    for spec in experiments:
        run_root = infer_run_root(spec, experiments_root=experiments_root)
        entry: Dict[str, object] = {
            'name': spec.name,
            'category': spec.category,
            'stage': spec.stage,
            'config': spec.config,
            'run_root': str(run_root),
            'description': spec.description,
            'commands': {},
        }
        if spec.category == 'detection':
            weights = infer_detection_weights(spec, experiments_root=experiments_root)
            entry['weights'] = str(weights)
            entry['commands']['train'] = f"python tools/train.py --config {_quote(spec.config)}"
            entry['commands']['eval'] = f"python tools/val.py --config {_quote(spec.config)} --weights {_quote(weights)}"
        else:
            detector_weights = infer_tracking_detector_weights(
                experiments_root=experiments_root,
                detector_run=detector_weight_run,
            )
            tracking_output_dir = infer_tracking_output_dir(spec, experiments_root=experiments_root)
            results_path = infer_tracking_results_path(spec, experiments_root=experiments_root)
            tracking_eval_dir = infer_tracking_eval_dir(spec, experiments_root=experiments_root)
            entry['weights'] = str(detector_weights)
            entry['tracking_output_dir'] = str(tracking_output_dir)
            entry['tracking_results'] = str(results_path)
            entry['tracking_eval_dir'] = str(tracking_eval_dir)
            rgb_value = _placeholder(source_rgb, '<path/to/rgb_frames>')
            ir_value = _placeholder(source_ir, '<path/to/ir_frames>')
            image_root_value = _placeholder(image_root, rgb_value)
            infer_command = (
                f"python tools/infer.py --config {_quote(spec.config)} --weights {_quote(detector_weights)} "
                f"--source_rgb {_quote(rgb_value)} --source_ir {_quote(ir_value)} --save_dir {_quote(tracking_output_dir)}"
            )
            eval_command = (
                f"python tools/val.py --config {_quote(track_eval_config)} "
                f"tracking_eval.results_path={_quote(results_path)} tracking_eval.image_root={_quote(image_root_value)}"
            )
            if tracking_gt:
                eval_command += f" tracking_eval.gt_path={_quote(tracking_gt)}"
            entry['commands']['infer'] = infer_command
            entry['commands']['track_eval'] = eval_command
        suite.append(entry)
    return suite


def _commands_for_mode(entry, mode):
    if mode == 'plan':
        return list(entry.get('commands', {}).values())
    if mode == 'train':
        return [entry['commands']['train']] if 'train' in entry.get('commands', {}) else []
    if mode == 'eval':
        return [entry['commands']['eval']] if 'eval' in entry.get('commands', {}) else []
    if mode == 'track_eval':
        commands = []
        if 'infer' in entry.get('commands', {}):
            commands.append(entry['commands']['infer'])
        if 'track_eval' in entry.get('commands', {}):
            commands.append(entry['commands']['track_eval'])
        return commands
    raise ValueError(f'Unsupported mode: {mode}')


def render_plan(suite, mode='plan'):
    lines: List[str] = []
    for entry in suite:
        lines.append(f"[{entry['category'].upper()}] {entry['name']}")
        lines.append(f"  stage: {entry['stage']}")
        lines.append(f"  config: {entry['config']}")
        lines.append(f"  run_root: {entry['run_root']}")
        if entry.get('weights'):
            lines.append(f"  weights: {entry['weights']}")
        if entry.get('tracking_output_dir'):
            lines.append(f"  tracking_output_dir: {entry['tracking_output_dir']}")
        commands = _commands_for_mode(entry, mode)
        if not commands:
            lines.append('  commands: <none for this mode>')
        else:
            lines.append('  commands:')
            for command in commands:
                lines.append(f'    {command}')
        lines.append('')
    return '\n'.join(lines).rstrip() + '\n'


def write_command_script(suite, output_path, mode='plan'):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if output_path.suffix.lower() == '.bat':
        lines.append('@echo off')
        lines.append('')
    for entry in suite:
        lines.append(f"# {entry['name']} ({entry['category']})")
        for command in _commands_for_mode(entry, mode):
            lines.append(command)
        lines.append('')
    output_path.write_text('\n'.join(lines).strip() + '\n', encoding='utf-8')
    return str(output_path)


def execute_suite(suite, mode='plan'):
    if mode == 'plan':
        return []
    executed = []
    for entry in suite:
        for command in _commands_for_mode(entry, mode):
            subprocess.run(command, shell=True, check=False)
            executed.append(command)
    return executed


def parse_args():
    parser = argparse.ArgumentParser(description='Plan or launch the final project experiment suite.')
    parser.add_argument('--mode', choices=['plan', 'train', 'eval', 'track_eval'], default='plan')
    parser.add_argument('--subset', choices=['detection', 'tracking', 'all'], default='all')
    parser.add_argument('--experiments-root', type=str, default='outputs/experiments')
    parser.add_argument('--detector-weight-run', type=str, default='full_project')
    parser.add_argument('--source-rgb', type=str, default='')
    parser.add_argument('--source-ir', type=str, default='')
    parser.add_argument('--tracking-gt', type=str, default='')
    parser.add_argument('--image-root', type=str, default='')
    parser.add_argument('--track-eval-config', type=str, default='configs/main/tracking_eval.yaml')
    parser.add_argument('--emit-script', type=str, default='')
    parser.add_argument('--execute', action='store_true', help='Actually execute the generated commands. Default is plan-only.')
    return parser.parse_args()


def main():
    args = parse_args()
    suite = build_experiment_suite(
        mode=args.mode,
        subset=args.subset,
        experiments_root=args.experiments_root,
        detector_weight_run=args.detector_weight_run,
        source_rgb=args.source_rgb,
        source_ir=args.source_ir,
        tracking_gt=args.tracking_gt,
        image_root=args.image_root,
        track_eval_config=args.track_eval_config,
    )
    print(render_plan(suite, mode=args.mode))
    if args.emit_script:
        script_path = write_command_script(suite, args.emit_script, mode=args.mode)
        print(f'Command script written to {script_path}')
    if args.execute:
        executed = execute_suite(suite, mode=args.mode)
        print(f'Executed {len(executed)} commands.')


if __name__ == '__main__':
    main()
