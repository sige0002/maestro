#!/usr/bin/env python3
"""Legacy cube-macro baseline: six trials with independent placement/home scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def call(command, log, cwd=ROOT, env=None, timeout=900):
    with log.open('a') as stream:
        return subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, timeout=timeout).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--so101-root', type=Path, default=ROOT.parent/'so101')
    args = parser.parse_args()
    so101 = args.so101_root.resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    relative = Path('runs')/'so101-benchmark'/stamp
    output = ROOT/relative
    compose = ['docker', 'compose', '-f', 'compose.so101.yaml', 'run', '--rm']
    subprocess.run(compose+['--entrypoint', '/usr/bin/install', 'maestro', '-d',
                           '-o', str(os.getuid()), '-g', str(os.getgid()), str(relative)], check=True, cwd=ROOT)
    files = [*sorted((ROOT/'maestro/so101').glob('*.py')), ROOT/'maestro/so101/prompt.txt',
             ROOT/'maestro/vlm.py', so101/'scripts/models/model.py']
    hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest = {'scope': 'dedicated_cube_tool_via_VLM_not_full_Maestro_reproduction',
                'source_sha256': hashes, 'cases': [], 'prior_layout': os.environ.get('SIM_CUBE_LAYOUT', 'default')}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'OUTPUT {output}', flush=True)
    japanese = {'red': '赤', 'green': '緑', 'blue': '青'}
    try:
        for layout in ('default', 'rotated'):
            for color in ('green', 'red', 'blue'):
                case = f'{layout}-{color}'
                folder = output/case
                folder.mkdir()
                env = dict(os.environ, SIM_CUBE_LAYOUT=layout)
                print(f'START {case}', flush=True)
                rc = call(['./so101', 'mujoco', 'up', '--force-recreate', '--no-deps'],
                          folder/'simulator.log', cwd=so101, env=env, timeout=120)
                if rc:
                    raise RuntimeError(f'Simulator setup failed for {case}')
                instruction = f'{japanese[color]}の立方体をトレーに入れ、最後に初期姿勢へ戻ってください。画像と関節状態で確認してください。'
                container = f'so101-bench-{stamp.lower()}-{case}'
                task = compose+['--name', container, '--entrypoint', '/bin/bash', 'maestro', '-c',
                                'source /opt/ros/jazzy/setup.bash; exec /opt/maestro/bin/python -m maestro.so101.runner "$@"',
                                '--', '--config', 'config/so101.yaml', '--record', '--max-steps', '3',
                                '--output', str(relative/case), '-i', instruction]
                timed_out = False
                try:
                    rc = call(task, folder/'trial.log', timeout=900)
                except subprocess.TimeoutExpired:
                    subprocess.run(['docker', 'stop', '-t', '10', container], check=True)
                    timed_out, rc = True, 124
                evaluator = compose+['--entrypoint', '/bin/bash', 'maestro', '-c',
                    'source /opt/ros/jazzy/setup.bash; exec /opt/maestro/bin/python -m maestro.so101.evaluate "$@"',
                    '--', '--color', color, '--output', str(relative/case/'evaluation')]
                evaluation_rc = call(evaluator, folder/'evaluation.log', timeout=90)
                evaluation_file = folder/'evaluation/evaluation.json'
                evaluation = json.loads(evaluation_file.read_text()) if evaluation_file.exists() else {}
                run_files = sorted(folder.glob('*/result.json'))
                result = json.loads(run_files[-1].read_text()) if run_files else {}
                entry = {'case': case, 'instruction': instruction, 'exit_code': rc,
                         'evaluation_exit_code': evaluation_rc, 'timed_out': timed_out,
                         'model_status': result.get('status'), 'success': evaluation.get('success', False),
                         'run': str(run_files[-1].parent.relative_to(ROOT)) if run_files else None}
                manifest['cases'].append(entry)
                (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
                print('RESULT '+json.dumps(entry, ensure_ascii=False), flush=True)
    finally:
        # Restore default scene configuration and stop using the benchmark layout.
        env = dict(os.environ, SIM_CUBE_LAYOUT=manifest['prior_layout'])
        call(['./so101', 'mujoco', 'up', '--force-recreate', '--no-deps'],
             output/'restore.log', cwd=so101, env=env, timeout=120)
    print(f'FINISHED {sum(c["success"] for c in manifest["cases"])}/{len(manifest["cases"])}', flush=True)


if __name__ == '__main__':
    main()
