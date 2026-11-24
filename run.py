"""Provide a plain interactive console for the existing KGC entry points."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
GROUPS = (
    ('Extraction and fusion', 'main_extraction.py', (
        ('extract', 'Extract geographic entities, events, and their relations.'),
        ('fuse', 'Merge duplicate entities and events in the graph.'),
        ('env', 'Extract hazard-forming environment descriptions.'),
        ('obs_pat', 'Extract sensor observation patterns.'),
        ('spatial', 'Geocode features and calculate spatial relationships.'),
        ('all', 'Run all extraction and construction stages.'),
    )),
    ('Knowledge generalization', 'main_generalization.py', (
        ('disaster_gen', 'Cluster disaster pairs and construct abstract events.'),
        ('spatial_rule', 'Generalize spatial relationships and environment rules.'),
        ('obs_gen', 'Generalize observation events and standard factors.'),
        ('stats', 'Export graph statistics and observation summaries.'),
        ('all', 'Run all generalization stages.'),
    )),
    ('Instantiation and mapping', 'main_instantiation.py', (
        ('instantiate', 'Instantiate regional disaster and observation events.'),
        ('map', 'Generate event and observation-factor distance fields.'),
        ('all', 'Run instantiation followed by mapping.'),
    )),
    ('Extraction validation', 'main_extraction_validation.py', (
        ('evaluate', 'Evaluate a prepared extraction workbook.'),
        ('plot', 'Plot extraction results across few-shot settings.'),
        ('table', 'Calculate the three extraction metrics in paper Table 4.'),
        ('prepare', 'Prepare a new evaluation workbook.'),
    )),
    ('Method validation and paper reproduction', 'main_validation.py', (
        ('reproduce', 'Replay archived experiments and verify Tables 4, 6, and 7.'),
        ('patterns', 'Verify the archived 1,656-pattern statistics.'),
        ('compare', 'Compare existing DOELG and OELG raster fields.'),
        ('instantiate', 'Construct the OELG baseline instance graph.'),
        ('map', 'Generate maps from the OELG instance graph.'),
    )),
)


def choose(title, items):
    """Return a numbered selection, or None when the user chooses Back."""
    print(f'\n{title}')
    for index, (name, description) in enumerate(items, 1):
        print(f'  {index}. {name:<16} {description}')
    print('  0. Back / Exit')
    while True:
        value = input('Select a number: ').strip()
        if value == '0':
            return None
        if value.isdigit() and 1 <= int(value) <= len(items):
            return int(value) - 1
        print('Please enter one of the listed numbers.')


def ask(label, kind='text', default=''):
    """Read an optional value and validate basic file, integer, and flag inputs."""
    while True:
        suffix = f' [Enter: {default}]' if default else ' [Enter: use script default]'
        value = input(label + suffix + ': ').strip()
        if kind == 'flag':
            if value.lower() in ('', 'n', 'no'):
                return False
            if value.lower() in ('y', 'yes'):
                return True
            print('Please enter y or n.')
            continue
        value = value.strip('"')
        if not value:
            if kind == 'required_file' or kind == 'required_output':
                print('A path is required.')
                continue
            return ''
        if kind in ('file', 'required_file', 'directory', 'required_output'):
            path = Path(value).expanduser()
            path = path if path.is_absolute() else ROOT / path
            if kind in ('file', 'required_file') and not path.is_file():
                print(f'File not found: {path}')
                continue
            if kind == 'directory' and not path.is_dir():
                print(f'Directory not found: {path}')
                continue
            if kind == 'required_output' and path.exists():
                print('Choose a new output file; this path already exists.')
                continue
            return str(path.resolve())
        if kind in ('integer', 'shots'):
            numbers = value.split()
            if not numbers or (kind == 'integer' and len(numbers) != 1) or not all(
                    number.isdigit() and int(number) > 0 for number in numbers):
                print('Enter positive whole numbers.' if kind == 'shots' else 'Enter a positive whole number.')
                continue
        return value


def setting(path, key):
    """Read a displayed default without exposing unrelated connection credentials."""
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        for part in key.split('.'):
            value = value[part]
        return str(value)
    except (OSError, ValueError, KeyError, TypeError):
        return 'script default'


def build_command(group_index, task):
    """Collect applicable parameters and build an argument list for an existing script."""
    script = GROUPS[group_index][1]
    command = [sys.executable, '-u', str(ROOT / 'main' / script), '-t', task]

    def add(flag, label, kind='text', default=''):
        """Append a supplied option while leaving blank values to the underlying script."""
        value = ask(label, kind, default)
        if kind == 'flag':
            if value:
                command.append(flag)
        elif value:
            command.append(flag)
            command.extend(value.split() if kind == 'shots' else [value])
        return value

    if group_index in (0, 1):
        config = add('--config', 'Construction configuration', 'file', 'construct_config.json')
        config = config or ROOT / 'construct_config.json'
        section = 'extraction' if group_index == 0 else 'generalization'
        add('--database', 'Neo4j database', default=setting(config, section + '.database'))
        add('--workers', 'Worker count', 'integer', setting(config, section + '.workers'))
        if group_index == 0 and task in ('extract', 'all'):
            add('--input_dir', 'Input JSON directory', 'directory', setting(config, 'extraction.input_dir'))
    elif group_index == 2:
        if task in ('instantiate', 'all'):
            add('--logic_database', 'Source logic database', default='events-logic-all-1zz-sensor-en-0904')
            add('--events', 'Initial event GeoJSON', 'file', 'data/geo_dataset/events_zhengzhou/onset.geojson')
        snapshot = ''
        if task == 'map':
            snapshot = add('--snapshot', 'Offline source snapshot (optional)', 'file', 'read Neo4j')
        if not snapshot:
            add('--ins_database', 'Instance database', default='events-ins-all-zhengzhou-test-1')
        if task in ('map', 'all'):
            add('--map-config', 'Mapping configuration', 'file', 'instantiation_config.json')
            add('--output-dir', 'New map output directory')
            add('--no-plots', 'Skip map PNGs? y/n', 'flag', 'n')
    elif group_index == 3:
        if task != 'table':
            index = choose('Evaluation target', (('event', 'Disaster and observation events.'),
                                                  ('geo', 'Geographic entities and spatial relations.'),
                                                  ('relation', 'Event causal and trigger relations.')))
            if index is None:
                return None
            kind = ('event', 'geo', 'relation')[index]
            command.extend(['--kind', kind])
        if task in ('prepare', 'evaluate'):
            source = add('--input', 'Input workbook', 'required_file')
            if task == 'prepare':
                destination = add('--output', 'New output workbook', 'required_output')
                if Path(source) == Path(destination):
                    raise ValueError('Input and output workbook paths must differ.')
                if kind != 'geo':
                    add('--only-raw', 'Prepare predictions without ground truth? y/n', 'flag', 'n')
        else:
            add('--base-dir', 'Workbook base directory', 'directory')
            add('--shots', 'Few-shot sample counts', 'shots', '1 5 10 30')
            add('--output' if task == 'table' else '--output-dir',
                'Output CSV file' if task == 'table' else 'Plot output directory')
    else:
        config_name = ('paper_experiment_config.json' if task in ('reproduce', 'patterns') else
                       'importance_comparison_config.json' if task == 'compare' else 'oelg_config.json')
        add('--config', 'Validation configuration', 'file', 'instantiation_validation/' + config_name)
        if task in ('instantiate', 'map', 'reproduce'):
            add('--output-dir', 'Output directory')
            add('--no-plots', 'Skip maps entirely? y/n' if task == 'instantiate' else 'Skip map PNGs? y/n',
                'flag', 'n')
        if task in ('compare', 'reproduce'):
            add('--overwrite', 'Replace existing comparison reports? y/n', 'flag', 'n')
    return command


def execute(command, dry_run=False):
    """Run a child in the project root, stream its logs, and stop it on interruption."""
    print('\nCommand: ' + subprocess.list2cmdline(command), flush=True)
    if dry_run:
        print('Preview only; no task was started.')
        return
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    process = subprocess.Popen(command, cwd=ROOT, env=environment)
    try:
        code = process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    print('Task completed.' if code == 0 else f'Task failed (exit code {code}). See the messages above.')


def main(argv=None):
    """Show the menu without initializing models or connecting to a database."""
    parser = argparse.ArgumentParser(description='Plain command-line launcher for KGC_main.')
    parser.add_argument('--dry-run', action='store_true', help='Preview menu commands without executing tasks')
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    print('DOELG - Disaster Observation Event Logic Graph Construction and Application')
    print('Extract knowledge, generalize event logic, instantiate events, and evaluate results.')
    print('Live graph tasks use Neo4j and the configured model/geocoding services.')
    print('Paper reproduction and archived pattern checks run offline.')
    print('Parameters apply to this run only. Configuration files are not edited.')
    print(f'Project: {ROOT}\nPython: {sys.executable}')
    print('Use the KGC conda environment. Select 0 to return or exit; Ctrl+C cancels a task.')
    if args.dry_run:
        print('PREVIEW MODE: commands will not execute.')
    while True:
        try:
            group = choose('Main menu', tuple((name, '') for name, _, _ in GROUPS))
            if group is None:
                return 0
            task_index = choose(GROUPS[group][0], GROUPS[group][2])
            if task_index is None:
                continue
            task, description = GROUPS[group][2][task_index]
            print(f'\n{task}: {description}')
            if group in (0, 1) or (group in (2, 4) and task in ('instantiate', 'all')):
                print('This task writes graph data to its selected Neo4j database.'
                      if not (group == 1 and task == 'stats') else 'This task reads Neo4j and exports statistics.')
            command = build_command(group, task)
            if command:
                execute(command, args.dry_run)
        except EOFError:
            print('\nConsole closed.')
            return 0
        except KeyboardInterrupt:
            print('\nOperation cancelled. Returning to the main menu.')
        except (OSError, ValueError) as error:
            print(f'Unable to run task: {error}')


if __name__ == '__main__':
    raise SystemExit(main())
