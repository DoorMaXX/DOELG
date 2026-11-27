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
        ('update', 'Update and map: copy an instance graph and infer from new initial events.'),
    )),
    ('Extraction validation', 'main_extraction_validation.py', (
        ('evaluate', 'Evaluate a prepared extraction workbook.'),
        ('plot', 'Plot extraction results across few-shot settings.'),
        ('table', 'Calculate the three extraction metrics in paper Table 4.'),
        ('prepare', 'Prepare a new evaluation workbook.'),
    )),
    ('Method validation and paper reproduction', 'main_validation.py', (
        ('instantiate', 'Construct the OELG baseline instance graph.'),
        ('map', 'Generate maps from the OELG instance graph.'),
        ('compare', 'Compare two databases; reuse rasters or generate missing maps.'),
        ('pattern', 'Summarize patterns from a logic database or verify archived pattern statistics.'),
        ('reproduction', 'Paper reproduction from archived inputs only (paper version).'),
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
        if key.split('.')[-1].endswith('database'):
            from src.construct_config import resolve_database_name
            value = resolve_database_name(value)
        return str(value)
    except (OSError, ValueError, KeyError, TypeError):
        return 'script default'


def build_command(group_index, task):
    """Collect applicable parameters and build an argument list for an existing script."""
    task = {'pattern': 'patterns', 'reproduction': 'reproduce'}.get(task, task)
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
        config = add('--config', 'Instantiation and mapping configuration', 'file', 'instantiation_config.json')
        config = config or ROOT / 'instantiation_config.json'
        add('--construct-config', 'Connection and model configuration', 'file', 'construct_config.json')
        if task == 'update':
            from src.instantiation_settings import load_update_settings
            defaults = load_update_settings(config)
            add('--source-database', 'Existing instance database to copy', default=defaults['source_database'])
            add('--logic_database', 'Source logic database', default=defaults['logic_database'])
            add('--events', 'New initial events GeoJSON', 'file', defaults['events_file'])
            add('--ins_database', 'New update instance database', default=defaults['instance_database'])
            add('--output-dir', 'New update output directory', default=defaults['update_output_dir'] + '/<new run>')
            add('--no-plots', 'Skip map PNGs? y/n', 'flag', 'n')
            return command
        if task in ('instantiate', 'all'):
            add('--logic_database', 'Source logic database', default=setting(config, 'instantiation.logic_database'))
            events_default = setting(config, 'instantiation.events_file')
            if events_default == 'script default':
                events_default = setting(config, 'region')
            add('--events', 'Initial event GeoJSON', 'file', events_default)
        snapshot = ''
        if task == 'map':
            snapshot = add('--snapshot', 'Offline source snapshot (optional)', 'file', 'read Neo4j')
        if not snapshot:
            add('--ins_database', 'Instance database', default=setting(config, 'database'))
        if task in ('map', 'all'):
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
        if task == 'patterns':
            source_index = choose('Validation data source', (
                ('Local Neo4j databases', 'Read existing result graphs without modifying graph data.'),
                ('Existing archives', 'Use frozen snapshots, pattern statistics or exported rasters.')))
            if source_index is None:
                return None
            source = 'database' if source_index == 0 else 'archive'
            command.extend(['--source', source])
        else:
            source = 'database' if task == 'compare' else 'archive' if task == 'reproduce' else None
            if source:
                command.extend(['--source', source])
        config_name = ('database_validation_config.json' if source == 'database' else
                       'paper_experiment_config.json' if task in ('reproduce', 'patterns') else
                       'importance_comparison_config.json' if task == 'compare' else 'oelg_config.json')
        config = add('--config', 'Validation configuration', 'file', 'instantiation_validation/' + config_name)
        if source == 'database':
            from instantiation_validation.validation_sources import load_database_profile
            config = config or ROOT / 'instantiation_validation' / config_name
            preview = load_database_profile(config)
            connection = add('--construct-config', 'Connection configuration', 'file', preview['construct_config'])
            defaults = load_database_profile(config, {'construct_config': connection or None})
            if task == 'compare':
                add('--doelg-database', 'DOELG instance database', default=defaults['doelg_database'])
                add('--oelg-database', 'OELG instance database', default=defaults['oelg_database'])
                add('--doelg-raster-dir', 'DOELG map root (reuse existing rasters)', default=defaults['doelg_raster_dir'])
                add('--oelg-raster-dir', 'OELG map root (reuse existing rasters)', default=defaults['oelg_raster_dir'])
            if task == 'patterns':
                add('--pattern-database', 'Generalized logic database', default=defaults['pattern_database'])
            add('--output-dir', 'New local evaluation output directory')
            if task != 'patterns':
                add('--no-plots', 'Skip map PNGs? y/n', 'flag', 'n')
        else:
            if task in ('instantiate', 'map'):
                selected_config = config or ROOT / 'instantiation_validation' / config_name
                add('--instance-database', 'OELG instance database', default=setting(selected_config, 'instance_database'))
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
    parser = argparse.ArgumentParser(description='Plain command-line launcher for DOELG_main.')
    parser.add_argument('--dry-run', action='store_true', help='Preview menu commands without executing tasks')
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    print('DOELG - Disaster Observation Event Logic Graph Construction and Application')
    print('Extract knowledge, generalize event logic, instantiate events, and evaluate results.')
    print('Live graph tasks use Neo4j and the configured model/geocoding services.')
    print('5.1 instantiates OELG; 5.3 compares methods; 5.4 summarizes patterns; 5.5 replays paper archives.')
    print('Write tasks create missing target databases in the configured Neo4j instance. {date} means YYYYMMDD in Shanghai.')
    print('Service settings are configured first; task parameter choices apply to this run only.')
    print(f'Project: {ROOT}\nPython: {sys.executable}')
    print('The menu uses the Python interpreter shown above. Select 0 to return or exit; Ctrl+C cancels a task.')
    if args.dry_run:
        print('PREVIEW MODE: commands will not execute; service settings will not be saved.')
    try:
        from src.startup_configuration import configure_services
        configure_services(ROOT / 'construct_config.json', persist=not args.dry_run)
    except (EOFError, KeyboardInterrupt):
        print('\nService configuration cancelled. No task was started.')
        return 130
    except (OSError, ValueError) as error:
        print(f'Service configuration failed: {error}')
        return 1
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
            if group in (0, 1) or (group in (2, 4) and task in ('instantiate', 'all', 'update')):
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
