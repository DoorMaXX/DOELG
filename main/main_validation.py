import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'instantiation_validation'))

SOURCE_TASKS = ('reproduce', 'patterns', 'compare')


def run_oelg(config, maps_only=False, no_plots=False, output_dir=None, instance_database=None):
    """Instantiate the OELG baseline or render its existing instance graph."""
    import instantiation_oelg
    arguments = ['--config', str(config)]
    if instance_database:
        arguments.extend(['--instance-database', instance_database])
    if maps_only:
        arguments.append('--maps-only')
    if no_plots:
        arguments.append('--no-plots')
    if output_dir:
        arguments.extend(['--map-output-dir' if maps_only else '--output-folder', str(output_dir)])
    return instantiation_oelg.main(arguments)


def run_comparison(config, overwrite=False):
    """Evaluate and compare raster archives described by the selected configuration."""
    import compare_importance_fields
    arguments = ['--config', str(config)]
    if overwrite:
        arguments.append('--overwrite')
    return compare_importance_fields.main(arguments)


def main(argv=None):
    """Select archive or read-only database inputs for evaluation tasks before execution."""
    parser = argparse.ArgumentParser(description='Graph validation from local databases or archived inputs')
    parser.add_argument('-t', '--task', required=True, choices=[*SOURCE_TASKS, 'instantiate', 'map', 'pattern', 'reproduction'])
    parser.add_argument('--instance-database', help='Instance database for OELG instantiate/map; supports {date}')
    parser.add_argument('--source', choices=['database', 'archive'], help='patterns: database/archive (default archive); compare: database/archive (default database); reproduce: archive only')
    parser.add_argument('--config', type=Path, help='Configuration for the selected task and source')
    parser.add_argument('--construct-config', type=Path, help='Override connection settings for database validation')
    parser.add_argument('--doelg-database', help='DOELG instance database for database validation')
    parser.add_argument('--oelg-database', help='OELG instance database for database validation')
    parser.add_argument('--oelg-snapshot', help='Optional advanced override; otherwise use cached raster metadata or discover the latest complete OELG result')
    parser.add_argument('--pattern-database', help='Generalized logic database for pattern statistics')
    parser.add_argument('--doelg-raster-dir', type=Path, help='Existing DOELG map root; missing rasters are generated from its database')
    parser.add_argument('--oelg-raster-dir', type=Path, help='Existing OELG map root; missing rasters are generated from its database')
    parser.add_argument('--no-plots', action='store_true', help='Skip map PNG generation')
    parser.add_argument('--output-dir', type=Path, help='Output directory for the selected workflow')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing comparison reports')
    parser.add_argument('--show-source', action='store_true', help='Print source settings and exit without connecting to services')
    args = parser.parse_args(argv)
    args.task = {'pattern': 'patterns', 'reproduction': 'reproduce'}.get(args.task, args.task)
    database_options = dict(construct_config=args.construct_config, doelg_database=args.doelg_database,
                            oelg_database=args.oelg_database, oelg_snapshot=args.oelg_snapshot,
                            pattern_database=args.pattern_database, doelg_raster_dir=args.doelg_raster_dir,
                            oelg_raster_dir=args.oelg_raster_dir)
    if args.task not in SOURCE_TASKS:
        if args.source or args.show_source or any(value is not None for value in database_options.values()):
            parser.error('Source selection and database overrides apply only to reproduce, patterns and compare.')
        if args.overwrite:
            parser.error('--overwrite is only available for evaluation tasks')
        return run_oelg(args.config or PROJECT_ROOT / 'instantiation_validation/oelg_config.json',
                        maps_only=args.task == 'map', no_plots=args.no_plots, output_dir=args.output_dir, instance_database=args.instance_database)
    if args.instance_database is not None:
        parser.error('--instance-database applies only to OELG instantiate/map.')
    source = args.source or ('database' if args.task == 'compare' else 'archive')
    if args.task == 'reproduce' and source != 'archive':
        parser.error('Paper reproduction (5.5) uses archived paper inputs only. Use compare (5.3) for database comparison.')
    if source == 'database':
        if args.task == 'patterns' and (args.no_plots or args.overwrite):
            parser.error('Database pattern statistics do not accept --no-plots or --overwrite.')
        if args.task == 'patterns' and any(database_options[key] is not None for key in ('doelg_database', 'oelg_database', 'oelg_snapshot', 'doelg_raster_dir', 'oelg_raster_dir')):
            parser.error('Pattern statistics use --pattern-database, not instance-database arguments.')
        if args.task == 'compare' and args.pattern_database is not None:
            parser.error('Comparison uses instance databases; --pattern-database is only for reproduce and patterns.')
        from validation_sources import DEFAULT_DATABASE_CONFIG, load_database_profile, run_database_validation
        config = args.config or DEFAULT_DATABASE_CONFIG
        if args.show_source:
            profile = load_database_profile(config, database_options)
            display = {key: profile[key] for key in ('config_path', 'construct_config', 'doelg_database',
                                                      'oelg_database', 'oelg_snapshot', 'pattern_database',
                                                      'doelg_raster_dir', 'oelg_raster_dir')}
            print(json.dumps(dict(display, source=source, task=args.task, database_writes=False), indent=2))
            return display
        return run_database_validation(args.task, config, args.output_dir, args.no_plots,
                                        args.overwrite, database_options)
    if any(value is not None for value in database_options.values()):
        parser.error('Database arguments require --source database; archive mode does not query Neo4j.')
    if args.task == 'patterns' and (args.no_plots or args.output_dir or args.overwrite):
        parser.error('Archive pattern statistics accept only --config and source-display options.')
    if args.task == 'compare' and (args.no_plots or args.output_dir):
        parser.error('Archive comparison uses raster paths and output_dir from its configuration.')
    filename = 'importance_comparison_config.json' if args.task == 'compare' else 'paper_experiment_config.json'
    config = args.config or PROJECT_ROOT / 'instantiation_validation' / filename
    if args.show_source:
        display = dict(source=source, task=args.task, config_path=str(config.resolve()), database_writes=False)
        print(json.dumps(display, indent=2))
        return display
    if args.task == 'compare':
        return run_comparison(config, args.overwrite)
    from paper_experiment import pattern_summary, reproduce
    if args.task == 'patterns':
        report = dict(pattern_summary(config), source='archive')
        print(json.dumps(report, indent=2))
        return report
    return reproduce(config, args.output_dir, args.no_plots, args.overwrite)


if __name__ == '__main__':
    main()
