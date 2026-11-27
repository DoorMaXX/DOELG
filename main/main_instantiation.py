import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.construct_config import configure_construct, DEFAULT_CONFIG_PATH
from src.neo4j_database import ensure_database
from src.instantiation_settings import (DEFAULT_CONFIG, load_instantiation_settings, load_update_settings,
                                        project_path, set_instantiation_settings)


def read_initial_events(settings):
    """Validate English event descriptions and polygon extents before graph copying or inference."""
    from shapely.geometry import Polygon, shape
    with Path(settings['events_file']).open(encoding='utf-8-sig') as stream:
        document = json.load(stream)
    if document.get('crs'):
        from pyproj import CRS
        crs = CRS.from_user_input(document['crs']['properties']['name'])
        if crs.to_epsg() != 4326 and crs != CRS.from_user_input('OGC:CRS84'):
            raise ValueError('Initial events must use EPSG:4326 / CRS84 longitude and latitude.')
    if document.get('type') != 'FeatureCollection':
        raise ValueError('The initial-event file must be a GeoJSON FeatureCollection.')
    features = []
    for index, feature in enumerate(document.get('features', []), 1):
        geometry = shape(feature['geometry'])
        if geometry.geom_type not in ('Polygon', 'MultiPolygon') or geometry.is_empty or not geometry.is_valid:
            raise ValueError(f'Initial event {index} must have a valid polygon or multipolygon extent.')
        if not isinstance(feature.get('properties', {}).get('event'), str) or not feature['properties']['event'].strip():
            raise ValueError(f'Initial event {index} is missing its event description.')
        features.append((feature, geometry))
    if not features:
        raise ValueError('The initial-event file contains no polygon or multipolygon events.')
    west, south, east, north = settings['study_area']
    region = Polygon([(west, north), (east, north), (east, south), (west, south), (west, north)])
    return region, features


def run_instantiation(settings, *, event_num_start=0, strict=False, initial_events=None):
    """Apply resolved settings and reuse the reference algorithm for fresh or appended initial events."""
    import instantiation
    from ontology.neo4jmodel import Neo4jDatabaseConnector, GlobalDisasterEvent

    print(f"\n{'=' * 50}\n[INSTANTIATION] Starting Graph Instantiation\n{'=' * 50}")
    start_time = time.time()
    instantiation.configure_runtime(settings)
    region_poly, features = initial_events or read_initial_events(settings)
    print(f"Logic database: {settings['logic_database']}; instance database: {settings['instance_database']}")
    print(f"Feature directory: {settings['features_folder']}; maximum depth: {settings['max_depth']}")
    print('Instantiation follows the sequential reference algorithm; workers is reserved.')
    instantiation.neo = Neo4jDatabaseConnector(settings['logic_database'])
    instantiation.ALL_EVENTS_CACHE = {event.event_description: event for event in GlobalDisasterEvent.nodes.all()}
    polygons, event_nodes = [], []
    for feature, geometry in features:
        event_name = feature['properties'].get('event', 'unknown')
        event_node = GlobalDisasterEvent.nodes.first_or_none(event_description=event_name)
        if not event_node:
            raise ValueError(f'Event not found: {event_name}')
        polygons.append(geometry)
        event_nodes.append(event_node)
        print(f'Event found: {event_name}, approximate affected area: {geometry.area:.6f}')
    ensure_database(settings['instance_database'])
    instantiation.construct_instantiation_database(
        ori_events=event_nodes, ori_extents=polygons, region_extent=region_poly,
        max_depth=settings['max_depth'], event_num_start=event_num_start, strict=strict)
    print(f'Instantiation completed in: {time.time() - start_time:.2f} seconds')
    return True


def run_mapping(target_db, config, snapshot=None, output_dir=None, no_plots=False):
    """Map the resolved database or an offline snapshot without loading instantiation clients."""
    from distance_fields import run
    print(f"\n{'=' * 50}\n[MAPPING] Database: {target_db}\n{'=' * 50}")
    start_time = time.time()
    run(config, snapshot, output_dir, no_plots, database=target_db)
    print(f'Mapping completed in: {time.time() - start_time:.2f} seconds')
    return True


def run_update_and_map(settings, config, output_dir=None, no_plots=False):
    """Copy a baseline graph, infer only new evidence and map the combined copied instance."""
    from src.instance_update import copy_instance_graph, instance_statistics
    initial_events = read_initial_events(settings)
    suffix = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    output = Path(output_dir) if output_dir else Path(settings['update_output_dir']) / suffix
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Choose an empty update output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    runtime = deepcopy(settings)
    runtime.update(geos_output_folder=str(output / 'inference'), event_trees_file=str(output / 'event_trees.pickle'),
                   execution_report=str(output / 'inference_execution.json'))
    report = dict(status='running', source_database=settings['source_database'],
                  target_database=settings['instance_database'], new_events_file=settings['events_file'],
                  new_initial_events=len(initial_events[1]), source_database_writes=False,
                  reference='KGC_2/instantiation_en.py:main_update_zhengzhou', output_dir=str(output))
    report_path = output / 'update_status.json'
    print(f'\n[UPDATE] {settings["source_database"]} -> {settings["instance_database"]}', flush=True)
    print(f'New initial events: {report["new_initial_events"]}; update output: {output}', flush=True)
    def save_report():
        """Record update progress without including credentials or model settings."""
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save_report()
    try:
        copied = copy_instance_graph(settings['source_database'], settings['instance_database'],
                                     batch_size=settings['copy_batch_size'])
        report.update(copy=copied, status='updating')
        save_report()
        set_instantiation_settings(runtime)
        run_instantiation(runtime, event_num_start=copied['next_event_num'], strict=True, initial_events=initial_events)
        after = instance_statistics(settings['instance_database'])
        if after['initial_events'] != copied['statistics']['initial_events'] + report['new_initial_events']:
            raise RuntimeError('Updated initial-event count does not match the copied baseline plus new evidence.')
        report.update(updated_statistics=after, status='mapping')
        save_report()
        run_mapping(settings['instance_database'], config, output_dir=str(output / 'maps'), no_plots=no_plots)
        report.update(status='complete', maps=str(output / 'maps'))
        save_report()
        print(f'Update and mapping completed: {report_path}', flush=True)
        return report
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__)
        save_report()
        raise


def parse_arguments(argv=None):
    """Parse explicit overrides and resolve all other defaults from the selected configuration."""
    parser = argparse.ArgumentParser(description='Instantiate graph events and map importance fields.')
    parser.add_argument('-t', '--task', required=True, choices=['instantiate', 'map', 'all', 'update'])
    parser.add_argument('--config', '--map-config', dest='map_config', default=str(DEFAULT_CONFIG),
                        help='Instantiation and mapping configuration; defaults to instantiation_config.json')
    parser.add_argument('--construct-config', default=str(DEFAULT_CONFIG_PATH),
                        help='Neo4j credentials and model settings; defaults to construct_config.json')
    parser.add_argument('--source-database', help='Existing DOELG instance database to copy for update')
    parser.add_argument('-ldb', '--logic_database', help='Override instantiation.logic_database')
    parser.add_argument('-idb', '--ins_database', help='Override database in the selected configuration')
    parser.add_argument('-w', '--workers', type=int, help='Reserved worker setting; reference execution is sequential')
    parser.add_argument('--events', help='Override the initial-event GeoJSON path')
    parser.add_argument('--max-depth', type=int, help='Override instantiation.max_depth')
    parser.add_argument('--snapshot', help='Map an offline snapshot instead of querying Neo4j')
    parser.add_argument('--output-dir', help='New update-run directory, or override the map output directory')
    parser.add_argument('--no-plots', action='store_true', help='Export GeoTIFFs without map PNGs')
    parser.add_argument('--show-config', action='store_true', help='Print effective settings and exit without connecting to services')
    args = parser.parse_args(argv)
    try:
        overrides = {'logic_database': args.logic_database, 'instance_database': args.ins_database,
                     'workers': args.workers, 'events_file': args.events, 'max_depth': args.max_depth}
        if args.task == 'update':
            if args.snapshot:
                parser.error('update copies a live instance database; --snapshot is for mapping only.')
            settings = load_update_settings(args.map_config, dict(overrides, source_database=args.source_database))
        else:
            if args.source_database:
                parser.error('--source-database is only available for update.')
            settings = load_instantiation_settings(args.map_config, overrides)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    args.map_config = settings['config_path']
    args.construct_config = str(project_path(args.construct_config))
    args.snapshot = str(project_path(args.snapshot)) if args.snapshot else None
    args.output_dir = str(project_path(args.output_dir)) if args.output_dir else None
    return args, settings


def main(argv=None):
    """Load connection settings first, then execute the requested task with resolved runtime values."""
    args, settings = parse_arguments(argv)
    configure_construct(args.construct_config)
    set_instantiation_settings(settings)
    if args.show_config:
        print(json.dumps(dict(settings, task=args.task, construct_config=args.construct_config,
                              snapshot=args.snapshot, output_dir=args.output_dir), indent=2))
        return 0
    if args.task == 'update':
        run_update_and_map(settings, args.map_config, args.output_dir, args.no_plots)
        return 0
    if args.task in ('instantiate', 'all'):
        run_instantiation(settings)
    if args.task in ('map', 'all'):
        run_mapping(settings['instance_database'], args.map_config, args.snapshot, args.output_dir, args.no_plots)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
