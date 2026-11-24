import argparse
import os
import sys
import time
from shapely.geometry import Polygon, shape
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import instantiation
import mapping
from ontology.neo4jmodel import Neo4jDatabaseConnector, GlobalDisasterEvent


def run_instantiation(workers, logic_db, target_db, events_path):
    """Build and persist instance events from a logic database and input GeoJSON."""
    print(f"\n{'=' * 50}")
    print(f"🌲 [PHASE 1] Starting Graph Instantiation")
    print(f"{'=' * 50}")
    start_time = time.time()

    instantiation.logic_graph_database = logic_db
    instantiation.ins_graph_database = target_db
    instantiation._EVENT_IMPACT_TYPE_CACHE.clear()
    instantiation._RELATE_EVENT_CACHE.clear()
    instantiation._OBSERVATION_TASK_CACHE.clear()

    print(f"Initializing connection to logic database: {logic_db}...")
    instantiation.neo = Neo4jDatabaseConnector(logic_db)

    instantiation.ALL_EVENTS_CACHE = {e.event_description: e for e in GlobalDisasterEvent.nodes.all()}

    zhengzhou_coords = [
        [113.549609, 34.791410],
        [113.657406, 34.791410],
        [113.657406, 34.702835],
        [113.549609, 34.702835],
        [113.549609, 34.791410]
    ]
    region_poly = Polygon(zhengzhou_coords)

    with open(events_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)

    polygons = []
    event_nodes = []

    for feature in geojson_data['features']:

        geometry = feature['geometry']

        shapely_obj = shape(geometry)

        if shapely_obj.geom_type in ('Polygon', 'MultiPolygon'):
            polygons.append(shapely_obj)

            event_name = feature['properties'].get('event', 'unknown')
            event_node = GlobalDisasterEvent.nodes.first_or_none(event_description=event_name)
            if not event_node:
                print(f"Event not found: {event_name}")
                raise ValueError(f"Event not found: {event_name}")
            else:
                event_nodes.append(event_node)

            print(f"Event found: {event_name}, approximate affected area: {shapely_obj.area:.6f}")

    print("Executing recursive instantiation algorithm...")

    instantiation.construct_instantiation_database(
        ori_events=event_nodes, ori_extents=polygons,
        region_extent=region_poly, max_depth=2
    )
    print(f"✅ Instantiation completed in: {time.time() - start_time:.2f} seconds")
    return True


def run_mapping(target_db, config, snapshot=None, output_dir=None, no_plots=False):
    """Run the distance-field mapping workflow for an instance database."""
    print(f"\n{'=' * 50}")
    print(f"🗺️ [PHASE 2] Starting Cartographic Mapping (Database: {target_db})")
    print(f"{'=' * 50}")
    start_time = time.time()

    mapping.ins_graph_database = target_db

    print("Executing Mapping calculations and plotting...")

    mapping.run_distance_fields(config, target_db, snapshot, output_dir, no_plots)

    print(f"✅ Mapping Phase completed in: {time.time() - start_time:.2f} seconds")
    return True


def main():
    """Parse command-line options and run the selected pipeline stage."""
    parser = argparse.ArgumentParser(
        description="Standardized Command Line Toolkit for Knowledge Graph Instantiation and Mapping.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    tasks_help = (
        "Select the execution stage profile:\n"
        "  instantiate : Execute recursive instantiation of abstract events to regional graph nodes\n"
        "  map         : Generate disaster risk maps and observation factor priority maps\n"
        "  all         : Execute instantiation followed by mapping sequentially"
    )

    parser.add_argument(
        '-t', '--task',
        type=str,
        required=True,
        choices=['instantiate', 'map', 'all'],
        help=tasks_help
    )

    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=10,
        help="Thread concurrency pool worker boundary constraint (Default: 10)"
    )

    parser.add_argument(
        '-ldb', '--logic_database',
        type=str,
        default="events-logic-all-1zz-sensor-en-0904",
        help="The Logic Neo4j database to read generic relations from."
    )

    parser.add_argument(
        '-idb', '--ins_database',
        type=str,
        default="events-ins-all-zhengzhou-test-1",
        help="The Instantiated Neo4j database to write specific instances to."
    )
    parser.add_argument('--map-config', default=os.path.join(project_root, 'instantiation_config.json'))
    parser.add_argument('--events', default=os.path.join(project_root, 'data/geo_dataset/events_zhengzhou/onset.geojson'),
                        help='GeoJSON with initial event names and impact geometries')
    parser.add_argument('--snapshot', help='Use an existing distance-field snapshot instead of Neo4j')
    parser.add_argument('--output-dir', help='New output directory for distance fields')
    parser.add_argument('--no-plots', action='store_true', help='Export rasters without PNGs')

    args = parser.parse_args()
    overall_start = time.time()

    if args.task == 'instantiate':
        run_instantiation(args.workers, args.logic_database, args.ins_database, args.events)
    elif args.task == 'map':
        run_mapping(args.ins_database, args.map_config, args.snapshot, args.output_dir, args.no_plots)
    elif args.task == 'all':
        print("\n🚀 [Pipeline Mode] Initiating End-to-End Instantiation & Mapping Pipeline...")
        run_instantiation(args.workers, args.logic_database, args.ins_database, args.events)
        run_mapping(args.ins_database, args.map_config, args.snapshot, args.output_dir, args.no_plots)
        print(
            f"\n🎉 [Pipeline Mode] Production run completed successfully! Total elapsed time: {time.time() - overall_start:.2f} seconds")


if __name__ == "__main__":
    main()
