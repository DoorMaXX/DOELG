import argparse
import os
import sys
import time
from shapely.geometry import Polygon
import concurrent.futures

# Ensure correct package imports based on project structure
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import the existing modules directly to avoid code duplication
import instantiation
import mapping
from ontology.neo4jmodel import Neo4jDatabaseConnector, GlobalDisasterEvent


def run_instantiation(workers, logic_db, target_db):
    """
    Sets up the environment and executes the recursive instantiation pipeline
    by invoking the existing functions from the instantiation module.
    """
    print(f"\n{'=' * 50}")
    print(f"🌲 [PHASE 1] Starting Graph Instantiation")
    print(f"{'=' * 50}")
    start_time = time.time()

    # Override module globals to match CLI arguments
    instantiation.logic_graph_database = logic_db
    instantiation.ins_graph_database = target_db

    print(f"Initializing connection to logic database: {logic_db}...")
    neo_logic = Neo4jDatabaseConnector(logic_db)

    # Pre-cache global events for the instantiation module
    instantiation.ALL_EVENTS_CACHE = {e.event_description: e for e in GlobalDisasterEvent.nodes.all()}

    # Initialize the worker pool within the instantiation module
    instantiation.GLOBAL_EXECUTOR = concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=instantiation._init_worker,
        initargs=(logic_db,)
    )

    # Initial coordinate boundary for Zhengzhou region extent
    zhengzhou_coords = [
        [113.549609, 34.791410],
        [113.657406, 34.791410],
        [113.657406, 34.702835],
        [113.549609, 34.702835],
        [113.549609, 34.791410]
    ]
    region_poly = Polygon(zhengzhou_coords)

    demo_events_desc = [
        "Extreme Precipitation,Heavy Rain",
        "Administrative Region,Urban Waterlogging"
    ]

    ori_events = []
    for desc in demo_events_desc:
        ev = GlobalDisasterEvent.nodes.first_or_none(event_description=desc)
        if ev:
            ori_events.append(ev)

    if not ori_events:
        print("Warning: No matching GlobalDisasterEvents found for instantiation.")
        instantiation.GLOBAL_EXECUTOR.shutdown(wait=True)
        return False

    ori_extents = [region_poly for _ in ori_events]

    print("Executing recursive instantiation algorithm...")

    # Trigger the core logic directly from the instantiation module
    instantiation.construct_instantiation_database(
        ori_events=ori_events,
        ori_extents=ori_extents,
        ori_features=[],
        region_extent=region_poly,
        max_depth=2,
        max_workers=workers
    )

    instantiation.GLOBAL_EXECUTOR.shutdown(wait=True)
    print(f"✅ Instantiation completed in: {time.time() - start_time:.2f} seconds")
    return True


def run_mapping(target_db):
    """
    Sets up the environment and triggers the cartographic mapping workflow
    by invoking the main process from the mapping module.
    """
    print(f"\n{'=' * 50}")
    print(f"🗺️ [PHASE 2] Starting Cartographic Mapping (Database: {target_db})")
    print(f"{'=' * 50}")
    start_time = time.time()

    # Sync module targets to the CLI arguments
    mapping.ins_graph_database = target_db

    print("Executing Mapping calculations and plotting...")
    # Trigger the main mapping execution logic
    mapping.main_init()

    print(f"✅ Mapping Phase completed in: {time.time() - start_time:.2f} seconds")
    return True


def main():
    """
    CLI Router for the Instantiation and Mapping pipeline.
    """
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
        default="events-logic-all-1zz-sensor-en",
        help="The Logic Neo4j database to read generic relations from."
    )

    parser.add_argument(
        '-idb', '--ins_database',
        type=str,
        default="events-ins-all-zhengzhou-1-sensor-en-parallel",
        help="The Instantiated Neo4j database to write specific instances to."
    )

    args = parser.parse_args()
    overall_start = time.time()

    if args.task == 'instantiate':
        run_instantiation(args.workers, args.logic_database, args.ins_database)

    elif args.task == 'map':
        run_mapping(args.ins_database)

    elif args.task == 'all':
        print("\n🚀 [Pipeline Mode] Initiating End-to-End Instantiation & Mapping Pipeline...")
        run_instantiation(args.workers, args.logic_database, args.ins_database)
        run_mapping(args.ins_database)
        print(
            f"\n🎉 [Pipeline Mode] Production run completed successfully! Total elapsed time: {time.time() - overall_start:.2f} seconds")


if __name__ == "__main__":
    main()