import argparse
import os
import sys
import time
import pickle

# Ensure correct package imports based on project structure
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.construct_config import configure_from_cli, get_config_path, get_configured_path, get_setting

configure_from_cli()

# Import ontology components and graph connectors
from ontology.neo4jmodel import Neo4jDatabaseConnector

# Import generalization scripts and logic
from generalization import (
    get_geo_impact_pair, similarity_calculation, process_abstract_clustering_to_graph,
    _link_envDes_to_origin_event, _append_cause_count_to_global_cause,
    calculate_global_event_causal_confidence, output_top_causal_pairs,
    generalize_spatial_relations_parallel, generate_env_rule_to_database_parallel,
    construct_generalized_observation_event, construct_global_disaster_arguments,
    update_global_DO_argument, generalize_observation_patterns, pattern_statistics,
    extract_standard_factor_observations, construct_observation_events_all,
    sensor_reclassify_parallel, extract_typical_key_paths, analyze_disaster_and_observation_stats,
    find_top_influence_disasters, find_top_hub_disasters, connect_disaster_trigger_relate_obs,
    export_disaster_observation_to_json
)
from other.factor_map import main_factor_map_parallel_1

# ==========================================
# 1. PHASE 1: DISASTER EVENT GENERALIZATION
# ==========================================

def run_disaster_generalization(workers):
    """
    Groups specific disaster event pairs into abstract GlobalDisasterEvents, calculates
    similarity clustering, maps the hierarchy to Neo4j, and calculates causal confidence.
    """
    print(f"\n{'=' * 50}")
    print(f"🌋 [PHASE 1] Starting Disaster Event Generalization")
    print(f"{'=' * 50}")
    start_time = time.time()

    print("-> Step 1/4: Extracting and mapping geo-impact pairs...")
    disaster_core = get_geo_impact_pair()

    print(f"-> Step 2/4: Performing Agglomerative Clustering on {len(disaster_core)} pairs...")
    # Calculates semantic, environmental, and spatial similarity
    disaster_pair_clustered = similarity_calculation(disaster_core)

    print("-> Step 3/4: Mapping cluster hierarchy (Global & Merge nodes) into Neo4j...")
    process_abstract_clustering_to_graph(disaster_pair_clustered)
    _link_envDes_to_origin_event()
    _append_cause_count_to_global_cause()

    print("-> Step 4/4: Calculating global causal confidence and ranking top pairs...")
    calculate_global_event_causal_confidence()
    output_top_causal_pairs(get_setting('generalization.statistics.causal_top_n'))

    print(f"✅ Disaster Event Generalization completed in: {time.time() - start_time:.2f} seconds")
    return disaster_pair_clustered


# ==========================================
# 2. PHASE 2: SPATIAL & ENVIRONMENTAL RULES
# ==========================================

def run_spatial_and_env_rules(disaster_pair_clustered, workers):
    """
    Induces prevalent spatial relationships (Topology, Distance, Orientation) for generalized
    disaster patterns using an LLM, and generates quantitative environmental rules.
    """
    print(f"\n{'=' * 50}")
    print(f"📐 [PHASE 2] Starting Spatial Relations & Environmental Rules Induction")
    print(f"{'=' * 50}")
    start_time = time.time()

    if not disaster_pair_clustered:
        # Load from disk if running this step in isolation
        pickle_path = get_configured_path('generalization.paths.clusters')
        if os.path.exists(pickle_path):
            print(f"Loading clusters from {pickle_path}...")
            with open(pickle_path, "rb") as f:
                disaster_pair_clustered = pickle.load(f)
        else:
            print("❌ Error: No clustered data found in memory or disk. Please run Phase 1 first.")
            return False

    print("-> Step 1/2: Generalizing spatial relationships via LLM...")
    generalize_spatial_relations_parallel(disaster_pair_clustered, workers)

    print("-> Step 2/2: Generating quantitative environmental rules...")
    generate_env_rule_to_database_parallel(max_workers=workers)

    print(f"✅ Spatial & Environmental Rules Induction completed in: {time.time() - start_time:.2f} seconds")
    return True


# ==========================================
# 3. PHASE 3: OBSERVATION EVENT GENERALIZATION
# ==========================================

def run_observation_generalization(neo_connector, workers):
    """
    Standardizes factor mappings, integrates existing observations, constructs new generalized
    public observations, and statistically summarizes observation patterns.
    """
    print(f"\n{'=' * 50}")
    print(f"🛰️ [PHASE 3] Starting Observation Event Generalization")
    print(f"{'=' * 50}")
    start_time = time.time()

    # Step 1: Aligning observable parameters (Factor Alignment)
    print("-> Step 1/6: Aligning observation factors to standard indicators...")
    main_factor_map_parallel_1(neo_connector, neo_connector.database_name, max_workers=workers)
    sensor_reclassify_parallel(max_workers=workers)

    # Step 2: Integrating existing observations
    print("-> Step 2/6: Integrating existing observation events into abstract schema...")
    construct_generalized_observation_event()
    construct_global_disaster_arguments()
    update_global_DO_argument()

    # Step 3: Generalizing Observation Patterns
    print("-> Step 3/6: Statistically abstracting observation patterns...")
    generalize_observation_patterns()

    # (Optional Step: Target DB Migration - Uncomment if explicitly required per specific setup)
    # print("-> Step 4/6: Migrating cross-database observation knowledge...")
    # migrate_observation_knowledge(source_db='events-do-gen-2', target_db=neo_connector.database_name)

    # Step 4 & 5: Constructing New Generalized Observations
    print("-> Step 4/6: Extracting baseline standard factor observations...")
    extract_standard_factor_observations()

    print("-> Step 5/6: Constructing new public observation events for coverage gaps...")
    construct_observation_events_all()

    print("-> Step 6/6: Connecting cascade triggering paths for observations...")
    connect_disaster_trigger_relate_obs()

    print(f"✅ Observation Event Generalization completed in: {time.time() - start_time:.2f} seconds")
    return True


# ==========================================
# 4. PHASE 4: STATISTICAL OUTPUT
# ==========================================

def run_statistics():
    """
    Extracts key operational data from the generalized graph: multi-hop causal chains,
    hub vulnerability analytics, and exports observation triggers to JSON.
    """
    print(f"\n{'=' * 50}")
    print(f"📊 [PHASE 4] Starting Statistical Summarization and Output")
    print(f"{'=' * 50}")
    start_time = time.time()

    print("-> Executing: Disaster Pattern Frequency Analytics...")
    pattern_statistics()

    path_settings = get_setting('generalization.statistics.paths')
    print(f"\n-> Executing: Multi-hop Causal Chain Extraction (Top {path_settings['top_n']} paths, {path_settings['hops']} hops)...")
    extract_typical_key_paths(hops=path_settings['hops'], top_n=path_settings['top_n'])

    print("\n-> Executing: Global Graph Degree and Hub Analytics...")
    analyze_disaster_and_observation_stats()
    find_top_influence_disasters()
    find_top_hub_disasters()

    print("\n-> Executing: Exporting Disaster-Observation hierarchies to JSON...")
    export_disaster_observation_to_json()

    print(f"✅ Statistical Output completed in: {time.time() - start_time:.2f} seconds")
    return True


# ==========================================
# 5. COMMAND LINE INTERFACE ROUTER
# ==========================================

def main():
    """
    Standardized Command Line Toolkit for Knowledge Graph Generalization tasks.
    Routes execution to specific modular workflows based on provided arguments.
    """
    parser = argparse.ArgumentParser(
        description="Standardized Command Line Toolkit for Knowledge Graph Generalization.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    tasks_help = (
        "Select the execution stage profile:\n"
        "  disaster_gen  : Group specific event pairs into abstract GlobalDisasterEvents and calculate cluster similarity\n"
        "  spatial_rule  : Induce spatial topological/distance relations and quantitative environmental rules via LLM\n"
        "  obs_gen       : Align factors, integrate and construct generalized observation events, and summarize patterns\n"
        "  stats         : Extract multi-hop causal chains, graph hub metrics, and export data to JSON\n"
        "  all           : Execute the entire generalization pipeline sequentially"
    )

    parser.add_argument(
        '-t', '--task',
        type=str,
        required=True,
        choices=['disaster_gen', 'spatial_rule', 'obs_gen', 'stats', 'all'],
        help=tasks_help
    )

    parser.add_argument(
        '--config',
        default=str(get_config_path()),
        help="Construction configuration file (default: construct_config.json in the project root)"
    )

    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=get_setting('generalization.workers'),
        help="Thread pool size; overrides generalization.workers in the configuration"
    )

    parser.add_argument(
        '-db', '--database',
        type=str,
        default=get_setting('generalization.database'),
        help="The Neo4j database name to connect to."
    )

    args = parser.parse_args()
    overall_start = time.time()

    print(f"Connecting to Neo4j database: {args.database}...")
    try:
        neo_connector = Neo4jDatabaseConnector(args.database)
    except Exception as e:
        print(f"Failed to connect to database '{args.database}'. Error: {e}")
        sys.exit(1)

    # Initialize shared variables for sequential execution
    disaster_clusters = None

    if args.task == 'disaster_gen':
        run_disaster_generalization(args.workers)
    elif args.task == 'spatial_rule':
        run_spatial_and_env_rules(None, args.workers)
    elif args.task == 'obs_gen':
        run_observation_generalization(neo_connector, args.workers)
    elif args.task == 'stats':
        run_statistics()
    elif args.task == 'all':
        print("\n🚀 [Pipeline Mode] Initiating End-to-End Generalization Pipeline...")

        disaster_clusters = run_disaster_generalization(args.workers)
        run_spatial_and_env_rules(disaster_clusters, args.workers)
        run_observation_generalization(neo_connector, args.workers)
        run_statistics()

        print(
            f"\n🎉 [Pipeline Mode] Generalization pipeline completed successfully! Total elapsed time: {time.time() - overall_start:.2f} seconds")


if __name__ == "__main__":
    main()
