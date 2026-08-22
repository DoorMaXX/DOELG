from langchain.agents import create_agent
from ontology.neo4jmodel import Neo4jDatabaseConnector, GeoObject, DisPhen, DisasterEvent, OriginText, Geometry, \
    MergeDisasterEvent, GlobalDisasterEvent, EnvironmentDescription, ObservationEvent, GlobalObservationEvent, \
    MergeObservationEvent, GeneralizedObservationEvent, Sensor, Factor, GeneralizedObservationPattern, EnvironmentRule, \
    StandardFactor
from tqdm import tqdm
from event_extraction import geo_types, impact_types, llm
from fusion import get_ollama_embedding
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from other.environment_descrption import get_env_rule_chain
import concurrent.futures
from instantiation_o import migrate_observation_knowledge
from other.factor_map import main_factor_map_parallel_1, indicator_mapping, main_factor_map_
import json
from neomodel import db

"""
Generalization to disaster event pairs and generalization to observed events
"""


def get_geo_impact_pair():
    """
    Extracts pairs of causally linked disaster events from the database and generalizes
    them into a dictionary grouping specific event pairs by their generic geographic and disaster Phenomenon types.
    """
    disaster_event_nodes = DisasterEvent.nodes.all()
    geo_impact_pair = {}
    for en in disaster_event_nodes:
        ca_geo_object = en.geo_argument.all()[0]
        ca_geo_name = ca_geo_object.geo_object_name
        ca_geo_type = ca_geo_object.geo_type

        ca_impact = en.impact_argument.all()[0]
        ca_impact_name = ca_impact.impact_name
        ca_impact_type = ca_impact.impact_type

        if ca_geo_type not in geo_types or ca_impact_type not in impact_types:
            continue

        ca_event_gen = f"{ca_geo_type},{ca_impact_type}"
        ca_event = en.event_description
        for re_en in en.cause.all():
            re_geo_object = re_en.geo_argument.all()[0]
            re_geo_name = re_geo_object.geo_object_name
            re_geo_type = re_geo_object.geo_type

            re_impact = re_en.impact_argument.all()[0]
            re_impact_name = re_impact.impact_name
            re_impact_type = re_impact.impact_type

            if re_geo_type not in geo_types or re_impact_type not in impact_types:
                continue

            re_event_gen = f"{re_geo_type},{re_impact_type}"
            re_event = re_en.event_description

            if ca_event_gen == re_event_gen:
                continue

            event_pair = [ca_event, re_event]

            event_pair_gen = f"{ca_event_gen}-{re_event_gen}"
            event_pairs = geo_impact_pair.get(event_pair_gen, [])
            if event_pair not in event_pairs:
                event_pairs.append(event_pair)
            geo_impact_pair[event_pair_gen] = event_pairs
    return geo_impact_pair


EMBEDDING_CACHE = {}


def get_cached_embedding(text):
    """
    Retrieves the embedding for a given text, utilizing an in-memory cache to avoid
    redundant API calls for identical strings.
    """
    if not text or not isinstance(text, str):
        return np.zeros(4096)
    if text not in EMBEDDING_CACHE:
        EMBEDDING_CACHE[text] = get_ollama_embedding(text)
    return EMBEDDING_CACHE[text]


def cosine_similarity(v1, v2):
    """
    Computes the cosine similarity between two numerical vectors.
    """
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0: return 0
    return np.dot(v1, v2) / norm


def prefetch_event_data(pairs):
    """
    Prefetches event descriptions, environmental descriptions, and spatial relationship
    data from the Neo4j database in a single pass to optimize similarity calculations.
    """
    prefetched = []
    unique_texts = set()

    for pair in pairs:
        ca_en = DisasterEvent.nodes.first_or_none(event_description=pair[0])
        re_en = DisasterEvent.nodes.first_or_none(event_description=pair[1])

        if not ca_en or not re_en:
            prefetched.append(None)
            continue

        event_desc = f"{ca_en.event_description}导致{re_en.event_description}"

        ca_env = ca_en.envDes_argument.all()[0].description if ca_en.envDes_argument.all() else ""
        re_env = re_en.envDes_argument.all()[0].description if re_en.envDes_argument.all() else ""
        env_desc = f"{ca_env}{re_env}"

        spatial_data = None
        rel = ca_en.spatial_to.relationship(re_en)
        if rel and rel.relations:
            sr = rel.relations
            # 只取需要的字段
            spatial_data = {
                "topo": str(sr.get("topology_computation", "")),
                "dist": sr.get("distance_computation"),
                "basin": f"{sr.get('basin_computation', {}).get('overlap', '')}-{sr.get('basin_computation', {}).get('up_down_stream', '')}",
                "sem": str(sr.get("semantics_path", ""))
            }

        unique_texts.update([event_desc, env_desc])
        if spatial_data:
            unique_texts.update([spatial_data["topo"], spatial_data["basin"], spatial_data["sem"]])

        prefetched.append({
            "event_desc": event_desc,
            "env_desc": env_desc,
            "spatial": spatial_data,
            "original_pair": pair
        })

    return prefetched


def calculate_pair_score(feat1, feat2):
    """
    Calculates a weighted similarity score between two event features in memory,
    evaluating their event descriptions, environmental descriptions, and spatial relationships.
    """
    if not feat1 or not feat2: return 0

    # 1. Event Argument Similarity
    s1 = cosine_similarity(get_cached_embedding(feat1["event_desc"]),
                           get_cached_embedding(feat2["event_desc"]))

    # 2. Hazard-forming Environment Similarity
    s2 = cosine_similarity(get_cached_embedding(feat1["env_desc"]),
                           get_cached_embedding(feat2["env_desc"]))

    # 3. Spatial Relation Similarity
    s3 = -1
    sp1, sp2 = feat1["spatial"], feat2["spatial"]

    if sp1 and sp2:
        if all([sp1["topo"], sp1["dist"], sp1["basin"], sp1["sem"],
                sp2["topo"], sp2["dist"], sp2["basin"], sp2["sem"]]):

            topo_sim = cosine_similarity(get_cached_embedding(sp1["topo"]), get_cached_embedding(sp2["topo"]))
            dist_sim = max(0, 1 - abs(sp1["dist"] - sp2["dist"]) / 10000)
            basin_sim = cosine_similarity(get_cached_embedding(sp1["basin"]), get_cached_embedding(sp2["basin"]))
            sem_sim = cosine_similarity(get_cached_embedding(sp1["sem"]), get_cached_embedding(sp2["sem"]))

            s3 = 0.3 * topo_sim + 0.15 * dist_sim + 0.25 * basin_sim + 0.3 * sem_sim
        elif sp1["sem"] and sp2["sem"]:
            s3 = cosine_similarity(get_cached_embedding(sp1["sem"]), get_cached_embedding(sp2["sem"]))

    if s3 >= 0:
        return 0.2 * s1 + 0.5 * s2 + 0.3 * s3
    return 0.3 * s1 + 0.7 * s2


def similarity_calculation(event_pairs: dict, threshold=0.65):
    """
    Performs Agglomerative Clustering on event pairs based on their similarity scores
    to group highly similar disaster event evolutions. Saves the results to a pickle file.
    """
    clustered_results = {}

    for key, pairs in tqdm(event_pairs.items(), desc="Total Progress"):
        n = len(pairs)
        if n == 0:
            clustered_results[key] = []
            continue
        if n == 1:
            clustered_results[key] = [pairs]
            continue

        features = prefetch_event_data(pairs)

        sim_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                score = calculate_pair_score(features[i], features[j])
                sim_matrix[i, j] = sim_matrix[j, i] = score

        dist_matrix = np.maximum(0, 1 - sim_matrix)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=1 - threshold
        )
        labels = clustering.fit_predict(dist_matrix)

        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(pairs[idx])
        clustered_results[key] = list(clusters.values())

    # 持久化
    with open('all-cluster.pickle', 'wb') as f:
        pickle.dump(clustered_results, f)

    return clustered_results


def process_abstract_clustering_to_graph(clustered_results):
    """
    Maps the clustered event pairs into the Neo4j graph by creating and linking
    GlobalDisasterEvent (abstract level) and MergeDisasterEvent (cluster level) nodes.
    """
    count = 0
    for pattern_label, clusters in tqdm(clustered_results.items()):
        parts = pattern_label.split("-")
        if len(parts) != 2:
            continue
        cause_desc_gen = parts[0]
        result_desc_gen = parts[1]

        if len(clusters) > 1:
            print("聚类结果：", clusters)

        for cluster_idx, pairs in enumerate(clusters):
            current_cluster_id = f"{pattern_label}_C{cluster_idx}"

            g_cause = GlobalDisasterEvent.nodes.first_or_none(event_description=cause_desc_gen)
            if not g_cause:
                g_cause = GlobalDisasterEvent(event_description=cause_desc_gen).save()
            g_result = GlobalDisasterEvent.nodes.first_or_none(event_description=result_desc_gen)
            if not g_result:
                g_result = GlobalDisasterEvent(event_description=result_desc_gen).save()

            m_cause = _get_or_create_merge_node(cause_desc_gen, current_cluster_id)
            m_result = _get_or_create_merge_node(result_desc_gen, current_cluster_id)

            if not g_cause.include.is_connected(m_cause):
                g_cause.include.connect(m_cause)
            if not g_result.include.is_connected(m_result):
                g_result.include.connect(m_result)

            rel_m = m_cause.cause.relationship(m_result)
            if rel_m:
                rel_m.relation_count += len(pairs)
                rel_m.save()
            else:
                m_cause.cause.connect(m_result, {'relation_count': len(pairs)})

            _connect_global_cause_with_pattern(g_cause, g_result, current_cluster_id)

            for specific_cause_text, specific_result_text in pairs:
                _link_to_original_event(m_cause, specific_cause_text, 'MergeDisasterEvent', 'DisasterEvent')
                _link_to_original_event(m_result, specific_result_text, 'MergeDisasterEvent', 'DisasterEvent')

                _link_factors_to_global(g_cause, specific_cause_text)
                _link_factors_to_global(g_result, specific_result_text)

    _link_envDes_to_origin_event()


def construct_generalized_observation_event():
    """
    Constructs GeneralizedObservationEvent nodes by aggregating observation data
    (sensors, platforms, factors, and geographic objects) triggered by GlobalDisasterEvents.
    """
    global_disaster_nodes = GlobalDisasterEvent.nodes.all()
    for global_disaster_node in global_disaster_nodes:
        merge_disaster_nodes = global_disaster_node.include.all()
        has_done_disasters = []
        for merge_disaster_node in merge_disaster_nodes:
            disaster_nodes = merge_disaster_node.include.all()
            for disaster_node in disaster_nodes:
                if disaster_node.node_id in has_done_disasters:
                    continue
                has_done_disasters.append(disaster_node.node_id)

                observation_nodes = disaster_node.trigger.all()
                for observation_node in observation_nodes:
                    sensor_node = observation_node.sensor_argument.all()[0]
                    sensor_type = sensor_node.sensor_type
                    sensor_platform = sensor_node.platform

                    factor_name = ""
                    factor_nodes = observation_node.factor_argument.all()
                    if factor_nodes:
                        factor_name = factor_nodes[0].factor_name

                    geo_type = ""
                    geometry = ""
                    geo_nodes = observation_node.geo_argument.all()
                    if geo_nodes:
                        geo_type = geo_nodes[0].geo_type
                        geometry = geo_nodes[0].geometry

                    generalized_observation_eventDes = \
                        f"[{sensor_type},{sensor_platform}]-[{geo_type},{factor_name}]-[{global_disaster_node.node_id}]"
                    generalized_observation_node = (
                        GeneralizedObservationEvent.nodes.first_or_none(
                            event_description=generalized_observation_eventDes))
                    if not generalized_observation_node:
                        generalized_observation_node = (
                            GeneralizedObservationEvent(event_description=generalized_observation_eventDes)).save()

                    if not generalized_observation_node.include.is_connected(observation_node):
                        generalized_observation_node.include.connect(observation_node)
                        print(
                            f"create [include]: ({generalized_observation_eventDes}) - ({observation_node.event_description})")

                    rel_m = global_disaster_node.trigger.relationship(generalized_observation_node)
                    if rel_m:
                        rel_m.relation_count += 1
                        rel_m.save()
                        print(
                            f"update [trigger]: ({global_disaster_node.event_description}) - ({generalized_observation_eventDes})")
                    else:
                        global_disaster_node.trigger.connect(generalized_observation_node)
                        print(
                            f"create [trigger]: ({global_disaster_node.event_description}) - ({generalized_observation_eventDes})")

                    generalized_sensor_node = (
                        Sensor.nodes.first_or_none(sensor_name=sensor_type,
                                                   sensor_type=sensor_type,
                                                   platform=sensor_platform))
                    if not generalized_sensor_node:
                        generalized_sensor_node = Sensor(sensor_name=sensor_type,
                                                         sensor_type=sensor_type,
                                                         platform=sensor_platform).save()
                    if not generalized_observation_node.sensor_argument.is_connected(generalized_sensor_node):
                        generalized_observation_node.sensor_argument.connect(generalized_sensor_node)
                        print(
                            f"create [obs sensor]: ({generalized_observation_eventDes}) - [{sensor_type},{sensor_platform}]")

                    if len(factor_nodes) > 0:
                        standard_factor_nodes = factor_nodes[0].relate.all()
                        if not generalized_observation_node.factor_argument.is_connected(factor_nodes[0]):
                            generalized_observation_node.factor_argument.connect(factor_nodes[0])
                            print(f"create [obs factor]: ({generalized_observation_eventDes}) - [{factor_name}]")

                        if len(standard_factor_nodes) > 0:
                            if not generalized_observation_node.std_factor_argument.is_connected(
                                    standard_factor_nodes[0]):
                                generalized_observation_node.std_factor_argument.connect(standard_factor_nodes[0])
                                print(
                                    f"cre`ate [obs std factor]: ({generalized_observation_eventDes}) - [{standard_factor_nodes[0].factor_name}]")

                        if not generalized_sensor_node.observe_factor.is_connected(factor_nodes[0]):
                            generalized_sensor_node.observe_factor.connect(factor_nodes[0])
                            print(f"create [sensor factor]: [{sensor_type},{sensor_platform}] - [{factor_name}]")

                        if not global_disaster_node.observed.is_connected(factor_nodes[0]):
                            global_disaster_node.observed.connect(factor_nodes[0])
                            print(
                                f"create [disaster factor]: ({global_disaster_node.event_description}) - [{factor_name}]")

                    if len(geo_nodes) > 0:
                        generalized_geo_node = (
                            GeoObject.nodes.first_or_none(geo_object_name=geo_type, geo_type=geo_type))
                        if not generalized_geo_node:
                            generalized_geo_node = (
                                GeoObject(geo_object_name=geo_type, geo_type=geo_type, geometry=geometry).save())
                        if not generalized_observation_node.geo_argument.is_connected(generalized_geo_node):
                            generalized_observation_node.geo_argument.connect(generalized_geo_node)
                            print(f"create [obs geo]: ({generalized_observation_eventDes}) - [{geo_type}]")
                        if not generalized_sensor_node.observe_geo.is_connected(generalized_geo_node):
                            generalized_sensor_node.observe_geo.connect(generalized_geo_node)
                            print(f"create [sensor geo]: [{sensor_type},{sensor_platform}] - [{geo_type}]")

                        if len(factor_nodes) > 0:
                            if not generalized_geo_node.has_factor.is_connected(factor_nodes[0]):
                                generalized_geo_node.has_factor.connect(factor_nodes[0])
                                print(f"creat [geo factor]: [{geo_type}] - [{factor_name}]")


def construct_global_disaster_arguments():
    """
    Creates and connects abstract geographic (GeoObject) and disaster Phenomenon (DisPhen) argument nodes
    to their corresponding GlobalDisasterEvent nodes.
    """
    global_disaster_nodes = GlobalDisasterEvent.nodes.all()
    for global_disaster_node in global_disaster_nodes:
        gd_geo_type = str(global_disaster_node.event_description).split(',')[0]
        gd_impact_type = str(global_disaster_node.event_description).split(',')[1]

        gd_geo_node = GeoObject.nodes.first_or_none(geo_object_name=gd_geo_type, geo_type=gd_geo_type)
        if not gd_geo_node:
            gd_geo_node = GeoObject(geo_object_name=gd_geo_type, geo_type=gd_geo_type).save()
        if not global_disaster_node.geo_argument.is_connected(gd_geo_node):
            global_disaster_node.geo_argument.connect(gd_geo_node)
            print(f"create [disaster geo]: ({global_disaster_node.event_description}) - ({gd_geo_type})")

        gd_impact_node = DisPhen.nodes.first_or_none(impact_name=gd_impact_type, impact_type=gd_impact_type)
        if not gd_impact_node:
            gd_impact_node = DisPhen(impact_name=gd_impact_type, impact_type=gd_impact_type).save()
        if not global_disaster_node.impact_argument.is_connected(gd_impact_node):
            global_disaster_node.impact_argument.connect(gd_impact_node)
            print(f"create [disaster impact]: ({global_disaster_node.event_description}) - ({gd_impact_type})")

        if not gd_geo_node.happen.is_connected(gd_impact_node):
            gd_geo_node.happen.connect(gd_impact_node)
            print(f"create [geo impact]: ({gd_geo_type}) - ({gd_impact_type})")

        merge_disaster_nodes = global_disaster_node.include.all()
        has_done_disasters = []
        for merge_disaster_node in merge_disaster_nodes:
            disaster_nodes = merge_disaster_node.include.all()
            for disaster_node in disaster_nodes:
                if disaster_node.node_id in has_done_disasters:
                    continue
                has_done_disasters.append(disaster_node.node_id)

                d_factor_nodes = disaster_node.observed.all()
                for d_factor_node in d_factor_nodes:
                    if not global_disaster_node.observed.is_connected(d_factor_node):
                        global_disaster_node.observed.connect(d_factor_node)
                        print(
                            f"create [disaster factor]: {global_disaster_node.event_description} - {d_factor_node.factor_name}")
                    if gd_geo_node:
                        if d_factor_node.factor_type == "environment_factors":
                            continue
                        if not gd_geo_node.has_factor.is_connected(d_factor_node):
                            r = gd_geo_node.has_factor.connect(d_factor_node)
                            print(
                                f"create [(d)geo factor]: {gd_geo_node.geo_object_name} - {d_factor_node.factor_name}")


def update_global_DO_argument():
    """
    Updates graph relationships to link global disaster arguments with generalized
    observation arguments, associating geographic locations with observed factors.
    """
    global_disaster_nodes = GlobalDisasterEvent.nodes.all()
    for global_disaster_node in global_disaster_nodes:

        d_geo_nodes = global_disaster_node.geo_argument.all()
        d_geo_node = None
        if d_geo_nodes:
            d_geo_node = d_geo_nodes[0]

        gen_obs_nodes = global_disaster_node.trigger.all()
        for gen_obs_node in gen_obs_nodes:
            o_geo_nodes = gen_obs_node.geo_argument.all()
            o_geo_node = None
            if o_geo_nodes:
                o_geo_node = o_geo_nodes[0]

            if o_geo_node and d_geo_node:
                if o_geo_node.node_id != d_geo_node.node_id:
                    if not d_geo_node.associate.is_connected(o_geo_node):
                        r = d_geo_node.associate.connect(o_geo_node)
                        print(f"create [(d)geo (o)geo]:{d_geo_node.geo_object_name} - {o_geo_node.geo_object_name}")

            o_factor_nodes = gen_obs_node.factor_argument.all()
            for o_factor_node in o_factor_nodes:
                if d_geo_node:
                    if o_factor_node.factor_type == "environment_factors":
                        continue
                    if not d_geo_node.has_factor.is_connected(o_factor_node):
                        r = d_geo_node.has_factor.connect(o_factor_node)
                        print(f"create [(d)geo factor]: {d_geo_node.geo_object_name} - {o_factor_node.factor_name}")
                if not global_disaster_node.observed.is_connected(o_factor_node):
                    r = global_disaster_node.observed.connect(o_factor_node)
                    print(
                        f"create [disaster factor]: {global_disaster_node.event_description} - {o_factor_node.factor_name}")


def calculate_global_event_causal_confidence():
    """
    Calculates and updates the causal confidence and lift score for 'CAUSE' relationships
    between GlobalDisasterEvent nodes based on relationship frequencies and path uniqueness.
    """
    query = """
        MATCH (m1:GlobalDisasterEvent)-[r:CAUSE]->(m2:GlobalDisasterEvent)
        OPTIONAL MATCH (m1)-[:CAUSE]->(mid:GlobalDisasterEvent)-[:CAUSE]->(m2)
        WITH m1, m2, r, count(distinct mid) as path_count
        RETURN m1.node_id as source_id, 
               m2.node_id as target_id, 
               r.relation_count as edge_count,
               path_count
        """
    results, _ = db.cypher_query(query)

    if not results:
        print("未找到 MergeDisasterEvent 间的 CAUSE 关系，请检查 count 属性是否已写入。")
        return

    out_counts_query = """
    MATCH (m:GlobalDisasterEvent)-[r:CAUSE]->()
    RETURN m.node_id as node_id, sum(r.relation_count) as total_out
    """
    out_results, _ = db.cypher_query(out_counts_query)
    out_dict = {row[0]: row[1] for row in out_results}

    total_instances_query = "MATCH (:GlobalDisasterEvent)-[r:CAUSE]->() RETURN sum(r.relation_count)"
    total_instances = db.cypher_query(total_instances_query)[0][0][0] or 1

    in_counts_query = """
    MATCH ()-[r:CAUSE]->(m:GlobalDisasterEvent)
    RETURN m.node_id as node_id, sum(r.count) as total_in
    """
    in_results, _ = db.cypher_query(in_counts_query)
    in_dict = {row[0]: row[1] for row in in_results}

    print(f"正在计算 {len(results)} 条事理模式对的置信度...")

    import math
    for row in tqdm(results, "灾害引发置信度计算"):
        s_id, t_id, count, path_count = row

        total_out = out_dict.get(s_id, 0)
        p_b_given_a = (count + 1) / (total_out + len(out_dict))

        total_in = in_dict.get(t_id, 0)
        p_b = total_in / total_instances
        lift = p_b_given_a / p_b if p_b > 0 else 1.0

        lift_factor = math.tanh(lift / 5.0)
        base_score = p_b_given_a * 0.7 + lift_factor * 0.3

        path_enhancement = 1 + 0.15 * math.log1p(path_count)

        final_conf = base_score * path_enhancement

        final_conf = min(final_conf, 1.0)

        update_query = """
            MATCH (m1:GlobalDisasterEvent {node_id: $s_id})-[r:CAUSE]->(m2:GlobalDisasterEvent {node_id: $t_id})
            SET r.confidence = $conf, 
                r.lift = $lift, 
                r.path_redundancy = $path_count
            """
        db.cypher_query(update_query, {
            's_id': s_id, 't_id': t_id,
            'conf': round(final_conf, 4),
            'lift': round(lift, 4),
            'path_count': path_count
        })

        print('confidence:', final_conf)

    print("事理对因果置信度计算完成。")


def output_top_causal_pairs(top_n=20):
    """
    Queries and prints the top N disaster causal pairs from the database,
    sorted descending by their calculated confidence scores.
    """
    query = """
    MATCH (m1:GlobalDisasterEvent)-[r:CAUSE]->(m2:GlobalDisasterEvent)
    RETURN m1.event_description AS cause_event, 
           m2.event_description AS result_event, 
           r.confidence AS confidence, 
           r.path_redundancy AS paths,
           r.relation_count AS count
    ORDER BY r.confidence DESC
    LIMIT $limit
    """
    results, _ = db.cypher_query(query, {'limit': top_n})

    print(f"{'原因事件 (Cause)':<30} | {'结果事件 (Result)':<30} | {'置信度':<8} | {'路径数':<5} | {'频次':<5}")
    print("-" * 90)

    for row in results:
        c_ev, r_ev, conf, paths, count = row
        # 截断过长的文字以便对齐
        c_ev = (c_ev[:28] + '..') if len(c_ev) > 30 else c_ev
        r_ev = (r_ev[:28] + '..') if len(r_ev) > 30 else r_ev

        print(f"{c_ev:<30} | {r_ev:<30} | {conf:<10.4f} | {paths:<7} | {count:<5}")


def _get_or_create_merge_node(desc, pattern):
    """
    Internal helper function to fetch or create a MergeDisasterEvent node
    identified by its generic description and cluster pattern ID.
    """
    query = """
    MERGE (n:MergeDisasterEvent {event_description: $desc, cluster_pattern: $pattern})
    ON CREATE SET n.node_id = randomUUID()
    RETURN n
    """
    res, _ = db.cypher_query(query, {'desc': desc, 'pattern': pattern})
    return MergeDisasterEvent.inflate(res[0][0])


def _link_factors_to_global(global_node, specific_desc, event='DisasterEvent'):
    """
    Internal helper function to find Factor nodes observed by a specific event
    and connect them to the parent global/abstract event node.
    """
    query = f"""
    MATCH (g:Global{event}), (d:{event})-[:OBSERVED]->(f:Factor)
    WHERE elementId(g) = $g_id AND d.event_description = $d_desc
    MERGE (g)-[:OBSERVED]->(f)
    """
    db.cypher_query(query, {'g_id': global_node.element_id, 'd_desc': specific_desc})



def _connect_global_cause_with_pattern(start_node, end_node, pattern):
    """
    Internal helper function to establish a 'CAUSE' relationship between two
    GlobalDisasterEvent nodes, tagged with a specific cluster pattern ID.
    """
    query = """
    MATCH (a:GlobalDisasterEvent), (b:GlobalDisasterEvent)
    WHERE elementId(a) = $a_id AND elementId(b) = $b_id
    MERGE (a)-[r:CAUSE {cluster_pattern: $pattern}]->(b)
    ON CREATE SET r.relation_type = 'GlobalCause', r.rel_id = randomUUID()
    """
    db.cypher_query(query, {
        'a_id': start_node.element_id,
        'b_id': end_node.element_id,
        'pattern': pattern
    })


def _append_cause_count_to_global_cause():
    """
    Internal helper function to aggregate relation counts from intermediate MergeDisasterEvent
    causal edges and append them to the corresponding GlobalDisasterEvent causal edges.
    """
    ca_gd_nodes = GlobalDisasterEvent.nodes.all()
    for ca_gd_node in ca_gd_nodes:
        ca_gd_eventDes = ca_gd_node.event_description

        re_gd_nodes = ca_gd_node.cause.all()
        for re_gd_node in re_gd_nodes:
            re_gd_eventDes = re_gd_node.event_description

            g_cause_relations = ca_gd_node.cause.all_relationships(re_gd_node)

            for gr in g_cause_relations:
                pattern = gr.cluster_pattern

                ca_md_node = MergeDisasterEvent.nodes.first_or_none(event_description=ca_gd_eventDes,
                                                                    cluster_pattern=pattern)
                re_md_node = MergeDisasterEvent.nodes.first_or_none(event_description=re_gd_eventDes,
                                                                    cluster_pattern=pattern)

                if ca_md_node and re_md_node:
                    m_cause_relation = ca_md_node.cause.relationship(re_md_node)

                    if m_cause_relation:
                        gr.relation_count = m_cause_relation.relation_count
                        gr.save()
                else:
                    print(f"Warning: Merge node not found for pattern {pattern}")



def _link_to_original_event(merge_node, specific_desc, merge_type, origin_type):
    """
    Internal helper function to link a generalized merge node to its original,
    concrete event node using an 'INCLUDE' relationship.
    """
    query = f"""
    MATCH (m:{merge_type}), (orig:{origin_type})
    WHERE elementId(m) = $m_id AND orig.event_description = $specific_desc
    MERGE (m)-[:INCLUDE]->(orig)
    """
    db.cypher_query(query, {'m_id': merge_node.element_id, 'specific_desc': specific_desc})


def _link_envDes_to_origin_event():
    """
    Internal helper function to aggregate environmental descriptions from concrete
    disaster events and attach a combined description to the parent MergeDisasterEvent node.
    """
    merger_nodes = MergeDisasterEvent.nodes.all()
    for merger_node in merger_nodes:
        event_nodes = merger_node.include.all()
        env_des_list = []
        for en in event_nodes:
            env_nodes = en.envDes_argument.all()
            for env_node in env_nodes:
                env_des = str(env_node.description).split("-")
                for ed in env_des:
                    if ed not in env_des_list:
                        env_des_list.append(ed)
        if len(env_des_list) == 0:
            continue
        merger_envDes_str = "-".join(env_des_list)
        merger_env_node = EnvironmentDescription.nodes.first_or_none(description=merger_envDes_str)
        if not merger_env_node:
            merger_env_node = EnvironmentDescription(description=merger_envDes_str)
            merger_env_node.save()
        if not merger_node.envDes_argument.is_connected(merger_env_node):
            merger_node.envDes_argument.connect(merger_env_node)
        print(
            f"{merger_node.event_description}\n--------------------{env_des_list}-----------------------")


def fetch_spatial_generalization_llm(pattern_label, cause_desc_gen, result_desc_gen, disaster_sr):
    """
    Invokes an LLM to analyze and generalize the prevalent spatial topology, distance,
    and orientation relationships for a specific disaster causal pattern.
    """
    if not disaster_sr:
        return None

    sys_prompt = f"""
            You are an expert in geospatial analysis and disaster mechanisms. Your task is to analyze the prevalent spatial relationships between the [Cause Event] and the [Result Event] within the disaster causal pattern named "{pattern_label}".

            ### Task Background
            We have identified a type of disaster evolution pattern: "{cause_desc_gen}" causes "{result_desc_gen}".
            Based on the provided list of specific instances (disaster_sr), you need to generalize the most typical spatial characteristics between the two event entities under this pattern.

            ### Analysis Dimension Guidelines
            Please conduct deep reasoning from the following three dimensions, relying not only on literal text matching but also on geographical laws and physical mechanisms:
            1. **Spatial Topology Relationship (Topology)**:
            - Analyze the hierarchy and extent between the two geographic entities in the causal disaster event pair.
            2. **Spatial Distance Relationship (Distance)**:
            - Determine the propagation distance of the disaster impact.
            3. **Spatial Orientation Relationship (Orientation)**:
            - Analyze the spatial orientation characteristics of the cause event relative to the result event.
        """
    from pydantic import BaseModel, Field
    from typing import List, Literal

    class SpatialInfo(BaseModel):
        topology: Literal[
            "Intersect", "Contains", "Contained By", "Equal", "Adjacent", "Disjoint", "Uncertain"] = Field(
            description=f"For the pattern [{pattern_label}], the spatial topology relationship of the cause event relative to the result event."
        )
        distance: Literal[
            "Very Close (<10m)", "Close (10m-100m)", "Medium (100m-1km)", "Far (1km-5km)", "Very Far (>5km)"] = Field(
            description=f"For the pattern [{pattern_label}], the typical spatial distance of impact propagation between events."
        )
        orientation: List[
            Literal["Upstream-Downstream", "Waterside-Landside", "Source-Channel-Sink", "High Elevation-Low Elevation",
            "Upslope-Downslope", "Sunny Slope-Shaded Slope", "Center-Edge", "Inside Region-Outside Region", "Upwind-Downwind", "No Obvious Orientation"]] = Field(
            description=f"For the pattern [{pattern_label}], the spatial orientation characteristics of the cause event relative to the result event (multiple choice).")
        evidence: str = Field(
            description="The comprehensive judging basis for constructing the above spatial relationships. Please explain using the input disaster instance causal events and the spatial relationship between events, combined with geographical laws (e.g., water flows to lower elevations) or specific instance place name relationships (e.g., A is the higher-level administrative region of B)."
        )

    try:
        agent = create_agent(llm, system_prompt=sys_prompt, response_format=SpatialInfo)
        user_content = f"Please analyze the following disaster event pair instances and generalize the spatial relationship characteristics of [{pattern_label}]:\n{disaster_sr}"
        output = agent.invoke({"messages": [{"role": "user", "content": user_content}]})

        return {
            "pattern_label": pattern_label,
            "structured_response": output["structured_response"],
            "status": "success"
        }
    except Exception as e:
        print(f"LLM Error on {pattern_label}: {e}")
        return None


def generalize_spatial_relations_parallel(clustered_results: dict, max_workers=1):
    """
    Manages multi-threaded execution to extract spatial instance data, calls the LLM
    for spatial generalization, and writes the structured results back into the graph database.
    """
    print("正在从数据库提取实例数据...")
    tasks = []

    for pattern_label, clusters in tqdm(clustered_results.items(), desc="数据预取"):
        parts = pattern_label.split("-")
        if len(parts) != 2: continue
        cause_desc_gen, result_desc_gen = parts[0], parts[1]

        for cluster_idx, pairs in enumerate(clusters):
            current_cluster_id = f"{pattern_label}_C{cluster_idx}"

            # 这里必须在主线程完成查询，确认是否已连接
            m_cause = MergeDisasterEvent.nodes.filter(event_description=cause_desc_gen,
                                                      cluster_pattern=current_cluster_id).first_or_none()
            m_result = MergeDisasterEvent.nodes.filter(event_description=result_desc_gen,
                                                       cluster_pattern=current_cluster_id).first_or_none()

            disaster_sr = []
            for pair in pairs:
                ca = DisasterEvent.nodes.first_or_none(event_description=pair[0])
                re = DisasterEvent.nodes.first_or_none(event_description=pair[1])
                if ca and re:
                    ca_geo = ca.geo_argument.all()[0]
                    re_geo = re.geo_argument.all()[0]

                    ca.spatial_to.connect(re)
                    r = ca.spatial_to.relationship(re)

                    disaster_sr.append({
                        'cause_disaster': {'subject': [ca_geo.geo_object_name, ca_geo.geo_type]},
                        'result_disaster': {'subject': [re_geo.geo_object_name, re_geo.geo_type]},
                        'relation': {'spatial_related': r.relations}
                    })

            tasks.append({
                "pattern_label": pattern_label,
                "cause_desc_gen": cause_desc_gen,
                "result_desc_gen": result_desc_gen,
                "current_cluster_id": current_cluster_id,
                "disaster_sr": disaster_sr
            })

    print(f"开始并行泛化分析 (并发数: {max_workers})...")
    final_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(fetch_spatial_generalization_llm,
                            t['pattern_label'], t['cause_desc_gen'],
                            t['result_desc_gen'], t['disaster_sr']): t
            for t in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="LLM 泛化中"):
            res = future.result()
            if res:
                task_info = future_to_task[future]
                res.update({
                    "current_cluster_id": task_info["current_cluster_id"],
                    "cause_desc_gen": task_info["cause_desc_gen"],
                    "result_desc_gen": task_info["result_desc_gen"]
                })
                final_results.append(res)

    print(f"正在写入归纳结果 (共 {len(final_results)} 条关系)...")
    for res in final_results:
        m_cause = MergeDisasterEvent.nodes.filter(event_description=res['cause_desc_gen'],
                                                  cluster_pattern=res['current_cluster_id']).first_or_none()
        m_result = MergeDisasterEvent.nodes.filter(event_description=res['result_desc_gen'],
                                                   cluster_pattern=res['current_cluster_id']).first_or_none()

        if m_cause and m_result:
            _connect_merge_event_spatial_relations(
                m_cause, m_result, res['current_cluster_id'], res['structured_response']
            )


def _connect_merge_event_spatial_relations(m_cause, m_result, pattern, sr):
    """
    Internal helper function to attach the LLM-generated spatial relationship JSON
    data to the spatial edge connecting two MergeDisasterEvent nodes.
    """
    if hasattr(sr, 'model_dump'):
        spatial_data = sr.model_dump()
    else:
        spatial_data = sr.dict()

    try:

        if m_cause and m_result:
            if m_cause.spatial_to.is_connected(m_result):
                rel = m_cause.spatial_to.relationship(m_result)

                rel.relations = spatial_data
                rel.save()
            else:
                m_cause.spatial_to.connect(m_result, {'relations': spatial_data})
                print(f"[Connect Spatial] Created relation for {pattern}")
        else:
            print(f"[Spatial Skip] Nodes not found for pattern: {pattern}")
    except Exception as e:
        print(f"[Connect Spatial Error] Failed for {pattern}: {e}")


def generalize_observation_patterns():
    """
    Generalizes specific observation metrics (spatial/temporal resolution, weather effects,
    and positions) into GeneralizedObservationPattern nodes and links them to global observation events.
    """
    global_disaster_nodes = GlobalDisasterEvent.nodes.all()
    for global_disaster_node in global_disaster_nodes:
        gen_obs_nodes = global_disaster_node.trigger.all()
        for gen_obs_node in gen_obs_nodes:
            obs_nodes = gen_obs_node.include.all()
            sensor_nodes = gen_obs_node.sensor_argument.all()
            if sensor_nodes:
                sensor = sensor_nodes[0].sensor_type
                platform = sensor_nodes[0].platform
            else:
                continue

            obs_pattern_nodes = []
            for obs_node in obs_nodes:
                obs_pattern_node = obs_node.obsPat_argument.all()
                if len(obs_pattern_node) > 0:
                    obs_pattern_nodes.append(obs_pattern_node[0])

            space_res_dict = {'Ultra-high Resolution': 0, 'High Resolution': 0, 'Medium Resolution': 0,
                              'Low Resolution': 0}
            time_res_dict = {'Ultra-high Frequency': 0, 'High Frequency': 0, 'Low Frequency': 0}
            obs_position_list = []
            weather_effect_list = []

            for obs_pattern_node in obs_pattern_nodes:
                space_res = obs_pattern_node.space_res
                time_res = obs_pattern_node.time_res
                obs_position = obs_pattern_node.obs_position
                weather_effect = obs_pattern_node.weather_effect

                if isinstance(space_res, (int, float)) and space_res > 0:
                    if space_res < 1:
                        space_res_dict['Ultra-high Resolution'] += 1
                    elif space_res < 10:
                        space_res_dict['High Resolution'] += 1
                    elif space_res < 100:
                        space_res_dict['Medium Resolution'] += 1
                    else:
                        space_res_dict['Low Resolution'] += 1

                if isinstance(time_res, (int, float)) and time_res > 0:
                    if time_res < 1:
                        time_res_dict['Ultra-high Frequency'] += 1
                    elif time_res <= 24:
                        time_res_dict['High Frequency'] += 1
                    else:
                        time_res_dict['Low Frequency'] += 1

                if obs_position != '' and obs_position not in obs_position_list:
                    obs_position_list.append(obs_position)

                if weather_effect_list != '' and weather_effect not in weather_effect_list:
                    weather_effect_list.append(weather_effect)

            def get_dominant_level(res_dict):
                # If all are 0, return empty, otherwise return the key with the max count
                if all(v == 0 for v in res_dict.values()):
                    return ''
                return max(res_dict, key=res_dict.get)

            gen_space_res = get_dominant_level(space_res_dict)
            if gen_space_res == '':
                gen_space_res = 'High Resolution'

            gen_time_res = get_dominant_level(time_res_dict)
            if gen_time_res == '':
                gen_time_res = 'High Frequency'

            if platform == 'Spaceborne Observation Platform' or platform == 'Airborne Observation Platform':
                gen_obs_position = 'Remote Sensing Observation'
            else:
                gen_obs_position = "-".join(obs_position_list) if obs_position_list else ''
                if gen_obs_position == '':
                    gen_obs_position = 'Proximal Point Observation'

            gen_weather_effect = ''

            if sensor == 'Passive Optical Sensor':
                if 'Obscured by Clouds and Fog' not in weather_effect_list:
                    weather_effect_list.append('Obscured by Clouds and Fog')
                if 'Limited by Illumination Conditions' not in weather_effect_list:
                    weather_effect_list.append('Limited by Illumination Conditions')
                if 'All-weather Observation' in weather_effect_list:
                    weather_effect_list.remove('All-weather Observation')
                gen_weather_effect = "-".join(weather_effect_list) if weather_effect_list else ''
            elif sensor == 'Active Microwave Sensor' or sensor == 'Passive Microwave Sensor':
                gen_weather_effect = 'All-weather Observation'
            elif sensor == 'Active Optical Sensor':
                if 'Obscured by Clouds and Fog' not in weather_effect_list:
                    weather_effect_list.append('Obscured by Clouds and Fog')
                if 'All-weather Observation' in weather_effect_list:
                    weather_effect_list.remove('All-weather Observation')
                gen_weather_effect = "-".join(weather_effect_list) if weather_effect_list else ''

            # Note: Changed '地面' to 'Ground' to match 'Ground-based Fixed Platforms' or 'Ground-based Mobile Platforms'
            if gen_weather_effect == '' and 'Ground' in platform:
                gen_weather_effect = 'All-weather Observation'

            # 3. Construct or get GeneralizedObservationPattern node
            gen_obsPat_node = GeneralizedObservationPattern.nodes.first_or_none(
                space_res=gen_space_res, time_res=gen_time_res,
                weather_effect=gen_weather_effect, obs_position=gen_obs_position
            )
            if not gen_obsPat_node:
                gen_obsPat_node = GeneralizedObservationPattern(
                    space_res=gen_space_res, time_res=gen_time_res,
                    weather_effect=gen_weather_effect, obs_position=gen_obs_position
                ).save()

            # Duplicate check from original code
            gen_obsPat_node = GeneralizedObservationPattern.nodes.first_or_none(
                space_res=gen_space_res, time_res=gen_time_res,
                weather_effect=gen_weather_effect, obs_position=gen_obs_position)
            if not gen_obsPat_node:
                gen_obsPat_node = GeneralizedObservationPattern(
                    space_res=gen_space_res, time_res=gen_time_res,
                    weather_effect=gen_weather_effect, obs_position=gen_obs_position)
                gen_obsPat_node.save()

            obsPat_did = gen_obsPat_node.disaster_ids
            if global_disaster_node.node_id not in obsPat_did:
                obsPat_did.append(global_disaster_node.node_id)
                gen_obsPat_node.disaster_ids = obsPat_did
                gen_obsPat_node.save()

            for obs_pattern_node in obs_pattern_nodes:
                if not gen_obsPat_node.include.is_connected(obs_pattern_node):
                    gen_obsPat_node.include.connect(obs_pattern_node)

            if not gen_obs_node.obsPat_argument.is_connected(gen_obsPat_node):
                gen_obs_node.obsPat_argument.connect(gen_obsPat_node)
            print(
                f"{global_disaster_node.event_description} - {gen_obs_node.event_description}\n------------------------------{[gen_space_res, gen_time_res, gen_obs_position, gen_weather_effect]}-------------------------------")


def get_all_relationship_details(src_desc, dst_desc):
    """
    Executes a Cypher query to retrieve all pattern IDs and confidence scores
    for the 'CAUSE' relationships connecting two specific GlobalDisasterEvents.
    """
    query = """
    MATCH (a:GlobalDisasterEvent {event_description: $src_desc})
    -[r:CAUSE]->(b:GlobalDisasterEvent {event_description: $dst_desc})
    RETURN r.cluster_pattern, r.confidence
    """
    params = {"src_desc": src_desc, "dst_desc": dst_desc}
    # results 是一个列表的列表，例如: [['PatternA', 0.85], ['PatternB', 0.92]]
    results, meta = db.cypher_query(query, params)
    return results


def pattern_statistics():
    """
    Aggregates statistical data on causal patterns across global disaster events,
    sorts them by frequency, and exports the results to a JSON file.
    """
    patterns = {}

    # 获取所有全局灾害节点
    g_disasters = GlobalDisasterEvent.nodes.all()
    for g_disaster in g_disasters:
        g_desc = g_disaster.event_description

        g_scd_nodes = g_disaster.cause.all()
        seen_in_this_level = set()

        for g_scd_node in g_scd_nodes:
            g_scd_desc = g_scd_node.event_description

            if g_scd_desc in seen_in_this_level:
                continue
            seen_in_this_level.add(g_scd_desc)

            cause_relations = get_all_relationship_details(g_desc, g_scd_desc)

            if not cause_relations:
                continue

            pair_name = f"({g_desc})-({g_scd_desc})"

            if pair_name not in patterns:
                patterns[pair_name] = {
                    "pattern_count": 0,
                    "patterns": []
                }

            for row in cause_relations:
                cluster_pattern = row[0]
                confidence = row[1]

                m_disaster = MergeDisasterEvent.nodes.first_or_none(
                    event_description=g_desc, cluster_pattern=cluster_pattern
                )
                m_scd_disaster = MergeDisasterEvent.nodes.first_or_none(
                    event_description=g_scd_desc, cluster_pattern=cluster_pattern
                )

                if m_disaster and m_scd_disaster:
                    sr = m_disaster.spatial_to.relationship(m_scd_disaster)

                    spatial_rels = getattr(sr, 'relations', []) if sr else []

                    patterns_in = {
                        "pattern_name": cluster_pattern,
                        "confidence": confidence,
                        "spatial_relations": spatial_rels
                    }

                    patterns[pair_name]["patterns"].append(patterns_in)
                    patterns[pair_name]["pattern_count"] += 1

    sorted_patterns = sorted(
        patterns.values(),
        key=lambda x: x['pattern_count'],
        reverse=True
    )

    output_file = "data/disaster_pattern_stats_07.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_patterns, f, ensure_ascii=False, indent=4)
        print(f"统计结果已成功保存至: {output_file}")
    except Exception as e:
        print(f"保存文件失败: {e}")

    return sorted_patterns


def extract_standard_factor_observations(output_path="data/standard_factor_observations.json"):
    """
    Queries the database to map StandardFactors to their corresponding ObservationEvents
    and exports this structured association data to a JSON file.
    """
    print("🔍 正在从数据库拉取 StandardFactor 与 Observation 的关联数据...")

    query = """
    MATCH (sf:StandardFactor)--(f:Factor)--(obs:ObservationEvent)

    // 提取观测事件的各个组成部分 (使用 OPTIONAL MATCH 防止部分节点缺失导致整条数据被丢弃)
    OPTIONAL MATCH (obs)--(s:Sensor)
    OPTIONAL MATCH (obs)--(geo:GeoObject)
    OPTIONAL MATCH (obs)--(pat:ObservationPattern)

    RETURN 
        sf.factor_name AS standard_factor_name,
        obs.event_description AS obs_description,
        s.sensor_type AS sensor_type,
        s.platform AS platform,
        geo.geo_type AS geo_type,
        geo.geo_object_name AS geo_name,
        pat.space_res AS space_res,
        pat.time_res AS time_res,
        pat.obs_position AS obs_position,
        pat.weather_effect AS weather_effect
    """

    try:
        results, meta = db.cypher_query(query)
    except Exception as e:
        print(f"❌ 数据库查询失败，请检查连接: {e}")
        return {}

    factor_obs_dict = {}

    for row in results:
        standard_factor_name = row[0]
        if not standard_factor_name:
            continue

        obs_info = {
            "observation_description": row[1] or "",
            "sensor": {
                "sensor_type": row[2] or "",
                "platform": row[3] or ""
            },
            "geo_object": {
                "geo_type": row[4] or "",
                "geo_name": row[5] or ""
            },
            "observation_pattern": {
                "space_res": row[6] or "",
                "time_res": row[7] or "",
                "obs_position": row[8] or "",
                "weather_effect": row[9] or ""
            }
        }

        if standard_factor_name not in factor_obs_dict:
            factor_obs_dict[standard_factor_name] = []

        if obs_info not in factor_obs_dict[standard_factor_name]:
            factor_obs_dict[standard_factor_name].append(obs_info)

    print(f"✅ 数据提取完成，共获取到 {len(factor_obs_dict)} 个标准指标的观测数据。")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(factor_obs_dict, f, ensure_ascii=False, indent=4)
        print(f"💾 结果已成功保存至: {output_path}")
    except Exception as e:
        print(f"❌ 保存 JSON 文件失败: {e}")


    return factor_obs_dict



def construct_public_gen_obs(factor, geo_type, factor_obs):
    """
    Constructs public GeneralizedObservationEvent nodes based on standard factor observations,
    generalizing the associated sensor, platform, and observation pattern details.
    """
    relate_obs = []
    obs_list = factor_obs.get(factor, [])
    for obs in obs_list:
        if obs['geo_object']['geo_type'] == geo_type:
            relate_obs.append(obs)

    if len(relate_obs) == 0:
        relate_obs = obs_list
    geo_type = ''

    relate_obs_dict = {}
    for obs in relate_obs:
        sensor = obs['sensor']['sensor_type']
        platform = obs['sensor']['platform']
        relate_obs_dict.setdefault(sensor, {}).setdefault(platform, []).append(obs)

    gen_obs_nodes = []
    for sensor, content in relate_obs_dict.items():
        for platform, obs_records in content.items():
            generalized_observation_eventDes = f"[{sensor},{platform}]-[{geo_type},{factor}]"

            generalized_observation_node = GeneralizedObservationEvent.nodes.first_or_none(
                event_description=generalized_observation_eventDes
            )
            if not generalized_observation_node:
                generalized_observation_node = GeneralizedObservationEvent(
                    event_description=generalized_observation_eventDes,
                    is_public=True
                ).save()
                print(f"create [gen_obs_event]: {generalized_observation_eventDes}")

            gen_obs_nodes.append(generalized_observation_node)

            # Calculate and generalize observation patterns (ObservationPattern)
            space_res_dict = {'Ultra-high Resolution': 0, 'High Resolution': 0, 'Medium Resolution': 0,
                              'Low Resolution': 0}
            time_res_dict = {'Ultra-high Frequency': 0, 'High Frequency': 0, 'Low Frequency': 0}
            obs_position_list = []
            weather_effect_list = []

            # Iterate over all specific observation records under this [Sensor-Platform] combination
            for o in obs_records:
                pat = o.get('observation_pattern', {})
                space_res = pat.get('space_res', '')
                time_res = pat.get('time_res', '')
                obs_position = pat.get('obs_position', '')
                weather_effect = pat.get('weather_effect', '')

                # Calculate spatial resolution (add exception handling for dirty data)
                try:
                    sr_val = float(space_res) if space_res else -1
                    if sr_val > 0:
                        if sr_val < 1:
                            space_res_dict['Ultra-high Resolution'] += 1
                        elif sr_val < 10:
                            space_res_dict['High Resolution'] += 1
                        elif sr_val < 100:
                            space_res_dict['Medium Resolution'] += 1
                        else:
                            space_res_dict['Low Resolution'] += 1
                except ValueError:
                    pass

                # Calculate time resolution
                try:
                    tr_val = float(time_res) if time_res else -1
                    if tr_val > 0:
                        if tr_val < 1:
                            time_res_dict['Ultra-high Frequency'] += 1
                        elif tr_val <= 24:
                            time_res_dict['High Frequency'] += 1
                        else:
                            time_res_dict['Low Frequency'] += 1
                except ValueError:
                    pass

                # Calculate observation position and weather effect
                if obs_position and obs_position not in obs_position_list:
                    obs_position_list.append(obs_position)
                if weather_effect and weather_effect not in weather_effect_list:
                    weather_effect_list.append(weather_effect)

            def get_dominant_level(res_dict):
                # If all are 0, return empty string, otherwise return the key with the max count
                if all(v == 0 for v in res_dict.values()):
                    return ''
                return max(res_dict, key=res_dict.get)

            gen_space_res = get_dominant_level(space_res_dict)
            if gen_space_res == '':
                gen_space_res = 'High Resolution'

            gen_time_res = get_dominant_level(time_res_dict)
            if gen_time_res == '':
                gen_time_res = 'High Frequency'

            if platform == 'Spaceborne Observation Platform' or platform == 'Airborne Observation Platform':
                gen_obs_position = 'Remote Sensing Observation'
            else:
                gen_obs_position = "-".join(obs_position_list) if obs_position_list else ''
                if gen_obs_position == '':
                    gen_obs_position = 'Proximal Point Observation'

            gen_weather_effect = ''

            if sensor == 'Passive Optical Sensor':
                if 'Obscured by Clouds and Fog' not in weather_effect_list:
                    weather_effect_list.append('Obscured by Clouds and Fog')
                if 'Limited by Illumination Condition' not in weather_effect_list:
                    weather_effect_list.append('Limited by Illumination Conditions')
                if 'All-weather Observation' in weather_effect_list:
                    weather_effect_list.remove('All-weather Observation')
                gen_weather_effect = "-".join(weather_effect_list) if weather_effect_list else ''
            elif sensor == 'Active Microwave Sensor' or sensor == 'Passive Microwave Sensor':
                gen_weather_effect = 'All-weather Observation'
            elif sensor == 'Active Optical Sensor':
                if 'Obscured by Clouds and Fog' not in weather_effect_list:
                    weather_effect_list.append('Obscured by Clouds and Fog')
                if 'All-weather Observation' in weather_effect_list:
                    weather_effect_list.remove('All-weather Observation')
                gen_weather_effect = "-".join(weather_effect_list) if weather_effect_list else ''

            # Note: Checking for 'Ground' to match 'Ground-based Fixed/Mobile Platforms'
            if gen_weather_effect == '' and 'Ground' in platform:
                gen_weather_effect = 'All-weather Observation'

            # 3. Construct or get GeneralizedObservationPattern node
            if len(generalized_observation_node.obsPat_argument.all()) == 0:
                gen_obsPat_node = GeneralizedObservationPattern.nodes.first_or_none(
                    space_res=gen_space_res, time_res=gen_time_res,
                    weather_effect=gen_weather_effect, obs_position=gen_obs_position
                )
                if not gen_obsPat_node:
                    gen_obsPat_node = GeneralizedObservationPattern(
                        space_res=gen_space_res, time_res=gen_time_res,
                        weather_effect=gen_weather_effect, obs_position=gen_obs_position
                    ).save()

                # 4. Establish connection between GeneralizedObservationEvent -> GeneralizedObservationPattern
                if not generalized_observation_node.obsPat_argument.is_connected(gen_obsPat_node):
                    generalized_observation_node.obsPat_argument.connect(gen_obsPat_node)
                    print(
                        f"connect [obs_event -> pattern]: {generalized_observation_eventDes} -> [{gen_space_res},{gen_time_res}]")

            # 5. Connect generalized event node with Factor and GeoObject nodes
            factor_node = StandardFactor.nodes.first_or_none(factor_name=factor)
            if not factor_node:
                factor_node = StandardFactor(factor_name=factor,
                                             relate_geos=indicator_mapping.get(factor, [])).save()
            if factor_node and not generalized_observation_node.std_factor_argument.is_connected(factor_node):
                generalized_observation_node.std_factor_argument.connect(factor_node)

            geo_node = GeoObject.nodes.first_or_none(geo_object_name=geo_type, geo_type=geo_type)
            if not geo_node:
                geo_node = GeoObject(geo_object_name=geo_type, geo_type=geo_type).save()
            if geo_node and not generalized_observation_node.geo_argument.is_connected(geo_node):
                generalized_observation_node.geo_argument.connect(geo_node)

            sensor_node = Sensor.nodes.first_or_none(sensor_type=sensor, platform=platform)
            if sensor_node and not generalized_observation_node.sensor_argument.is_connected(sensor_node):
                generalized_observation_node.sensor_argument.connect(sensor_node)

    return gen_obs_nodes


def construct_observation_events_all():
    """
    Globally assesses disaster nodes to identify missing trigger relationships for standard
    factors, generating and connecting new public generalized observations where needed.
    """
    g_disaster_nodes = GlobalDisasterEvent.nodes.all()
    # factor_obs = extract_standard_factor_observations()
    with open('./data/standard_factor_observations_1.json', 'r', encoding='utf-8') as f:
        factor_obs = json.load(f)

    for g_disaster_node in g_disaster_nodes:
        g_disaster_desc = g_disaster_node.event_description
        g_geo_node = g_disaster_node.geo_argument.all()[0]
        g_geo_type = g_geo_node.geo_type
        print(g_disaster_desc)

        now_factors = []
        g_obs_nodes = g_disaster_node.trigger.all()
        for g_obs_node in g_obs_nodes:
            now_factor_nodes = g_obs_node.factor_argument.all()
            for now_factor in now_factor_nodes:
                now_factors.append(now_factor.factor_name)

        factor_nodes = g_disaster_node.observed.all()
        for factor_node in factor_nodes:
            standard_factor = factor_node.relate.all()
            if len(standard_factor) > 0:
                factor_name = standard_factor[0].factor_name
            else:
                continue

            if factor_name in now_factors:
                continue

            target_suffix = f"-[{g_geo_type},{factor_name}]"

            existing_obs_nodes = GeneralizedObservationEvent.nodes.filter(
                event_description__endswith=target_suffix).all()

            if existing_obs_nodes:
                for gen_obs in existing_obs_nodes:
                    if not g_disaster_node.trigger.is_connected(gen_obs):
                        g_disaster_node.trigger.connect(gen_obs)
                        print(f"🔗 [直接连接 existing_trigger]: ({g_disaster_desc}) -> ({gen_obs.event_description})")
            else:
                new_gen_obs_nodes = construct_public_gen_obs(factor_name, g_geo_type, factor_obs)

                # 将新构建出来的事件列表与灾害节点进行连接
                for gen_obs in new_gen_obs_nodes:
                    if not g_disaster_node.trigger.is_connected(gen_obs):
                        g_disaster_node.trigger.connect(gen_obs)
                        print(f"✨ [构建并连接 new_trigger]: ({g_disaster_desc}) -> ({gen_obs.event_description})")


def fetch_env_rule(event_description, envDes, addition_info):
    """
    Executes a thread-safe LLM chain call to generate environmental rules based on
    event descriptions without performing any database operations.
    """
    env_rules = get_env_rule_chain(event_description, envDes, addition_info)

    if hasattr(env_rules, 'dict'):
        return env_rules.dict()
    elif hasattr(env_rules, 'model_dump'):  # Pydantic v2
        return env_rules.model_dump()
    else:
        return env_rules


def generate_env_rule_to_database_parallel(elevation=120, max_workers=10):
    """
    Coordinates parallel LLM requests to generate environmental rules for disaster
    events and safely writes the returned rules back to the database in the main thread.
    """
    disasters = MergeDisasterEvent.nodes.all()
    tasks_payload = []

    print("开始扫描数据库，准备任务...")
    for disaster in disasters:
        scd_disaster = disaster.cause.all()
        for scd in scd_disaster:
            envDes = ""
            disaster_envDes = scd.envDes_argument.all()

            if len(disaster_envDes) > 0:
                disaster_envDes_node = disaster_envDes[0]
                envDes = disaster_envDes_node.description
                rules = scd.envRule_argument.all()

                # 如果没有现成的 rules，将任务信息暂存到内存中
                if len(rules) == 0:
                    tasks_payload.append({
                        'disaster_node': scd,  # 暂存节点对象引用
                        'event_description': scd.event_description,
                        'envDes': envDes,
                        'addition_info': {'elevation': elevation}
                    })

    print(f"数据库扫描完毕，共发现 {len(tasks_payload)} 个待处理任务。")

    collected_results = []

    if tasks_payload:
        print("开始并发请求大模型...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}

            # 分发任务
            for task in tasks_payload:
                future = executor.submit(
                    fetch_env_rule,
                    task['event_description'],
                    task['envDes'],
                    task['addition_info']
                )
                future_to_task[future] = task

            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    env_rules = future.result()
                    collected_results.append((task['disaster_node'], env_rules))
                except Exception as exc:
                    print(f"节点 [{task['event_description']}] 的网络请求或转换报错: {exc}")

    if collected_results:
        print(f"网络请求完毕，准备将 {len(collected_results)} 条结果写入数据库...")
        for disaster, env_rules in collected_results:
            try:
                rule_node = EnvironmentRule(rules=env_rules).save()

                if not disaster.envRule_argument.is_connected(rule_node):
                    disaster.envRule_argument.connect(rule_node)

                print(f"成功保存规则并建立连接: {env_rules}")
            except Exception as exc:
                print(f"将规则写入数据库时发生错误: {exc}")

    print("全部任务处理完毕！")



from pydantic import BaseModel, Field
from typing import Literal


class ClassifyInfo(BaseModel):
    sensor_type: Literal['主动光学传感器', '被动光学传感器', '主动微波传感器', '被动微波传感器', '其它传感器'] = Field(
        description="传感器类别"
    )
    reasoning: str = Field(..., description="简述判别理由")


def fetch_llm_classification(sensor_id, sensor_name):
    """
    Prompts the LLM to classify a sensor name into specific predefined categories
    (e.g., Active Optical, Passive Microwave) in a thread-safe manner.
    """
    try:
        sys_prompt = """
            你是一名顶尖的地理空间传感器网络专家，你的任务是进行传感器类别的判断。
            ### 角色与目标
            你的核心目标是严格判断用户输入中的传感器名称属于
            [主动光学传感器, 被动光学传感器, 主动微波传感器, 被动微波传感器, 其它传感器]中的哪一项。
        """

        agent = create_agent(llm, system_prompt=sys_prompt, response_format=ClassifyInfo)

        output = agent.invoke(
            {"messages": [{"role": "user", "content": f"name=‘{sensor_name}’"}]}
        )
        sensor_type = output["structured_response"].sensor_type
        return sensor_id, sensor_type, None  # 返回 ID, 结果, 无错误
    except Exception as e:
        return sensor_id, None, str(e)  # 返回错误信息


def sensor_reclassify_parallel(max_workers=10):
    """
    Manages concurrent LLM calls to reclassify all sensor entities in the database
    and sequentially applies the updated classifications to the Neo4j nodes.
    """
    sensors = Sensor.nodes.all()

    sensor_map = {sensor.element_id: sensor for sensor in sensors}

    classification_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(fetch_llm_classification, s.element_id, s.sensor_name)
            for s in sensors
        ]

        for future in concurrent.futures.as_completed(futures):
            sensor_id, sensor_type, error = future.result()
            if error:
                print(f"[LLM 异常] 节点 {sensor_id}: {error}")
            elif sensor_type in ['主动光学传感器', '被动光学传感器', '主动微波传感器', '被动微波传感器', '其它传感器']:
                classification_results.append((sensor_id, sensor_type))
            else:
                print(f"[跳过] 节点 {sensor_id} 类别不在预期内: {sensor_type}")

    print(f"LLM 推理完成，准备将 {len(classification_results)} 条结果写入数据库...")

    for sensor_id, target_type in classification_results:
        try:
            target_sensor = sensor_map[sensor_id]
            target_sensor.sensor_type = target_type
            target_sensor.save()
        except Exception as db_err:
            print(f"[数据库写入异常] 节点 {sensor_id}: {db_err}")


def extract_typical_key_paths(hops=4, top_n=20):
    """
    Uses APOC path expansion to extract multi-hop causal chains among global disaster events,
    deduplicates them, and returns the top paths sorted by cumulative confidence and frequency.
    """
    print(f"\n正在提取固定的 {hops} 跳典型关键链式路径 (自动去重)...")

    start_nodes_query = """
    MATCH (start:GlobalDisasterEvent)-[:CAUSE]->()
    RETURN elementId(start) AS start_id
    """
    start_results, _ = db.cypher_query(start_nodes_query)

    best_chains_dict = {}

    for row in tqdm(start_results, desc='路径读取 (APOC + 去重)'):
        start_id = row[0]

        path_query = """
        MATCH (start:GlobalDisasterEvent)
        WHERE elementId(start) = $start_id

        CALL apoc.path.expandConfig(start, {
            relationshipFilter: 'CAUSE>',
            labelFilter: '+GlobalDisasterEvent',
            minLevel: $hops,
            maxLevel: $hops,
            uniqueness: 'NODE_PATH',
            limit: 10000  // 防止单节点超级发散
        }) YIELD path AS p

        WITH p, 
             [n IN nodes(p) | n.event_description] AS event_chain,
             relationships(p) AS rels

        // 剔除单步置信度过低的关系 (可选，这里设为0.01可过滤掉明显的垃圾数据)
        WHERE ALL(r IN rels WHERE coalesce(r.confidence, 0.001) >= 0.01)

        WITH p, event_chain, rels,
             reduce(score = 1.0, r IN rels | score * coalesce(r.confidence, 0.001)) AS path_confidence,
             reduce(total = 0, r IN rels | total + coalesce(r.relation_count, 0)) AS path_freq

        RETURN event_chain, path_confidence, path_freq
        """

        params = {
            'start_id': start_id,
            'hops': hops
        }

        try:
            paths_for_node, _ = db.cypher_query(path_query, params)

            if paths_for_node:
                for path_row in paths_for_node:
                    event_chain, path_confidence, path_freq = path_row

                    chain_tuple = tuple(event_chain)

                    if chain_tuple not in best_chains_dict:
                        best_chains_dict[chain_tuple] = (path_confidence, path_freq)
                    else:
                        existing_conf, existing_freq = best_chains_dict[chain_tuple]
                        if (path_confidence, path_freq) > (existing_conf, existing_freq):
                            best_chains_dict[chain_tuple] = (path_confidence, path_freq)

        except Exception as e:
            print(f"节点 {start_id} 路径提取失败: {e}")
            continue

    all_unique_paths = []
    for chain_tuple, (conf, freq) in best_chains_dict.items():
        all_unique_paths.append((conf, freq, hops, list(chain_tuple)))

    all_unique_paths.sort(reverse=True)
    top_paths = all_unique_paths[:top_n]

    print("-" * 120)
    print(f"{'灾害演化链 (Event Chain)':<70} | {'综合置信度':<12} | {'总频次':<8} | {'跳数'}")
    print("-" * 120)

    for item in top_paths:
        path_confidence, path_freq, hop_count, event_chain = item
        chain_str = " -> ".join(event_chain)
        display_chain = (chain_str[:67] + '...') if len(chain_str) > 70 else chain_str

        print(f"{display_chain:<70} | {path_confidence:<15.6f} | {path_freq:<10} | {hop_count}")

    print("-" * 120)

    return [(item[3], item[0], item[1], item[2]) for item in top_paths]


def analyze_disaster_and_observation_stats():
    """
    Calculates and prints summary statistics for the graph's generalization layer,
    including average node degrees and the average number of triggered observations per disaster.
    """
    degree_query = """
    MATCH (n:GlobalDisasterEvent)
    OPTIONAL MATCH (n)-[out:CAUSE]->()
    OPTIONAL MATCH (n)<-[in:CAUSE]-()
    WITH n, count(distinct out) as out_deg, count(distinct in) as in_deg
    RETURN 
        count(n) as total_nodes,
        sum(out_deg) as total_out_degree,
        sum(in_deg) as total_in_degree,
        avg(out_deg) as avg_out_degree,
        avg(in_deg) as avg_in_degree
    """
    results, _ = db.cypher_query(degree_query)

    total_nodes = results[0][0] or 0
    total_out = results[0][1] or 0
    total_in = results[0][2] or 0
    avg_out_degree = results[0][3] or 0
    avg_in_degree = results[0][4] or 0

    trigger_query = """
    MATCH (d:GlobalDisasterEvent)
    OPTIONAL MATCH (d)-[:TRIGGER]->(o:GeneralizedObservationEvent)
    WITH d, count(distinct o) as obs_count
    RETURN 
        count(distinct CASE WHEN obs_count > 0 THEN d END) as disaster_with_trigger,
        sum(obs_count) as total_triggers,
        avg(obs_count) as avg_obs_per_disaster
    """
    trigger_results, _ = db.cypher_query(trigger_query)

    disaster_with_trigger = trigger_results[0][0] or 0
    total_triggers = trigger_results[0][1] or 0
    avg_obs_per_disaster = trigger_results[0][2] or 0

    print("\n## 灾害事理图谱泛化层统计报告 ##")
    print("-" * 50)
    print(f"泛化灾害事件(GlobalDisasterEvent)总数: {total_nodes}")
    print(f"平均出度 (Average Out-degree):        {avg_out_degree:.4f}")
    print(f"平均入度 (Average In-degree):         {avg_in_degree:.4f}")
    print("-" * 50)
    print(f"存在触发观测能力的灾害事件个数:        {disaster_with_trigger}")
    print(f"总观测触发关系(TRIGGER)数量:          {total_triggers}")
    print(f"平均每个灾害事件可触发观测事件个数:     {avg_obs_per_disaster:.4f}")
    print("-" * 50)

    return {
        "total_nodes": total_nodes,
        "avg_out": avg_out_degree,
        "avg_in": avg_in_degree,
        "trigger_count": disaster_with_trigger,
        "avg_obs": avg_obs_per_disaster
    }


import pandas as pd


def find_top_influence_disasters(top_n=10):
    """
    Identifies and prints the top 'source' disasters (highest out-degree) and
    'composite' disasters (highest in-degree) to evaluate disaster influence and vulnerability.
    """
    query = """
    MATCH (n:GlobalDisasterEvent)
    OPTIONAL MATCH (n)-[out:CAUSE]->()
    OPTIONAL MATCH (n)<-[in:CAUSE]-()
    WITH n.event_description AS event_type, 
         count(distinct out) AS out_degree, 
         count(distinct in) AS in_degree
    RETURN event_type, out_degree, in_degree
    """

    results, _ = db.cypher_query(query)

    df = pd.DataFrame(results, columns=['事件类型', '出度(诱发)', '入度(受灾)'])

    source_disasters = df.sort_values(by='出度(诱发)', ascending=False).head(top_n)

    composite_disasters = df.sort_values(by='入度(受灾)', ascending=False).head(top_n)

    print(f"\n=== Top {top_n} 源头灾害 (影响力最强，易诱发其他灾害) ===")
    print(source_disasters[['事件类型', '出度(诱发)']].to_string(index=False))

    print(f"\n=== Top {top_n} 复合灾害 (脆弱性最高，易受多种灾害影响) ===")
    print(composite_disasters[['事件类型', '入度(受灾)']].to_string(index=False))

    return source_disasters, composite_disasters


def find_top_hub_disasters(top_n=10):
    """
    Identifies and prints the top 'hub' disasters, defined as those with the highest
    combined out-degree and in-degree, representing critical nexus points in disaster chains.
    """

    query = """
    MATCH (n:GlobalDisasterEvent)
    OPTIONAL MATCH (n)-[out:CAUSE]->()
    OPTIONAL MATCH (n)<-[in:CAUSE]-()
    WITH n.event_description AS event_type, 
         count(distinct out) AS out_degree, 
         count(distinct in) AS in_degree
    RETURN event_type, 
           out_degree, 
           in_degree, 
           (out_degree + in_degree) AS total_degree
    ORDER BY total_degree DESC
    LIMIT $limit
    """

    results, _ = db.cypher_query(query, {'limit': top_n})

    # 转换为 DataFrame 展示
    df = pd.DataFrame(results, columns=['事件类型', '出度(诱发)', '入度(受灾)', '总度数(枢纽强度)'])

    print(f"\n=== Top {top_n} 核心枢纽灾害 (灾害链中关联最密集的节点) ===")
    print("-" * 75)
    print(df.to_string(index=False))
    print("-" * 75)

    return df


def connect_disaster_trigger_relate_obs():
    """
    Links specific administrative disaster events to their corresponding observation events
    based on shared factors, ensuring observation coverage cascades appropriately.
    """
    _RELATE_EVENT_CACHE = {}

    disasters = GlobalDisasterEvent.nodes.all()

    for disaster in disasters:
        geo_name = disaster.geo_argument.all()[0].geo_object_name
        impact_name = disaster.impact_argument.all()[0].impact_name

        relate_event = None
        if geo_name != '行政区':
            relate_event_desc = f"行政区,{impact_name}"
            if relate_event_desc not in _RELATE_EVENT_CACHE:
                _RELATE_EVENT_CACHE[relate_event_desc] = GlobalDisasterEvent.nodes.first_or_none(
                    event_description=relate_event_desc)
            relate_event = _RELATE_EVENT_CACHE[relate_event_desc]

        factors = []
        for factor_node in disaster.observed.all():
            f_name = factor_node.factor_name
            factors.append(f_name)

        if relate_event:
            for obs_event in relate_event.trigger.all():
                obs_factors = []
                for factor_node in obs_event.factor_argument.all():
                    f_name = factor_node.factor_name
                    if f_name in factors:
                        continue
                if not disaster.trigger.is_connected(obs_event):
                    disaster.trigger.connect(obs_event)


def export_disaster_observation_to_json(output_path="data/disaster_observations_export_3zz.json"):
    """
    Exports the complete hierarchy of GlobalDisasterEvents along with their triggered
    GeneralizedObservationEvents (including sensors, factors, and patterns) to a JSON file.
    """
    export_data = []

    g_disaster_nodes = GlobalDisasterEvent.nodes.all()

    print(f"开始导出 {len(g_disaster_nodes)} 个全局灾害事件的数据...")

    for g_disaster_node in tqdm(g_disaster_nodes, desc="导出进度"):
        disaster_info = {
            "disaster_event_description": g_disaster_node.event_description,
            "triggered_observations": []
        }

        gen_obs_nodes = g_disaster_node.trigger.all()

        for gen_obs_node in gen_obs_nodes:
            obs_info = {
                "sensors": [],
                "factors": [],
                "observation_patterns": []
            }

            for sensor in gen_obs_node.sensor_argument.all():
                obs_info["sensors"].append({
                    "sensor_type": getattr(sensor, "sensor_type", ""),
                    "platform": getattr(sensor, "platform", ""),
                    "sensor_name": getattr(sensor, "sensor_name", "")
                })

            for factor in gen_obs_node.std_factor_argument.all():
                obs_info["factors"].append({
                    "factor_name": getattr(factor, "factor_name", ""),
                })

            for obs_pattern in gen_obs_node.obsPat_argument.all():
                obs_info["observation_patterns"].append({
                    "space_res": getattr(obs_pattern, "space_res", ""),
                    "time_res": getattr(obs_pattern, "time_res", ""),
                    "weather_effect": getattr(obs_pattern, "weather_effect", ""),
                    "obs_position": getattr(obs_pattern, "obs_position", "")
                })

            disaster_info["triggered_observations"].append(obs_info)

        export_data.append(disaster_info)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=4)
        print(f"\n导出成功！数据已保存至: {output_path}")
    except Exception as e:
        print(f"\n保存 JSON 文件时发生错误: {e}")


if __name__ == '__main__':
    # database = 'events-zhengzhou-ifs-nocause'
    # database = 'events-do-gen'
    database = 'events-logic-all-1zz-ori'
    neo = Neo4jDatabaseConnector(database)

    main_factor_map_parallel_1(neo, database)

    # 执行灾害事件对泛化
    disaster_core = get_geo_impact_pair()
    print(disaster_core)
    disaster_pair_clustered = similarity_calculation(disaster_core)
    disaster_pair_clustered = {}
    for key, value in disaster_core.items():
        disaster_pair_clustered[key] = [value]
    with open("all-cluster.pickle", "rb") as f:
        disaster_pair_clustered = pickle.load(f)
    process_abstract_clustering_to_graph(disaster_pair_clustered)
    generalize_spatial_relations_parallel(disaster_pair_clustered, 20)
    _link_envDes_to_origin_event()
    _append_cause_count_to_global_cause()
    calculate_global_event_causal_confidence()
    output_top_causal_pairs(50)
    generate_env_rule_to_database_parallel()

    # 执行观测事件泛化
    construct_generalized_observation_event()
    construct_global_disaster_arguments()
    update_global_DO_argument()
    generalize_observation_patterns()

    pattern_statistics()
    # construct_observation_events()

    extract_typical_key_paths(hops=3, top_n=100)
    analyze_disaster_and_observation_stats()
    find_top_influence_disasters()
    find_top_hub_disasters()
    connect_disaster_trigger_relate_obs()

    migrate_observation_knowledge(source_db='events-do-gen-2', target_db='events-logic-all-1zz')

    construct_observation_events_all()
    export_disaster_observation_to_json()

    with open("data/pickles/factor_map.pickle", "rb") as f:
        factor_map = pickle.load(f)
    main_factor_map_(neo, 'events-logic-all-1zz-sensor', factor_map)
    extract_standard_factor_observations("data/standard_factor_observations_1.json")

    sensor_reclassify_parallel()
