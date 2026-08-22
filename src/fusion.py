from neomodel import StructuredNode
from ontology.neo4jmodel import Neo4jDatabaseConnector, GeoObject, DisPhen, DisasterEvent, Time, Factor, OriginText, \
    neo4j_db, \
    Sensor, ObservationEvent
from langchain.agents import create_agent
import numpy as np
import ollama
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any
from event_extraction import llm
import pickle
from event_extraction import geo_types, impact_types, sensor_types, platform_types
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_ollama_embedding(text: str, model_name: str = "quentinz/bge-large-zh-v1.5:latest"):
    """
    Retrieves the text embedding vector using the specified Ollama model.
    Returns the embedding list on success or None if the API call fails.
    """
    try:
        response = ollama.embeddings(
            model=model_name,
            prompt=text
        )
        return response['embedding']
    except Exception as e:
        print(f"Ollama API call failed: {e}")
        return None


def merge_nodes(alias_node: StructuredNode, canonical_node: StructuredNode, node_type: str):
    """
    Merges two Neo4j nodes by redirecting all incoming and outgoing relationships
    from an alias node to a canonical node using APOC procedures.
    Deletes the alias node after refactoring.
    """
    if alias_node.node_id == canonical_node.node_id:
        print("Cannot merge a node with itself.")
        return

    alias_id = alias_node.node_id
    canonical_id = canonical_node.node_id

    cypher_incoming = f"""
        MATCH (alias:{node_type} {{node_id: $alias_id}})
        MATCH (canonical:{node_type} {{node_id: $canonical_id}})
        MATCH (other) -[r]-> (alias)
        CALL apoc.refactor.to(r, canonical) YIELD input, output
        DELETE r
        RETURN count(r) AS incoming_refactored
        """

    cypher_outgoing = f"""
        MATCH (alias:{node_type} {{node_id: $alias_id}})
        MATCH (canonical:{node_type} {{node_id: $canonical_id}})
        MATCH (alias) -[r]-> (other)
        CALL apoc.refactor.from(r, canonical) YIELD input, output
        DELETE r
        RETURN count(r) AS outgoing_refactored
        """

    params = {'alias_id': alias_id, 'canonical_id': canonical_id}

    try:
        results_in, _ = neo4j_db.cypher_query(cypher_incoming, params)
        refactored_in_count = results_in[0][0] if results_in and results_in[0] else 0

        results_out, _ = neo4j_db.cypher_query(cypher_outgoing, params)
        refactored_out_count = results_out[0][0] if results_out and results_out[0] else 0

    except Exception as e:
        print(f"Relationship redirection failed. Ensure the APOC library is installed: {e}")
        return

    alias_node.delete()


def get_dbscan_cluster(texts, all_info, eps=0.15, min_samples=2, metric='precomputed'):
    """
    Calculates cosine similarities for a list of text embeddings and applies the DBSCAN
    algorithm to group similar texts into clusters.
    Returns a dictionary grouping the associated entity information by cluster labels.
    Safely bypasses execution if inputs or valid embeddings are empty.
    """
    if not texts or not all_info:
        return {}

    valid_info = []
    vectors_list = []

    for i, text in enumerate(texts):
        if text and str(text).strip():
            vec = get_ollama_embedding(text)
            if vec is not None:
                valid_info.append(all_info[i])
                vectors_list.append(vec)

    if not vectors_list:
        print("No valid embeddings were retrieved. Skipping clustering.")
        return {}

    vectors = np.array(vectors_list)

    if len(vectors) < 2:
        return {}

    print("Calculating similarity matrix and performing DBSCAN clustering...")
    similarity_matrix = cosine_similarity(vectors)
    distance_matrix = 1 - similarity_matrix

    if np.any(distance_matrix < 0):
        distance_matrix[distance_matrix < 0] = 0

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    clustering.fit(distance_matrix)

    labels = clustering.labels_
    clusters = {}

    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(valid_info[i])

    return clusters


def geo_is_cluster_ok(cluster: list):
    """
    Uses an LLM agent to evaluate if a cluster of geographic entities refers to
    the same real-world object.
    """
    sys_prompt = """
        你是一名资深的地理知识图谱（GeoKG）专家，你的核心任务是对给定的地理实体名称列表进行同义实体聚类和归并。
        请基于以下原则进行判断：
        1. 同义/别名判断：如果多个名称是同一实体的全称、简称、别名或常见错误拼写，则它们应归并。
        2. 类型兼容性：归并的实体必须具有相同或高度兼容的类型。
        3. 几何兼容性：归并的实体应具有相同的核心几何形状,或其是空间尺度不同造成的表达差异。
        4. 统一名称：在归并时，应选择列表中最完整、最准确或最权威的名称作为 cluster_geo_name。
    """
    from pydantic import BaseModel, Field

    class ClusterInfo(BaseModel):
        is_all_cluster: bool = Field(description="是否所有名称都指向同一个地物")
        cluster_geo_name: str = Field(description="统一标准名称")
        cluster_geo_type: str = Field(description=f"地物类型，必须从可选项中选择：{geo_types}")
        cluster_geometry: str = Field(description="几何类型：'point'、'linestring'、'polygon'")
        excluded_geo: List[str] = Field(description="不属于最大归并集的名称列表")

    agent = create_agent(llm, system_prompt=sys_prompt, response_format=ClusterInfo)
    output = agent.invoke({"messages": [{"role": "user", "content": f"geo_names=‘{cluster}’"}]})
    return output["structured_response"]


def impact_is_cluster_ok(cluster: list):
    """
    Uses an LLM agent to validate if a cluster of disaster phenomena represents
    the same homogeneous phenomenon.
    """
    sys_prompt = """
        你是一名顶尖的灾害知识图谱专家，核心任务是对给定的灾害现象名称列表进行严格的多类别区分和同义实体聚类。
        1. 致灾因子：如降水、暴雨。
        2. 受灾现象/后果：如洪涝、洪水。
        3. 抽象状态/事件：如受灾、险情。
        严禁跨类别合并。
    """
    from pydantic import BaseModel, Field

    class ClusterInfo(BaseModel):
        is_all_cluster: bool = Field(description="是否所有名称同义")
        cluster_impact_name: str = Field(description="统一标准名称")
        cluster_impact_type: str = Field(description=f"灾害地理现象类型：{impact_types}")
        excluded_impact: List[str] = Field(description="不属于最大归并集的名称列表")

    agent = create_agent(llm, system_prompt=sys_prompt, response_format=ClusterInfo)
    output = agent.invoke({"messages": [{"role": "user", "content": f"impacts=‘{cluster}’"}]})
    return output["structured_response"]


def sensor_is_cluster_ok(cluster: list):
    """
    Uses an LLM agent to assess a cluster of sensor names, ensuring they refer
    to the identical sensor model.
    """
    sys_prompt = """
        你是一名资深的传感器观测知识图谱专家，任务是对给定的传感器名称列表进行同义实体聚类和归并。
    """
    from pydantic import BaseModel, Field

    class ClusterInfo(BaseModel):
        is_all_cluster: bool = Field(description="是否所有名称同义")
        cluster_sensor_name: str = Field(description="统一标准名称")
        cluster_sensor_type: str = Field(description=f"传感器类型：{sensor_types}")
        cluster_platform: str = Field(description=f"所搭载平台：{platform_types}")
        excluded_sensor: List[str] = Field(description="不属于最大归并集的名称列表")

    agent = create_agent(llm, system_prompt=sys_prompt, response_format=ClusterInfo)
    output = agent.invoke({"messages": [{"role": "user", "content": f"sensor_names=‘{cluster}’"}]})
    return output["structured_response"]


def factor_is_cluster_ok(cluster: list):
    """
    Uses an LLM agent to verify if a list of observable factor names refers
    to the identical abstract property.
    """
    sys_prompt = """
        你是一名资深的地理知识图谱专家，任务是对可观测因子列表进行同义实体聚类和归并。
    """
    from pydantic import BaseModel, Field

    class ClusterInfo(BaseModel):
        is_all_cluster: bool = Field(description="是否所有名称同义")
        cluster_factor_name: str = Field(description="统一标准名称")
        cluster_factor_type: str = Field(description="类型：'environment factors' 或 'feature factors'")
        excluded_factor: List[str] = Field(description="不属于最大归并集的名称列表")

    agent = create_agent(llm, system_prompt=sys_prompt, response_format=ClusterInfo)
    output = agent.invoke({"messages": [{"role": "user", "content": f"factors=‘{cluster}’"}]})
    return output["structured_response"]


def generic_cluster_identify(
        clusters: dict,
        task_name: str,
        validate_func: callable,
        extract_name_func: callable,
        construct_node_func: callable,
        attr_map: dict,
):
    """
    Executes a multi-threaded, multi-round pipeline to validate and refine entity clusters
    using LLMs. Re-clusters rejected nodes and aggregates valid groups.
    """
    fusion_nodes = {}
    delete_nodes = []
    MAX_ROUNDS = 2
    current_clusters = clusters
    next_round_nodes = []

    if not current_clusters:
        return (fusion_nodes, delete_nodes) if task_name == '因子' else fusion_nodes

    def thread_validate_task(cluster_key, cluster_value):
        if cluster_key == -1:
            return -1, cluster_value, None
        try:
            cluster_ok = validate_func(cluster_value)
            return cluster_key, cluster_value, cluster_ok
        except Exception as e:
            print(f"Error evaluating cluster [{cluster_key}]: {e}")
            return cluster_key, cluster_value, None

    for round_idx in range(MAX_ROUNDS):
        round_desc = "Initial Phase" if round_idx == 0 else f"Re-clustering Round {round_idx}"
        print(f"\n=== Starting {task_name} {round_desc} ===")
        next_round_nodes = []
        llm_results = []

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(thread_validate_task, k, v) for k, v in current_clusters.items()]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{round_desc} Validation"):
                try:
                    llm_results.append(future.result())
                except Exception as e:
                    print(f"Thread result error: {e}")

        for key, value, cluster_ok in llm_results:
            if key == -1 or cluster_ok is None:
                next_round_nodes.extend(value)
                continue

            excluded_list = getattr(cluster_ok, attr_map['excluded'], [])
            valid_group_nodes = []

            if not cluster_ok.is_all_cluster and len(excluded_list) > 0:
                for v in value:
                    v_name = extract_name_func(v)
                    if v_name in excluded_list:
                        next_round_nodes.append(v)
                    else:
                        valid_group_nodes.append(v)
            else:
                valid_group_nodes = value

            if not valid_group_nodes:
                continue

            cluster_name = getattr(cluster_ok, attr_map['cluster_name'], '')
            is_valid_cluster = False

            if task_name == '地物':
                is_valid_cluster = bool(
                    cluster_name and getattr(cluster_ok, 'cluster_geo_type', None) and getattr(cluster_ok,
                                                                                               'cluster_geometry',
                                                                                               None))
            elif task_name == '传感器':
                is_valid_cluster = bool(
                    cluster_name and getattr(cluster_ok, 'cluster_sensor_type', None) and getattr(cluster_ok,
                                                                                                  'cluster_platform',
                                                                                                  None))
            elif task_name == '因子':
                is_valid_cluster = bool(cluster_name and getattr(cluster_ok, 'cluster_factor_type', None))
            else:
                is_valid_cluster = bool(cluster_name and getattr(cluster_ok, 'cluster_impact_type', None))

            if is_valid_cluster:
                clustered_node_data = construct_node_func(cluster_ok)
                if cluster_name in fusion_nodes:
                    fusion_nodes[cluster_name]['origin_nodes'].extend(valid_group_nodes)
                else:
                    fusion_nodes[cluster_name] = {
                        "clustered_node": clustered_node_data,
                        "origin_nodes": valid_group_nodes
                    }
            else:
                next_round_nodes.extend(valid_group_nodes)

        if round_idx < MAX_ROUNDS - 1:
            if not next_round_nodes:
                break
            print(f"🔄 Re-clustering {len(next_round_nodes)} remaining nodes...")
            texts_for_clustering = [extract_name_func(n) for n in next_round_nodes]
            current_clusters = get_dbscan_cluster(
                texts=texts_for_clustering,
                all_info=next_round_nodes,
                eps=0.15,
                min_samples=2
            )
        else:
            print("Reached maximum retry rounds.")

    if next_round_nodes:
        for node_n in next_round_nodes:
            node_name = extract_name_func(node_n)
            if node_name not in fusion_nodes:
                fusion_nodes[node_name] = {"clustered_node": node_n, "origin_nodes": [node_n]}
            else:
                fusion_nodes[node_name]['origin_nodes'].append(node_n)

    return (fusion_nodes, delete_nodes) if task_name == '因子' else fusion_nodes


def geo_cluster_identify(clusters: dict):
    """
    Wrapper for generic_cluster_identify specifically configured for geographic objects.
    """
    return generic_cluster_identify(
        clusters=clusters,
        task_name="GeoObject",
        validate_func=geo_is_cluster_ok,
        extract_name_func=lambda x: x[0],
        construct_node_func=lambda res: [res.cluster_geo_name, res.cluster_geo_type, res.cluster_geometry],
        attr_map={
            'excluded': 'excluded_geo',
            'cluster_name': 'cluster_geo_name',
            'insert_name': 'insert_geo_name',
            'fusion_name': 'fusion_geo_name',
            'fusion_type': 'fusion_geo_type',
            'fusion_geometry': 'fusion_geo_geometry'
        }
    )


def impact_cluster_identify(clusters: dict):
    """
    Wrapper for generic_cluster_identify specifically configured for disaster phenomenon entities.
    """
    return generic_cluster_identify(
        clusters=clusters,
        task_name="DisPhen",
        validate_func=impact_is_cluster_ok,
        extract_name_func=lambda x: x[0],
        construct_node_func=lambda res: [res.cluster_impact_name, res.cluster_impact_type],
        attr_map={
            'excluded': 'excluded_impact',
            'cluster_name': 'cluster_impact_name',
            'insert_name': 'insert_impact_name',
            'fusion_name': 'fusion_impact_name',
            'fusion_type': 'fusion_impact_type'
        }
    )


def sensor_cluster_identify(clusters: dict):
    """
    Wrapper for generic_cluster_identify specifically configured for sensor entities.
    """
    return generic_cluster_identify(
        clusters=clusters,
        task_name="Sensor",
        validate_func=sensor_is_cluster_ok,
        extract_name_func=lambda x: x[0],
        construct_node_func=lambda res: [res.cluster_sensor_name, res.cluster_sensor_type, res.cluster_platform],
        attr_map={
            'excluded': 'excluded_sensor',
            'cluster_name': 'cluster_sensor_name',
            'insert_name': 'insert_sensor_name',
            'fusion_name': 'fusion_sensor_name',
            'fusion_type': 'fusion_sensor_type',
            'fusion_platform': 'fusion_platform'
        }
    )


def factor_cluster_identify(clusters: dict):
    """
    Wrapper for generic_cluster_identify specifically configured for observable factor entities.
    """
    return generic_cluster_identify(
        clusters=clusters,
        task_name="ObsFactor",
        validate_func=factor_is_cluster_ok,
        extract_name_func=lambda x: x[0],
        construct_node_func=lambda res: [res.cluster_factor_name, res.cluster_factor_type],
        attr_map={
            'excluded': 'excluded_factor',
            'cluster_name': 'cluster_factor_name',
            'insert_name': 'insert_factor_name',
            'is_valid_check': 'is_factor',
            'fusion_name': 'fusion_factor_name',
            'fusion_type': 'fusion_factor_type'
        }
    )


def geo_fusion_in_graph(fusion_geo_nodes: dict):
    """
    Creates or updates geographic nodes in the Neo4j graph using the unified fusion results.
    """
    if not fusion_geo_nodes:
        return

    for key, value in fusion_geo_nodes.items():
        if value['clustered_node']:
            f_geo_node = GeoObject.nodes.first_or_none(geo_object_name=key)
            if not f_geo_node:
                f_geo_node = GeoObject(geo_object_name=value['clustered_node'][0], geo_type=value['clustered_node'][1],
                                       geometry=value['clustered_node'][2]).save()
            if len(value['origin_nodes']) > 1:
                for v in value['origin_nodes']:
                    o_geo_node = GeoObject.nodes.first_or_none(geo_object_name=v[0])
                    if o_geo_node:
                        merge_nodes(o_geo_node, f_geo_node, 'GeoObject')
            else:
                f_geo_node.geo_type = value['clustered_node'][1]
                f_geo_node.geometry = value['clustered_node'][2]
                f_geo_node.save()


def impact_fusion_in_graph(fusion_impact_nodes: dict):
    """
    Creates or updates disaster phenomenon nodes in the Neo4j graph based on fusion results.
    """
    if not fusion_impact_nodes:
        return

    for key, value in fusion_impact_nodes.items():
        if value['clustered_node']:
            f_impact_node = DisPhen.nodes.first_or_none(impact_name=key)
            if not f_impact_node:
                f_impact_node = DisPhen(impact_name=value['clustered_node'][0],
                                        impact_type=value['clustered_node'][1]).save()
            if len(value['origin_nodes']) > 1:
                for v in value['origin_nodes']:
                    o_impact_node = DisPhen.nodes.first_or_none(impact_name=v[0])
                    if o_impact_node:
                        merge_nodes(o_impact_node, f_impact_node, 'DisPhen')


def sensor_fusion_in_graph(fusion_sensor_nodes: dict):
    """
    Creates or updates sensor nodes in the Neo4j graph utilizing fusion results.
    """
    if not fusion_sensor_nodes:
        return

    for key, value in fusion_sensor_nodes.items():
        if value['clustered_node']:
            f_sensor_node = Sensor.nodes.first_or_none(sensor_name=key)
            if not f_sensor_node:
                f_sensor_node = Sensor(sensor_name=value['clustered_node'][0], sensor_type=value['clustered_node'][1],
                                       platform=value['clustered_node'][2]).save()
            if len(value['origin_nodes']) > 1:
                for v in value['origin_nodes']:
                    o_sensor_node = Sensor.nodes.first_or_none(sensor_name=v[0])
                    if o_sensor_node:
                        merge_nodes(o_sensor_node, f_sensor_node, 'Sensor')
            else:
                f_sensor_node.sensor_type = value['clustered_node'][1]
                f_sensor_node.platform = value['clustered_node'][2]
                f_sensor_node.save()


def factor_fusion_in_graph(fusion_impact_nodes: dict):
    """
    Creates or updates observable factor nodes in the Neo4j graph.
    """
    if not fusion_impact_nodes:
        return

    for key, value in fusion_impact_nodes.items():
        if value['clustered_node']:
            f_factor_node = Factor.nodes.first_or_none(factor_name=key)
            if not f_factor_node:
                f_factor_node = Factor(factor_name=value['clustered_node'][0],
                                       factor_type=value['clustered_node'][1]).save()
            if len(value['origin_nodes']) > 1:
                for v in value['origin_nodes']:
                    o_factor_node = Factor.nodes.first_or_none(factor_name=v[0])
                    if o_factor_node:
                        merge_nodes(o_factor_node, f_factor_node, 'Factor')


def factor_delete_in_graph(delete_list):
    """
    Deletes observable factor nodes from the Neo4j graph that were marked for removal.
    """
    if not delete_list:
        return

    for i in delete_list:
        delete_nodes = Factor.nodes.filter(factor_name=i[0])
        for delete_node in delete_nodes:
            delete_node.delete()


def disaster_event_fusion_in_graph(fusion_event_nodes: dict):
    """
    Merges identical DisasterEvent alias nodes into canonical event nodes within Neo4j.
    """
    if not fusion_event_nodes:
        return

    for key, value in fusion_event_nodes.items():
        if value['clustered_node']:
            f_event_node = DisasterEvent.nodes.first_or_none(event_description=key)
            if not f_event_node:
                f_event_node = DisasterEvent(event_description=key).save()
            if len(value['origin_nodes']) > 1:
                for v in value['origin_nodes']:
                    o_event_node = DisasterEvent.nodes.first_or_none(node_id=v)
                    if o_event_node:
                        merge_nodes(o_event_node, f_event_node, 'DisasterEvent')


def obs_event_fusion_in_graph(fusion_event_nodes: dict):
    """
    Merges identical ObservationEvent alias nodes into canonical event nodes within Neo4j.
    """
    if not fusion_event_nodes:
        return

    for key, value in fusion_event_nodes.items():
        if value['clustered_node']:
            f_event_node = ObservationEvent.nodes.first_or_none(event_description=key)
            if not f_event_node:
                f_event_node = ObservationEvent(event_description=key).save()
            if len(value['origin_nodes']) > 1:
                for v in value['origin_nodes']:
                    o_event_node = ObservationEvent.nodes.first_or_none(node_id=v)
                    if o_event_node:
                        merge_nodes(o_event_node, f_event_node, 'ObservationEvent')


def set_disaster_event_description(event_nodes):
    """
    Generates and assigns a standardized string description to DisasterEvent nodes
    based on their linked geographic and impact arguments.
    """
    for event_node in event_nodes:
        if len(event_node.geo_argument) == 1 and len(event_node.impact_argument) == 1:
            geo_node = event_node.geo_argument.all()[0]
            impact_node = event_node.impact_argument.all()[0]
            event_desc = f"{geo_node.geo_object_name},{impact_node.impact_name}"
            event_node.event_description = event_desc
            event_node.save()


def set_obs_event_description(event_nodes):
    """
    Generates and assigns a standardized string description to ObservationEvent nodes
    based on their linked sensor, geographic object, and factor arguments.
    """
    for event_node in event_nodes:
        if len(event_node.sensor_argument) >= 1 and (
                len(event_node.factor_argument) >= 1 or len(event_node.geo_argument) >= 1):
            sensor_node = event_node.sensor_argument.all()[0]
            sensor_name = sensor_node.sensor_name

            factor_name = ""
            factor_nodes = event_node.factor_argument.all()
            if factor_nodes:
                factor_name = factor_nodes[0].factor_name

            geo_name = ""
            geo_nodes = event_node.geo_argument.all()
            if geo_nodes:
                geo_name = geo_nodes[0].geo_object_name
            event_desc = f"{sensor_name},[{geo_name}, {factor_name}]"
            event_node.event_description = event_desc
            event_node.save()


def disaster_event_cluster(event_nodes):
    """
    Groups DisasterEvent nodes by their exact string descriptions to identify
    identical events for fusion.
    """
    fusion_event_nodes = {}
    clustered_nodes = []
    for event_node in event_nodes:
        event_desc = event_node.event_description
        event_id = event_node.node_id
        if event_id in clustered_nodes:
            continue
        should_clustered_nodes = DisasterEvent.nodes.filter(event_description=event_desc)
        should_clustered_nodes_id = []
        for en in should_clustered_nodes:
            should_clustered_nodes_id.append(en.node_id)
            clustered_nodes.append(en.node_id)
        fusion_event_nodes[event_desc] = {"clustered_node": event_id, "origin_nodes": should_clustered_nodes_id}
    return fusion_event_nodes


def obs_event_cluster(event_nodes):
    """
    Groups ObservationEvent nodes by their exact string descriptions to identify
    identical events for fusion.
    """
    fusion_event_nodes = {}
    clustered_nodes = []
    for event_node in event_nodes:
        event_desc = event_node.event_description
        event_id = event_node.node_id
        if event_id in clustered_nodes:
            continue
        should_clustered_nodes = ObservationEvent.nodes.filter(event_description=event_desc)
        should_clustered_nodes_id = []
        for en in should_clustered_nodes:
            should_clustered_nodes_id.append(en.node_id)
            clustered_nodes.append(en.node_id)
        fusion_event_nodes[event_desc] = {"clustered_node": event_id, "origin_nodes": should_clustered_nodes_id}
    return fusion_event_nodes


def find_observe_origin_disaster(event_node):
    """
    Executes a Cypher query to locate and return the originating DisasterEvent
    that triggered a given ObservationEvent.
    """
    query = """
    MATCH (d:DisasterEvent)-[:TRIGGER]->(o:ObservationEvent)
    WHERE o.node_id = $node_id
    RETURN d LIMIT 1
    """
    params = {"node_id": event_node.node_id}
    results, meta = neo4j_db.cypher_query(query, params)

    if results:
        return DisasterEvent.inflate(results[0][0])
    return None


def filter_geo_object():
    """
    Identifies and deletes GeoObject nodes from the database that have no connections.
    """
    all_geos = GeoObject.nodes.all()
    delete_count = 0

    for geo in all_geos:
        query = """
        MATCH (n:GeoObject) WHERE n.node_id = $node_id
        RETURN EXISTS((:DisasterEvent)-[:ARGUMENT]->(n)) OR 
               EXISTS((:ObservationEvent)-[:ARGUMENT]->(n)) as is_connected
        """
        results, _ = neo4j_db.cypher_query(query, {"node_id": geo.node_id})
        is_connected = results[0][0]

        if not is_connected:
            geo.delete()
            delete_count += 1


def filter_factor():
    """
    Identifies and deletes Factor nodes from the database that are isolated.
    """
    all_factors = Factor.nodes.all()
    delete_count = 0

    for factor in all_factors:
        query = """
        MATCH (n:Factor) WHERE n.node_id = $node_id
        RETURN EXISTS((:DisasterEvent)-[:OBSERVED]->(n)) OR 
               EXISTS((:ObservationEvent)-[:OBSERVED]->(n)) as is_connected
        """
        results, _ = neo4j_db.cypher_query(query, {"node_id": factor.node_id})
        is_connected = results[0][0]

        if not is_connected:
            factor.delete()
            delete_count += 1


def filter_sensor():
    """
    Identifies and deletes Sensor nodes from the database that are not connected to any events.
    """
    all_sensors = Sensor.nodes.all()
    delete_count = 0

    for sensor in all_sensors:
        query = """
        MATCH (n:Sensor) WHERE n.node_id = $node_id
        RETURN EXISTS((:ObservationEvent)-[:ARGUMENT]->(n)) as is_connected
        """
        results, _ = neo4j_db.cypher_query(query, {"node_id": sensor.node_id})
        is_connected = results[0][0]

        if not is_connected:
            sensor.delete()
            delete_count += 1


def filter_obs_event():
    """
    Cleans up invalid ObservationEvents missing sensor connections or observation targets.
    """
    event_nodes = ObservationEvent.nodes.all()
    delete_nodes = []

    for event_node in event_nodes:
        o_geo_nodes = event_node.geo_argument.all()
        o_geo_node = o_geo_nodes[0] if o_geo_nodes else None

        o_factor_nodes = event_node.factor_argument.all()
        o_factor_node = o_factor_nodes[0] if o_factor_nodes else None

        o_sensor_nodes = event_node.sensor_argument.all()
        o_sensor_node = o_sensor_nodes[0] if o_sensor_nodes else None

        if not o_sensor_node:
            delete_nodes.append(event_node)
            continue

        if not o_factor_node and not o_geo_node:
            origin_disaster = find_observe_origin_disaster(event_node)
            if not origin_disaster:
                delete_nodes.append(event_node)
                continue
            else:
                d_geo_nodes = origin_disaster.geo_argument.all()
                d_geo_node = d_geo_nodes[0] if d_geo_nodes else None

                d_factor_nodes = origin_disaster.observed.all()
                d_factor_node = d_factor_nodes[0] if d_factor_nodes else None

                if d_geo_node:
                    event_node.geo_argument.connect(d_geo_node)
                if d_factor_node:
                    event_node.factor_argument.connect(d_factor_node)

    for delete_node in delete_nodes:
        delete_node.delete()

    filter_geo_object()
    filter_factor()
    filter_sensor()


def geo_fusion():
    """
    High-level function that triggers the DBSCAN clustering, LLM verification,
    and graph database fusion pipeline for GeoObject nodes.
    """
    geo_nodes = GeoObject.nodes.all()
    if not geo_nodes:
        return

    geo_names = [geo_node.geo_object_name for geo_node in geo_nodes]
    geo_info = [[geo_node.geo_object_name, geo_node.geo_type, geo_node.geometry] for geo_node in geo_nodes]

    clusters = get_dbscan_cluster(geo_names, geo_info, 0.15)
    if clusters:
        geo_clusters = geo_cluster_identify(clusters)
        geo_fusion_in_graph(geo_clusters)


def impact_fusion():
    """
    High-level function that triggers the DBSCAN clustering, LLM verification,
    and graph database fusion pipeline for DisPhen nodes.
    """
    impact_nodes = DisPhen.nodes.all()
    if not impact_nodes:
        return

    impact_names = [impact_node.impact_name for impact_node in impact_nodes]
    impact_info = [[impact_node.impact_name, impact_node.impact_type] for impact_node in impact_nodes]

    clusters = get_dbscan_cluster(impact_names, impact_info, 0.15)
    if clusters:
        impact_clusters = impact_cluster_identify(clusters)
        impact_fusion_in_graph(impact_clusters)


def factor_fusion():
    """
    High-level function that orchestrates the clustering, LLM verification,
    graph fusion, and deletion of invalid Factor nodes.
    """
    factor_nodes = Factor.nodes.all()
    if not factor_nodes:
        return

    factor_names = [factor_node.factor_name for factor_node in factor_nodes]
    factor_info = [[factor_node.factor_name, factor_node.factor_type] for factor_node in factor_nodes]

    clusters = get_dbscan_cluster(factor_names, factor_info, 0.15)
    if clusters:
        factor_clusters, d_nodes = factor_cluster_identify(clusters)
        factor_fusion_in_graph(factor_clusters)
        factor_delete_in_graph(d_nodes)


def sensor_fusion():
    """
    High-level function that manages the clustering, verification,
    and graph fusion process for Sensor nodes.
    """
    sensor_nodes = Sensor.nodes.all()
    if not sensor_nodes:
        return

    sensor_names = [sensor_node.sensor_name for sensor_node in sensor_nodes]
    sensor_info = [[sensor_node.sensor_name, sensor_node.sensor_type, sensor_node.platform]
                   for sensor_node in sensor_nodes]

    clusters = get_dbscan_cluster(sensor_names, sensor_info, 0.15)
    if clusters:
        sensor_clusters = sensor_cluster_identify(clusters)
        sensor_fusion_in_graph(sensor_clusters)


def disaster_event_fusion():
    """
    Standardizes descriptions and fuses duplicate DisasterEvent nodes in the database.
    """
    event_nodes = DisasterEvent.nodes.all()
    if not event_nodes:
        return

    set_disaster_event_description(event_nodes)
    event_clusters = disaster_event_cluster(event_nodes)
    disaster_event_fusion_in_graph(event_clusters)


def obs_event_fusion():
    """
    Filters invalid events, standardizes descriptions, and fuses duplicate
    ObservationEvent nodes in the database.
    """
    filter_obs_event()
    event_nodes = ObservationEvent.nodes.all()
    if not event_nodes:
        return

    set_obs_event_description(event_nodes)
    event_clusters = obs_event_cluster(event_nodes)
    obs_event_fusion_in_graph(event_clusters)


def relation_deduplication_count(relation_to_count: str = "CAUSE", batch_size: int = 1000):
    """
    Uses APOC periodic iterate to find duplicate relationships of a specified type,
    aggregates their weights into a single relation count, and deletes redundancies.
    """
    cypher_weighted_dedupe = f"""
            CALL apoc.periodic.iterate(
                "MATCH (s)-[r]->(t) 
                 WHERE type(r) = '{relation_to_count}' 
                 RETURN s, t, collect(r) AS rels",

                "WITH head(rels) AS r_keep, rels, size(rels) AS count
                 WHERE count > 1
                 SET r_keep.relation_count = coalesce(r_keep.relation_count, 1) + (count - 1)
                 WITH r_keep, tail(rels) AS r_delete
                 UNWIND r_delete AS rel_to_delete
                 DELETE rel_to_delete",
                {{batchSize: {batch_size}, parallel: false}} 
            ) YIELD batches, total, timeTaken
            RETURN total, batches, timeTaken
            """
    try:
        neo4j_db.cypher_query(cypher_weighted_dedupe)
    except Exception as e:
        print(f"[{relation_to_count}] relation deduplication failed: {e}")


def relation_deduplication(relation_to_count_1: str = 'CAUSE', relation_to_count_2: str = 'TRIGGER',
                           relation_to_inter: str = "SPATIAL",
                           batch_size: int = 1000):
    """
    Orchestrates relationship deduplication across the graph, applying weighted summation
    for causal/trigger relations, string merging for spatial relations, and simple deduplication
    for all others.
    """
    relation_deduplication_count(relation_to_count_1, batch_size)
    relation_deduplication_count(relation_to_count_2, batch_size)

    cypher_dedupe_and_merge = f"""
    CALL apoc.periodic.iterate(
        "MATCH (s)-[r:{relation_to_inter}]->(t)
         RETURN s, t, collect(r) AS rels",

        "WITH head(rels) AS r_keep, tail(rels) AS rels_to_delete, rels
         WHERE size(rels) > 1
         UNWIND rels AS r_all
         WITH r_keep, rels_to_delete, collect(DISTINCT r_all.relation_name) AS distinct_names
         SET r_keep.relation_name = apoc.text.join(distinct_names, ';')
         WITH rels_to_delete
         UNWIND rels_to_delete AS rel_to_delete
         DELETE rel_to_delete",
         {{batchSize: {batch_size}, parallel: false}}
    ) YIELD batches, total, timeTaken
    RETURN total, batches, timeTaken
    """

    try:
        neo4j_db.cypher_query(cypher_dedupe_and_merge)
    except Exception as e:
        print(f"[{relation_to_inter}] spatial relation deduplication failed: {e}")

    cypher_simple_dedupe = f"""
        CALL apoc.periodic.iterate(
            "MATCH (s)-[r]->(t) 
             WHERE type(r) <> '{relation_to_count_1}' AND  type(r) <> '{relation_to_count_2}'
             RETURN s, type(r) AS type, t, collect(r) AS rels",

            "WITH head(rels) AS r_keep, rels, size(rels) AS count
             WHERE count > 1
             WITH r_keep, tail(rels) AS r_delete
             UNWIND r_delete AS rel_to_delete
             DELETE rel_to_delete",
            {{batchSize: {batch_size}, parallel: false}} 
        ) YIELD batches, total, timeTaken
        RETURN total, batches, timeTaken
        """

    try:
        neo4j_db.cypher_query(cypher_simple_dedupe)
    except Exception as e:
        print(f"Standard relation deduplication failed: {e}")
        return

    print("\n⭐ 所有关系后处理完成。")


if __name__ == "__main__":
    # neo = Neo4jDatabaseConnector("events-all-1-mini")

    # 论元节点融合
    geo_fusion()
    impact_fusion()
    factor_fusion()
    sensor_fusion()

    # 事件节点融合
    disaster_event_fusion()
    obs_event_fusion()

    # 关系去重
    relation_deduplication()

    with open("fusion_geo_nodes.pickle", "rb") as f:
        clusters = pickle.load(f)
    print(clusters)
    # factor_fusion_in_graph(factor_clusters)
    # factor_delete_in_graph(d_nodes)

    with open("fusion_sensor_nodes.pickle", "rb") as f:
        clusters = pickle.load(f)
    print(clusters)
    sensor_fusion_in_graph(clusters)
    print(get_ollama_embedding("111"))
