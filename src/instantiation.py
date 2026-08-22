import os
import warnings
import json
import pickle
import datetime
import concurrent.futures
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import transform
from shapely import wkt
from neomodel import db
from tqdm import tqdm

from ontology.neo4jmodel import (Neo4jDatabaseConnector, GlobalDisasterEvent, MergeDisasterEvent,
                                 GeneralizedObservationEvent,
                                 ObservationEvent, GeoObject, Geometry, DisasterEvent, EnvironmentRule, DisPhen,
                                 OSMFeature,
                                 Sensor, GeneralizedObservationPattern, Factor)
from other.environment_descrption import get_env_rule_chain, evaluate_feature_against_rule

warnings.filterwarnings('ignore')

logic_graph_database = "events-logic-all-1zz-sensor-en"
ins_graph_database = "events-ins-all-zhengzhou-1-sensor-en-parallel"

geos_output_folder = "./data/output_geojson_zhengzhou_all_o_1"
geos_input_folder = "./data/output_with_elevation_and_slope_en"

neo = None
elevation = 120
ori_importance = 100
importance_punish = 30
i_fac = 1
importance_percentile = 0

ALL_EVENTS_CACHE = {}
GLOBAL_EXECUTOR = None

file_num = 0

LEVELS = [0, 10, 100, 1000, 5000, 10000]
IMP = [100, 80, 60, 40, 20, 5]

MAX_DIST_MAPPING = {
    "Very Close": 10,
    "Close": 100,
    "Medium": 1000,
    "Far": 5000,
    "Very Far": 10000
}

try:
    HYBAS_GDF = gpd.read_file("data/basin/henan_basin-2.shp")
    BASIN_ADJ_MAP = dict(zip(HYBAS_GDF['HYBAS_ID'], HYBAS_GDF['NEXT_DOWN']))
except Exception as e:
    print(f"Warning: Basin data load failed: {e}")
    HYBAS_GDF = None
    BASIN_ADJ_MAP = {}

_GEOJSON_CACHE = {}
_RELATE_EVENT_CACHE = {}
_OBSERVATION_TASK_CACHE = {}


def load_geojson_data(geo_name, region_extent=None):
    """
    Loads GeoJSON data, assigns a unique OSM ID to each feature, and optionally clips the data to a specified region extent.
    """
    file_path = f"{geos_input_folder}/{geo_name}.json"
    if not os.path.exists(file_path):
        return None

    try:
        gdf = gpd.read_file(file_path)
        if gdf.empty:
            return gdf

        gdf['osm_id'] = [f"{geo_name}_{i}" for i in range(len(gdf))]

        if region_extent is not None:
            if isinstance(region_extent, (Polygon, MultiPolygon)):
                region_gdf = gpd.GeoDataFrame({'geometry': [region_extent]}, crs="EPSG:4326")
            elif isinstance(region_extent, gpd.GeoDataFrame):
                region_gdf = region_extent
            else:
                region_gdf = None

            if region_gdf is not None:
                if gdf.crs != region_gdf.crs:
                    region_gdf = region_gdf.to_crs(gdf.crs)
                gdf = gpd.clip(gdf, region_gdf)

        return gdf

    except Exception as e:
        print(f"Error loading and clipping {file_path}: {e}")
        return None


def get_cached_geojson(geo_name, region_extent=None):
    """
    Retrieves and caches GeoJSON data, projecting it to EPSG:3857 to facilitate metric spatial calculations.
    """
    if geo_name not in _GEOJSON_CACHE:
        gdf = load_geojson_data(geo_name, region_extent=region_extent)

        if gdf is not None and not gdf.empty:
            _GEOJSON_CACHE[geo_name] = gdf.to_crs(epsg=3857)
        else:
            _GEOJSON_CACHE[geo_name] = None

    return _GEOJSON_CACHE[geo_name]


def get_downstream_path(start_ids, adjacency_map, max_depth=1000):
    """
    Computes a set of all downstream basin IDs starting from the given initial basin IDs using the adjacency map.
    """
    downstream_ids = set()
    queue = list(start_ids)
    visited = set(start_ids)

    depth = 0
    while queue and depth < max_depth:
        next_queue = []
        for current_id in queue:
            downstream_ids.add(current_id)

            next_id = adjacency_map.get(current_id, 0)
            if next_id != 0 and next_id not in visited:
                visited.add(next_id)
                next_queue.append(next_id)

        queue = next_queue
        depth += 1

    return downstream_ids


def calculate_up_down_stream(ori_geom_proj, scd_gdf_proj, scd_geo_name):
    """
    Identifies and evaluates secondary disaster geographic features located within the downstream basins of the original disaster geometry.
    """
    results = []

    if HYBAS_GDF is None or scd_gdf_proj.empty:
        return results

    ori_gs_wgs84 = gpd.GeoSeries([ori_geom_proj], crs="EPSG:3857").to_crs(epsg=4326)
    ori_geom_wgs84 = ori_gs_wgs84.iloc[0]

    scd_gdf_wgs84 = scd_gdf_proj.to_crs(epsg=4326)

    source_basins = HYBAS_GDF[HYBAS_GDF.intersects(ori_geom_wgs84)]
    if source_basins.empty:
        return []

    source_ids = set(source_basins['HYBAS_ID'])
    risk_basin_ids = get_downstream_path(source_ids, BASIN_ADJ_MAP)
    risk_basins_gdf = HYBAS_GDF[HYBAS_GDF['HYBAS_ID'].isin(risk_basin_ids)]
    affected_scd_wgs84 = gpd.sjoin(scd_gdf_wgs84, risk_basins_gdf, how='inner', predicate='intersects')

    if affected_scd_wgs84.empty:
        return []

    affected_scd_proj = affected_scd_wgs84.to_crs(epsg=3857)
    project_back = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform

    for idx, row in affected_scd_proj.iterrows():
        dist = row.geometry.distance(ori_geom_proj)

        if dist < 1000:
            priority = 1
            importance = 50
        elif dist < 5000:
            priority = 2
            importance = 30
        else:
            priority = 3
            importance = 10

        clean_props = {}
        orig_cols = scd_gdf_proj.columns
        for k, v in row.items():
            if k in orig_cols and k != 'geometry':
                if pd.isna(v):
                    clean_props[k] = None
                elif hasattr(v, 'isoformat'):
                    clean_props[k] = v.isoformat()
                else:
                    clean_props[k] = v

        clean_props['_geo_unique_id'] = idx
        geom_wgs84 = transform(project_back, row.geometry)

        results.append({
            'scd_event_type': scd_geo_name,
            'geometry': geom_wgs84,
            'properties': clean_props,
            'priority': priority,
            'spa_priority': priority,
            'importance': importance,
            'topology': 'Located Downstream',
            'distance_desc': f"Downstream (Dist: {int(dist)}m)"
        })

    return results


def calculate_topology_distance(ori_geom_proj, scd_gdf_proj, scd_geo_name, topology, distance_desc):
    """
    Identifies geographic features that intersect with buffered zones originating from the initial disaster geometry, applying varying importance levels.
    """
    max_dist = 0
    results = []
    for key, val in MAX_DIST_MAPPING.items():
        if key in distance_desc:
            max_dist = val
            break
    if max_dist == 0: max_dist = 10000

    if topology in ["Intersect", "Contains", "Contained By", "Equal", "Adjacent"]:
        start_level_idx = 0
    else:
        start_level_idx = 1

    active_levels = [d for d in LEVELS if d <= max_dist]
    active_levels = [d for d in active_levels if LEVELS.index(d) >= start_level_idx]

    processed_indices = set()

    for dist in active_levels:
        if dist == 0:
            current_search_geom = ori_geom_proj
        else:
            current_search_geom = ori_geom_proj.buffer(dist)

        candidates = scd_gdf_proj[scd_gdf_proj.geometry.intersects(current_search_geom)]
        new_hits = candidates[~candidates.index.isin(processed_indices)]

        if new_hits.empty:
            continue

        processed_indices.update(new_hits.index)
        current_priority = LEVELS.index(dist)
        importance = IMP[current_priority]
        new_hits_geo = new_hits.to_crs(epsg=4326)

        for idx, row in new_hits_geo.iterrows():
            clean_props = {}
            for k, v in row.drop('geometry').to_dict().items():
                if pd.isna(v):
                    clean_props[k] = None
                elif hasattr(v, 'isoformat'):
                    clean_props[k] = v.isoformat()
                else:
                    clean_props[k] = v

            clean_props['_geo_unique_id'] = idx

            results.append({
                'scd_event_type': scd_geo_name,
                'geometry': row.geometry,
                'properties': clean_props,
                'spa_priority': current_priority,
                'importance': importance,
                'topology': topology,
                'distance_desc': f"Buffer {dist}m" if dist > 0 else "Original Extent"
            })

    return results


def calculate_topology_distance_single(ori_geom_proj, scd_gdf_proj, scd_geo_name, topology, distance_desc='Very Close'):
    """
    Identifies secondary disaster features exclusively based on spatial intersection with the original extent, bypassing any distance buffering.
    """
    results = []

    intersecting_gdf = scd_gdf_proj[scd_gdf_proj.geometry.intersects(ori_geom_proj)]

    if intersecting_gdf.empty:
        return results

    current_priority = 0
    importance = IMP[0] if 'IMP' in globals() else 1.0
    new_hits_geo = intersecting_gdf.to_crs(epsg=4326)

    for idx, row in new_hits_geo.iterrows():
        clean_props = {}
        for k, v in row.drop('geometry').to_dict().items():
            if pd.isna(v):
                clean_props[k] = None
            elif hasattr(v, 'isoformat'):
                clean_props[k] = v.isoformat()
            else:
                clean_props[k] = v

        clean_props['_geo_unique_id'] = idx

        results.append({
            'scd_event_type': scd_geo_name,
            'geometry': row.geometry,
            'properties': clean_props,
            'spa_priority': current_priority,
            'importance': importance,
            'topology': topology,
            'distance_desc': "Original Extent"
        })

    return results


def _init_worker(db_name):
    """
    Initializes a worker process by establishing an independent database connection and warming up the global event cache.
    """
    global neo, ALL_EVENTS_CACHE
    neo = Neo4jDatabaseConnector(db_name)
    ALL_EVENTS_CACHE = {e.event_description: e for e in GlobalDisasterEvent.nodes.all()}


def _calc_single_scd_worker(args):
    """
    Performs spatial calculations for a single secondary disaster event within an isolated worker process.
    """
    (scd_disaster_des, ori_disaster_des, ori_geom_wkt, region_geom_wkt,
     event_confidence, geo_importance, db_name) = args

    neo_conn = Neo4jDatabaseConnector(db_name)
    ori_geom_proj = wkt.loads(ori_geom_wkt) if ori_geom_wkt else None
    region_geom_proj = wkt.loads(region_geom_wkt) if region_geom_wkt else None
    region_gdf_proj = gpd.GeoDataFrame({'geometry': [region_geom_proj]}, crs="EPSG:3857") if region_geom_proj else None

    ori_disaster = GlobalDisasterEvent.nodes.first_or_none(event_description=ori_disaster_des)
    scd_disaster = GlobalDisasterEvent.nodes.first_or_none(event_description=scd_disaster_des)

    results = {}
    unique_features = {}

    if not ori_disaster or not scd_disaster:
        return results

    ori_geo_name = ori_disaster_des.split(',')[0]
    scd_geo_list = scd_disaster.geo_argument.all()
    if not scd_geo_list:
        return results
    scd_geo_name = scd_geo_list[0].geo_object_name

    if ori_geo_name == scd_geo_name:
        return results

    full_scd_gdf_proj = get_cached_geojson(scd_geo_name)
    if full_scd_gdf_proj is None or full_scd_gdf_proj.empty:
        return results

    if region_gdf_proj is not None:
        try:
            scd_gdf_proj = gpd.clip(full_scd_gdf_proj, region_gdf_proj)
        except Exception as e:
            scd_gdf_proj = gpd.GeoDataFrame(columns=full_scd_gdf_proj.columns, crs=full_scd_gdf_proj.crs)
    else:
        scd_gdf_proj = full_scd_gdf_proj

    if scd_gdf_proj.empty:
        return results

    rs = ori_disaster.cause.all_relationships(scd_disaster)

    for r in rs:
        confidence = r.confidence
        if confidence < event_confidence:
            continue
        pattern = r.cluster_pattern
        ori_merge = MergeDisasterEvent.nodes.first_or_none(event_description=ori_disaster_des, cluster_pattern=pattern)
        scd_merge = MergeDisasterEvent.nodes.first_or_none(event_description=scd_disaster_des, cluster_pattern=pattern)

        if ori_merge and scd_merge:
            envDes = ""
            scd_disaster_envDes = scd_merge.envDes_argument.all()
            if len(scd_disaster_envDes) > 0:
                envDes = scd_disaster_envDes[0].description
                print(envDes)

            r_spatial = ori_merge.spatial_to.relationship(scd_merge)
            spatial = r_spatial.relations
            topology = spatial['topology']
            distance_desc = spatial['distance']
            up_down_stream = spatial['orientation']

            print(f"Analyze: {scd_disaster_des} | Relation: {topology} | Dist: {distance_desc}")

            up_down_stream_result = []
            if 'Upstream-Downstream' in up_down_stream:
                up_down_stream_result = calculate_up_down_stream(ori_geom_proj, scd_gdf_proj, scd_geo_name)

            topo_dis_result = calculate_topology_distance(ori_geom_proj, scd_gdf_proj, scd_geo_name, topology,
                                                          distance_desc)

            combined_results = topo_dis_result + up_down_stream_result
            unique_feature_keys = list(unique_features.keys()).copy()

            env_rules = {}
            scd_disaster_envRule = scd_merge.envRule_argument.all()
            if len(scd_disaster_envRule) > 0:
                env_rules = scd_disaster_envRule[0].rules
            else:
                addition_info = {'elevation': elevation}
                env_rules = get_env_rule_chain(scd_disaster_envDes, envDes, addition_info)

                if hasattr(env_rules, 'dict'):
                    env_rules = env_rules.dict()
                elif hasattr(env_rules, 'model_dump'):
                    env_rules = env_rules.model_dump()

                rule_node = EnvironmentRule(rules=env_rules).save()

                if not scd_merge.envRule_argument.is_connected(rule_node):
                    scd_merge.envRule_argument.connect(rule_node)

            for feature in combined_results:
                f_id = feature['properties']['_geo_unique_id']
                p_val = feature.get('priority', feature.get('spa_priority', 999))
                feature['spa_priority'] = p_val
                i_val = feature.get('importance', 0)
                feature['importance'] = i_val

                if f_id not in unique_features:
                    unique_features[f_id] = feature
                else:
                    current_stored_p = unique_features[f_id].get('spa_priority', 999)
                    current_stored_i = unique_features[f_id].get('importance', 0)
                    if p_val < current_stored_p:
                        unique_features[f_id] = feature
                    if f_id not in unique_feature_keys:
                        unique_features[f_id]['importance'] = i_val + current_stored_i

            for f_id, feature in unique_features.items():
                if 'env_mismatch' in feature['properties']: continue

                if env_rules:
                    is_match = evaluate_feature_against_rule(feature, env_rules)
                    if not is_match:
                        feature['spa_priority'] += 2
                        feature['importance'] = max(0, feature['importance'] - importance_punish)
                        feature['properties']['env_mismatch'] = "True"
                        feature['properties']['mismatch_reason'] = str(env_rules.get('spatial_rule', ''))

                feature['importance'] = i_fac * feature['importance']

    final_features = list(unique_features.values())
    results[scd_disaster_des] = sorted(final_features, key=lambda x: x['spa_priority'])

    if final_features:
        df_features = pd.DataFrame(final_features)
        df_features = df_features[df_features['importance'] >= geo_importance]

        if not df_features.empty:
            gdf_features = gpd.GeoDataFrame(df_features, geometry='geometry', crs="EPSG:4326")
            gdf_proj = gdf_features.to_crs(epsg=3857)

            invalid_mask = ~gdf_proj.is_valid
            if invalid_mask.any():
                gdf_proj.loc[invalid_mask, 'geometry'] = gdf_proj[invalid_mask].geometry.buffer(0)

            gdf_proj = gdf_proj[~gdf_proj.is_empty]
            is_poly = gdf_proj.geom_type.isin(['Polygon', 'MultiPolygon'])
            valid_poly_mask = is_poly & gdf_proj.geometry.apply(
                lambda g: len(g.exterior.coords) >= 4 if g.geom_type == 'Polygon' else True)
            gdf_proj = gdf_proj[(~is_poly) | valid_poly_mask]

            if not gdf_proj.empty:
                buffered_geos = gdf_proj.geometry.buffer(50)
                merged_geom_proj = buffered_geos.unary_union

                if region_geom_proj is not None:
                    try:
                        if not merged_geom_proj.bounds or not region_geom_proj.intersects(merged_geom_proj):
                            merged_geom_proj = None
                        else:
                            merged_geom_proj = merged_geom_proj.intersection(region_geom_proj)
                    except Exception as e:
                        print(f"Intersection clip failed: {e}")

                if merged_geom_proj is not None and not merged_geom_proj.is_empty:
                    merged_geom_wgs84 = gpd.GeoSeries([merged_geom_proj], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]

                    results[f"{scd_disaster_des}_TOTAL_RANGE"] = [{
                        'scd_event_type': 'DisPhen_Area',
                        'geometry': merged_geom_wgs84,
                        'properties': {'description': f'{scd_disaster_des} Total Affected Area'},
                        'spa_priority': 0,
                        'importance': 0,
                        'topology': '',
                        'distance_desc': ''
                    }]
    return results


def calculate_second_disaster_extent_parallel(ori_disaster_des: str,
                                              ori_extent=None,
                                              region_extent=None,
                                              event_confidence=0,
                                              geo_importance=30):
    """
    Distributes the spatial extent calculations for secondary disasters to the global worker pool, returning the accumulated geometry dictionaries.
    """
    if ori_extent is None:
        print("Warning: ori_extent is None.")
        return {}

    if isinstance(ori_extent, (Polygon, MultiPolygon)):
        ori_gdf = gpd.GeoDataFrame({'geometry': [ori_extent]}, crs="EPSG:4326")
    elif isinstance(ori_extent, gpd.GeoDataFrame):
        ori_gdf = ori_extent
    else:
        return {}

    ori_gdf_proj = ori_gdf.to_crs(epsg=3857)
    ori_geom_proj = ori_gdf_proj.geometry.iloc[0]

    region_geom_proj = None
    if region_extent is not None:
        try:
            if isinstance(region_extent, (Polygon, MultiPolygon)):
                region_gdf = gpd.GeoDataFrame({'geometry': [region_extent]}, crs="EPSG:4326")
            elif isinstance(region_extent, gpd.GeoDataFrame):
                region_gdf = region_extent
            else:
                print("Warning: Unsupported region_extent format.")
                region_gdf = None

            if region_gdf is not None:
                region_gdf_proj = region_gdf.to_crs(epsg=3857)
                region_geom_proj = region_gdf_proj.geometry.iloc[0]
        except Exception as e:
            print(f"Warning: region_extent processing failed: {e}")

    ori_geom_wkt = ori_geom_proj.wkt if ori_geom_proj else None
    region_geom_wkt = region_geom_proj.wkt if region_geom_proj else None

    ori_disaster = GlobalDisasterEvent.nodes.first_or_none(event_description=ori_disaster_des)
    scd_disasters = ori_disaster.cause.all()

    tasks = []
    for scd_disaster in scd_disasters:
        tasks.append((
            scd_disaster.event_description,
            ori_disaster_des,
            ori_geom_wkt,
            region_geom_wkt,
            event_confidence,
            geo_importance,
            logic_graph_database
        ))

    results = {}

    if GLOBAL_EXECUTOR is not None and tasks:
        futures = [GLOBAL_EXECUTOR.submit(_calc_single_scd_worker, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                results.update(res)
            except Exception as e:
                print(f"Exception during spatial calculation: {e}")

    return results


def export_secondary_disasters_to_geojson(results: dict, geos_output_folder_in):
    """
    Exports the calculated secondary disaster regions and geographical features into standard GeoJSON files.
    """
    if not results or len(results) == 0:
        print("No secondary disaster data found, skipping export.")
        return

    if not os.path.exists(geos_output_folder_in):
        os.makedirs(geos_output_folder_in)

    generated_files = []

    for scd_event_des, items_list in results.items():
        if not items_list:
            continue

        if "_TOTAL_RANGE" in scd_event_des:
            file_prefix = "RANGE_"
        else:
            file_prefix = "EVENT_"

        features = []
        for item in items_list:
            clean_properties = {}
            for k, v in item['properties'].items():
                if pd.isna(v):
                    clean_properties[k] = None
                elif isinstance(v, (pd.Timestamp, datetime.datetime)):
                    clean_properties[k] = v.isoformat()
                else:
                    clean_properties[k] = v

            feature = {
                "type": "Feature",
                "geometry": item['geometry'].__geo_interface__,
                "properties": {
                    **clean_properties,
                    "target_geo_type": item['scd_event_type'],
                    "spa_priority": item['spa_priority'],
                    "importance": item['importance'],
                    "analysis_topology": item['topology'],
                    "analysis_distance": item['distance_desc']
                }
            }
            features.append(feature)

        feature_collection = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features
        }

        safe_event_name = scd_event_des.replace(',', '_').replace(' ', '_')
        file_name = f"{file_prefix}{safe_event_name}.geojson"
        file_path = os.path.join(geos_output_folder_in, file_name)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(feature_collection, f, ensure_ascii=False, indent=2)

            generated_files.append(file_path)
            print(f"Exported secondary disaster file: {file_path}")
        except Exception as e:
            print(f"Export failed for {file_name}: {e}")

    return generated_files


def build_event_node(current_event, current_features, current_extent=None, region_extent=None, visited=None,
                     depth=0, max_depth=2, output_folder=''):
    """
    Recursively constructs an event node tree capturing the propagation of disaster events, their extents, and associated observation tasks.
    """
    if current_extent is None:
        return None

    if visited is None:
        visited = set()

    event_desc = str(current_event.event_description)

    if event_desc in visited:
        return None
    visited.add(event_desc)

    parts = event_desc.split(',', 1)
    geo_name = parts[0]
    impact_name = parts[1] if len(parts) > 1 else ""

    relate_event = None
    if geo_name != 'Administrative Region':
        relate_event_desc = f"Administrative Region,{impact_name}"
        if relate_event_desc not in _RELATE_EVENT_CACHE:
            _RELATE_EVENT_CACHE[relate_event_desc] = GlobalDisasterEvent.nodes.first_or_none(
                event_description=relate_event_desc)
        relate_event = _RELATE_EVENT_CACHE[relate_event_desc]

    observation_tasks = []

    def get_cached_obs_task(obs_event, is_general=False):
        cache_key = (obs_event.event_description)
        if not is_general:
            is_general = obs_event.is_public
        if cache_key not in _OBSERVATION_TASK_CACHE:
            _OBSERVATION_TASK_CACHE[cache_key] = {
                "obs_event_desc": obs_event.event_description,
                "sensors": [{"name": s.sensor_name, "type": s.sensor_type, "platform": s.platform} for s in
                            obs_event.sensor_argument.all()],
                "factors": [{"name": f.factor_name, "type": ''} for f in obs_event.std_factor_argument.all()],
                "patterns": [{"space_res": p.space_res, "time_res": p.time_res, "weather_effect": p.weather_effect,
                              "obs_pos": p.obs_position} for p in obs_event.obsPat_argument.all()],
                "geo_objects": [{"name": g.geo_object_name, "type": g.geo_type} for g in obs_event.geo_argument.all()],
                "is_general": is_general
            }
        return _OBSERVATION_TASK_CACHE[cache_key]

    for obs_event in current_event.trigger.all():
        observation_tasks.append(get_cached_obs_task(obs_event))

    if len(observation_tasks) == 0 and relate_event:
        for obs_event in relate_event.trigger.all():
            observation_tasks.append(get_cached_obs_task(obs_event, True))

    node = {
        "event_description": event_desc,
        "current_depth": depth,
        "affect_features": current_features,
        "affect_extent": current_extent,
        "observation_tasks": observation_tasks,
        "children": [],
        "confidence": 0.0
    }

    output_folder_new = f"{output_folder}/{event_desc}"

    extent_results = calculate_second_disaster_extent_parallel(event_desc, current_extent, region_extent=region_extent)

    if depth >= max_depth:
        return node

    cypher_query = """
        MATCH (p:GlobalDisasterEvent)-[r:CAUSE]->(c:GlobalDisasterEvent)
        WHERE elementId(p) = $parent_id
        RETURN c.event_description AS child_desc, r.confidence AS confidence
        """
    results, meta = db.cypher_query(cypher_query, {'parent_id': current_event.element_id})

    seen_in_this_level = set()

    for row in results:
        scd_desc = row[0]
        child_confidence = float(row[1]) if row[1] is not None else 0.0

        if scd_desc.startswith('Administrative Region,'):
            continue

        if scd_desc in seen_in_this_level:
            continue
        seen_in_this_level.add(scd_desc)

        scd_event = ALL_EVENTS_CACHE.get(scd_desc)
        if not scd_event:
            continue

        geo_extent = None
        extent = extent_results.get(f"{scd_desc}_TOTAL_RANGE", None)
        if extent and len(extent) > 0:
            geo_extent = extent[0]["geometry"]

        child_node = build_event_node(
            scd_event,
            current_features=extent_results.get(scd_desc, []),
            current_extent=geo_extent,
            region_extent=region_extent,
            visited=visited.copy(),
            depth=depth + 1,
            max_depth=max_depth,
            output_folder=output_folder_new
        )

        if child_node:
            child_node["confidence"] = child_confidence
            node["children"].append(child_node)

    return node


def save_tree_to_neo4j(node, ori_event_desc, parent_node, parent_node_path: list, event_num, parent_prob=1.0):
    """
    Recursively stores the constructed event node tree structure, mapping disaster relationships and generating graph nodes in the Neo4j database.
    """
    event_desc = node["event_description"]
    depth = node.get("current_depth", 0)
    confidence = node.get("confidence", 0)

    current_probability = parent_prob * (confidence * 7)

    cur_node = DisasterEvent.nodes.first_or_none(ori_event_desc=ori_event_desc,
                                                 parent_path=parent_node_path,
                                                 event_description=event_desc,
                                                 event_num=event_num)
    if not cur_node:
        cur_node = DisasterEvent(event_description=event_desc,
                                 ori_event_desc=ori_event_desc,
                                 parent_path=parent_node_path,
                                 event_num=event_num,
                                 depth=depth,
                                 probability=current_probability).save()
    else:
        cur_node.probability = current_probability
        cur_node.save()

    construct_disaster_arguments(cur_node, node)
    construct_observation(cur_node, node)

    if parent_node is not None:
        if not parent_node.cause.is_connected(cur_node):
            parent_node.cause.connect(cur_node, {"confidence": confidence})

    new_path = parent_node_path + [event_desc]
    for child in node.get("children", []):
        save_tree_to_neo4j(child, ori_event_desc, cur_node, new_path, event_num, current_probability)


def construct_observation(disaster_node, node: dict):
    """
    Generates GeneralizedObservationEvent nodes and links sensor, factor, and geometry arguments from the extracted observation tasks.
    """
    if "observation_tasks" in node:
        for task in node["observation_tasks"]:
            obs_desc = task["obs_event_desc"]
            is_general = task["is_general"]

            if not is_general:
                obs_node = ObservationEvent.nodes.first_or_none(event_description=obs_desc,
                                                                disaster_id=disaster_node.node_id)
                if not obs_node:
                    obs_node = ObservationEvent(event_description=obs_desc, disaster_id=disaster_node.node_id,
                                                is_public=False).save()
            else:
                obs_node = ObservationEvent.nodes.first_or_none(event_description=obs_desc,
                                                                disaster_id='000000')
                if not obs_node:
                    obs_node = ObservationEvent(event_description=obs_desc, disaster_id='000000', is_public=True).save()

            if not disaster_node.trigger.is_connected(obs_node):
                disaster_node.trigger.connect(obs_node)

            for s_info in task.get("sensors", []):
                s_node = Sensor.nodes.get_or_none(sensor_name=s_info["name"], platform=s_info["platform"])
                if not s_node:
                    s_node = Sensor(sensor_name=s_info["name"], sensor_type=s_info["type"],
                                    platform=s_info["platform"]).save()
                if not obs_node.sensor_argument.is_connected(s_node):
                    obs_node.sensor_argument.connect(s_node)

            for f_info in task.get("factors", []):
                f_node = Factor.nodes.get_or_none(factor_name=f_info["name"])
                if not f_node:
                    f_node = Factor(factor_name=f_info["name"], factor_type=f_info["type"]).save()
                if not obs_node.factor_argument.is_connected(f_node):
                    obs_node.factor_argument.connect(f_node)

            for p_info in task.get("patterns", []):
                p_node = GeneralizedObservationPattern.nodes.get_or_none(
                    space_res=p_info["space_res"], time_res=p_info["time_res"])
                if not p_node:
                    p_node = GeneralizedObservationPattern(
                        space_res=p_info["space_res"], time_res=p_info["time_res"],
                        weather_effect=p_info["weather_effect"], obs_position=p_info["obs_pos"]).save()
                if not obs_node.gen_obsPat_argument.is_connected(p_node):
                    obs_node.gen_obsPat_argument.connect(p_node)

            for g_node in disaster_node.geo_argument.all():
                if not obs_node.geo_argument.is_connected(g_node):
                    obs_node.geo_argument.connect(g_node)


def export_disaster_observation_to_json_ins(output_path="data/obs_capability/disaster_observations_export_ins_t.json"):
    """
    Exports the generated hierarchy of instantiated disaster events and their triggered observation requirements into a JSON file format.
    """
    export_data = []
    disaster_nodes = DisasterEvent.nodes.all()

    print(f"Exporting {len(disaster_nodes)} disaster event nodes...")

    for disaster_node in tqdm(disaster_nodes, desc="Export Progress"):
        geo_node = disaster_node.geo_argument.all()[0]
        geometry_nodes = geo_node.has_geometry.all()
        if geometry_nodes:
            geometry_node = geometry_nodes[0]
            geometry = geometry_node.geometry
        else:
            geometry = ""

        disaster_info = {
            "disaster_event_description": disaster_node.event_description,
            "disaster_event_extent": geometry,
            "triggered_observations": []
        }

        obs_nodes = disaster_node.trigger.all()

        for obs_node in obs_nodes:
            obs_info = {
                "sensors": [],
                "factors": [],
                "observation_patterns": []
            }

            for sensor in obs_node.sensor_argument.all():
                obs_info["sensors"].append({
                    "sensor_type": getattr(sensor, "sensor_type", ""),
                    "platform": getattr(sensor, "platform", "")
                })

            for factor in obs_node.factor_argument.all():
                obs_info["factors"].append({
                    "factor_name": getattr(factor, "factor_name", "")
                })

            for obs_pattern in obs_node.gen_obsPat_argument.all():
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
        print(f"\nExport successful! Data saved to: {output_path}")
    except Exception as e:
        print(f"\nError saving JSON file: {e}")


def construct_disaster_arguments(disaster_node, node: dict):
    """
    Creates mapping relationships linking a given DisasterEvent node to its respective GeoObject, OSMFeature, Geometry, and Phenomenon arguments.
    """
    gd_geo_type = str(disaster_node.event_description).split(',')[0]
    gd_impact_type = str(disaster_node.event_description).split(',')[1]

    gd_geo_node = GeoObject.nodes.first_or_none(geo_object_name=gd_geo_type,
                                                geo_type=gd_geo_type,
                                                event_id=disaster_node.node_id)
    if not gd_geo_node:
        gd_geo_node = GeoObject(geo_object_name=gd_geo_type, geo_type=gd_geo_type,
                                event_id=disaster_node.node_id).save()
    if not disaster_node.geo_argument.is_connected(gd_geo_node):
        disaster_node.geo_argument.connect(gd_geo_node)

    gd_geometries = gd_geo_node.has_geometry.all()
    if len(gd_geometries) == 0:
        gd_geometry_node = Geometry(geometry=str(node['affect_extent'])).save()
        gd_geo_node.has_geometry.connect(gd_geometry_node)

    node_features = node.get('affect_features', [])

    if node_features:
        disaster_id_str = str(disaster_node.node_id)

        osm_ids = [f['properties']['osm_id'] for f in node_features]

        existing_features = OSMFeature.nodes.filter(osm_id__in=osm_ids)
        feature_map = {f.osm_id: f for f in existing_features}

        existing_geoms = Geometry.nodes.filter(osm_id__in=osm_ids)
        geom_map = {g.osm_id: g for g in existing_geoms}

        with db.transaction:
            for feature in node_features:
                feature_osm_id = feature['properties']['osm_id']

                feature_node = feature_map.get(feature_osm_id)
                if not feature_node:
                    feature_node = OSMFeature(osm_id=feature_osm_id,
                                              properties=feature['properties']).save()
                    feature_map[feature_osm_id] = feature_node

                feature_prop_dict = feature_node.properties or {}
                if disaster_id_str not in feature_prop_dict:
                    feature_prop_dict[disaster_id_str] = {
                        'importance': feature['importance'],
                        'priority': feature['spa_priority'],
                        'topology': feature['topology']
                    }
                    feature_node.properties = feature_prop_dict
                    feature_node.save()

                if feature_osm_id not in geom_map:
                    geometry_node = Geometry(osm_id=feature_osm_id,
                                             geometry=str(feature['geometry'])).save()
                    geom_map[feature_osm_id] = geometry_node

        rel_data = [{'osm_id': f['properties']['osm_id']} for f in node_features]

        cypher_query = """
                UNWIND $rel_data AS row
                MATCH (geo:GeoObject {geo_object_name: $geo_name, event_id: $event_id})
                MATCH (feat:OSMFeature {osm_id: row.osm_id})
                MATCH (geom:Geometry {osm_id: row.osm_id})
                MERGE (geo)-[:HAS_FEATURE]->(feat)
                MERGE (feat)-[:HAS_GEOMETRY]->(geom)
                """

        db.cypher_query(cypher_query, {
            'rel_data': rel_data,
            'geo_name': gd_geo_type,
            'event_id': disaster_node.node_id
        })

    gd_impact_node = DisPhen.nodes.first_or_none(impact_name=gd_impact_type, impact_type=gd_impact_type)
    if not gd_impact_node:
        gd_impact_node = DisPhen(impact_name=gd_impact_type, impact_type=gd_impact_type).save()
    if not disaster_node.impact_argument.is_connected(gd_impact_node):
        disaster_node.impact_argument.connect(gd_impact_node)

    if not gd_geo_node.happen.is_connected(gd_impact_node):
        gd_geo_node.happen.connect(gd_impact_node)


def construct_ori_disaster_arguments(disaster_node, node: dict):
    """
    Constructs and securely links the fundamental environmental and geographic argument nodes strictly for the origin (root) disaster event.
    """
    gd_geo_type = str(disaster_node.event_description).split(',')[0]
    gd_impact_type = str(disaster_node.event_description).split(',')[1]

    gd_geo_node = GeoObject.nodes.first_or_none(geo_object_name=gd_geo_type,
                                                geo_type=gd_geo_type,
                                                event_id=disaster_node.node_id)
    if not gd_geo_node:
        gd_geo_node = GeoObject(geo_object_name=gd_geo_type, geo_type=gd_geo_type,
                                event_id=disaster_node.node_id).save()
    if not disaster_node.geo_argument.is_connected(gd_geo_node):
        disaster_node.geo_argument.connect(gd_geo_node)

    gd_geometries = gd_geo_node.has_geometry.all()
    if len(gd_geometries) == 0:
        gd_geometry_node = Geometry(geometry=str(node['affect_extent'])).save()
        gd_geo_node.has_geometry.connect(gd_geometry_node)
    else:
        gd_geometry_node = gd_geometries[0]

    feature_osm_id = f"{gd_geo_type}_{disaster_node.node_id}"
    feature_node = OSMFeature.nodes.first_or_none(osm_id=feature_osm_id)
    if not feature_node:
        feature_node = OSMFeature(osm_id=feature_osm_id).save()
    feature_prop_dict = feature_node.properties
    if str(disaster_node.node_id) not in feature_prop_dict:
        feature_prop_dict[disaster_node.node_id] = {'importance': ori_importance,
                                                    'priority': 0}
        feature_node.properties = feature_prop_dict
        feature_node.save()

    if not gd_geo_node.has_Feature.is_connected(feature_node):
        gd_geo_node.has_Feature.connect(feature_node)

    if not feature_node.has_geometry.is_connected(gd_geometry_node):
        feature_node.has_geometry.connect(gd_geometry_node)

    gd_impact_node = DisPhen.nodes.first_or_none(impact_name=gd_impact_type, impact_type=gd_impact_type)
    if not gd_impact_node:
        gd_impact_node = DisPhen(impact_name=gd_impact_type, impact_type=gd_impact_type).save()
    if not disaster_node.impact_argument.is_connected(gd_impact_node):
        disaster_node.impact_argument.connect(gd_impact_node)

    if not gd_geo_node.happen.is_connected(gd_impact_node):
        gd_geo_node.happen.connect(gd_impact_node)


def construct_instantiation_database(ori_events: list, ori_extents: list = None,
                                     ori_features: list = None, region_extent=None,
                                     max_depth=2, max_workers=10):
    """
    Orchestrates the entire logic for initializing event nodes, calculating extents recursively, and persisting the graph instances back to Neo4j.
    """
    import time
    from datetime import datetime
    start_time = time.time()

    event_trees = []
    for idx, ori_event in enumerate(ori_events):
        tree = build_event_node(ori_event,
                                current_features=ori_features,
                                current_extent=ori_extents[idx],
                                region_extent=region_extent,
                                max_depth=max_depth,
                                output_folder=geos_output_folder)
        if tree:
            event_trees.append(tree)

    end_time = time.time()
    execution_duration_seconds = round(end_time - start_time, 2)
    completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_data = {
        "status": "success",
        "completion_time": completion_time,
        "execution_duration_seconds": execution_duration_seconds,
        "execution_duration_formatted": f"{execution_duration_seconds} seconds",
        "event_trees_generated": len(event_trees)
    }

    output_filename = "data/execution_report.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Execution complete. Time taken: {execution_duration_seconds} s. Report saved to {output_filename}")

    filename = 'event_trees_rain_zhengzhou-all-1-sensor-en-parallel.pickle'
    try:
        with open(filename, 'wb') as file:
            pickle.dump(event_trees, file)
    except Exception as e:
        print(f"Error saving pickle: {e}")

    with open("event_trees_rain_zhengzhou-all-1-sensor-en-parallel.pickle", "rb") as f:
        event_trees = pickle.load(f)

    neo.connect_database(ins_graph_database)
    event_num = 0
    for event_tree in event_trees:
        try:
            ori_node = DisasterEvent.nodes.first_or_none(event_description=event_tree["event_description"],
                                                         is_happen=True, depth=0, event_num=event_num)
            if not ori_node:
                ori_node = DisasterEvent(event_description=event_tree["event_description"], depth=0,
                                         is_happen=True, event_num=event_num).save()

            construct_ori_disaster_arguments(ori_node, event_tree)
            construct_observation(ori_node, event_tree)

            parent_node_path = [event_tree["event_description"]]
            for child in event_tree.get("children", []):
                save_tree_to_neo4j(child, event_tree["event_description"], ori_node, parent_node_path, event_num)
            event_num += 1
        except Exception as e:
            print(f"Persistence failed for {event_tree['event_description']}: {e}")
    return event_trees


def migrate_observation_knowledge(source_db='events-do-gen-2', target_db='events-logic-2'):
    """
    Extracts sensor and pattern attributes associated with general observation events from a source database and maps them cleanly into a target graph.
    """
    neo.connect_database(source_db)
    print(f"Extracting observation knowledge from {source_db}...")

    knowledge_data = []

    gde_nodes = GlobalDisasterEvent.nodes.all()

    for gde in gde_nodes:
        observation_events = gde.trigger.all()

        for goe in observation_events:
            event_data = {
                "gde_desc": gde.event_description,
                "goe_desc": goe.event_description,
                "sensors": [
                    {
                        "name": s.sensor_name, "type": s.sensor_type,
                        "platform": s.platform, "res": s.space_res
                    } for s in goe.sensor_argument.all()
                ],
                "factors": [
                    {"name": f.factor_name, "type": f.factor_type}
                    for f in goe.factor_argument.all()
                ],
                "geo_objects": [
                    {"name": g.geo_object_name, "type": g.geo_type, "geom": g.geometry}
                    for g in goe.geo_argument.all()
                ],
                "patterns": [
                    {"space": p.space_res, "time": p.time_res, "weather": p.weather_effect, "obs_pos": p.obs_position}
                    for p in goe.obsPat_argument.all()
                ]
            }
            knowledge_data.append(event_data)

    neo.connect_database(target_db)
    print(f"Injecting knowledge into {target_db}...")

    for data in knowledge_data:
        target_gde = GlobalDisasterEvent.nodes.first_or_none(event_description=data["gde_desc"])
        if not target_gde:
            continue

        new_goe = GeneralizedObservationEvent.nodes.first_or_none(event_description=data["goe_desc"])
        if not new_goe:
            new_goe = GeneralizedObservationEvent(event_description=data["goe_desc"]).save()

        rel_m = target_gde.trigger.relationship(new_goe)
        if not rel_m:
            target_gde.trigger.connect(new_goe)

        for s_info in data["sensors"]:
            s_node = Sensor.nodes.get_or_none(
                sensor_name=s_info["name"],
                sensor_type=s_info["type"],
                platform=s_info["platform"]
            )
            if not s_node:
                s_node = Sensor(
                    sensor_name=s_info["name"],
                    sensor_type=s_info["type"],
                    platform=s_info["platform"],
                    space_res=s_info["res"]
                ).save()

            if not new_goe.sensor_argument.is_connected(s_node):
                new_goe.sensor_argument.connect(s_node)

        for f_info in data["factors"]:
            f_node = Factor.nodes.get_or_none(
                factor_name=f_info["name"],
                factor_type=f_info["type"]
            )
            if not f_node:
                f_node = Factor(factor_name=f_info["name"], factor_type=f_info["type"]).save()

            std_f_nodes = f_node.relate.all()
            std_f_node = None
            if std_f_nodes:
                std_f_node = std_f_nodes[0]

            if not new_goe.factor_argument.is_connected(f_node):
                new_goe.factor_argument.connect(f_node)

            if std_f_node:
                if not new_goe.std_factor_argument.is_connected(std_f_node):
                    new_goe.std_factor_argument.connect(std_f_node)

            if not s_node.observe_factor.is_connected(f_node):
                s_node.observe_factor.connect(f_node)

            if not target_gde.observed.is_connected(f_node):
                target_gde.observed.connect(f_node)

        for g_info in data["geo_objects"]:
            g_node = GeoObject.nodes.get_or_none(
                geo_object_name=g_info["name"],
                geo_type=g_info["type"]
            )
            if not g_node:
                g_node = GeoObject(
                    geo_object_name=g_info["name"],
                    geo_type=g_info["type"],
                    geometry=g_info.get("geom", "")
                ).save()

            if not new_goe.geo_argument.is_connected(g_node):
                new_goe.geo_argument.connect(g_node)

            if not s_node.observe_geo.is_connected(g_node):
                s_node.observe_geo.connect(g_node)

            for f_info in data["factors"]:
                f_node = Factor.nodes.first_or_none(factor_name=f_info["name"])
                if not g_node.has_factor.is_connected(f_node):
                    g_node.has_factor.connect(f_node)

        for p_info in data["patterns"]:
            p_node = GeneralizedObservationPattern.nodes.get_or_none(
                space_res=p_info["space"],
                time_res=p_info["time"],
                weather_effect=p_info["weather"],
                obs_position=p_info["obs_pos"]
            )

            if not p_node:
                p_node = GeneralizedObservationPattern(
                    space_res=p_info["space"],
                    time_res=p_info["time"],
                    weather_effect=p_info["weather"],
                    obs_position=p_info["obs_pos"]
                ).save()

            if not new_goe.obsPat_argument.is_connected(p_node):
                new_goe.obsPat_argument.connect(p_node)

    print("Knowledge migration complete.")


def get_standard_factor_to_disaster():
    """
    Identifies standard observation factors bound to instantiated disaster events and returns a mapping from standard factors to incident event lists.
    """
    factor_to_disaster = {}

    disaster_nodes = DisasterEvent.nodes.all()
    for disaster_node in disaster_nodes:
        event_desc = disaster_node.event_description
        obs_nodes = disaster_node.trigger.all()
        geo_type = disaster_node.geo_argument.all()[0].geo_type
        for obs_node in obs_nodes:
            factor_nodes = obs_node.factor_argument.all()
            if len(factor_nodes) > 0:
                factor_node = factor_nodes[0]
                standard_factor_nodes = factor_node.relate.all()
                for sf_node in standard_factor_nodes:

                    sf_relate_geos = sf_node.relate_geos
                    if len(sf_relate_geos) > 0 and geo_type not in sf_relate_geos:
                        continue

                    sf_name = sf_node.factor_name
                    sf_to_disaster = factor_to_disaster.get(sf_name, [])
                    if event_desc not in sf_to_disaster:
                        sf_to_disaster.append(event_desc)
                    factor_to_disaster[sf_name] = sf_to_disaster

    return factor_to_disaster


def get_factor_to_disaster():
    """
    Builds a primary dictionary mapping directly tracking normal generalized observation factors against their associated specific disaster events.
    """
    factor_to_disaster = {}

    disaster_nodes = DisasterEvent.nodes.all()
    for disaster_node in disaster_nodes:
        event_desc = disaster_node.event_description
        obs_nodes = disaster_node.trigger.all()
        geo_type = disaster_node.geo_argument.all()[0].geo_type
        for obs_node in obs_nodes:
            factor_nodes = obs_node.factor_argument.all()
            if len(factor_nodes) > 0:
                factor_node = factor_nodes[0]
                sf_name = factor_node.factor_name
                sf_to_disaster = factor_to_disaster.get(sf_name, [])
                if event_desc not in sf_to_disaster:
                    sf_to_disaster.append(event_desc)
                factor_to_disaster[sf_name] = sf_to_disaster

    return factor_to_disaster