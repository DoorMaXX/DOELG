"""OELG 3.4 节的空间实例化基线；参照 instantiation_en.py 的数据与命名。

python instantiation_oelg.py --config oelg_config.json
默认将实例化结果写入独立 Neo4j 实例库，制图重新从实例库读取数据。
缺少孕灾环境约束时，proxy 模式标记为简化基线；strict 模式拒绝运行。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
import shutil

import numpy as np
from pyproj import CRS, Transformer
from shapely import make_valid
from shapely.geometry import shape, mapping, Polygon, GeometryCollection
from shapely.ops import transform, unary_union
from shapely.prepared import prep
from src.construct_config import resolve_database_name
from src.neo4j_database import ensure_database
from standard_observation_factors import STANDARD_FACTORS, canonical_factors
from oelg_environment import (MissingEnvironmentData, evaluate_spatial_rule,
                              validate_environment_rule, number)


def read_json(path):
    with open(path, encoding='utf-8-sig') as stream:
        return json.load(stream)


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    temporary.replace(path)


def normalized(value):
    return ' '.join(str(value or '').split()).casefold()


def phenomenon(description):
    return str(description).rsplit(',', 1)[-1].strip()


def file_key(name):
    # 对应 instantiation_en.py 的逐现象栅格文件名。
    return ''.join(c for c in name if c.isalnum())


def uid(*parts):
    return uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(parts, ensure_ascii=False)).hex


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def valid_geometry(geometry):
    if geometry is None or geometry.is_empty:
        return GeometryCollection()
    return geometry if geometry.is_valid else make_valid(geometry)


def read_extent(path):
    data = read_json(path)
    crs_name = data.get('crs', {}).get('properties', {}).get('name', 'EPSG:4326')
    crs = CRS.from_user_input(crs_name)
    if crs.to_epsg() != 4326 and crs != CRS.from_user_input('OGC:CRS84'):
        raise ValueError(f'范围文件必须为经纬度 EPSG:4326 / CRS84：{path}')
    geometries = ([f['geometry'] for f in data['features']] if data['type'] == 'FeatureCollection'
                  else [data['geometry']] if data['type'] == 'Feature' else [data])
    geometry = valid_geometry(unary_union([shape(g) for g in geometries if g]))
    if geometry.is_empty or geometry.geom_type not in ('Polygon', 'MultiPolygon'):
        raise ValueError(f'范围必须是非空面：{path}')
    return geometry


class OELGKnowledgeRepository:
    """复用项目连接配置，通过显式数据库 session 读取不同知识库。"""

    def __init__(self, database_name):
        from distance_fields import open_driver
        self.driver = open_driver({'connection_module': str(Path(__file__).resolve().parents[1] / 'ontology' / 'neo4jmodel.py')})

    def query(self, database, query, **parameters):
        with self.driver.session(database=database, default_access_mode='READ') as session:
            return [record.data() for record in session.run(query, **parameters)]

    def read_abstract_graph(self, database, dataset):
        """Read current OELG ARGUMENT types, including imported Appendix mappings."""
        rows = self.query(database, '''
            MATCH (event:DisasterEvent)
            OPTIONAL MATCH (event)-[mapping:ARGUMENT]->(geo:GeoObject)
            RETURN event.node_id AS node_id, event.event_description AS name,
                   collect(DISTINCT CASE WHEN coalesce(mapping.oelg_dbbody_enabled, true)
                       THEN coalesce(geo.geo_type, geo.geo_object_name) END) AS geo_types,
                   count(mapping) AS geographic_mapping_count,
                   sum(CASE WHEN mapping.oelg_review_version IS NOT NULL THEN 1 ELSE 0 END)
                       AS reviewed_mapping_count
        ''')
        # 只取矩阵导入器生成的节点，隔离同库其他事件。
        nodes = {}
        for row in rows:
            expected = uid('predefined-causal-matrix', dataset, row['name'])
            # 原导入器将三个元素放在一个列表中；uid 的元组 JSON 表示与之相同。
            if row['node_id'] == expected:
                nodes[row['name']] = {'node_id': row['node_id'],
                                      'geo_types': sorted(t for t in row['geo_types'] if t),
                                      'dbbody_review_complete': bool(row.get('geographic_mapping_count'))
                                          and row.get('reviewed_mapping_count') == row.get('geographic_mapping_count')}
        edges = self.query(database, '''
            MATCH (source:DisasterEvent)-[relation:CAUSE]->(target:DisasterEvent)
            RETURN source.node_id AS source_id, target.node_id AS target_id,
                   source.event_description AS source, target.event_description AS target,
                   relation.relation_type AS relation_type
        ''')
        edges = [e for e in edges if e['source'] in nodes and e['target'] in nodes
                 and e['source_id'] == nodes[e['source']]['node_id']
                 and e['target_id'] == nodes[e['target']]['node_id']]
        if not nodes or not edges:
            raise ValueError('未读取到指定 dataset 的矩阵节点和关系，请先运行矩阵导入器')
        return {'nodes': dict(sorted(nodes.items())),
                'edges': sorted(edges, key=lambda e: (e['source'], e['target'], e['relation_type']))}

    def read_environment_rules(self, database, nodes):
        """Read EnvironmentRule JSONProperty from the same OELG event nodes."""
        rows = self.query(database, '''
            MATCH (event:DisasterEvent)-[:ARGUMENT]->(rule:EnvironmentRule)
            WHERE event.node_id IN $event_ids
            RETURN event.node_id AS event_id, rule.node_id AS rule_id, rule.rules AS rules
        ''', event_ids=[node['node_id'] for node in nodes.values()])
        names = {node['node_id']: name for name, node in nodes.items()}
        result = {}
        for row in rows:
            name = names[row['event_id']]
            if name in result:
                raise ValueError(f'{name} 在 OELG 中关联多个环境规则，需先消除歧义')
            rule = json.loads(row['rules']) if isinstance(row['rules'], str) else row['rules']
            validate_environment_rule(rule)
            if rule['phenomenon'] != name:
                raise ValueError(f'OELG 环境规则与事件不一致：{name}')
            result[name] = rule
        missing = set(nodes) - result.keys()
        if missing:
            raise ValueError(f'OELG 缺少 DPEnvironment 规则：{sorted(missing)}')
        return result

    def read_observation_knowledge(self, database):
        # Match the Impact node itself; event descriptions are not a join key.
        rows = self.query(database, """
            MATCH (event:GlobalDisasterEvent)-[:ARGUMENT]->(impact:Impact)
            MATCH (event)-[:TRIGGER]->(obs:GeneralizedObservationEvent)
            MATCH (obs)-[:OBSERVED]->(factor:StandardFactor)
            OPTIONAL MATCH (obs)-[:ARGUMENT]->(sensor:Sensor)
            RETURN CASE WHEN trim(coalesce(impact.impact_type, '')) = ''
                        THEN impact.impact_name ELSE trim(impact.impact_type) END AS impact_type,
                   obs.node_id AS node_id, obs.event_description AS description,
                   collect(DISTINCT event.node_id) AS source_event_ids,
                   collect(DISTINCT factor.factor_name) AS factors,
                   collect(DISTINCT sensor.sensor_name) AS candidate_sensors
        """)
        result = defaultdict(list)
        for row in rows:
            name = str(row['impact_type'] or '').strip()
            if not name or not row['node_id']:
                raise ValueError('Impact type or observation node_id is missing.')
            row['factors'] = canonical_factors(row['factors'])
            if not row['factors']:
                continue
            row['factor_schema'] = 'StandardFactor'
            # Geographic eligibility belongs exclusively to the OELG ARGUMENT
            # mapping. This observation query does not look up GeoObject nodes.
            row['source_geo_type'] = None
            row['source_geo_types'] = []
            row['geo_types'] = []
            for key in ('source_event_ids', 'source_geo_types', 'candidate_sensors'):
                row[key] = sorted(set(row.get(key) or []))
            result[name].append(row)
        return {name: sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
                for name, items in sorted(result.items())}


def load_instantiation_knowledge(repository, config):
    """Always refresh geographic eligibility from the configured OELG graph.

    An optional JSON snapshot may supply the causal structure, but never its
    old geographic lists. Missing OELG nodes fail instead of querying another
    graph or falling back to a cached geographic mapping.
    """
    current = repository.read_abstract_graph(config['abstract_database'], config['dataset'])
    if config.get('environment_rule_source', 'config') == 'oelg':
        current['environment_rules'] = repository.read_environment_rules(config['abstract_database'], current['nodes'])
    if not config.get('knowledge_json'):
        return current
    cached = read_json(config['knowledge_json'])['knowledge']
    missing = sorted(set(cached['nodes']) - current['nodes'].keys())
    if missing:
        raise ValueError(f'OELG 图谱缺少快照中的灾害现象，不能匹配承灾地物：{missing}')
    result = dict(cached, nodes={name: dict(node, **current['nodes'][name])
                                for name, node in cached['nodes'].items()})
    if 'environment_rules' in current:
        result['environment_rules'] = current['environment_rules']
    return result


class GeographicFeatureStore:
    """保持原脚本的 GeoType_行号 标识，读取前不改变地物顺序。"""

    def __init__(self, folder, region, projected_crs='EPSG:3857', missing_policy='error'):
        self.folder = Path(folder)
        self.crs = CRS.from_user_input(projected_crs)
        if not self.crs.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-8
                                           for a in self.crs.axis_info):
            raise ValueError('projected_crs 必须使用米制投影')
        self.to_metric = Transformer.from_crs(4326, self.crs, always_xy=True).transform
        self.to_wgs84 = Transformer.from_crs(self.crs, 4326, always_xy=True).transform
        self.region = transform(self.to_metric, region)
        self.cache = {}
        self.input_hashes = {}
        self.missing_policy = missing_policy
        self.missing_layers = set()

    def load(self, geo_type):
        if geo_type in self.cache:
            return self.cache[geo_type]
        path = self.folder / f'{geo_type}.json'
        if not path.is_file():
            if self.missing_policy == 'report_and_skip':
                self.missing_layers.add(geo_type)
                self.cache[geo_type] = []
                return []
            raise FileNotFoundError(f'地物类型缺少输入文件，不能当作无候选区域：{path}')
        data = read_json(path)
        self.input_hashes[str(path.resolve())] = sha256(path)
        source_crs = data.get('crs', {}).get('properties', {}).get('name', 'EPSG:4326')
        project = Transformer.from_crs(source_crs, self.crs, always_xy=True).transform
        features = []
        for index, feature in enumerate(data['features']):
            if not feature.get('geometry'):
                continue
            geometry = valid_geometry(transform(project, shape(feature['geometry'])))
            if not geometry.is_empty and geometry.intersects(self.region):
                geometry = valid_geometry(geometry.intersection(self.region))
                if not geometry.is_empty:
                    features.append({'osm_id': f'{geo_type}_{index}', 'geo_type': geo_type,
                                     'geometry': geometry, 'properties': feature.get('properties') or {}})
        self.cache[geo_type] = features
        return features


def screen_causal_graph(knowledge, root, max_depth):
    """有界可达图；删除自环和任一共同父节点的两个直接子节点之间的边。"""
    if root not in knowledge['nodes']:
        raise ValueError(f'知识库中不存在起始现象：{root}')
    edges = {}
    for edge in knowledge['edges']:
        edge = dict(edge)
        edge['relation_type'] = {'immediate': 'Immediate', 'altering': 'Altering'}.get(
            str(edge['relation_type']).casefold(), edge['relation_type'])
        if edge['relation_type'] not in ('Immediate', 'Altering'):
            raise ValueError(f'未识别的矩阵因果关系：{edge}')
        key = (edge['source'], edge['target'])
        if key in edges and edges[key]['relation_type'] != edge['relation_type']:
            raise ValueError(f'同一现象对存在冲突关系：{key}')
        edges[key] = edge
    adjacency = defaultdict(set)
    for source, target in edges:
        if source != target:
            adjacency[source].add(target)
    depth = {root: 0}
    queue = deque([root])
    while queue:
        source = queue.popleft()
        if depth[source] >= max_depth:
            continue
        for target in sorted(adjacency[source]):
            if target not in depth:
                depth[target] = depth[source] + 1
                queue.append(target)
    reachable_edges = {pair: edge for pair, edge in edges.items()
                       if pair[0] in depth and depth[pair[0]] < max_depth and pair[1] in depth}
    # 一次性根据筛选前的邻接集合判定，避免依赖遍历顺序。
    removed, kept = [], []
    for pair, edge in sorted(reachable_edges.items()):
        source, target = pair
        parents = [p for p, children in adjacency.items()
                   if p in depth and depth[p] < max_depth and source in children and target in children]
        if source == target or parents:
            removed.append(dict(edge, reason='self_loop' if source == target else 'common_parent',
                                common_parents=sorted(parents)))
        else:
            kept.append(edge)
    return kept, removed


class OELGInstantiationBuilder:
    """BFS 实例化：父灾害范围 ∩ 承灾体范围 ∩ 孕灾环境范围。"""

    def __init__(self, config, knowledge, observations=None):
        self.config = config
        self.knowledge = knowledge
        self.observations = {name:[dict(task,factors=canonical_factors(task.get('factors',[])),
                                      geo_types=[],
                                      factor_schema='StandardFactor') for task in tasks]
                             for name,tasks in (observations or {}).items()}
        self.region_wgs84 = read_extent(config['region'])
        rainfall_data = read_json(config['rainfall'])
        for feature in rainfall_data.get('features', []):
            description = (feature.get('properties') or {}).get('event')
            if description and normalized(phenomenon(description)) != normalized(config['root_phenomenon']):
                raise ValueError(f'降水输入含不同起始灾害，请拆分为独立快照：{description}')
        self.rain_wgs84 = read_extent(config['rainfall']).intersection(self.region_wgs84)
        if self.rain_wgs84.is_empty:
            raise ValueError('降水范围与研究区无交集')
        self.features = GeographicFeatureStore(config['features_folder'], self.region_wgs84,
                                              config['projected_crs'], config.get('missing_layer_policy', 'error'))
        self.region = self.features.region
        self.rain = transform(self.features.to_metric, self.rain_wgs84)
        self.environment_cache = {}
        self.environment_rules = (knowledge.get('environment_rules', {})
                                  if config.get('environment_rule_source', 'config') == 'oelg'
                                  else config.get('environment_rules', {}))
        if config.get('environment_rule_source') == 'oelg':
            missing = set(knowledge['nodes']) - self.environment_rules.keys()
            if missing:
                raise ValueError(f'OELG 缺少 DPEnvironment 规则：{sorted(missing)}')
        self.regional_mean_elevation = number(config.get('regional_mean_elevation_m', 120))
        self.environment_audit = {}
        self.scenario_proxy_phenomena = set()
        for name, rule in self.environment_rules.items():
            if 'spatial_rule' in rule:
                validate_environment_rule(rule)
                self.environment_audit[name] = dict(version=rule['version'], description=rule['description'],
                    quality=rule['quality'], status='not_reached', passed=0, rejected=0, unknown=0,
                    source_passed=0, source_rejected=0, source_unknown=0,
                    missing_fields={}, unknown_examples=[])
        self.potential_features = {}
        self.buffered_features = {}
        self.type_unions = {}
        self.potential_extents = {}
        self.proxy_phenomena = set()
        self.pruned = []

    def environment_mask(self, name):
        if name in self.environment_cache:
            return self.environment_cache[name]
        rule = self.environment_rules.get(name)
        if rule is None:
            if self.config['environment_mode'] == 'strict':
                raise ValueError(f'{name} 缺少 DPEnvironment 规则；请配置环境空间图层或地物属性条件')
            self.proxy_phenomena.add(name)
            mask = self.region
        elif 'spatial_rule' in rule:
            self.environment_audit[name]['status'] = 'evaluated'
            if rule['quality'] == 'scenario_proxy':
                self.scenario_proxy_phenomena.add(name)
            environment = rule.get('environment')
            if environment:
                pieces = []
                for geo_type in environment['geo_types']:
                    for feature in self.features.load(geo_type):
                        if self._matches_spatial_environment(name, feature['properties'],
                                environment['spatial_rule'], feature['osm_id'], source=True):
                            geometry = feature['geometry']
                            if environment['buffer_m']:
                                geometry = geometry.buffer(environment['buffer_m'])
                            pieces.append(geometry)
                mask = unary_union(pieces)
                self.environment_audit[name]['missing_source_layers'] = sorted(
                    set(environment['geo_types']) & self.features.missing_layers)
                self.environment_audit[name]['source_extent_empty'] = mask.is_empty
            else:
                mask = self.region
        elif not rule.get('mask') and not rule.get('all') and not rule.get('unrestricted_reason'):
            raise ValueError(f'{name} 的环境规则为空；无约束必须显式说明 unrestricted_reason')
        else:
            mask = (transform(self.features.to_metric, read_extent(rule['mask']))
                    if rule.get('mask') else self.region)
            if rule.get('mask'):
                self.features.input_hashes[str(Path(rule['mask']).resolve())] = sha256(rule['mask'])
        self.environment_cache[name] = valid_geometry(mask.intersection(self.region))
        return self.environment_cache[name]

    def _matches_spatial_environment(self, name, properties, spatial_rule, feature_id=None, source=False):
        audit = self.environment_audit[name]
        prefix = 'source_' if source else ''
        try:
            accepted = evaluate_spatial_rule(properties, spatial_rule, self.regional_mean_elevation)
        except MissingEnvironmentData as exc:
            audit[prefix + 'unknown'] += 1
            for field in str(exc).split(', '):
                audit['missing_fields'][field] = audit['missing_fields'].get(field, 0) + 1
            if len(audit['unknown_examples']) < 5:
                audit['unknown_examples'].append(dict(feature_id=feature_id, fields=str(exc), source=source))
            if self.config.get('environment_missing_policy', 'error') == 'error':
                raise ValueError(f'{name} 的孕灾条件缺少或含无效地物属性：{exc}') from exc
            return False
        audit[prefix + ('passed' if accepted else 'rejected')] += 1
        return accepted

    def matches_environment(self, name, properties, feature_id=None):
        """all 为经人工指定的属性阈值，缺字段报错，禁止静默默认满足。"""
        rule = self.environment_rules.get(name, {})
        if 'spatial_rule' in rule:
            return self._matches_spatial_environment(name, properties, rule['spatial_rule'], feature_id)
        for condition in rule.get('all', []):
            field, operator, expected = condition['field'], condition['op'], condition['value']
            actual = properties.get(field)
            if actual is None:
                raise ValueError(f'{name} 的孕灾条件缺少地物属性：{field}')
            if operator == 'in':
                accepted = actual in expected
            elif operator == 'eq':
                accepted = actual == expected
            else:
                actual, expected = float(actual), float(expected)
                if not math.isfinite(actual) or not math.isfinite(expected):
                    raise ValueError(f'{name} 的环境属性非有限数：{field}')
                tests = {'lt': actual < expected, 'le': actual <= expected,
                         'gt': actual > expected, 'ge': actual >= expected}
                if operator not in tests:
                    raise ValueError(f'不支持的环境条件运算符：{operator}')
                accepted = tests[operator]
            if not accepted:
                return False
        return True

    def calculate_disaster_extent(self, name, parent_extent):
        if name in self.potential_features:
            prepared_parent = prep(parent_extent)
            if prepared_parent.covers(self.potential_extents[name]):
                return self.potential_features[name], self.potential_extents[name]
            selected = []
            groups = defaultdict(list)
            for feature in self.potential_features[name]:
                # 地物本体须位于父事件范围，缓冲不能引入父事件范围外的新地物。
                if prepared_parent.intersects(feature['geometry']):
                    footprint = (feature['footprint'] if prepared_parent.covers(feature['footprint'])
                                 else valid_geometry(feature['footprint'].intersection(parent_extent)))
                    if not footprint.is_empty:
                        selected.append(dict(feature, footprint=footprint))
                        groups[feature['geo_type']].append(feature['osm_id'])
            extents = []
            for geo_type, ids in groups.items():
                key = (geo_type, tuple(ids))
                if key not in self.type_unions:
                    self.type_unions[key] = unary_union([self.buffered_features[i] for i in ids])
                extents.append(self.type_unions[key])
            extent = unary_union(extents).intersection(self.environment_mask(name)).intersection(parent_extent)
            return selected, valid_geometry(extent)
        types = [t for t in self.knowledge['nodes'][name]['geo_types']
                 if t not in self.config.get('exclude_geo_types', [])]
        if not types:
            if self.knowledge['nodes'][name].get('dbbody_review_complete'):
                # No defensible DBbody in this review is a modeling limitation,
                # not proof that the real-world phenomenon cannot occur.
                return [], GeometryCollection()
            database = self.config.get('abstract_database', 'oelg-1')
            raise ValueError(f'{name} 在 OELG 库 {database} 中缺少可用承灾地物 ARGUMENT 关联；'
                             f'请用 oelg_graph_store.py --database {database} 补充映射')
        mask = valid_geometry(self.environment_mask(name).intersection(parent_extent))
        prepared_mask = prep(mask)
        selected = []
        type_extents = []
        for geo_type in types:
            selected_for_type = []
            for feature in self.features.load(geo_type):
                if not prepared_mask.intersects(feature['geometry']):
                    continue
                if not self.matches_environment(name, feature['properties'], feature['osm_id']):
                    continue
                # 50 m 是为与 instantiation_en.py 输出口径一致的可配置参数，非论文阈值。
                cache_key = feature['osm_id']
                if cache_key not in self.buffered_features:
                    self.buffered_features[cache_key] = (
                        feature['geometry'].buffer(self.config['feature_buffer_m'])
                        if self.config['feature_buffer_m'] else feature['geometry'])
                footprint = self.buffered_features[cache_key]
                if not prepared_mask.covers(footprint):
                    footprint = valid_geometry(footprint.intersection(mask))
                if not footprint.is_empty:
                    selected.append(dict(feature, footprint=footprint))
                    selected_for_type.append(feature['osm_id'])
            key = (geo_type, tuple(selected_for_type))
            if key not in self.type_unions:
                self.type_unions[key] = unary_union([self.buffered_features[i] for i in selected_for_type])
            type_extents.append(self.type_unions[key].intersection(mask))
        extent = valid_geometry(unary_union(type_extents))
        return selected, extent

    def observation_tasks(self, name, selected, extent):
        tasks = []
        seen = set()
        for task in self.observations.get(name, []):
            if not task.get('factors'):
                continue
            key = task.get('node_id') or json.dumps(task, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            tasks.append(dict(task, geometry=extent,
                              sensor_status='candidate_only_no_time_coverage_validation'))
        return tasks

    def instantiate(self):
        root_name = self.config['root_phenomenon']
        root_rule = self.environment_rules.get(root_name, {})
        if 'spatial_rule' in root_rule:
            if root_rule.get('observed_root_bypass'):
                self.environment_audit[root_name]['status'] = 'observed_root_input_preserved'
            else:
                root_mask = self.environment_mask(root_name)
                pieces = []
                for i, feature in enumerate(read_json(self.config['rainfall']).get('features', [])):
                    if feature.get('geometry') and self.matches_environment(root_name,
                            feature.get('properties') or {}, f'root_{i}'):
                        pieces.append(transform(self.features.to_metric, shape(feature['geometry'])))
                self.rain = valid_geometry(unary_union(pieces).intersection(root_mask).intersection(self.rain))
                if self.rain.is_empty:
                    raise ValueError('起始事件无满足孕灾环境的输入范围；请检查根事件条件及属性')
        max_depth = self.config['max_depth']
        preliminary, preliminary_removed = screen_causal_graph(self.knowledge, root_name, max_depth)
        reachable_names = {e['target'] for e in preliminary + preliminary_removed} - {root_name}
        viable = {root_name}
        for name in sorted(reachable_names):
            print(f'Checking candidate disaster and spatial conditions: {name}', flush=True)
            selected, extent = self.calculate_disaster_extent(name, self.region)
            self.potential_features[name] = selected
            self.potential_extents[name] = extent
            if not extent.is_empty:
                viable.add(name)
            else:
                reviewed_out = (self.knowledge['nodes'][name].get('dbbody_review_complete')
                                and not self.knowledge['nodes'][name]['geo_types'])
                if reviewed_out and name in self.environment_audit:
                    self.environment_audit[name]['status'] = 'no_reviewed_dbbody_mapping'
                audit = self.environment_audit.get(name, {})
                unknown_environment = (audit.get('unknown', 0) or audit.get('source_unknown', 0)
                                       or audit.get('missing_source_layers'))
                self.pruned.append({'phenomenon': name, 'reason':
                    'no_reviewed_dbbody_mapping' if reviewed_out else
                    'environment_data_incomplete' if unknown_environment else 'no_occurring_conditions_in_study_area'})
        screened = {'nodes': {n: v for n, v in self.knowledge['nodes'].items() if n in viable},
                    'edges': [e for e in self.knowledge['edges']
                              if e['source'] in viable and e['target'] in viable]}
        edges, removed = screen_causal_graph(screened, root_name, max_depth)
        adjacency = defaultdict(list)
        for edge in edges:
            adjacency[edge['source']].append(edge)
        root = {'id': uid(self.config['snapshot'], root_name), 'phenomenon': root_name,
                'event_description': f'Administrative Region,{root_name}', 'current_depth': 0,
                'priority': 1, 'is_happen': True, 'affect_extent': self.rain,
                'affect_features': [], 'parent_id': None, 'path': [root_name],
                'relation_type': None, 'observation_tasks': []}
        root['observation_tasks'] = self.observation_tasks(root_name, [], self.rain)
        nodes, queue = [root], deque([root])
        while queue:
            parent = queue.popleft()
            if parent['current_depth'] >= max_depth:
                continue
            print(f"Instantiating depth {parent['current_depth'] + 1}: {parent['phenomenon']}", flush=True)
            for edge in adjacency[parent['phenomenon']]:
                name = edge['target']
                if name in parent['path']:
                    self.pruned.append({'parent': parent['id'], 'phenomenon': name, 'reason': 'path_cycle'})
                    continue
                selected, extent = self.calculate_disaster_extent(name, parent['affect_extent'])
                if extent.is_empty:
                    self.pruned.append({'parent': parent['id'], 'phenomenon': name,
                                        'reason': 'empty_bearing_environment_parent_intersection'})
                    continue
                path = parent['path'] + [name]
                node = {'id': uid(self.config['snapshot'], path), 'phenomenon': name,
                        'event_description': name, 'current_depth': parent['current_depth'] + 1,
                        'priority': parent['priority'] + 1, 'is_happen': False,
                        'affect_extent': extent, 'affect_features': selected,
                        'parent_id': parent['id'], 'path': path,
                        'relation_type': edge['relation_type'],
                        'observation_tasks': self.observation_tasks(name, selected, extent)}
                nodes.append(node)
                queue.append(node)
                if len(nodes) > self.config.get('max_instances', 10000):
                    raise RuntimeError('实例数量超限，请检查深度和知识图结构；未截断输出冒充完整结果')
        self.nodes, self.removed_edges = nodes, removed
        return nodes

    def export_results(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        save_json(folder / 'environment_audit.json', dict(
            regional_mean_elevation_m=self.regional_mean_elevation,
            missing_policy=self.config.get('environment_missing_policy', 'error'),
            source=self.config.get('environment_rule_source', 'config'),
            rules=self.environment_rules, evaluations=self.environment_audit))
        project_back = self.features.to_wgs84
        event_features, feature_rows, tasks = [], [], []
        features_output = defaultdict(dict)
        factor_to_disaster = defaultdict(set)
        for node in self.nodes:
            selected_types = {feature['geo_type'] for feature in node['affect_features']}
            event_features.append({'type': 'Feature', 'geometry': mapping(transform(project_back, node['affect_extent'])),
                                   'properties': {k: node[k] for k in ('id', 'phenomenon', 'event_description',
                                                  'current_depth', 'priority', 'is_happen', 'parent_id', 'path', 'relation_type')}})
            for feature in node['affect_features']:
                event_desc = f"{feature['geo_type']},{node['phenomenon']}"
                # 二值 eligibility=100 仅为旧绘图字段兼容，不是概率或风险评分。
                record = {event_desc: 100.0}
                existing = features_output[feature['geo_type']].setdefault(feature['osm_id'], [])
                if record not in existing:
                    existing.append(record)
                feature_rows.append({'event_id': node['id'], 'phenomenon': node['phenomenon'],
                                     'geo_type': feature['geo_type'], 'osm_id': feature['osm_id'],
                                     'depth': node['current_depth'], 'eligible': 1})
            for index, task in enumerate(node['observation_tasks']):
                task_copy = {k: v for k, v in task.items() if k != 'geometry'}
                task_copy.update(id=uid(node['id'], 'obs', index), disaster_id=node['id'],
                                 phenomenon=node['phenomenon'], priority=node['priority'])
                task_copy['geometry_reference'] = {'file': 'events.geojson', 'event_id': node['id']}
                tasks.append(task_copy)
                for factor in task.get('factors', []):
                    for geo_type in selected_types:
                        if not task.get('geo_types') or geo_type in task['geo_types']:
                            factor_to_disaster[factor].add(f"{geo_type},{node['phenomenon']}")
        save_json(folder / 'events.geojson', {'type': 'FeatureCollection', 'features': event_features})
        save_json(folder / 'observation_tasks.json', tasks)
        save_json(folder / 'features_output.json', features_output)
        save_json(folder / 'factor_to_disaster.json', {k: sorted(v) for k, v in factor_to_disaster.items()})
        save_json(folder / 'pruning_report.json', {'removed_abstract_edges': self.removed_edges,
                                                 'pruned_instances': self.pruned})
        with open(folder / 'affected_features.csv', 'w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=['event_id', 'phenomenon', 'geo_type', 'osm_id', 'depth', 'eligible'])
            writer.writeheader()
            writer.writerows(feature_rows)
        return {'event_instances': len(self.nodes), 'causal_edges': len(self.nodes) - 1,
                'observation_instances': len(tasks), 'feature_event_records': len(feature_rows),
                'unique_phenomena': len({n['phenomenon'] for n in self.nodes}),
                'proxy_phenomena': sorted(self.proxy_phenomena),
                'scenario_environment_phenomena': sorted(self.scenario_proxy_phenomena),
                'environment_unknown_phenomena': sorted(n for n, a in self.environment_audit.items()
                    if a['unknown'] or a['source_unknown'] or a.get('missing_source_layers')),
                'missing_geographic_layers': sorted(self.features.missing_layers),
                'method_status': ('simplified_OELG_baseline' if self.proxy_phenomena else
                                  'scenario_environment_OELG_baseline' if self.scenario_proxy_phenomena
                                  else 'configured_OELG_spatial_baseline'),
                'sensor_matching': 'candidate_metadata_only',
                'knowledge_gaps': {'phenomena_without_observation_tasks': sorted({
                    n['phenomenon'] for n in self.nodes if not n['observation_tasks']})}}


def draw_geometry(ax, geometry, color, alpha=0.6, edgecolor=None, linewidth=0.5):
    from matplotlib.path import Path as PlotPath
    from matplotlib.patches import PathPatch
    from shapely.geometry.polygon import orient
    if geometry.is_empty:
        return
    if geometry.geom_type == 'Polygon':
        geometry = orient(geometry, sign=1)
        vertices, codes = [], []
        for ring in [geometry.exterior, *geometry.interiors]:
            points = list(ring.coords)
            vertices.extend(points)
            codes.extend([PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 2) + [PlotPath.CLOSEPOLY])
        ax.add_patch(PathPatch(PlotPath(vertices, codes), facecolor=color,
                              edgecolor=edgecolor or color, alpha=alpha, linewidth=linewidth))
    elif geometry.geom_type in ('MultiPolygon', 'GeometryCollection', 'MultiLineString', 'MultiPoint'):
        for part in geometry.geoms:
            draw_geometry(ax, part, color, alpha, edgecolor, linewidth)
    elif geometry.geom_type == 'LineString':
        x, y = geometry.xy
        ax.plot(x, y, color=edgecolor or color, alpha=alpha, linewidth=linewidth)
    elif geometry.geom_type == 'Point':
        ax.scatter([geometry.x], [geometry.y], color=color, s=5)


def visualize_results(builder, folder, config):
    """输出相同范围的逐现象地图、实例传播图及逐因子二值观测范围。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import networkx as nx
    folder = Path(folder)
    maps = folder / 'event_maps'
    maps.mkdir(exist_ok=True)
    phenomenon_geometries = defaultdict(list)
    for node in builder.nodes:
        phenomenon_geometries[node['phenomenon']].append(node['affect_extent'])
    regions = {name: unary_union(items) for name, items in phenomenon_geometries.items()}
    bounds = builder.region.bounds

    def setup_axis(ax):
        draw_geometry(ax, builder.region, 'none', 1, '#344054', 1)
        draw_geometry(ax, builder.rain, 'none', 1, '#2474b5', 1)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect('equal')
        ax.ticklabel_format(style='plain', useOffset=False)
        ax.set_xlabel(f"Easting (m; {config['projected_crs']})")
        ax.set_ylabel('Northing (m)')

    for name, geometry in sorted(regions.items()):
        fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
        draw_geometry(ax, geometry, '#c77b36', 0.75)
        setup_axis(ax)
        ax.set_title(f'OELG: {name}\nBinary candidate extent (not risk probability)')
        fig.savefig(maps / f'{file_key(name)}.png', dpi=160)
        plt.close(fig)
    graph = nx.DiGraph()
    for node in builder.nodes:
        graph.add_node(node['id'], subset=node['current_depth'])
        if node['parent_id']:
            graph.add_edge(node['parent_id'], node['id'])
    positions = nx.multipartite_layout(graph, subset_key='subset', align='vertical')
    levels = Counter(n['current_depth'] for n in builder.nodes)
    fig, ax = plt.subplots(figsize=(15, max(5, max(levels.values()) * 0.35)), layout='constrained')
    nx.draw_networkx_edges(graph, positions, ax=ax, alpha=0.25, arrowsize=8)
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=70,
                           node_color=[builder.nodes[i]['current_depth'] for i in range(len(builder.nodes))], cmap='YlOrBr')
    nx.draw_networkx_labels(graph, positions,
                           labels={n['id']: n['phenomenon'] for n in builder.nodes}, font_size=7, ax=ax)
    ax.set_title('OELG instantiated disaster cascade\nLevels show causal sequence, not probability')
    ax.axis('off')
    fig.savefig(folder / 'cascade.png', dpi=180)
    plt.close(fig)
    factor_geometries = defaultdict(dict)
    task_extent_cache = {}
    for node in builder.nodes:
        for task in node['observation_tasks']:
            allowed_types = task.get('geo_types') or []
            key = (node['id'], tuple(sorted(allowed_types)))
            if key not in task_extent_cache:
                if not allowed_types:
                    task_extent_cache[key] = node['affect_extent']
                else:
                    task_extent_cache[key] = unary_union([
                        f['footprint'] for f in node['affect_features'] if f['geo_type'] in allowed_types])
            extent = task_extent_cache[key]
            if not extent.is_empty:
                for factor in task.get('factors', []):
                    factor_geometries[factor][key] = extent
    factor_maps = folder / 'factor_maps'
    factor_maps.mkdir(exist_ok=True)
    for name, items in sorted(factor_geometries.items()):
        extent = unary_union(list(items.values()))
        fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
        draw_geometry(ax, extent, '#357d82', 0.75)
        setup_axis(ax)
        ax.set_title(f'OELG observation requirement: {name}\nBinary eligibility (not importance score)')
        fig.savefig(factor_maps / f'{file_key(name)}.png', dpi=160)
        plt.close(fig)
    save_json(folder / 'factor_extents.geojson', {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {'factor': name},
         'geometry': mapping(transform(builder.features.to_wgs84, unary_union(list(items.values()))))}
        for name, items in sorted(factor_geometries.items())]})
    return regions


def export_rasters_and_compare(builder, regions, folder, config):
    """二值栅格对比；不同方法的风险评分不可直接视为同尺度数值。"""
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder = Path(folder)
    raster_folder = folder / 'event_rasters'
    raster_folder.mkdir(exist_ok=True)
    xmin, ymin, xmax, ymax = builder.region.bounds
    resolution = float(config['grid_resolution_m'])
    width, height = math.ceil((xmax - xmin) / resolution), math.ceil((ymax - ymin) / resolution)
    if width * height > 25_000_000:
        raise ValueError('输出网格过大，请增大 grid_resolution_m')
    affine = from_origin(xmin, ymax, resolution, resolution)
    grid_shape = (height, width)
    grid_crs = builder.features.crs
    reference_folder = config.get('doelg_rasters')
    reference_index = {}
    if reference_folder:
        for path in sorted(Path(reference_folder).glob('*.tif')):
            name = path.stem.removesuffix('_importance')
            key = normalized(file_key(name))
            if key in reference_index:
                raise ValueError(f'对比目录中现象文件名规范化后重复：{path}')
            reference_index[key] = path
        if not reference_index:
            raise FileNotFoundError(f'DOELG 对比目录没有 tif：{reference_folder}')
        if config.get('comparison_grid', 'doelg') == 'doelg':
            with rasterio.open(next(iter(reference_index.values()))) as src:
                if src.crs is None or src.transform.b != 0 or src.transform.d != 0:
                    raise ValueError('DOELG 参考网格必须有 CRS 且为正北向')
                affine, grid_crs = src.transform, src.crs
                height, width = src.height, src.width
                grid_shape = (height, width)
                xmin, ymin, xmax, ymax = src.bounds
    grid_project = Transformer.from_crs(builder.features.crs, grid_crs, always_xy=True).transform
    grid_region = transform(grid_project, builder.region)
    valid = rasterize([(grid_region, 1)], out_shape=grid_shape, transform=affine).astype(bool)
    if not valid.any():
        raise ValueError('对比网格与研究区不相交')
    # 面积用等面积投影计算每个网格面积，避免 Web Mercator 面积膨胀。
    equal_area = Transformer.from_crs(grid_crs, 6933, always_xy=True).transform
    cell_areas = np.zeros(grid_shape)
    for row in range(height):
        for col in range(width):
            if valid[row, col]:
                x0, y0 = affine * (col, row)
                x1, y1 = affine * (col + 1, row + 1)
                cell_areas[row, col] = transform(equal_area, Polygon(
                    [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])).area / 1e6
    selected_features = defaultdict(dict)
    for node in builder.nodes:
        for feature in node['affect_features']:
            selected_features[node['phenomenon']][feature['osm_id']] = feature['geometry']
    names = sorted(set(regions) | {name for name in builder.knowledge['nodes']
                                  if normalized(file_key(name)) in reference_index})
    rows = []
    skipped_geometry_types = Counter()
    comparison_folder = folder / 'comparison'
    comparison_folder.mkdir(exist_ok=True)
    for name in names:
        geometry = transform(grid_project, regions.get(name, GeometryCollection()))
        extent_mask = (rasterize([(geometry, 1)], out_shape=grid_shape, transform=affine,
                          all_touched=config['all_touched']).astype(bool)
                if not geometry.is_empty else np.zeros(grid_shape, bool)) & valid
        oelg = np.zeros(grid_shape, bool)
        if name == config['root_phenomenon']:
            oelg = extent_mask.copy()
        else:
            # 与原绘图函数一致：面栅格化，线按 2*pixel_width 采样，点落格。
            for original_geometry in selected_features[name].values():
                feature_geometry = transform(grid_project, original_geometry)
                if feature_geometry.geom_type in ('Polygon', 'MultiPolygon'):
                    oelg |= rasterize([(feature_geometry, 1)], out_shape=grid_shape,
                                      transform=affine, all_touched=False).astype(bool)
                elif feature_geometry.geom_type in ('Point', 'LineString'):
                    if feature_geometry.geom_type == 'Point' or feature_geometry.length == 0:
                        points = [feature_geometry if feature_geometry.geom_type == 'Point' else feature_geometry.interpolate(0)]
                    else:
                        points = [feature_geometry.interpolate(d) for d in np.arange(
                            0, feature_geometry.length, 2 * abs(affine.a))]
                    for point in points:
                        col, row_index = ~affine * (point.x, point.y)
                        col, row_index = math.floor(col), math.floor(row_index)
                        if 0 <= row_index < height and 0 <= col < width:
                            oelg[row_index, col] = True
                # 原 instantiation_en.py 对 MultiLineString/MultiPoint 也未落格。
                else:
                    skipped_geometry_types[feature_geometry.geom_type] += 1
        oelg &= valid
        for suffix, mask in [('extent', extent_mask), ('features', oelg)]:
            raster = np.where(valid, mask.astype('uint8'), 255).astype('uint8')
            with rasterio.open(raster_folder / f'{file_key(name)}_{suffix}_eligible.tif', 'w', driver='GTiff',
                           width=width, height=height, count=1, dtype='uint8',
                           crs=grid_crs, transform=affine, nodata=255,
                           compress='deflate') as dst:
                dst.write(raster, 1)
                dst.update_tags(method='OELG', semantics=f'binary {suffix} eligibility; not risk score')
        # 原函数在 GeoTIFF 导出之前执行 Gaussian filtering（默认 sigma=5）。
        # 二值支持域使用相同核；只比较支持范围，不把二值平滑值当作风险分数。
        sigma = float(config.get('reference_blur_sigma', 0))
        if sigma < 0:
            raise ValueError('reference_blur_sigma 不能为负')
        if sigma:
            radius = int(4 * sigma + 0.5)
            offsets = np.arange(-radius, radius + 1)
            kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
            kernel /= kernel.sum()
            smoothed = oelg.astype(float)
            for axis in (0, 1):
                padding = [(0, 0), (0, 0)]
                padding[axis] = (radius, radius)
                padded = np.pad(smoothed, padding, mode='symmetric')
                smoothed = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode='valid'), axis, padded)
            oelg = (smoothed > 0) & valid
        row = {'phenomenon': name, 'oelg_area_km2': float(cell_areas[oelg].sum()),
               'comparison_status': 'no_reference', 'doelg_area_km2': None,
               'intersection_km2': None, 'union_km2': None, 'iou': None,
               'oelg_only_km2': None, 'doelg_only_km2': None,
               'reference_valid_fraction': None}
        reference = reference_index.get(normalized(file_key(name)))
        if reference:
            with rasterio.open(reference) as src:
                if src.crs is None:
                    raise ValueError(f'DOELG 栅格缺少 CRS：{reference}')
                values = src.read(1, masked=True)
                # 先在原始网格二值化，再用最近邻投影，不插值制造风险值。
                known = ~np.ma.getmaskarray(values) & np.isfinite(values.data)
                binary = np.where(known, values.data > config['doelg_threshold'], 255).astype('uint8')
                target = np.full(grid_shape, 255, dtype='uint8')
                reproject(binary, target, src_transform=src.transform, src_crs=src.crs,
                          src_nodata=255, dst_transform=affine, dst_crs=grid_crs,
                          dst_nodata=255, resampling=Resampling.nearest)
            common_valid = valid & (target != 255)
            reference_fraction = float(cell_areas[common_valid].sum() / cell_areas[valid].sum())
            doelg = (target == 1) & common_valid
            current = oelg & common_valid
            both, union = current & doelg, current | doelg
            union_area = float(cell_areas[union].sum())
            row.update(comparison_status='common_valid_footprint',
                       reference_valid_fraction=reference_fraction,
                       oelg_area_km2=float(cell_areas[current].sum()),
                       doelg_area_km2=float(cell_areas[doelg].sum()),
                       intersection_km2=float(cell_areas[both].sum()), union_km2=union_area,
                       iou=float(cell_areas[both].sum() / union_area) if union_area else None,
                       oelg_only_km2=float(cell_areas[current & ~doelg].sum()),
                       doelg_only_km2=float(cell_areas[doelg & ~current].sum()))
            from matplotlib.colors import ListedColormap
            fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
            arrays = [np.ma.array(current, mask=~common_valid), np.ma.array(doelg, mask=~common_valid),
                      np.ma.array(current.astype('uint8') + 2 * doelg.astype('uint8'), mask=~common_valid)]
            for ax, values, title in zip(axes, arrays, [f'OELG feature support (sigma={sigma:g})',
                                                         f'DOELG score > {config["doelg_threshold"]}',
                                                         'Orange OELG / Blue DOELG / Purple both']):
                cmap = ListedColormap(['#f2f4f7', '#c77b36', '#3182bd', '#7b4c9a']) if ax is axes[2] else ListedColormap(['#f2f4f7', '#c77b36' if ax is axes[0] else '#3182bd'])
                ax.imshow(values, extent=(xmin, xmin + width * affine.a, ymax + height * affine.e, ymax),
                          origin='upper', cmap=cmap, vmin=0, vmax=3 if ax is axes[2] else 1,
                          interpolation='nearest')
                ax.set_title(title, fontsize=10)
                ax.set_aspect('equal')
                ax.ticklabel_format(style='plain', useOffset=False)
                ax.set_xlabel('Longitude' if CRS.from_user_input(grid_crs).is_geographic else 'Easting (m)')
            title_iou = f'{row["iou"]:.3f}' if row['iou'] is not None else 'undefined (both empty)'
            fig.suptitle(f'{name} | spatial agreement IoU = {title_iou}\n'
                         'Agreement is not instantiation_validation accuracy; verify matching DOELG input and snapshot')
            fig.savefig(comparison_folder / f'{file_key(name)}_comparison.png', dpi=150)
            plt.close(fig)
            builder.features.input_hashes[str(reference.resolve())] = sha256(reference)
        rows.append(row)
    with open(folder / 'comparison.csv', 'w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(folder / 'comparison_protocol.json', {
        'representation': 'selected feature footprints using instantiation_en.py raw rasterization',
        'grid_crs': str(grid_crs), 'grid_shape': list(grid_shape), 'grid_transform': list(affine),
        'doelg_threshold_exclusive': config['doelg_threshold'],
        'matched_gaussian_sigma_pixels': config.get('reference_blur_sigma', 0),
        'gaussian_truncate': 4.0,
        'reference_input_status': config.get('doelg_comparison_status', 'unverified'),
        'geometry_types_not_rasterized_by_legacy_protocol': dict(skipped_geometry_types),
        'area_unit': 'km2, cells transformed to EPSG:6933',
        'scope': 'spatial agreement, not predictive accuracy or probability calibration'})
    return rows


def save_instance_database(repository, builder, database):
    """Commit the complete graph atomically; exports are audit copies only."""
    from oelg_graph_store import write_builder
    return write_builder(repository.driver, builder, database)


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    base = path.parent
    for key in ('abstract_database', 'observation_database', 'instance_database'):
        config[key] = resolve_database_name(config[key])
    for key in ('rainfall', 'region', 'features_folder', 'output_folder', 'knowledge_json', 'doelg_rasters',
                'risk_basemap', 'risk_maps_folder', 'factor_maps_folder'):
        if config.get(key):
            config[key] = str((base / config[key]).resolve())
    for rule in config.get('environment_rules', {}).values():
        if rule.get('mask'):
            rule['mask'] = str((base / rule['mask']).resolve())
    if config['environment_mode'] not in ('proxy', 'strict'):
        raise ValueError('environment_mode 必须为 proxy 或 strict')
    if config.get('environment_rule_source', 'config') not in ('config', 'oelg'):
        raise ValueError('environment_rule_source 必须为 config 或 oelg')
    if config.get('environment_rule_source') == 'oelg' and config.get('environment_rules'):
        raise ValueError('使用 OELG 环境规则时不得同时设置本地 environment_rules')
    if config.get('environment_missing_policy', 'error') not in ('error', 'report_and_skip'):
        raise ValueError('environment_missing_policy 必须为 error 或 report_and_skip')
    if config.get('environment_rule_source') == 'oelg' and 'regional_mean_elevation_m' not in config:
        raise ValueError('必须显式指定 regional_mean_elevation_m')
    if 'regional_mean_elevation_m' in config:
        number(config['regional_mean_elevation_m'])
    if config.get('missing_layer_policy', 'error') not in ('error', 'report_and_skip'):
        raise ValueError('missing_layer_policy 必须为 error 或 report_and_skip')
    if not isinstance(config['max_depth'], int) or config['max_depth'] < 0:
        raise ValueError('max_depth 必须为非负整数')
    if config['feature_buffer_m'] < 0 or config['grid_resolution_m'] <= 0:
        raise ValueError('缓冲距离须非负，网格分辨率须大于零')
    datetime.fromisoformat(config['snapshot_time'])
    return config


def build_risk_features_output(nodes):
    """按现象合并被选中的地物，100 表示候选，不代表发生概率。"""
    output = defaultdict(dict)
    for node in nodes:
        for feature in node['affect_features']:
            event = f"{feature['geo_type']},{node['phenomenon']}"
            items = output[feature['geo_type']].setdefault(feature['osm_id'], [])
            record = {event: 100.0}
            if record not in items:
                items.append(record)
    return dict(output)


def calculate_threshold_coverage(grid, valid, fraction=0.25):
    """按图像网格统计严格高于轮廓阈值的范围占比，不含图片白边。"""
    grid = np.asarray(grid, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if grid.ndim != 2 or grid.shape != valid.shape or not valid.any():
        raise ValueError('风险网格与研究区掩膜须尺寸一致，且研究区至少包含一个像元')
    if not 0 < fraction < 1 or not np.isfinite(grid[valid]).all():
        raise ValueError('阈值比例须在 0 与 1 之间，研究区风险值须为有限数')
    low, high = float(grid[valid].min()), float(grid[valid].max())
    threshold = low + fraction * (high - low)
    above = int(np.count_nonzero(valid & (grid > threshold)))
    valid_count = int(np.count_nonzero(valid))
    return {
        'min_value': low, 'max_value': high, 'threshold': threshold,
        'above_threshold_pixels': above, 'total_map_pixels': int(grid.size),
        'valid_region_pixels': valid_count,
        'above_threshold_map_percent': 100.0 * above / grid.size,
        'above_threshold_region_percent': 100.0 * above / valid_count,
    }


def build_factor_to_disaster(nodes):
    """保留观测任务的地理类型限制，建立因子与具体灾害类型的映射。"""
    result = defaultdict(set)
    for node in nodes:
        geo_types = {feature['geo_type'] for feature in node['affect_features']}
        for task in node['observation_tasks']:
            allowed = task.get('geo_types') or []
            for geo_type in geo_types:
                if allowed and geo_type not in allowed:
                    continue
                for factor in canonical_factors(task.get('factors', [])):
                    result[factor].add(f"{geo_type},{node['phenomenon']}")
    return {factor: sorted(events) for factor, events in sorted(result.items())}


def aggregate_factor_features(features_output, factor_to_disaster):
    """同一因子、同一地物取相关灾害分值最大值，避免路径重复计数。"""
    event_to_factors = defaultdict(set)
    for original_factor, events in factor_to_disaster.items():
        factor = canonical_factors([original_factor])[0]
        for event in events:
            event_to_factors[event].add(factor)
    output = {}
    for geo_type, features in features_output.items():
        selected = {}
        for osm_id, records in features.items():
            scores = {}
            for record in records:
                for event, score in record.items():
                    if float(score) <= 0:
                        continue
                    for factor in event_to_factors.get(event, ()):
                        scores[factor] = max(scores.get(factor, 0.0), float(score))
            if scores:
                selected[osm_id] = [{factor: score} for factor, score in sorted(scores.items())]
        if selected:
            output[geo_type] = selected
    return output


def plot_observation_factor_maps(features_output, factor_to_disaster, config):
    """复用灾害图的底图、栅格化、平滑、轮廓线和阈值占比统计。"""
    if not factor_to_disaster:
        raise ValueError('缺少观测因子—灾害映射，请先读取观测知识并生成 factor_to_disaster.json')
    factor_features = aggregate_factor_features(features_output, factor_to_disaster)
    mapped = {factor for features in factor_features.values() for records in features.values()
              for record in records for factor in record}
    skipped = sorted(set(factor_to_disaster) - mapped)
    if skipped:
        print(f'No selected features for the following observation factors; maps skipped: {skipped}', flush=True)
    factor_config = dict(config)
    factor_config['risk_maps_folder'] = config.get('factor_maps_folder') or str(
        Path(config['risk_maps_folder']).parent / 'output_factor_maps_grid')
    return plot_disaster_risk_maps(factor_features, factor_config, map_kind='factor')


def plot_disaster_risk_maps(features_output, config, map_kind='phenomenon'):
    """输出真实平滑场 GeoTIFF，并可选绘制底图、色阶和轮廓 PNG。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib import font_manager
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    for filename in ('msyh.ttc', 'simhei.ttf', 'simsun.ttc'):
        font = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / filename
        if font.is_file():
            font_manager.fontManager.addfont(str(font))
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'SimSun']
    matplotlib.rcParams['font.sans-serif'] = [n for n in preferred if n in available] + ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    folder = Path(config.get('risk_maps_folder') or Path(config['output_folder']) / 'risk_maps')
    folder.mkdir(parents=True, exist_ok=True)
    render_png = config.get('risk_render_png', True)
    if render_png:
        background_path = Path(config['risk_basemap'])
        if not background_path.is_file():
            raise FileNotFoundError(f'缺少风险图底图：{background_path}')
        background = mpimg.imread(background_path)
    region = read_extent(config['region'])
    xmin, ymin, xmax, ymax = region.bounds
    size = int(config.get('risk_grid_size', 200))
    sigma = float(config.get('risk_blur_sigma', 5))
    fraction = float(config.get('risk_contour_fraction', 0.25))
    if size < 2 or sigma < 0 or not 0 < fraction < 1:
        raise ValueError('风险图网格须至少 2，sigma 须非负，轮廓比例须在 0 与 1 之间')
    affine = from_bounds(xmin, ymin, xmax, ymax, size, size)
    valid = rasterize([(region, 1)], out_shape=(size, size), transform=affine).astype(bool)
    store = GeographicFeatureStore(config['features_folder'], region, config['projected_crs'])
    grouped = defaultdict(lambda: defaultdict(dict))
    for geo_type, ids in features_output.items():
        for osm_id, records in ids.items():
            for record in records:
                for event, score in record.items():
                    if float(score) > 0:
                        name = event if map_kind == 'factor' else phenomenon(event)
                        grouped[name][geo_type][osm_id] = max(
                            float(score), grouped[name][geo_type].get(osm_id, 0))
    if not grouped:
        raise ValueError('没有可绘制的地物分布数据')
    # 每个地物图层只读取、投影一次；生成所有现象共享的 WGS84 几何索引。
    geometries = {}
    for geo_type in sorted({t for groups in grouped.values() for t in groups}):
        geometries[geo_type] = {f['osm_id']: transform(store.to_wgs84, f['geometry'])
                                for f in store.load(geo_type)}
        store.cache.pop(geo_type, None)

    def parts(geometry):
        if geometry.geom_type in ('MultiLineString', 'MultiPoint', 'GeometryCollection'):
            for part in geometry.geoms:
                yield from parts(part)
        else:
            yield geometry

    extent = [xmin, xmax, ymin, ymax]
    pixel_width, pixel_height = (xmax - xmin) / size, (ymax - ymin) / size
    lons = xmin + (np.arange(size) + 0.5) * pixel_width
    lats = ymax - (np.arange(size) + 0.5) * pixel_height
    missing_ids = []
    outputs = []
    coverage_rows = []
    for name, types in sorted(grouped.items()):
        grid = np.zeros((size, size), dtype=float)
        for geo_type, ids in types.items():
            for osm_id, value in ids.items():
                geometry = geometries[geo_type].get(osm_id)
                if geometry is None:
                    missing_ids.append(osm_id)
                    continue
                for part in parts(geometry):
                    if part.geom_type in ('Polygon', 'MultiPolygon'):
                        grid = np.maximum(grid, rasterize([(part, value)], out_shape=grid.shape,
                                                         transform=affine, dtype='float64'))
                    elif part.geom_type in ('Point', 'LineString'):
                        points = ([part] if part.geom_type == 'Point' else
                                  [part.interpolate(d) for d in np.arange(0, max(part.length, 1e-12), 2 * pixel_width)])
                        for point in points:
                            col, row = math.floor((point.x - xmin) / pixel_width), math.floor((ymax - point.y) / pixel_height)
                            if 0 <= row < size and 0 <= col < size:
                                grid[row, col] = max(grid[row, col], value)
        if sigma:
            radius = int(4 * sigma + 0.5)
            kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
            kernel /= kernel.sum()
            for axis in (0, 1):
                padding = [(0, 0), (0, 0)]
                padding[axis] = (radius, radius)
                grid = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode='valid'), axis,
                                          np.pad(grid, padding, mode='symmetric'))
        tif_path = export_importance_geotiff(grid, valid, affine, name, folder, map_kind, sigma)
        field = np.ma.array(grid, mask=~valid)
        coverage = calculate_threshold_coverage(grid, valid, fraction)
        low, high = coverage['min_value'], coverage['max_value']
        threshold = coverage['threshold']
        # 与原图一致逐图显示；该色阶用于分布展示，不能跨现象直接比较概率。
        filename = folder / f'{file_key(name)}_field.png'
        if render_png:
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(background, extent=extent, origin='upper', zorder=1)
            ax.imshow(field, extent=extent, origin='upper', cmap='YlOrRd',
                      alpha=0.65, vmin=low, vmax=high if high > low else low + 1, zorder=2)
            if high > low:
                ax.contour(lons + (xmax-xmin)*0.003, lats - (ymax-ymin)*0.003, field,
                           levels=[threshold], colors=['#333333'], linewidths=4, alpha=0.4, zorder=3)
                ax.contour(lons, lats, field, levels=[threshold], colors=['#E65100'], linewidths=2.5, zorder=4)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_axis_off()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            fig.savefig(filename, dpi=int(config.get('risk_dpi', 300)), bbox_inches='tight', pad_inches=0.06)
            plt.close(fig)
        outputs.append(str(filename if render_png else tif_path))
        coverage_rows.append({map_kind: name, **coverage,
                              'image_file': filename.name if render_png else '', 'geotiff_file': str(tif_path)})
        print(f'GeoTIFF exported: {tif_path}', flush=True)
        print(f"  Threshold: {threshold:.6g}；Strictly above threshold: {coverage['above_threshold_pixels']} pixels; "
              f"of all cells: {coverage['above_threshold_map_percent']:.2f}%；"
              f"of valid study area: {coverage['above_threshold_region_percent']:.2f}%", flush=True)
    if missing_ids:
        raise ValueError(f'有 {len(set(missing_ids))} 个地物 ID 未在原图层找到，不能视为完整结果：{sorted(set(missing_ids))[:10]}')
    report_path = folder / ('factor_threshold_coverage.csv' if map_kind == 'factor' else 'risk_threshold_coverage.csv')
    with open(report_path, 'w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)
    print(f'Threshold coverage saved: {report_path}', flush=True)
    return outputs


def export_importance_geotiff(grid, valid, affine, name, folder, map_kind, sigma):
    """与 instantiation_en 的目录/命名一致；不将显示阈值或色阶写入栅格。"""
    import re
    import rasterio
    grid, valid = np.asarray(grid), np.asarray(valid, dtype=bool)
    if grid.ndim != 2 or grid.shape != valid.shape or not valid.any():
        raise ValueError('GeoTIFF 网格与有效区掩膜须尺寸一致且有效区非空')
    if not np.isfinite(grid[valid]).all():
        raise ValueError('有效区内存在非有限重要性值')
    if map_kind not in ('factor', 'phenomenon'):
        raise ValueError('map_kind 须为 factor 或 phenomenon')
    folder = Path(folder)
    target = Path(str(folder) + '_geotiff') if map_kind == 'factor' else folder / 'geotiff'
    target.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', name).strip() if map_kind == 'factor' else file_key(name)
    suffix = '_importance_field.tif' if map_kind == 'factor' else '_importance.tif'
    path = target / f'{safe_name}{suffix}'
    values = np.where(valid, grid, -9999.).astype('float32')
    with rasterio.open(path, 'w', driver='GTiff', height=grid.shape[0], width=grid.shape[1],
                       count=1, dtype='float32', crs='EPSG:4326', transform=affine,
                       nodata=-9999., compress='deflate') as dst:
        dst.write(values, 1)
        dst.update_tags(method='OELG', field_name=name, field_kind=map_kind,
                        blur_sigma=str(sigma), value_semantics='smoothed selected-feature score; not probability',
                        processing='existing OELG rasterization and smoothing; no display clipping')
    return path


def refresh_standard_factor_exports(config, target, repository):
    """Refresh factor links of saved instances without changing spatial eligibility."""
    source,target = Path(config['output_folder']),Path(target).resolve()
    required = ['events.geojson','affected_features.csv','features_output.json',
                'observation_tasks.json','knowledge_snapshot.json']
    for name in required:
        if not (source/name).is_file():
            raise FileNotFoundError(source/name)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f'Standard-factor target must be empty: {target}')
    observations = repository.read_observation_knowledge(config['observation_database'])
    index = {}
    for name,tasks in observations.items():
        for task in tasks:
            key = (name,task['node_id'],task.get('source_geo_type'))
            if key in index and index[key]['factors']!=task['factors']:
                raise ValueError(f'Conflicting observation StandardFactor associations: {key}')
            index[key] = task
    events = read_json(source/'events.geojson')['features']
    nodes = {e['properties']['id']:dict(observation_tasks=[],affect_features=[],
               phenomenon=e['properties']['phenomenon']) for e in events}
    types = defaultdict(set)
    with (source/'affected_features.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['event_id'] not in nodes:
                raise ValueError(f'Unknown event in affected_features.csv: {row["event_id"]}')
            if int(row['eligible']):
                types[row['event_id']].add(row['geo_type'])
    for key,node in nodes.items():
        node['affect_features'] = [{'geo_type':geo} for geo in sorted(types[key])]
    refreshed,unmapped = [],[]
    for task in read_json(source/'observation_tasks.json'):
        key = (task['phenomenon'],task['node_id'],task.get('source_geo_type'))
        standard = index.get(key)
        if standard is None:
            unmapped.append(dict(task_id=task['id'],node_id=task['node_id'],
                                 reason='No linked StandardFactor in current observation knowledge'))
            continue
        updated = dict(task,factors=canonical_factors(standard['factors']),factor_schema='StandardFactor')
        refreshed.append(updated)
        nodes[task['disaster_id']]['observation_tasks'].append(updated)
    if not refreshed:
        raise ValueError('No saved observation tasks could be linked to StandardFactor.')
    factor_map = build_factor_to_disaster(list(nodes.values()))
    if not factor_map:
        raise ValueError('No StandardFactor has eligible feature/event associations.')
    target.mkdir(parents=True,exist_ok=True)
    for name in ('events.geojson','affected_features.csv','features_output.json','pruning_report.json'):
        if (source/name).is_file():
            shutil.copy2(source/name,target/name)
    knowledge = read_json(source/'knowledge_snapshot.json')
    knowledge.update(observations=observations,factor_schema='StandardFactor')
    save_json(target/'knowledge_snapshot.json',knowledge)
    save_json(target/'observation_tasks.json',refreshed)
    save_json(target/'factor_to_disaster.json',factor_map)
    report = dict(source=str(source),target=str(target),factor_schema='StandardFactor',
                  allowed_factors=list(STANDARD_FACTORS),linked_factors=sorted(factor_map),
                  refreshed_tasks=len(refreshed),unmapped_tasks=unmapped,
                  observation_database=config['observation_database'],
                  unchanged_input_hashes={name:sha256(target/name) for name in
                                         ('events.geojson','affected_features.csv','features_output.json')})
    save_json(target/'standard_factor_report.json',report)
    print(f'StandardFactor refresh: {len(refreshed)} tasks, {len(factor_map)} factors, '
          f'{len(unmapped)} unmapped tasks reported -> {target}',flush=True)
    return report


def run_suffix():
    return datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]


def available_output_folder(path):
    """Keep existing output intact, including partially completed runs."""
    path = Path(path)
    candidate = path
    while candidate.exists() and (not candidate.is_dir() or any(candidate.iterdir())):
        candidate = path.with_name(f'{path.name}-{run_suffix()}')
    return candidate


def prepare_run_destination(config, repository):
    """Select a fresh snapshot and audit destination before generating IDs.

    A different output path also changes the run signature, so an existing
    database snapshot must never be reused for a new audit export.
    """
    if config['instance_database'] in {config['abstract_database'], config['observation_database']}:
        raise ValueError('Instance database must differ from both knowledge databases.')
    result = dict(config)
    base_folder = Path(config['output_folder'])
    while True:
        folder = Path(result['output_folder'])
        occupied = folder.exists() and (not folder.is_dir() or any(folder.iterdir()))
        rows = repository.query(config['instance_database'], '''
            OPTIONAL MATCH (snapshot:OELGSnapshot {name:$snapshot})
            WITH count(snapshot) AS snapshots
            OPTIONAL MATCH (event:DisasterEvent {oelg_snapshot:$snapshot})
            RETURN snapshots, count(event) AS events
        ''', snapshot=result['snapshot'])
        if not rows or len(rows) != 1:
            raise ValueError('无法确认实例快照是否已存在，未开始写入')
        if not occupied and not rows[0]['snapshots'] and not rows[0]['events']:
            return result
        suffix = run_suffix()
        result['snapshot'] = f"{config['snapshot']}-{suffix}"
        result['output_folder'] = str(base_folder.with_name(f'{base_folder.name}-{suffix}'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance-database', help='Override the OELG instance database; {date} uses the Shanghai calendar')
    parser.add_argument('--config', default=str(Path(__file__).with_name('oelg_config.json')))
    parser.add_argument('--write-database', action='store_true', help='兼容旧命令；现在默认写入 Neo4j')
    parser.add_argument('--no-plots', action='store_true', help='只完成实例入库与审计导出')
    parser.add_argument('--output-folder', type=Path, help='实例审计输出目录；已使用时自动添加运行后缀')
    parser.add_argument('--map-output-dir', type=Path, help='地图输出目录；非空时自动添加运行后缀')
    parser.add_argument('--all-outputs', action='store_true', help='兼容旧命令；现在默认导出完整审计文件')
    map_options = parser.add_mutually_exclusive_group()
    map_options.add_argument('--risk-maps-only', action='store_true', help='从 Neo4j 重绘事件距离衰减场')
    map_options.add_argument('--factor-maps-only', action='store_true', help='从 Neo4j 重绘因子距离衰减场')
    map_options.add_argument('--maps-only', action='store_true', help='从 Neo4j 重绘事件及因子距离衰减场')
    map_options.add_argument('--rasters-only', action='store_true', help='从 Neo4j 导出距离衰减 GeoTIFF，不生成 PNG')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.instance_database:
        config['instance_database'] = resolve_database_name(args.instance_database)
    if args.output_folder:
        config['output_folder'] = str(args.output_folder.resolve())

    def plot_from_graph():
        from plot_oelg_distance_fields import run
        map_config = Path(args.config).resolve().parent / config.get('distance_map_config', 'oelg_distance_config.json')
        kind = 'event' if args.risk_maps_only else 'factor' if args.factor_maps_only else None
        requested = (args.map_output_dir.resolve() if args.map_output_dir else
                     (map_config.parent / read_json(map_config)['output_dir']).resolve())
        output = available_output_folder(requested)
        print(f'Map output directory: {output}', flush=True)
        return run(map_config, output, args.rasters_only or args.no_plots,
                   oelg_override=config, only_kind=kind)

    if args.risk_maps_only or args.factor_maps_only or args.maps_only or args.rasters_only:
        plot_from_graph()
        return
    started = time.perf_counter()
    repository = OELGKnowledgeRepository(config['abstract_database'])
    try:
        if config['instance_database'].casefold() in {config['abstract_database'].casefold(), config['observation_database'].casefold()}:
            raise ValueError('Instance database must differ from both knowledge databases.')
        ensure_database(config['instance_database'], driver=repository.driver)
        config = prepare_run_destination(config, repository)
        folder = Path(config['output_folder'])
        print(f'Instance snapshot: {config["snapshot"]}\nAudit output directory: {folder}', flush=True)
        # OELG owns geographic matching; the observation database supplies only
        # factors and candidate sensors, never geographic types or fallbacks.
        knowledge = load_instantiation_knowledge(repository, config)
        snapshot = {'knowledge': knowledge,
                    'observations': repository.read_observation_knowledge(config['observation_database'])}
        builder = OELGInstantiationBuilder(config, knowledge, snapshot['observations'])
        builder.instantiate()
        from oelg_graph_store import FACTOR_RULE, SCHEMA_VERSION
        signature_data = {'config': config, 'knowledge': snapshot, 'graph_schema': SCHEMA_VERSION,
                          'factor_rule': FACTOR_RULE, 'input_hashes': builder.features.input_hashes,
                          'rainfall_hash': sha256(config['rainfall']), 'region_hash': sha256(config['region'])}
        builder.run_signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True,
                                              ensure_ascii=False).encode('utf-8')).hexdigest()
        counts = save_instance_database(repository, builder, config['instance_database'])
        print(f'Neo4j committed: {config["instance_database"]} / {config["snapshot"]}: {counts}', flush=True)
        summary = builder.export_results(folder)
        save_json(folder / 'knowledge_snapshot.json', snapshot)
        save_json(folder / 'run_manifest.json', dict(summary, configuration=config,
            inputs=signature_data['input_hashes'], run_signature=builder.run_signature,
            database_written=True, database=config['instance_database'], snapshot=config['snapshot'],
            factor_rule=FACTOR_RULE, graph_counts=counts, elapsed_seconds=time.perf_counter()-started))
    finally:
        repository.driver.close()
    if not args.no_plots:
        plot_from_graph()


if __name__ == '__main__':
    main()
