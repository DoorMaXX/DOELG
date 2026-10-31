"""Atomic OELG snapshots and mappings for existing causal-matrix events.

python oelg_graph_store.py --dry-run
python oelg_graph_store.py --database oelg-1
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from standard_observation_factors import canonical_factors

SCHEMA_VERSION = 2
FACTOR_RULE = 'Impact.impact_type <- ARGUMENT - GlobalDisasterEvent - TRIGGER -> GeneralizedObservationEvent - OBSERVED -> StandardFactor; union over all matching global events'
DEFAULT_PHENOMENON_GEO_MAPPING = Path(__file__).parent / 'data' / 'dictionary' / 'phenomenon_geo_mapping.json'



def identifier(*parts):
    return uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(parts, ensure_ascii=False)).hex


def prepare_rows(builder):
    """Persist each selected geometry once; retain every instance-feature edge."""
    from shapely.ops import transform

    snapshot = builder.config['snapshot']
    events, tasks, links, features = [], [], [], {}
    for index, node in enumerate(builder.nodes):
        event_id = node['id']
        events.append(dict(id=event_id, description=node['event_description'],
                           phenomenon=node['phenomenon'], depth=node['current_depth'],
                           priority=node['priority'], is_happen=node['is_happen'],
                           parent=node['parent_id'], path=node['path'],
                           relation_type=node['relation_type'], event_num=index,
                           geo_id=identifier(event_id, 'space'),
                           geometry_id=identifier(event_id, 'geometry'),
                           geometry=transform(builder.features.to_wgs84, node['affect_extent']).wkt))
        for index, task in enumerate(node['observation_tasks']):
            tasks.append(dict(id=identifier(event_id, 'obs', index), event_id=event_id,
                              description=task.get('description'), source_id=task.get('node_id'),
                              factors=canonical_factors(task['factors']),
                              source_event_ids=task.get('source_event_ids', []),
                              source_geo_types=task.get('source_geo_types', []),
                              impact_type=task.get('impact_type', node['phenomenon']),
                              geo_types=task.get('geo_types', []),
                              candidate_sensors=task.get('candidate_sensors', [])))
        for feature in node['affect_features']:
            osm_id = feature['osm_id']
            feature_id = identifier(snapshot, 'feature', osm_id)
            if osm_id not in features:
                features[osm_id] = dict(id=feature_id, osm_id=osm_id, geo_type=feature['geo_type'],
                    geometry_id=identifier(snapshot, 'feature-geometry', osm_id),
                    geometry=transform(builder.features.to_wgs84, feature['geometry']).wkt)
            links.append(dict(event_id=event_id, geo_id=identifier(event_id, feature['geo_type']),
                              geo_type=feature['geo_type'], feature_id=feature_id))
    return events, list(features.values()), links, tasks


def write_builder(driver, builder, database):
    config = builder.config
    if database in {config['abstract_database'], config['observation_database']}:
        raise ValueError('Instance database must differ from both knowledge databases.')
    events, features, links, tasks = prepare_rows(builder)
    counts = dict(event_count=len(events), feature_count=len(features),
                  link_count=len(links), task_count=len(tasks))
    params = dict(snapshot=config['snapshot'], signature=builder.run_signature,
                  schema=SCHEMA_VERSION, time=config['snapshot_time'],
                  observation_database=config['observation_database'], factor_rule=FACTOR_RULE,
                  config_json=json.dumps(config, ensure_ascii=False), **counts)
    with driver.session(database=database) as session:
        # Constraints apply only to this writer's namespaced records.
        session.run('CREATE CONSTRAINT oelg_record_id IF NOT EXISTS FOR (n:OELGRecord) REQUIRE n.node_id IS UNIQUE').consume()
        session.run('CREATE CONSTRAINT oelg_snapshot_name IF NOT EXISTS FOR (n:OELGSnapshot) REQUIRE n.name IS UNIQUE').consume()

        def write(tx):
            previous = tx.run('MATCH (s:OELGSnapshot {name:$snapshot}) RETURN properties(s) AS s', **params).single()
            if previous:
                meta = previous['s']
                if meta.get('signature') != builder.run_signature or meta.get('schema_version') != SCHEMA_VERSION:
                    raise ValueError('Snapshot already exists with different inputs/schema; choose a new snapshot name.')
                if meta.get('status') == 'complete':
                    return counts
            legacy = tx.run('MATCH (e:DisasterEvent {oelg_snapshot:$snapshot}) RETURN count(e) AS n', **params).single()['n']
            if legacy:
                raise ValueError('Unversioned or incomplete snapshot exists; choose a new snapshot name.')
            tx.run('''MERGE (s:OELGSnapshot {name:$snapshot})
                SET s.status='writing', s.signature=$signature, s.schema_version=$schema,
                    s.snapshot_time=$time, s.observation_database=$observation_database,
                    s.factor_rule=$factor_rule, s.config_json=$config_json''', **params).consume()
            tx.run('''UNWIND $rows AS row
                MERGE (e:OELGRecord {node_id:row.id}) SET e:DisasterEvent,
                    e.event_description=row.description, e.depth=row.depth, e.event_num=row.event_num,
                    e.is_happen=row.is_happen, e.oelg_priority=row.priority,
                    e.oelg_phenomenon=row.phenomenon, e.oelg_path=row.path,
                    e.oelg_snapshot=$snapshot, e.oelg_run_signature=$signature,
                    e.oelg_snapshot_time=$time
                MERGE (g:OELGRecord {node_id:row.geo_id}) SET g:GeoObject,
                    g.geo_type='OELG composite disaster space', g.event_id=row.id
                MERGE (m:OELGRecord {node_id:row.geometry_id}) SET m:Geometry,
                    m.geometry=row.geometry, m.crs='EPSG:4326'
                MERGE (e)-[:ARGUMENT]->(g) MERGE (g)-[:HASGEOMETRY]->(m)
                WITH e MATCH (s:OELGSnapshot {name:$snapshot}) MERGE (s)-[:CONTAINS]->(e)
                ''', rows=events, **params).consume()
            for offset in range(0, len(features), 500):
                tx.run('''UNWIND $rows AS row
                    MERGE (f:OELGRecord {node_id:row.id}) SET f:OSMFeature,
                        f.osm_id=row.osm_id, f.geo_type=row.geo_type, f.oelg_snapshot=$snapshot
                    MERGE (m:OELGRecord {node_id:row.geometry_id}) SET m:Geometry,
                        m.geometry=row.geometry, m.crs='EPSG:4326'
                    MERGE (f)-[:HASGEOMETRY]->(m)''', rows=features[offset:offset+500], **params).consume()
            for offset in range(0, len(links), 2000):
                tx.run('''UNWIND $rows AS row
                    MATCH (e:OELGRecord {node_id:row.event_id}), (f:OELGRecord {node_id:row.feature_id})
                    MERGE (g:OELGRecord {node_id:row.geo_id}) SET g:GeoObject,
                        g.geo_type=row.geo_type, g.geo_object_name=row.geo_type, g.event_id=row.event_id
                    MERGE (e)-[:ARGUMENT]->(g) MERGE (g)-[r:HAS_FEATURE]->(f) SET r.eligible=1
                    ''', rows=links[offset:offset+2000]).consume()
            tx.run('''UNWIND $rows AS row
                MATCH (p:OELGRecord {node_id:row.parent}), (e:OELGRecord {node_id:row.id})
                MERGE (p)-[r:CAUSE]->(e) SET r.relation_type=row.relation_type
                ''', rows=[e for e in events if e['parent']]).consume()
            for offset in range(0, len(tasks), 500):
                tx.run('''UNWIND $rows AS row
                    MATCH (e:OELGRecord {node_id:row.event_id})
                    MERGE (o:OELGRecord {node_id:row.id}) SET o:ObservationEvent,
                        o.event_description=row.description, o.disaster_id=row.event_id,
                        o.oelg_snapshot=$snapshot, o.source_observation_id=row.source_id,
                        o.source_global_event_ids=row.source_event_ids, o.impact_type=row.impact_type,
                        o.source_geo_types=row.source_geo_types, o.geo_types=row.geo_types,
                        o.oelg_candidate_sensors=row.candidate_sensors
                    MERGE (e)-[:TRIGGER]->(o)
                    WITH e,o,row MATCH (e)-[:ARGUMENT]->(g:GeoObject {geo_type:'OELG composite disaster space'})
                    MERGE (o)-[:ARGUMENT]->(g)
                    WITH e,o,row UNWIND row.factors AS name
                    MERGE (f:StandardFactor {factor_name:name}) ON CREATE SET f.node_id=randomUUID()
                    MERGE (o)-[:OBSERVED]->(f) MERGE (e)-[:OBSERVED]->(f)
                    ''', rows=tasks[offset:offset+500], **params).consume()
            tx.run('''MATCH (s:OELGSnapshot {name:$snapshot}) SET s.status='complete',
                s.event_count=$event_count, s.feature_count=$feature_count,
                s.link_count=$link_count, s.task_count=$task_count, s.completed_at=datetime()
                ''', **params).consume()
            return counts
        return session.execute_write(write)


def read_graph(driver, database, snapshot):
    """No export-file or original-layer fallback. Read one complete graph snapshot."""
    from shapely import wkt

    with driver.session(database=database, default_access_mode='READ') as session:
        def read(tx):
            result = tx.run('MATCH (s:OELGSnapshot {name:$snapshot}) RETURN properties(s) AS s', snapshot=snapshot).single()
            if not result or result['s'].get('status') != 'complete' or result['s'].get('schema_version') != SCHEMA_VERSION:
                raise ValueError('No complete OELG graph snapshot. Run instantiation_oelg.py first.')
            meta = result['s']
            rows = tx.run('''MATCH (e:DisasterEvent {oelg_snapshot:$snapshot})
                OPTIONAL MATCH (p:DisasterEvent)-[r:CAUSE]->(e)
                RETURN e.node_id AS id,e.oelg_phenomenon AS phenomenon,e.depth AS current_depth,
                    e.oelg_path AS path,p.node_id AS parent_id,r.relation_type AS relation_type
                ''', snapshot=snapshot).data()
            events = dict(type='FeatureCollection', features=[dict(type='Feature', geometry=None, properties=r) for r in rows])
            links = tx.run('''MATCH (e:DisasterEvent {oelg_snapshot:$snapshot})-[:ARGUMENT]->(g:GeoObject)-[r:HAS_FEATURE]->(f:OSMFeature {oelg_snapshot:$snapshot})
                RETURN e.node_id AS event_id,e.oelg_phenomenon AS phenomenon,e.depth AS depth,
                    g.geo_type AS geo_type,f.osm_id AS osm_id,r.eligible AS eligible
                ''', snapshot=snapshot).data()
            tasks = tx.run('''MATCH (e:DisasterEvent {oelg_snapshot:$snapshot})-[:TRIGGER]->(o:ObservationEvent)
                OPTIONAL MATCH (o)-[:OBSERVED]->(f:StandardFactor)
                RETURN o.node_id AS id,e.node_id AS disaster_id,o.geo_types AS geo_types,
                    o.source_observation_id AS source_observation_id,
                    o.source_global_event_ids AS source_event_ids, o.source_geo_types AS source_geo_types,
                    collect(DISTINCT f.factor_name) AS factors
                ''', snapshot=snapshot).data()
            geometries = {}
            for r in tx.run('''MATCH (f:OSMFeature {oelg_snapshot:$snapshot})
                OPTIONAL MATCH (f)-[:HASGEOMETRY]->(g:Geometry)
                RETURN f.osm_id AS id,g.geometry AS geometry,g.crs AS crs''', snapshot=snapshot):
                if r['id'] in geometries or r['crs'] != 'EPSG:4326' or not r['geometry']:
                    raise ValueError(f'Missing, duplicate or non-WGS84 graph geometry: {r["id"]}')
                geometry = wkt.loads(r['geometry'])
                if geometry.is_empty or not geometry.is_valid:
                    raise ValueError(f'Invalid graph geometry: {r["id"]}')
                geometries[r['id']] = geometry
            counts = dict(event_count=len(rows), link_count=len(links), task_count=len(tasks), feature_count=len(geometries))
            if any(meta.get(k) != v for k, v in counts.items()):
                raise ValueError(f'Graph snapshot counts disagree with committed manifest: {counts}')
            if any(r['osm_id'] not in geometries for r in links):
                raise ValueError('Graph feature link has no stored geometry.')
            for task in tasks:
                task['factors'] = canonical_factors(task['factors'])
                if not task['factors']:
                    raise ValueError('Graph observation has no direct StandardFactor.')
            provenance = dict(source='neo4j', database=database, snapshot=snapshot,
                run_signature=meta['signature'], schema_version=SCHEMA_VERSION,
                factor_rule=meta['factor_rule'], observation_database=meta['observation_database'],
                read_at=datetime.now(timezone.utc).isoformat(), **counts)
            return events, links, tasks, geometries, provenance
        return session.execute_read(read)


def load_graph(oelg, connection_config):
    from distance_fields import open_driver
    connection_config = dict(connection_config)
    connection_config.setdefault('connection_module', str(Path(__file__).resolve().parents[1] / 'ontology' / 'neo4jmodel.py'))
    with open_driver(connection_config) as driver:
        return read_graph(driver, oelg['instance_database'], oelg['snapshot'])



def load_phenomenon_geo_mapping(path=DEFAULT_PHENOMENON_GEO_MAPPING):
    """Read the portable copy of the Appendix C/A mapping, without database access."""
    document = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(document, dict) or document.get('schema_version') != 1:
        raise ValueError('Mapping JSON must have schema_version=1.')
    records = document.get('mappings')
    if not isinstance(records, list) or not records:
        raise ValueError('Mapping JSON must contain a non-empty mappings list.')
    return records


def _mapping_rows(records):
    """Validate and deduplicate pairs before opening a database session."""
    pairs = {}
    mapping_ids = {}
    for number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f'Mapping row {number} must be an object.')
        row = {}
        for key in ('phenomenon', 'geo_type', 'mapping_id', 'relation', 'condition'):
            value = record.get(key, '')
            if not isinstance(value, str):
                raise ValueError(f'Mapping row {number}: {key} must be text.')
            row[key] = value.strip()
        if not row['phenomenon'] or not row['geo_type']:
            raise ValueError(f'Mapping row {number}: phenomenon and geo_type are required.')
        if row['relation'] not in ('可能发生于', '可能作用于'):
            raise ValueError(f'Mapping row {number}: unsupported possibility relation.')
        pair = (_mapping_name(row['phenomenon']), _mapping_name(row['geo_type']))
        if pair in pairs and pairs[pair] != row:
            raise ValueError(f'Conflicting duplicate mapping: {pair!r}')
        if row['mapping_id']:
            prior = mapping_ids.setdefault(row['mapping_id'], pair)
            if prior != pair:
                raise ValueError(f'Mapping ID reused: {row["mapping_id"]}')
        pairs[pair] = row
    if not pairs:
        raise ValueError('At least one phenomenon-geographic mapping is required.')
    return list(pairs.values())


def _mapping_name(value):
    # Existing oelg-1 uses Earth Bank / Telecom Line; Appendix A uses lower case.
    return ' '.join(str(value or '').split()).casefold()


def write_phenomenon_geo_mapping(driver, records, database='oelg-1',
                                dataset='predefined_causal_temporal_v1',
                                source_file='', dry_run=False):
    """Add Appendix mappings to existing DisasterEvent -ARGUMENT-> GeoObject.

    Reuse the causal-matrix events and type nodes. Never create DisasterEvent or
    Impact nodes, modify CAUSE edges, or remove existing ARGUMENT relationships.
    A dry run reads the target graph and reports exactly what would be added.
    All data writes and the final pair verification share one transaction.
    """
    rows = _mapping_rows(records)
    if not isinstance(database, str) or not database.strip() or database.strip().lower() == 'system':
        raise ValueError('A non-system target database is required.')
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError('The existing causal-matrix dataset is required.')
    database, dataset = database.strip(), dataset.strip()
    with driver.session(database=database, default_access_mode='READ' if dry_run else 'WRITE') as session:
        def apply(tx):
            event_names = {_mapping_name(row['phenomenon']) for row in rows}
            geo_names = {_mapping_name(row['geo_type']) for row in rows}
            events, geos = {}, {}
            for event in tx.run('''MATCH (e:DisasterEvent)
                    RETURN e.node_id AS id, e.event_description AS name''').data():
                name = _mapping_name(event['name'])
                if name not in event_names:
                    continue
                expected = identifier('predefined-causal-matrix', dataset, event['name'])
                if event['id'] != expected:
                    continue
                if name in events:
                    raise ValueError(f'Duplicate matrix event: {event["name"]}')
                events[name] = event['id']
            missing = sorted(event_names - events.keys())
            if missing:
                raise ValueError(f'Existing causal-matrix events are missing: {missing}')
            for geo in tx.run('''MATCH (g:GeoObject)
                    WHERE NOT 'event_id' IN keys(g) AND NOT 'geometry' IN keys(g)
                    RETURN g.node_id AS id, g.geo_type AS type, g.geo_object_name AS name''').data():
                name = _mapping_name(geo['type'] or geo['name'])
                if name not in geo_names or _mapping_name(geo['name']) != name:
                    continue
                if not geo['id'] or name in geos:
                    raise ValueError(f'Missing ID or duplicate geographic type: {geo["name"]}')
                geos[name] = geo['id']
            new_geos = {}
            links = []
            for row in rows:
                geo_name = _mapping_name(row['geo_type'])
                if geo_name not in geos:
                    # Same identity convention as import_causal_matrix.py.
                    geos[geo_name] = identifier('causal-matrix-geographic-type', row['geo_type'])
                    new_geos[geo_name] = dict(id=geos[geo_name], name=row['geo_type'])
                event_id, geo_id = events[_mapping_name(row['phenomenon'])], geos[geo_name]
                links.append(dict(row, event_id=event_id, geo_node_id=geo_id,
                                  rel_id=identifier('appendix-ca-argument', event_id, geo_id)))
            existing = {}
            for edge in tx.run('''MATCH (e:DisasterEvent)-[r:ARGUMENT]->(g:GeoObject)
                    WHERE e.node_id IN $event_ids
                    RETURN e.node_id AS event_id, g.node_id AS geo_id, count(r) AS count
                    ''', event_ids=list(events.values())).data():
                existing[(edge['event_id'], edge['geo_id'])] = edge['count']
            pairs = {(row['event_id'], row['geo_node_id']) for row in links}
            if any(existing.get(pair, 0) > 1 for pair in pairs):
                raise ValueError('Target graph has duplicate ARGUMENT relationships for a mapping pair.')
            report = dict(database=database, dataset=dataset, dry_run=dry_run,
                          phenomenon_count=len(events), geographic_object_count=len(geos),
                          mapping_count=len(links), new_geographic_objects=len(new_geos),
                          new_argument_relationships=sum(pair not in existing for pair in pairs),
                          existing_mapping_relationships=sum(pair in existing for pair in pairs))
            if dry_run:
                return report
            tx.run('''UNWIND $rows AS row
                MERGE (g:GeoObject {node_id:row.id})
                ON CREATE SET g.geo_object_name=row.name, g.geo_type=row.name
                ''', rows=list(new_geos.values())).consume()
            for offset in range(0, len(links), 500):
                tx.run('''UNWIND $rows AS row
                    MATCH (e:DisasterEvent {node_id:row.event_id})
                    MATCH (g:GeoObject {node_id:row.geo_node_id})
                    MERGE (e)-[r:ARGUMENT]->(g)
                    ON CREATE SET r.rel_id=row.rel_id,
                        r.relation_type='Argument', r.relation_name='论元'
                    SET r.appendix_mapping_id=row.mapping_id, r.appendix_relation=row.relation,
                        r.appendix_condition=row.condition, r.appendix_source=$source_file
                    ''', rows=links[offset:offset + 500], source_file=str(source_file)).consume()
            result = tx.run('''UNWIND $rows AS row
                MATCH (e:DisasterEvent {node_id:row.event_id})-[r:ARGUMENT]->
                    (g:GeoObject {node_id:row.geo_node_id})
                WHERE r.appendix_mapping_id=row.mapping_id AND r.appendix_relation=row.relation
                    AND r.appendix_condition=row.condition AND r.appendix_source=$source_file
                RETURN count(r) AS count
                ''', rows=links, source_file=str(source_file)).single()
            if not result or result['count'] != len(links):
                raise RuntimeError('Mapping verification failed; transaction rolled back.')
            return dict(report, verified_mapping_relationships=result['count'])
        return session.execute_read(apply) if dry_run else session.execute_write(apply)


def apply_phenomenon_geo_review(driver, review, database='oelg-1', dry_run=False):
    """Annotate existing ARGUMENTs with DBbody eligibility; never delete edges."""
    if (not isinstance(review, dict) or review.get('schema_version') != 1
            or not isinstance(review.get('review_version'), str) or not review['review_version'].strip()):
        raise ValueError('Review must specify schema_version=1 and review_version.')
    if database != review.get('database') or database == 'system':
        raise ValueError('Review database does not match the requested target.')
    rows = review.get('records')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Review must contain records.')
    pairs = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Each review record must be an object.')
        for key in ('source_id', 'target_id', 'rel_id', 'phenomenon', 'graph_geo_type',
                    'status', 'role', 'reason', 'basis', 'reference'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'Review record lacks {key}.')
        if type(row.get('enabled')) is not bool:
            raise ValueError('Review enabled must be a boolean.')
        roles = {'保留': 'DBbody', '转为环境候选': 'DPEnvironment',
                 '仅作空间范围': 'disasterSpace', '待核查': 'Unresolved',
                 '剔除承灾体映射': 'NotDBbody'}
        if roles.get(row['status']) != row['role']:
            raise ValueError('Review status and role disagree.')
        if row['enabled'] != (row['status'] == '保留' and row['role'] == 'DBbody'):
            raise ValueError('Only retained DBbody mappings may be enabled.')
        pair = (row['source_id'], row['target_id'])
        if pair in pairs:
            raise ValueError('Duplicate review pair.')
        pairs.add(pair)
    enabled = sum(row['enabled'] for row in rows)
    with driver.session(database=database, default_access_mode='READ' if dry_run else 'WRITE') as session:
        def apply(tx):
            scope = tx.run('''MATCH (e:DisasterEvent)-[r:ARGUMENT]->(:GeoObject)
                WHERE e.node_id IN $event_ids RETURN count(r) AS count''',
                event_ids=sorted({row['source_id'] for row in rows})).single()
            matched = tx.run('''UNWIND $rows AS row
                MATCH (e:DisasterEvent {node_id:row.source_id})-[r:ARGUMENT]->
                    (g:GeoObject {node_id:row.target_id})
                WHERE r.rel_id=row.rel_id AND e.event_description=row.phenomenon
                    AND g.geo_type=row.graph_geo_type
                RETURN count(r) AS count''', rows=rows).single()
            if not scope or not matched or scope['count'] != len(rows) or matched['count'] != len(rows):
                raise ValueError('Graph mappings changed since review; export and review again before applying.')
            report = dict(database=database, review_version=review['review_version'],
                          reviewed_count=len(rows), enabled_count=enabled,
                          disabled_count=len(rows)-enabled, dry_run=dry_run)
            if dry_run:
                return report
            tx.run('''UNWIND $rows AS row
                MATCH (:DisasterEvent {node_id:row.source_id})-[r:ARGUMENT]->
                    (:GeoObject {node_id:row.target_id})
                WHERE r.rel_id=row.rel_id
                SET r.oelg_dbbody_enabled=row.enabled, r.oelg_mapping_role=row.role,
                    r.oelg_review_status=row.status, r.oelg_review_reason=row.reason,
                    r.oelg_review_basis=row.basis, r.oelg_review_reference=row.reference,
                    r.oelg_review_version=$version''', rows=rows, version=review['review_version']).consume()
            verified = tx.run('''UNWIND $rows AS row
                MATCH (:DisasterEvent {node_id:row.source_id})-[r:ARGUMENT]->
                    (:GeoObject {node_id:row.target_id})
                WHERE r.rel_id=row.rel_id AND r.oelg_dbbody_enabled=row.enabled
                    AND r.oelg_mapping_role=row.role AND r.oelg_review_status=row.status
                    AND r.oelg_review_reason=row.reason AND r.oelg_review_basis=row.basis
                    AND r.oelg_review_reference=row.reference AND r.oelg_review_version=$version
                RETURN count(r) AS count''', rows=rows, version=review['review_version']).single()
            if not verified or verified['count'] != len(rows):
                raise RuntimeError('Review annotation verification failed; transaction rolled back.')
            return report
        return session.execute_read(apply) if dry_run else session.execute_write(apply)


def write_environment_rules(driver, catalog, database='oelg-1',
                            dataset='predefined_causal_temporal_v1', dry_run=False):
    """Reuse OELG DisasterEvents and the existing Environment* / ARGUMENT schema."""
    from oelg_environment import validate_environment_rule
    if database == 'system' or catalog.get('schema_version') != 1:
        raise ValueError('Invalid environment catalog or database')
    rules = catalog.get('rules')
    if not isinstance(rules, list) or not rules:
        raise ValueError('Environment catalog must have rules')
    rows, names = [], set()
    for rule in rules:
        validate_environment_rule(rule)
        name = rule['phenomenon']
        if name in names:
            raise ValueError('Duplicate environment phenomenon')
        names.add(name)
        event_id = identifier('predefined-causal-matrix', dataset, name)
        rows.append(dict(event_id=event_id, name=name, description=rule['description'],
                         rule_id=identifier('oelg-environment-rule', dataset, name),
                         description_id=identifier('oelg-environment-description', dataset, name),
                         rules=json.dumps(rule, ensure_ascii=False, sort_keys=True), version=rule['version'],
                         rule_rel_id=identifier('oelg-environment-rule-link', dataset, name),
                         description_rel_id=identifier('oelg-environment-description-link', dataset, name)))
    required_types = sorted({t for r in rules for t in r.get('environment', {}).get('geo_types', [])})
    with driver.session(database=database, default_access_mode='READ' if dry_run else 'WRITE') as session:
        def write(tx):
            found = tx.run('''UNWIND $rows AS row
                OPTIONAL MATCH (e:DisasterEvent {node_id:row.event_id, event_description:row.name})
                RETURN row.name AS name, count(e) AS count''', rows=rows).data()
            if len(found) != len(rows) or any(row['count'] != 1 for row in found):
                raise ValueError('Environment rules must match one existing OELG event each')
            types = tx.run('MATCH (g:GeoObject) RETURN collect(DISTINCT g.geo_type) AS types').single()['types']
            if set(required_types) - set(types):
                raise ValueError(f'Environment geographic types missing in OELG: {set(required_types)-set(types)}')
            conflicts = tx.run('''UNWIND $rows AS row
                MATCH (:DisasterEvent {node_id:row.event_id})-[:ARGUMENT]->(r:EnvironmentRule)
                WHERE r.node_id <> row.rule_id OR r.node_id IS NULL
                RETURN count(r) AS count''', rows=rows).single()['count']
            if conflicts:
                raise ValueError('Unmanaged environment rules already exist; reconcile before importing')
            report = dict(database=database, event_count=len(rows), rule_count=len(rows),
                          description_count=len(rows), dry_run=dry_run)
            if dry_run:
                return report
            tx.run('''UNWIND $rows AS row
                MATCH (e:DisasterEvent {node_id:row.event_id})
                MERGE (d:EnvironmentDescription {node_id:row.description_id})
                SET d.description=row.description, d.oelg_environment_version=row.version
                MERGE (r:EnvironmentRule {node_id:row.rule_id})
                SET r.rules=row.rules, r.oelg_environment_version=row.version
                MERGE (e)-[dr:ARGUMENT]->(d)
                ON CREATE SET dr.rel_id=row.description_rel_id, dr.relation_type='Argument', dr.relation_name='论元'
                MERGE (e)-[rr:ARGUMENT]->(r)
                ON CREATE SET rr.rel_id=row.rule_rel_id, rr.relation_type='Argument', rr.relation_name='论元'
                ''', rows=rows).consume()
            verified = tx.run('''UNWIND $rows AS row
                MATCH (:DisasterEvent {node_id:row.event_id})-[:ARGUMENT]->
                      (r:EnvironmentRule {node_id:row.rule_id})
                MATCH (:DisasterEvent {node_id:row.event_id})-[:ARGUMENT]->
                      (d:EnvironmentDescription {node_id:row.description_id})
                WHERE r.rules=row.rules AND d.description=row.description
                RETURN count(*) AS count''', rows=rows).single()['count']
            if verified != len(rows):
                raise RuntimeError('Environment rule verification failed; transaction rolled back')
            return report
        return session.execute_read(write) if dry_run else session.execute_write(write)


def main(argv=None):
    parser = argparse.ArgumentParser(description='为已有 OELG 灾害现象补充地物 ARGUMENT 映射')
    parser.add_argument('--mapping', type=Path, default=DEFAULT_PHENOMENON_GEO_MAPPING)
    parser.add_argument('--database', default='oelg-1')
    parser.add_argument('--dataset', default='predefined_causal_temporal_v1')
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--review', type=Path, help='应用论文复核 JSON，仅标记现有关联的承灾体资格')
    operation.add_argument('--environment-rules', type=Path, help='保存各事件的孕灾环境描述及规则')
    parser.add_argument('--dry-run', action='store_true', help='只读核对目标图谱，统计待新增节点和关系')
    args = parser.parse_args(argv)
    if args.environment_rules:
        from oelg_environment import load_environment_catalog
        catalog = load_environment_catalog(args.environment_rules)
    elif args.review:
        review = json.loads(args.review.read_text(encoding='utf-8-sig'))
    else:
        records = load_phenomenon_geo_mapping(args.mapping)
        _mapping_rows(records)
    from distance_fields import open_driver
    with open_driver({'connection_module': str(Path(__file__).resolve().parents[1] / 'ontology' / 'neo4jmodel.py')}) as driver:
        if args.environment_rules:
            report = write_environment_rules(driver, catalog, args.database, args.dataset, args.dry_run)
        elif args.review:
            report = apply_phenomenon_geo_review(driver, review, args.database, args.dry_run)
        else:
            report = write_phenomenon_geo_mapping(
                driver, records, args.database, args.dataset, args.mapping.resolve(), args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    main()
