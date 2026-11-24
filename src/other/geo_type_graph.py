"""Build and verify the geographic type graph defined by Appendices A and B."""

from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
from uuid import NAMESPACE_URL, uuid5

from src.construct_config import get_neo4j_url, get_setting, resolve_project_path


def rules_path():
    """Resolve the configured, project-local Appendix A/B catalog."""
    return resolve_project_path(get_setting('geocoding').get('type_rules', 'data/dictionary/geo_type_rules.json'))


@lru_cache(maxsize=4)
def load_geo_type_rules(path):
    """Read a frozen Appendix A/B catalog without requiring Word at runtime."""
    with Path(path).open(encoding='utf-8-sig') as stream:
        catalog = json.load(stream)
    if catalog.get('schema_version') != 1:
        raise ValueError('Unsupported geographic type catalog version.')
    return catalog


def canonical_geo_type(value, catalog=None):
    """Resolve Appendix capitalization and the Road alias to the project's English types."""
    catalog = catalog if catalog is not None else load_geo_type_rules(str(rules_path()))
    names = {node['geo_type'].casefold(): node['geo_type'] for node in catalog['nodes']}
    aliases = {key.casefold(): target for key, target in catalog.get('aliases', {}).items()}
    text = str(value or '').strip()
    return names.get(text.casefold()) or aliases.get(text.casefold())


def prepare_geo_type_graph(catalog):
    """Expand grouped endpoints, deduplicate rules, and retain original appendix row provenance."""
    nodes = []
    names = set()
    for entry in catalog['nodes']:
        name = entry['geo_type']
        if name in names or not name or re.search(r'[\u3400-\u9fff]', name):
            raise ValueError('Geographic type names must be unique, nonempty English terms.')
        names.add(name)
        nodes.append(dict(node_id=uuid5(NAMESPACE_URL, 'kgc/geo-type/' + name).hex,
                          geo_type=name, geo_object_name=name, is_location=False, geometry='',
                          category=entry['category'], appendix_name=entry['appendix_name'],
                          appendix_a_row=entry['appendix_a_row']))
    edges = {}
    filled_names = []
    for row in catalog['rows']:
        heads = [canonical_geo_type(part, catalog) for part in row['head'].split('/')]
        tails = [canonical_geo_type(part, catalog) for part in row['tail'].split('/')]
        if not all(heads) or not all(tails):
            raise ValueError(f"Unknown endpoint in Appendix B row {row['appendix_b_row']}.")
        kind = row['relation_type'].strip()
        name = row['relation_name'].strip()
        if name == '-':
            filled_names.append(row['appendix_b_row'])
            name = kind
        if not kind or not name or re.search(r'[\u3400-\u9fff]', kind + name):
            raise ValueError('Appendix relations require English types and names.')
        for head in heads:
            for tail in tails:
                key = (head, tail, kind, name)
                if key not in edges:
                    identifier = uuid5(NAMESPACE_URL, 'kgc/geo-type-edge/' + json.dumps(key)).hex
                    edges[key] = dict(source=head, target=tail, properties=dict(
                        rel_id=identifier, relation_type=kind, relation_name=name,
                        appendix_b_rows=[], appendix_relation_name=row['relation_name']))
                edges[key]['properties']['appendix_b_rows'].append(row['appendix_b_row'])
    connected = {name for edge in edges for name in edge[:2]}
    return dict(nodes=nodes, edges=list(edges.values()),
                source_document=catalog['source_document'],
                summary=dict(node_count=len(nodes), relationship_count=len(edges),
                             appendix_b_rows=len(catalog['rows']),
                             duplicate_expanded_rules=sum(len(e['properties']['appendix_b_rows']) - 1
                                                          for e in edges.values()),
                             filled_relation_name_rows=filled_names,
                             isolated_types=sorted(names - connected)))


def open_geo_type_driver():
    """Open the configured Neo4j server without embedding credentials in reports."""
    from neo4j import GraphDatabase
    parsed = urlsplit(get_neo4j_url())
    host = f'[{parsed.hostname}]' if ':' in parsed.hostname else parsed.hostname
    return GraphDatabase.driver(f'{parsed.scheme}://{host}:{parsed.port or 7687}',
                                auth=(unquote(parsed.username or ''), unquote(parsed.password or '')),
                                connection_timeout=15, max_transaction_retry_time=0)


def read_graph(runner):
    """Snapshot all nodes and directed relationships from the selected auxiliary database."""
    nodes = [row.data() for row in runner.run(
        'MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties')]
    edges = [row.data() for row in runner.run(
        'MATCH (a)-[r]->(b) RETURN elementId(a) AS source, elementId(b) AS target, '
        'type(r) AS type, properties(r) AS properties')]
    return dict(nodes=nodes, relationships=edges)


def validate_geo_type_graph(snapshot, prepared):
    """Compare every stored type and relationship against the complete expanded appendix catalog."""
    nodes = snapshot['nodes']
    names = [node['properties'].get('geo_type') for node in nodes]
    expected_names = {node['geo_type'] for node in prepared['nodes']}
    names_by_id = {node['id']: node['properties'].get('geo_type') for node in nodes}
    actual = Counter((names_by_id.get(edge['source']), names_by_id.get(edge['target']),
                      edge['properties'].get('relation_type'), edge['properties'].get('relation_name'))
                     for edge in snapshot['relationships'])
    expected = Counter((edge['source'], edge['target'], edge['properties']['relation_type'],
                        edge['properties']['relation_name']) for edge in prepared['edges'])
    errors = []
    if len(names) != len(expected_names) or set(names) != expected_names:
        errors.append('Stored geographic types differ from Appendix A.')
    if any(node['labels'] != ['GeoObject'] or node['properties'].get('geo_object_name') != node['properties'].get('geo_type')
           for node in nodes):
        errors.append('Node labels or English geographic object names are inconsistent.')
    expected_node_properties = {node['geo_type']: node for node in prepared['nodes']}
    if any(any(node['properties'].get(key) != value for key, value in
               expected_node_properties.get(node['properties'].get('geo_type'), {}).items()) for node in nodes):
        errors.append('Node categories or Appendix A provenance differ from the catalog.')
    if any(edge['type'] != 'SPATIAL' for edge in snapshot['relationships']):
        errors.append('Unexpected relationship types exist.')
    if actual != expected:
        errors.append('Stored relationships differ from Appendix B or contain duplicates.')
    expected_properties = {(edge['source'], edge['target'], edge['properties']['relation_type'],
                            edge['properties']['relation_name']): edge['properties'] for edge in prepared['edges']}
    if any(any(edge['properties'].get(key) != value for key, value in expected_properties.get(
            (names_by_id.get(edge['source']), names_by_id.get(edge['target']),
             edge['properties'].get('relation_type'), edge['properties'].get('relation_name')), {}).items())
           for edge in snapshot['relationships']):
        errors.append('Relationship properties or Appendix B row provenance differ from the catalog.')
    if re.search(r'[\u3400-\u9fff]', json.dumps(snapshot, ensure_ascii=False)):
        errors.append('Chinese property values remain in the auxiliary graph.')
    return dict(valid=not errors, errors=errors, node_count=len(nodes),
                relationship_count=len(snapshot['relationships']),
                missing_types=sorted(expected_names - set(names)),
                unexpected_types=sorted(set(names) - expected_names, key=str),
                missing_relationship_count=sum((expected - actual).values()),
                unexpected_relationship_count=sum((actual - expected).values()))


def rebuild_geo_type_graph(driver, database, prepared, output_dir):
    """Back up the named auxiliary graph, then replace and validate it in one transaction."""
    if database != 'geos-type-1':
        raise ValueError('Rebuilding is restricted to the explicitly designated geos-type-1 database.')
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S-%f')
    with driver.session(database=database) as session:
        previous = session.execute_read(read_graph)
        if any(node['labels'] != ['GeoObject'] for node in previous['nodes']):
            raise ValueError('Unexpected node labels found; refusing to replace a non-type graph.')
        if any(edge['type'] != 'SPATIAL' for edge in previous['relationships']):
            raise ValueError('Unexpected relationship types found; refusing to replace this database.')
        backup = folder / f'{database}-before-rebuild-{stamp}.json'
        backup_payload = dict(schema_version=1, database=database,
                              created_at=datetime.now(timezone.utc).isoformat(), **previous)
        backup_text = json.dumps(backup_payload, ensure_ascii=False, indent=2)
        with backup.open('x', encoding='utf-8') as stream:
            stream.write(backup_text)
        if json.loads(backup.read_text(encoding='utf-8')) != backup_payload:
            raise ValueError('The pre-rebuild backup could not be verified.')
        print(f'Backup saved: {backup}', flush=True)
        with session.begin_transaction() as transaction:
            transaction.run('MATCH (n) DETACH DELETE n').consume()
            transaction.run('UNWIND $rows AS row CREATE (n:GeoObject) SET n = row',
                            rows=prepared['nodes']).consume()
            transaction.run(
                'UNWIND $rows AS row MATCH (a:GeoObject {geo_type: row.source}) '
                'MATCH (b:GeoObject {geo_type: row.target}) '
                'CREATE (a)-[r:SPATIAL]->(b) SET r = row.properties', rows=prepared['edges']).consume()
            validation = validate_geo_type_graph(read_graph(transaction), prepared)
            if not validation['valid']:
                raise ValueError(f'Geographic type graph validation failed: {validation["errors"]}')
            transaction.commit()
        validation = validate_geo_type_graph(session.execute_read(read_graph), prepared)
    report = dict(database=database, source_document=prepared['source_document'],
                  backup=str(backup.resolve()), backup_sha256=hashlib.sha256(backup.read_bytes()).hexdigest(),
                  previous_node_count=len(previous['nodes']),
                  previous_relationship_count=len(previous['relationships']),
                  catalog_summary=prepared['summary'], validation=validation)
    report_path = folder / f'{database}-validation-{stamp}.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Validation report saved: {report_path}', flush=True)
    if not validation['valid']:
        raise ValueError('Post-commit validation failed; inspect the validation report.')
    return report
