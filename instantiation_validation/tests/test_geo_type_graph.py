"""Validate Appendix A/B expansion, rebuild guards, and path-name compatibility."""

import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

from src.other import geo_type_graph as graph
from instantiation_validation.tests.test_graph_language import graph_namespace, load_function, vocabularies

ROOT = Path(__file__).resolve().parents[2]


def prepared_snapshot(prepared):
    """Represent a prepared graph using the same shape returned by the live Neo4j reader."""
    ids = {node['geo_type']: node['node_id'] for node in prepared['nodes']}
    return dict(nodes=[dict(id=node['node_id'], labels=['GeoObject'], properties=deepcopy(node))
                       for node in prepared['nodes']],
                relationships=[dict(source=ids[edge['source']], target=ids[edge['target']],
                                    type='SPATIAL', properties=deepcopy(edge['properties']))
                               for edge in prepared['edges']])


class GeoTypeCatalogTests(unittest.TestCase):
    """Check vocabulary alignment and faithful expansion of grouped appendix rows."""

    def setUp(self):
        """Load the actual frozen appendix catalog for each isolated check."""
        self.catalog = deepcopy(graph.load_geo_type_rules(str(ROOT / 'data/dictionary/geo_type_rules.json')))
        self.prepared = graph.prepare_geo_type_graph(self.catalog)

    def test_appendix_types_match_extraction_vocabulary(self):
        """Cover every distinct geographic category produced by current extraction."""
        self.assertEqual({node['geo_type'] for node in self.prepared['nodes']},
                         set(vocabularies()['geo_types_en']))
        self.assertEqual(len(self.prepared['nodes']), 78)
        self.assertEqual(self.prepared['summary']['appendix_b_rows'], 126)
        self.assertEqual(len(self.prepared['edges']), 401)

    def test_aliases_preserve_existing_project_capitalization(self):
        """Resolve the appendix's Road spelling and capitalization without adding new types."""
        for name, expected in [('Road', 'Highway'), ('Earth bank', 'Earth Bank'),
                               ('In-building Parking', 'In-Building Parking'), ('Telecom line', 'Telecom Line')]:
            self.assertEqual(graph.canonical_geo_type(name, self.catalog), expected)
        self.assertIsNone(graph.canonical_geo_type('Unknown category', self.catalog))

    def test_group_expansion_and_duplicate_provenance(self):
        """Expand endpoint groups and retain the source rows of collapsed duplicate rules."""
        edges = {(edge['source'], edge['target'], edge['properties']['relation_type'],
                  edge['properties']['relation_name']): edge for edge in self.prepared['edges']}
        self.assertIn(('Tunnel', 'Underground Building', 'Spatial Topology', 'Connect'), edges)
        duplicate = edges[('Administrative Region', 'Highway', 'Spatial Topology', 'Overlap/Connect')]
        self.assertEqual(duplicate['properties']['appendix_b_rows'], [2, 4])
        missing_name = edges[('Railway', 'Traffic Intersection', 'Rely on', 'Rely on')]
        self.assertEqual(missing_name['properties']['appendix_relation_name'], '-')
        self.assertEqual(self.prepared['summary']['duplicate_expanded_rules'], 12)

    def test_isolated_appendix_types_are_not_given_invented_edges(self):
        """Keep unconnected Appendix A types instead of adding unsupported rules."""
        self.assertEqual(self.prepared['summary']['isolated_types'],
                         ['Cave', 'Coastline', 'Earth Bank', 'Sewer Vent', 'Sinkhole',
                          'Storage Tank', 'Valley', 'Water Tower'])

    def test_unknown_rule_endpoints_are_rejected(self):
        """Reject a malformed catalog before any database connection or deletion."""
        self.catalog['rows'][0]['tail'] = 'Unknown category'
        with self.assertRaisesRegex(ValueError, 'Unknown endpoint'):
            graph.prepare_geo_type_graph(self.catalog)

    def test_validation_checks_complete_graph_and_provenance(self):
        """Accept the exact graph and detect missing edges, duplicates, and changed source rows."""
        snapshot = prepared_snapshot(self.prepared)
        self.assertTrue(graph.validate_geo_type_graph(snapshot, self.prepared)['valid'])
        for variant in ['missing', 'duplicate', 'provenance', 'language']:
            changed = deepcopy(snapshot)
            if variant == 'missing':
                changed['relationships'].pop()
            elif variant == 'duplicate':
                changed['relationships'].append(deepcopy(changed['relationships'][0]))
            elif variant == 'provenance':
                changed['relationships'][0]['properties']['appendix_b_rows'] = [999]
            else:
                changed['nodes'][0]['properties']['geo_type'] = chr(27827)
            with self.subTest(variant=variant):
                self.assertFalse(graph.validate_geo_type_graph(changed, self.prepared)['valid'])

    def test_path_lookup_accepts_appendix_aliases(self):
        """Make the production path lookup query the same English types stored by the builder."""
        namespace = graph_namespace()
        first = namespace['GeoObject'](geo_type='In-Building Parking').save()
        second = namespace['GeoObject'](geo_type='Highway').save()
        query = Mock(return_value=([[object()]], []))
        fake_db = Mock(cypher_query=query)
        namespace.update(canonical_geo_type=lambda value: graph.canonical_geo_type(value, self.catalog),
                         get_setting=lambda key: 6, extract_path_triples=lambda *args: [['Parking', 'Adjacent', 'Highway']])
        run = load_function('src/geo_extraction.py', 'find_geo_type_path', namespace)
        with patch('neomodel.db', fake_db):
            self.assertEqual(run('In-building Parking', 'Road'), [['Parking', 'Adjacent', 'Highway']])
        self.assertEqual(query.call_args.args[1], {'start_id': first.node_id, 'end_id': second.node_id})

    def test_equivalent_aliases_do_not_query_a_common_endpoint(self):
        """Avoid Neo4j's common-endpoint shortest-path error for equivalent type aliases."""
        namespace = graph_namespace()
        namespace['canonical_geo_type'] = lambda value: graph.canonical_geo_type(value, self.catalog)
        query = Mock()
        run = load_function('src/geo_extraction.py', 'find_geo_type_path', namespace)
        with patch('neomodel.db', Mock(cypher_query=query)):
            self.assertEqual(run('Road', 'Highway'), [])
        query.assert_not_called()


class GeoTypeRebuildTests(unittest.TestCase):
    """Protect unrelated databases and verify backup-before-replacement ordering."""

    def setUp(self):
        """Prepare mocked transactions and actual local backup/report files."""
        self.prepared = graph.prepare_geo_type_graph(
            graph.load_geo_type_rules(str(ROOT / 'data/dictionary/geo_type_rules.json')))
        self.snapshot = prepared_snapshot(self.prepared)
        self.old = dict(nodes=[dict(id='old', labels=['GeoObject'], properties={'geo_type': chr(27827)})], relationships=[])
        self.driver = MagicMock()
        self.session = self.driver.session.return_value.__enter__.return_value
        self.session.execute_read.side_effect = [self.old, self.snapshot]
        self.transaction = self.session.begin_transaction.return_value.__enter__.return_value
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        redirect = contextlib.redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def test_other_databases_cannot_be_rebuilt(self):
        """Reject a user-data database before making any connection."""
        with self.assertRaisesRegex(ValueError, 'restricted'):
            graph.rebuild_geo_type_graph(self.driver, 'test-1', self.prepared, self.folder)
        self.driver.session.assert_not_called()

    def test_unexpected_labels_prevent_replacement(self):
        """Refuse deletion when the selected auxiliary database contains other node labels."""
        self.old['nodes'][0]['labels'] = ['DisasterEvent']
        with self.assertRaisesRegex(ValueError, 'Unexpected node labels'):
            graph.rebuild_geo_type_graph(self.driver, 'geos-type-1', self.prepared, self.folder)
        self.session.begin_transaction.assert_not_called()

    def test_backup_is_verified_before_transaction(self):
        """Save the old graph before starting replacement and commit only after validation."""
        def begin():
            """Require a readable backup to exist before the first write transaction."""
            files = list(self.folder.glob('*before-rebuild*.json'))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text(encoding='utf-8'))['nodes'], self.old['nodes'])
            return transaction_context
        transaction_context = self.session.begin_transaction.return_value
        self.session.begin_transaction.side_effect = begin
        with patch.object(graph, 'read_graph', return_value=self.snapshot):
            report = graph.rebuild_geo_type_graph(self.driver, 'geos-type-1', self.prepared, self.folder)
        self.transaction.commit.assert_called_once()
        self.assertTrue(report['validation']['valid'])
        backup = Path(report['backup'])
        self.assertEqual(hashlib.sha256(backup.read_bytes()).hexdigest(), report['backup_sha256'])

    def test_failed_validation_does_not_commit(self):
        """Leave the transaction uncommitted if the rebuilt graph fails appendix validation."""
        with patch.object(graph, 'read_graph', return_value=self.old):
            with self.assertRaisesRegex(ValueError, 'validation failed'):
                graph.rebuild_geo_type_graph(self.driver, 'geos-type-1', self.prepared, self.folder)
        self.transaction.commit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
