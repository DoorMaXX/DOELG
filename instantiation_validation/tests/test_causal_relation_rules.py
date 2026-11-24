"""Validate Appendix E parity and causal filtering without writing to Neo4j."""

import concurrent.futures
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from src.causal_relation_matrix import (
    active_relation_counts, calculate_relation_confidence, get_relation_code,
    get_relation_metadata, canonical_impact_type, RULES_PATH,
)
from src.other.causal_relation_rules import classify_event_cause, enforce_existing_causal_rules
from test_graph_language import graph_namespace, load_function

ROOT = Path(__file__).resolve().parents[2]


def event(namespace, impact, name, geo=None):
    """Create a typed in-memory disaster with one geographic argument."""
    node = namespace['DisasterEvent'](event_description=f'{name},{impact}').save()
    node.impact_argument.connect(namespace['Impact'](impact_type=impact))
    node.geo_argument.connect(geo or namespace['GeoObject'](geo_object_name=name, geo_type='行政区'))
    return node


class CausalRuleTests(unittest.TestCase):
    """Check class-level rule coverage and event-level graph behavior."""

    def test_all_workbook_records_match_runtime_rules(self):
        """Verify the source workbook, all active pairs, and all excluded Remove records."""
        source = json.loads(RULES_PATH.read_text(encoding='utf-8'))
        workbook = RULES_PATH.with_name('Appendix_E.xlsx')
        self.assertEqual(hashlib.sha256(workbook.read_bytes()).hexdigest(), source['source_sha256'])
        ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        with ZipFile(workbook) as archive:
            strings = [''.join(node.itertext()) for node in ET.fromstring(
                archive.read('xl/sharedStrings.xml')).findall('m:si', ns)]
            records = []
            for row in ET.fromstring(archive.read('xl/worksheets/sheet2.xml')).findall('m:sheetData/m:row', ns):
                values = {}
                for cell in row.findall('m:c', ns):
                    node = cell.find('m:v', ns)
                    value = node.text if node is not None else ''
                    if cell.attrib.get('t') == 's':
                        value = strings[int(value)]
                    values[''.join(c for c in cell.attrib['r'] if c.isalpha())] = value
                if values.get('C') in ('I', 'A'):
                    records.append(values)
        self.assertEqual(len(records), 197)
        for row in records:
            metadata = get_relation_metadata(row['A'], row['B'])
            if row['G'] == 'Remove':
                self.assertIsNone(metadata)
            else:
                self.assertEqual(metadata['matrix_relation_code'], row['C'])
                self.assertEqual(metadata['relation_type'], row['D'])
                self.assertEqual(metadata['affectType'], row['D'])
                self.assertEqual(metadata['matrix_review_status'], row['G'])
        self.assertEqual(active_relation_counts(), (87, 107, 194))
        self.assertEqual(len(source['instance_guards']), 34)

    def test_only_case_and_whitespace_are_normalized(self):
        """Recognize dictionary capitalization without accepting unrelated surface names."""
        self.assertEqual(canonical_impact_type('Plant Pests And Diseases'), 'Plant Pests and Diseases')
        self.assertEqual(get_relation_code(' Extreme   Precipitation ', 'Waterlogging'), 'I')
        self.assertIsNone(get_relation_code('Flood', 'Waterlogging'))
        self.assertIsNone(get_relation_code('Casualties', 'Extreme Precipitation'))

    def test_confidence_coefficients_are_unchanged(self):
        """Apply the same numeric coefficient after renaming T/F to I/A."""
        self.assertEqual(calculate_relation_confidence(.8, 'I'), .8)
        self.assertAlmostEqual(calculate_relation_confidence(.8, 'A'), .52)
        self.assertEqual(calculate_relation_confidence(.8, 'T'), .8)
        self.assertAlmostEqual(calculate_relation_confidence(.8, 'F'), .52)

    def test_same_type_requires_distinct_events_and_objects(self):
        """Retain preliminary status while rejecting reflexive and same-object edges."""
        namespace = graph_namespace()
        first = event(namespace, 'Waterlogging', 'first district')
        second = event(namespace, 'Waterlogging', 'second district')
        self.assertIsNone(classify_event_cause(first, first))
        same_object = event(namespace, 'Waterlogging', 'duplicate event', first.geo_argument.all()[0])
        self.assertIsNone(classify_event_cause(first, same_object))
        duplicate_object = event(namespace, 'Waterlogging', 'first district')
        self.assertIsNone(classify_event_cause(first, duplicate_object))
        self.assertEqual(classify_event_cause(first, second)['matrix_review_status'], 'Preliminary')

    def test_existing_filter_preserves_evidence_confidence_and_trigger(self):
        """Remove unsupported CAUSE edges without deleting observations or valid evidence."""
        namespace = graph_namespace()
        rain = event(namespace, 'Extreme Precipitation', 'city')
        flood = event(namespace, 'Waterlogging', 'city')
        casualties = event(namespace, 'Casualties', 'city')
        retained = rain.cause.connect(flood, dict(evidence='text evidence', relation_count=7, confidence=.8))
        rain.cause.connect(casualties)
        observation = namespace['ObservationEvent'](event_description='observation')
        rain.trigger.connect(observation)
        summary = enforce_existing_causal_rules([rain, flood, casualties])
        self.assertEqual(summary, dict(removed=1, updated=1))
        self.assertFalse(rain.cause.is_connected(casualties))
        self.assertEqual((retained.evidence, retained.relation_count, retained.confidence), ('text evidence', 7, .8))
        self.assertEqual(retained.relation_type, 'Immediate')
        self.assertTrue(rain.trigger.is_connected(observation))
        self.assertEqual(enforce_existing_causal_rules([rain]), dict(removed=0, updated=0))

    def test_relation_recognition_filters_before_connecting(self):
        """Run the real extraction stage with mocked model outputs and in-memory graph writes."""
        namespace = graph_namespace()
        rain = event(namespace, 'Extreme Precipitation', 'city')
        flood = event(namespace, 'Waterlogging', 'city')
        landslide = event(namespace, 'Landslide', 'slope')
        casualties = event(namespace, 'Casualties', 'city')
        origin = namespace['OriginText'](text_content='Source evidence').save()
        for node in (rain, flood, landslide, casualties):
            origin.exist.connect(node)
        observation = namespace['ObservationEvent'](event_description='observation').save()
        observation.sensor_argument.connect(namespace['Sensor'](sensor_name='radar'))
        origin.exist_obs.connect(observation)
        jobs = [dict(head_id=rain.node_id, tail_id=target.node_id, evidence='LLM evidence')
                for target in (flood, landslide, casualties, rain)]
        namespace.update(concurrent=concurrent, print=Mock(),
                         enforce_existing_causal_rules=enforce_existing_causal_rules,
                         classify_event_cause=classify_event_cause,
                         _worker_cause_relation=lambda *args: jobs,
                         _worker_trigger_relation=lambda *args: [dict(
                             head_id=rain.node_id, tail_id=observation.node_id, evidence='observation evidence')])
        run = load_function('main/main_extraction.py', 'run_relation_recognition', namespace)
        run(1)
        self.assertEqual(rain.cause.relationship(flood).relation_type, 'Immediate')
        self.assertEqual(rain.cause.relationship(flood).matrix_relation_code, 'I')
        self.assertEqual(rain.cause.relationship(landslide).affectType, 'Altering')
        self.assertEqual(rain.cause.relationship(landslide).matrix_relation_code, 'A')
        self.assertEqual(rain.cause.relationship(flood).evidence, 'LLM evidence')
        self.assertFalse(rain.cause.is_connected(casualties))
        self.assertFalse(rain.cause.is_connected(rain))
        self.assertTrue(rain.trigger.is_connected(observation))
        self.assertFalse(hasattr(rain.trigger.relationship(observation), 'affectType'))

    def test_generalization_uses_the_same_rule_and_labels(self):
        """Keep relation categories on merge edges and on the Cypher-written global edges."""
        namespace = graph_namespace()
        db = Mock()
        namespace['db'] = db
        connect = load_function('src/generalization.py', '_connect_global_cause_with_pattern', namespace)
        first = SimpleNamespace(event_description='Administrative Region,Extreme Precipitation', element_id='first')
        second = SimpleNamespace(event_description='Mountain,Landslide', element_id='second')
        connect(first, second, 'pattern')
        params = db.cypher_query.call_args.args[1]
        self.assertEqual(params['metadata']['relation_type'], 'Altering')
        self.assertEqual(params['metadata']['matrix_relation_code'], 'A')
        self.assertIn('SET r += $metadata', db.cypher_query.call_args.args[0])

    def test_oelg_accepts_new_and_legacy_names_with_identical_scores(self):
        """Allow old lowercase archives while exporting canonical Appendix E names."""
        from instantiation_oelg import screen_causal_graph
        from plot_oelg_distance_fields import score_events
        from test_plot_oelg_distance_fields import OELGDistanceTests
        fixture = OELGDistanceTests()
        fixture.setUp()
        old = score_events(fixture.events)
        updated = deepcopy(fixture.events)
        for feature in updated['features']:
            if feature['properties']['relation_type'] == 'immediate':
                feature['properties']['relation_type'] = 'Immediate'
        self.assertEqual(score_events(updated), old)
        graph = dict(nodes={'Rain': {}, 'Flood': {}}, edges=[dict(
            source='Rain', target='Flood', relation_type='Immediate')])
        edges, _ = screen_causal_graph(graph, 'Rain', 2)
        self.assertEqual(edges[0]['relation_type'], 'Immediate')
