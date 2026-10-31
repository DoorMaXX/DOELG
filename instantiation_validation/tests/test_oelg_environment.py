import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from shapely.geometry import box, mapping
from oelg_environment import (load_environment_catalog, validate_environment_rule,
                              evaluate_spatial_rule, MissingEnvironmentData)
from instantiation_oelg import (OELGInstantiationBuilder, OELGKnowledgeRepository,
                                load_instantiation_knowledge)
import test_instantiation_oelg as baseline_tests


class EnvironmentRuleTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_environment_catalog()
        self.rules = {r['phenomenon']: r for r in self.catalog['rules']}

    def test_catalog_covers_every_existing_phenomenon(self):
        review = json.loads((Path(__file__).resolve().parents[1] / 'data/dictionary/phenomenon_geo_mapping_review.json').read_text(encoding='utf-8'))
        self.assertEqual(set(self.rules), {s['name'] for s in review['summary']})
        self.assertEqual(len(self.rules), 34)
        self.assertEqual(self.catalog['regional_mean_elevation_m'], 120)

    def test_relative_elevation_and_slope_boundaries(self):
        rule = self.rules['Waterlogging']['spatial_rule']
        self.assertTrue(evaluate_spatial_rule({'avg_elevation': 120, 'avg_slope': 5}, rule, 120))
        self.assertFalse(evaluate_spatial_rule({'avg_elevation': 120.1, 'avg_slope': 5}, rule, 120))
        self.assertFalse(evaluate_spatial_rule({'avg_elevation': 100, 'avg_slope': 5.1}, rule, 120))
        self.assertTrue(evaluate_spatial_rule({'avg_elevation': 120.1, 'avg_slope': 5}, rule, 130))

    def test_missing_invalid_measurements_never_become_zero(self):
        rule = self.rules['Waterlogging']['spatial_rule']
        for value in (None, 'bad', float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(MissingEnvironmentData):
                evaluate_spatial_rule({'avg_elevation': value, 'avg_slope': 1}, rule, 120)
        with self.assertRaises(MissingEnvironmentData):
            evaluate_spatial_rule({'avg_elevation': 100, 'avg_slope': -9999}, rule, 120)

    def test_three_valued_and_or(self):
        rule = self.rules['Structural Fracture']['spatial_rule']
        self.assertTrue(evaluate_spatial_rule({'avg_slope': 10}, rule, 120))
        with self.assertRaises(MissingEnvironmentData):
            evaluate_spatial_rule({'avg_slope': 1}, rule, 120)
        rule = self.rules['Waterlogging']['spatial_rule']
        self.assertFalse(evaluate_spatial_rule({'avg_elevation': 200}, rule, 120))

    def test_between_and_malformed_rules(self):
        rule = self.rules['Plant Pests and Diseases']['spatial_rule']
        self.assertTrue(evaluate_spatial_rule({'soil_moisture': .4, 'air_temperature': 20}, rule, 120))
        self.assertFalse(evaluate_spatial_rule({'soil_moisture': .4, 'air_temperature': 31}, rule, 120))
        for change in ({'operator': 'nonsense'}, {'factor': 'unknown'}, {'unit': 'km'},
                       {'operator': 'between', 'value': [5, 1]}):
            value = copy.deepcopy(self.rules['Waterlogging'])
            value['spatial_rule']['constraints'][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_environment_rule(value)


class EnvironmentInstantiationTests(unittest.TestCase):
    setUp = baseline_tests.OELGInstantiationTests.setUp
    write = baseline_tests.OELGInstantiationTests.write

    def use_rule(self, name):
        rule = next(r for r in load_environment_catalog()['rules'] if r['phenomenon'] == name)
        rule = dict(rule, phenomenon='Flood')
        self.config.update(environment_rules={'Flood': rule}, regional_mean_elevation_m=120,
                           environment_missing_policy='report_and_skip')
        return rule

    def test_actual_feature_filter_and_audit(self):
        self.use_rule('Waterlogging')
        path = self.folder / 'Road.json'
        data = json.loads(path.read_text())
        for feature, elevation in zip(data['features'], [100, 110, 130, 110]):
            feature['properties']['avg_elevation'] = elevation
        path.write_text(json.dumps(data))
        builder = OELGInstantiationBuilder(self.config, self.knowledge)
        nodes = builder.instantiate()
        self.assertEqual([f['osm_id'] for f in nodes[1]['affect_features']], ['Road_1'])
        self.assertEqual(builder.environment_audit['Flood']['rejected'], 1)
        self.assertFalse(builder.proxy_phenomena)
        self.assertEqual(builder.scenario_proxy_phenomena, {'Flood'})
        builder.export_results(self.folder / 'export')
        self.assertEqual(json.loads((self.folder/'export/environment_audit.json').read_text(encoding='utf-8'))['regional_mean_elevation_m'], 120)

    def test_unknown_candidates_reported_or_raise(self):
        self.use_rule('Waterlogging')
        builder = OELGInstantiationBuilder(self.config, self.knowledge)
        self.assertEqual(len(builder.instantiate()), 1)
        self.assertEqual(builder.environment_audit['Flood']['unknown'], 3)
        self.assertEqual(builder.pruned[0]['reason'], 'environment_data_incomplete')
        self.config['environment_missing_policy'] = 'error'
        with self.assertRaisesRegex(ValueError, 'rel_elevation'):
            OELGInstantiationBuilder(self.config, self.knowledge).instantiate()

    def test_source_slope_is_not_tested_on_flat_bearing_feature(self):
        rule = self.use_rule('Landslide')
        rule['environment']['geo_types'] = ['Mountain']
        source = dict(type='Feature', geometry=mapping(box(113.51, 34.71, 113.53, 34.73)),
                      properties={'avg_slope': 20})
        (self.folder/'Mountain.json').write_text(json.dumps(dict(type='FeatureCollection', features=[source])))
        builder = OELGInstantiationBuilder(self.config, self.knowledge)
        node = builder.instantiate()[1]
        self.assertEqual([f['osm_id'] for f in node['affect_features']], ['Road_1'])
        self.assertEqual(builder.environment_audit['Flood']['source_passed'], 1)
        self.assertLess(node['affect_extent'].difference(builder.rain).area, 1e-6)

    def test_missing_source_layer_is_not_unrestricted_environment(self):
        rule = self.use_rule('Landslide')
        rule['environment']['geo_types'] = ['Mountain']
        self.config['missing_layer_policy'] = 'report_and_skip'
        builder = OELGInstantiationBuilder(self.config, self.knowledge)
        self.assertEqual(len(builder.instantiate()), 1)
        self.assertEqual(builder.pruned[0]['reason'], 'environment_data_incomplete')
        self.assertEqual(builder.environment_audit['Flood']['missing_source_layers'], ['Mountain'])

    def test_observed_root_is_preserved_and_no_dbbody_not_reenabled(self):
        rule = self.use_rule('Extreme Precipitation')
        self.config['environment_rules']['Rain'] = dict(rule, phenomenon='Rain')
        self.knowledge['nodes']['Flood'].update(geo_types=[], dbbody_review_complete=True)
        builder = OELGInstantiationBuilder(self.config, self.knowledge)
        self.assertEqual(len(builder.instantiate()), 1)
        self.assertEqual(builder.environment_audit['Rain']['status'], 'observed_root_input_preserved')
        self.assertEqual(builder.environment_audit['Flood']['status'], 'no_reviewed_dbbody_mapping')

    def test_environment_rules_come_from_current_oelg_even_with_cached_knowledge(self):
        rule = self.use_rule('Waterlogging')
        knowledge = dict(self.knowledge, environment_rules={'Flood': rule})
        repository = Mock()
        repository.read_abstract_graph.return_value = self.knowledge
        repository.read_environment_rules.return_value = knowledge['environment_rules']
        stale = self.folder/'knowledge.json'
        stale.write_text(json.dumps(dict(knowledge=dict(self.knowledge, environment_rules={'old': {}}))))
        config = dict(self.config, abstract_database='oelg-1', dataset='test', knowledge_json=str(stale),
                      environment_rule_source='oelg', environment_rules={})
        result = load_instantiation_knowledge(repository, config)
        self.assertEqual(result['environment_rules'], knowledge['environment_rules'])
        repository.read_environment_rules.assert_called_once_with('oelg-1', self.knowledge['nodes'])

    def test_graph_json_rules_and_ambiguous_or_missing_rules(self):
        rule = self.use_rule('Waterlogging')
        repository = OELGKnowledgeRepository.__new__(OELGKnowledgeRepository)
        row = dict(event_id='event', rule_id='rule', rules=json.dumps(rule))
        repository.query = Mock(return_value=[row])
        nodes = {'Flood': {'node_id': 'event'}}
        self.assertEqual(repository.read_environment_rules('oelg-1', nodes)['Flood'], rule)
        repository.query.return_value = [row, row]
        with self.assertRaisesRegex(ValueError, '多个环境规则'):
            repository.read_environment_rules('oelg-1', nodes)
        repository.query.return_value = []
        with self.assertRaisesRegex(ValueError, '缺少 DPEnvironment'):
            repository.read_environment_rules('oelg-1', nodes)


if __name__ == '__main__':
    unittest.main()

