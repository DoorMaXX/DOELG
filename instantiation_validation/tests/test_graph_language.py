"""Exercise English graph writes and lookups without a database or model service."""

import ast
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import Mock

from src import graph_language as language
from src.standard_observation_factors import STANDARD_FACTORS
from src.causal_relation_matrix import get_relation_code, get_relation_metadata

ROOT = Path(__file__).resolve().parents[2]


def load_function(path, name, namespace):
    """Load a real pipeline function with isolated graph and service dependencies."""
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8-sig'))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), 'exec'), namespace)
    return namespace[name]


class Relations:
    """Record relationship writes for an in-memory graph fixture."""

    def __init__(self):
        """Initialize the connected-node registry."""
        self.links = {}

    def connect(self, node, properties=None):
        """Connect a node and return mutable relationship properties."""
        relation = SimpleNamespace(**{'relation_name': '', 'save': lambda: None, **(properties or {})})
        self.links[node] = relation
        return relation

    def is_connected(self, node):
        """Check whether the target node is connected."""
        return node in self.links

    def all(self):
        """Return all connected target nodes."""
        return list(self.links)

    def __len__(self):
        """Return the number of connected target nodes."""
        return len(self.links)

    def relationship(self, node):
        """Return relationship properties for the target node."""
        return self.links.get(node)

    def all_relationships(self, node):
        """Return the relationship list for a connected target."""
        return [self.links[node]] if node in self.links else []

    def disconnect(self, node):
        """Remove the recorded relationship to a target."""
        self.links.pop(node, None)


class Manager:
    """Record lookup conditions and saved nodes for one graph label."""

    def __init__(self):
        """Initialize independent query and node registries."""
        self.rows = []
        self.queries = []

    def first_or_none(self, **conditions):
        """Find the first node whose properties match the query."""
        self.queries.append(conditions)
        return next((row for row in self.rows
                     if all(getattr(row, key, None) == value for key, value in conditions.items())), None)

    def all(self):
        """Return all saved nodes for this label."""
        return list(self.rows)


class Node:
    """Provide the graph operations used by the construction functions under test."""

    def __init__(self, **properties):
        """Initialize supplied properties and lazy relationship containers."""
        self.__dict__.update(properties)
        self.node_id = str(id(self))
        self.disaster_ids = []

    def __getattr__(self, name):
        """Create a relationship container on first access."""
        relation = Relations()
        setattr(self, name, relation)
        return relation

    def save(self):
        """Save this node once and return it."""
        if self not in self.nodes.rows:
            self.nodes.rows.append(self)
        return self


def graph_namespace():
    """Build isolated graph labels and the actual normalization helpers."""
    labels = ('OriginText', 'GeoObject', 'DisPhen', 'Impact', 'Factor', 'Time', 'DisasterEvent',
              'Sensor', 'ObservationEvent', 'ObservationPattern', 'GlobalDisasterEvent',
              'MergeDisasterEvent')
    namespace = {name: type(name, (Node,), {'nodes': Manager()}) for name in labels}
    namespace.update(english_disaster=language.english_disaster,
                     english_observation=language.english_observation,
                     english_geo_results=language.english_geo_results,
                     english_term=language.english_term,
                     split_event_pair=language.split_event_pair,
                     get_relation_code=get_relation_code, get_relation_metadata=get_relation_metadata,
                     update_D_argument_relations=lambda node: None,
                     tqdm=lambda iterable, **kwargs: iterable)
    return namespace


def vocabularies():
    """Read the actual Chinese and English extraction vocabularies without model initialization."""
    tree = ast.parse((ROOT / 'src/event_extraction.py').read_text(encoding='utf-8-sig'))
    return {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.endswith(('types', 'types_en'))}


class GraphLanguageTests(unittest.TestCase):
    """Check vocabulary parity, graph insertion, and downstream matching."""

    def test_all_extraction_categories_follow_original_dictionary(self):
        """Ensure every controlled category uses the original experiment translation."""
        vocabulary = vocabularies()
        for name in ('geo_types', 'impact_types', 'sensor_types', 'platform_types'):
            self.assertEqual([language.english_term(value, strict=True) for value in vocabulary[name]],
                             vocabulary[name + '_en'])

    def test_unknown_controlled_category_does_not_silently_remain_chinese(self):
        """Translate missing categories once while preserving proper geographic names."""
        self.assertEqual(language.english_term('贾鲁河'), '贾鲁河')
        language._translate_missing_term.cache_clear()
        with patch.object(language, '_translate_term_with_model', return_value='New Category') as translate:
            self.assertEqual(language.english_term('未登记的类别', strict=True), 'New Category')
            self.assertEqual(language.english_term('未登记的类别', strict=True), 'New Category')
            translate.assert_called_once_with('未登记的类别')
        self.assertEqual(language.english_term('Administrative Region', strict=True), 'Administrative Region')

    def test_translation_failure_is_nonfatal_and_cached(self):
        """Avoid repeated requests or exceptions when the model cannot translate a term."""
        language._translate_missing_term.cache_clear()
        with patch.object(language, '_translate_term_with_model', side_effect=TimeoutError) as translate:
            self.assertEqual(language.english_term('缺词且服务超时', strict=True), '')
            self.assertEqual(language.english_term('缺词且服务超时', strict=True), '')
            translate.assert_called_once()
        for index, response in enumerate(('', None, '仍然是中文')):
            with patch.object(language, '_translate_term_with_model', return_value=response):
                self.assertEqual(language.english_term(f'无效翻译{index}', strict=True), '')

    def test_failed_feature_translation_does_not_block_valid_features(self):
        """Skip an untranslated feature and its edges while inserting the other features."""
        language._translate_missing_term.cache_clear()
        namespace = graph_namespace()
        insert = load_function('src/geo_extraction.py', 'insert_database_geo', namespace)
        raw = dict(features=[dict(name='Unknown site', type='缺失地物类型', geometry='polygon'),
                             dict(name='A', type='河流', geometry='linestring'),
                             dict(name='B', type='道路', geometry='linestring')],
                   spatial_relations=[dict(subject='Unknown site', object='A', relation_type='临近'),
                                      dict(subject='A', object='B', relation_type='缺失空间谓词'),
                                      dict(subject='A', object='B', relation_type='相交')])
        with patch.object(language, '_translate_term_with_model', side_effect=TimeoutError):
            self.assertTrue(insert(raw, 'Source corpus', 'Title'))
        self.assertEqual([node.geo_type for node in namespace['GeoObject'].nodes.rows], ['River', 'Highway'])
        first, second = namespace['GeoObject'].nodes.rows
        self.assertEqual(first.spatial.relationship(second).relation_name, 'Intersect')

    def test_failed_disaster_translation_does_not_block_later_events(self):
        """Return an explicit skip count and permit the next disaster to be written."""
        language._translate_missing_term.cache_clear()
        namespace = graph_namespace()
        insert = load_function('main/main_extraction.py', 'insert_database', namespace)
        disaster = dict(feature=dict(name='site', geo_type='缺失灾害地物类型', geometry='polygon'),
                        impact=dict(name='water', impact_type='积水内涝'), factors={})
        with patch.object(language, '_translate_term_with_model', side_effect=TimeoutError):
            self.assertEqual(insert(disaster, 'Source', 'Title', 1), 0)
        self.assertEqual(namespace['DisasterEvent'].nodes.rows, [])
        disaster['feature']['geo_type'] = '道路'
        self.assertEqual(insert(disaster, 'Source', 'Title', 2), 1)
        self.assertEqual(len(namespace['DisasterEvent'].nodes.rows), 1)

    def test_failed_sensor_translation_returns_a_skip_count(self):
        """Keep processing after a missing sensor category cannot be translated."""
        language._translate_missing_term.cache_clear()
        namespace = graph_namespace()
        insert = load_function('main/main_extraction.py', 'insert_database_obs', namespace)
        observation = dict(sensor=dict(name='device', type='缺失传感器类型', platform='地面固定平台'),
                           observed_object=None, factors={'feature_factors': ['水位']})
        with patch.object(language, '_translate_term_with_model', side_effect=TimeoutError):
            self.assertEqual(insert(observation, 'Source', 1), 0)
        self.assertEqual(namespace['ObservationEvent'].nodes.rows, [])
        observation['sensor']['type'] = '其它传感器'
        self.assertEqual(insert(observation, 'Source', 2), 1)

    def test_unavailable_dictionary_can_use_model_fallback(self):
        """Continue translation even when the configured dictionary file is unavailable."""
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            dictionary = language._read_dictionary(str(Path(folder) / 'missing.json'))
            language._translate_missing_term.cache_clear()
            with patch.object(language, 'translation_dictionary', return_value=dictionary), \
                    patch.object(language, '_translate_term_with_model', return_value='River'):
                self.assertEqual(language.english_term('河流', strict=True), 'River')

    def test_file_ingestion_continues_after_translation_failure(self):
        """Process valid candidates after failed candidates in both real file-ingestion loops."""
        import concurrent.futures
        import os
        for stage in ('geo', 'event'):
            with self.subTest(stage=stage):
                language._translate_missing_term.cache_clear()
                namespace = graph_namespace()
                namespace.update(os=os, concurrent=concurrent, print=Mock(),
                                 parse_tasks_from_json=lambda path: [0, 1])
                if stage == 'geo':
                    load_function('src/geo_extraction.py', 'insert_database_geo', namespace)
                    namespace['process_geo_item'] = lambda index: dict(
                        results=dict(features=[dict(name=f'site{index}',
                                                    type='无法翻译的地物' if index == 0 else '道路',
                                                    geometry='polygon')], spatial_relations=[]),
                        original_text=f'text{index}', title_key='Corpus')
                    function = 'run_geo_extraction'
                else:
                    load_function('main/main_extraction.py', 'insert_database', namespace)
                    load_function('main/main_extraction.py', 'insert_database_obs', namespace)
                    namespace['process_event_item'] = lambda index: dict(
                        disasters=[dict(feature=dict(name=f'site{index}',
                                                     geo_type='无法翻译的地物' if index == 0 else '道路',
                                                     geometry='polygon'),
                                        impact=dict(name='water', impact_type='积水内涝'), factors={})],
                        observations=[], original_text=f'text{index}', title_key='Corpus', index=index)
                    function = 'run_event_extraction'
                run = load_function('main/main_extraction.py', function, namespace)
                with patch.object(os, 'listdir', return_value=['corpus.json']), \
                        patch.object(language, '_translate_term_with_model', side_effect=TimeoutError):
                    run('input', 1)
                self.assertEqual(len(namespace['GeoObject'].nodes.rows), 1)
                if stage == 'event':
                    self.assertEqual(len(namespace['DisasterEvent'].nodes.rows), 1)
                summaries = [str(call.args[0]) for call in namespace['print'].call_args_list]
                self.assertTrue(any('1 skipped' in line or '1 candidate events skipped' in line
                                    for line in summaries))

    def test_cleaned_dictionary_terms_keep_the_original_translation(self):
        """Translate terms after the punctuation cleanup used by the extraction workers."""
        cases = {
            '行,蓄,滞洪区': 'Flood Storage Area',
            '行，蓄，滞洪区': 'Flood Storage Area',
            '行, 蓄, 滞洪区': 'Flood Storage Area',
            '河,湖岛': 'Islet',
            '渗透(孔隙)水压力': 'Seepage Pressure',
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(language.english_term(value, strict=True), expected)
        for name in ('geo_types', 'impact_types', 'sensor_types', 'platform_types'):
            for term in vocabularies()[name]:
                cleaned = term.replace('、', ',').replace('，', ',').replace('（', '(').replace('）', ')')
                self.assertEqual(language.english_term(cleaned, strict=True),
                                 language.english_term(term, strict=True))
        self.assertEqual(language.english_term('蒙洼，蓄洪区'), '蒙洼，蓄洪区')

    def test_geographic_worker_cleanup_can_be_inserted_in_english(self):
        """Replay the reported failure through cleanup, parsing, translation, and insertion."""
        import re
        namespace = graph_namespace()
        namespace['re'] = re
        namespace['geo_ner_re'] = lambda text: (
            "results = [Feature('蒙洼蓄洪区', '行、蓄、滞洪区', Geometry('polygon'))]"
        )
        load_function('src/geo_extraction.py', 'parse_geo_string_to_dict', namespace)
        process = load_function('main/main_extraction.py', 'process_geo_item', namespace)
        insert = load_function('src/geo_extraction.py', 'insert_database_geo', namespace)
        result = process({'text': '蒙洼蓄洪区受淹。', 'title': 'Corpus title'})
        self.assertIsNotNone(result)
        self.assertEqual(result['results']['features'][0]['type'], '行,蓄,滞洪区')
        insert(result['results'], result['original_text'], result['title_key'])
        node = namespace['GeoObject'].nodes.rows[0]
        self.assertEqual(node.geo_type, 'Flood Storage Area')
        self.assertEqual(node.geo_object_name, '蒙洼蓄洪区')

    def test_disaster_lookup_and_write_use_english_categories(self):
        """Normalize categories before deduplication without rewriting the source corpus."""
        namespace = graph_namespace()
        insert = load_function('main/main_extraction.py', 'insert_database', namespace)
        disaster = dict(feature=dict(name='道路', geo_type='道路', geometry='linestring'),
                        impact=dict(name='发生积水', impact_type='积水内涝'), factors={})
        original = deepcopy(disaster)
        insert(disaster, '道路发生积水。', '原始标题', 1)
        geo = namespace['GeoObject'].nodes.rows[0]
        self.assertEqual((geo.geo_object_name, geo.geo_type), ('Highway', 'Highway'))
        self.assertEqual(namespace['Impact'].nodes.rows[0].impact_type, 'Waterlogging')
        self.assertEqual(namespace['DisasterEvent'].nodes.rows[0].event_description, 'Highway,发生积水')
        self.assertEqual(namespace['OriginText'].nodes.rows[0].text_content, '道路发生积水。')
        self.assertEqual(namespace['GeoObject'].nodes.queries[0], {'geo_object_name': 'Highway'})
        self.assertEqual(disaster, original)

    def test_observation_lookup_uses_same_english_values_as_write(self):
        """Translate the sensor, platform, and geographic category before lookup."""
        namespace = graph_namespace()
        insert = load_function('main/main_extraction.py', 'insert_database_obs', namespace)
        observation = dict(sensor=dict(name='相机', type='被动光学传感器', platform='地面固定平台'),
                           observed_object=dict(name='河流', geo_type='河流', geometry='linestring'),
                           factors={'feature_factors': ['水位']})
        insert(observation, '相机观测水位。', 1)
        sensor = namespace['Sensor'].nodes.rows[0]
        self.assertEqual(sensor.sensor_type, 'Passive Optical Sensor')
        self.assertEqual(sensor.platform, 'Ground-based Fixed Platform')
        self.assertEqual(namespace['Sensor'].nodes.queries[0]['sensor_type'], sensor.sensor_type)
        self.assertEqual(namespace['GeoObject'].nodes.rows[0].geo_object_name, 'River')
        self.assertEqual(namespace['Factor'].nodes.rows[0].factor_name, '水位')

    def test_geographic_relations_keep_english_endpoints_consistent(self):
        """Resolve translated generic names to the same nodes used by spatial edges."""
        namespace = graph_namespace()
        insert = load_function('src/geo_extraction.py', 'insert_database_geo', namespace)
        result = dict(features=[dict(name='道路', type='道路', geometry='linestring'),
                                dict(name='贾鲁河', type='河流', geometry='linestring')],
                      spatial_relations=[dict(subject='道路', object='贾鲁河', relation_type='临近')])
        insert(result, '道路临近贾鲁河。', 'Title')
        road, river = namespace['GeoObject'].nodes.rows
        self.assertEqual(road.spatial.relationship(river).relation_name, 'Near')
        self.assertEqual(road.geo_object_name, 'Highway')
        self.assertEqual(river.geo_object_name, '贾鲁河')

    def test_unknown_spatial_predicate_uses_translation_and_checks_output(self):
        """Use the model only for predicates absent from the fixed vocabulary."""
        with patch.object(language, '_translate_relation_with_model', return_value='Situated Between') as translate:
            self.assertEqual(language.english_relation('居于二者之间'), 'Situated Between')
            translate.assert_called_once_with('居于二者之间')
        self.assertEqual(language.english_relation('相邻;临近;相邻'), 'Adjacent;Near')

    def test_observation_pattern_translation_precedes_lookup(self):
        """Convert the Chinese structured response before finding or saving a pattern."""
        namespace = graph_namespace()
        obs = namespace['ObservationEvent'](event_description='observation').save()
        disaster = namespace['DisasterEvent'](event_description='disaster').save()
        origin = namespace['OriginText'](text_content='原始语料').save()
        origin.exist.connect(disaster)
        obs.origin.connect(origin)
        namespace['get_event_obsPat_llm'] = lambda *args: SimpleNamespace(
            space_res=1, time_res=2, weather_effect='受云雾遮挡', obs_position='遥感观测')
        save = load_function('src/other/observation_pattern.py', 'get_save_event_obsPat', namespace)
        save(obs, disaster)
        pattern = namespace['ObservationPattern'].nodes.rows[0]
        self.assertEqual(pattern.weather_effect, 'Obscured by Clouds and Fog')
        self.assertEqual(pattern.obs_position, 'Remote Sensing Observation')
        self.assertEqual(namespace['ObservationPattern'].nodes.queries[0]['weather_effect'], pattern.weather_effect)

    def test_english_event_pair_survives_hyphenated_geo_category(self):
        """Keep the category's internal hyphen when creating abstract event clusters."""
        vocabulary = vocabularies()
        namespace = graph_namespace()
        namespace.update(geo_types=vocabulary['geo_types_en'], impact_types=vocabulary['impact_types_en'],
                         _get_or_create_merge_node=lambda desc, pattern: namespace['MergeDisasterEvent'](
                             event_description=desc, cluster_pattern=pattern).save(),
                         _connect_global_cause_with_pattern=lambda *args: None,
                         _link_to_original_event=lambda *args: None,
                         _link_factors_to_global=lambda *args: None,
                         _link_envDes_to_origin_event=lambda: None)
        construct = load_function('src/generalization.py', 'process_abstract_clustering_to_graph', namespace)
        label = 'In-Building Parking,Waterlogging-River,Submergence'
        construct({label: [[('specific cause', 'specific result')]]})
        self.assertEqual([node.event_description for node in namespace['GlobalDisasterEvent'].nodes.rows],
                         ['In-Building Parking,Waterlogging', 'River,Submergence'])
        self.assertTrue(all(node.cluster_pattern == label + '_C0'
                            for node in namespace['MergeDisasterEvent'].nodes.rows))

    def test_english_disasters_are_not_filtered_by_chinese_taxonomy(self):
        """Include newly constructed English nodes in disaster-pair generalization."""
        vocabulary = vocabularies()
        namespace = graph_namespace()
        namespace.update(geo_types=vocabulary['geo_types_en'], impact_types=vocabulary['impact_types_en'])
        first = namespace['DisasterEvent'](event_description='first').save()
        second = namespace['DisasterEvent'](event_description='second').save()
        for event, geo_type, impact_type in ((first, 'River', 'Water Overflow'),
                                             (second, 'Highway', 'Waterlogging')):
            event.geo_argument.connect(namespace['GeoObject'](geo_object_name='proper name', geo_type=geo_type))
            event.impact_argument.connect(namespace['DisPhen'](impact_name='surface form', impact_type=impact_type))
        first.cause.connect(second)
        get_pairs = load_function('src/generalization.py', 'get_geo_impact_pair', namespace)
        self.assertEqual(get_pairs(), {'River,Water Overflow-Highway,Waterlogging': [['first', 'second']]})

    def test_administrative_observations_match_english_description(self):
        """Find Administrative Region events through the English query condition."""
        namespace = graph_namespace()
        administrative = namespace['GlobalDisasterEvent'](event_description='Administrative Region,Subsidence').save()
        bridge = namespace['GlobalDisasterEvent'](event_description='Bridge,Subsidence').save()
        impact = namespace['DisPhen'](impact_name='Subsidence')
        for event, name in ((administrative, 'Administrative Region'), (bridge, 'Bridge')):
            event.geo_argument.connect(namespace['GeoObject'](geo_object_name=name))
            event.impact_argument.connect(impact)
        observation = namespace['ObservationEvent'](event_description='observation')
        administrative.trigger.connect(observation)
        connect = load_function('src/generalization.py', 'connect_disaster_trigger_relate_obs', namespace)
        connect()
        self.assertTrue(bridge.trigger.is_connected(observation))
        self.assertEqual(namespace['GlobalDisasterEvent'].nodes.queries,
                         [{'event_description': 'Administrative Region,Subsidence'}])

    def test_sensor_classification_returns_english_for_chinese_prompt(self):
        """Keep the Chinese prompt while returning the English graph category."""
        namespace = dict(english_term=language.english_term, llm=None, ClassifyInfo=object,
                         create_agent=lambda *args, **kwargs: SimpleNamespace(
                             invoke=lambda *args, **kwargs: {'structured_response': SimpleNamespace(
                                 sensor_type='主动微波传感器')}))
        classify = load_function('src/generalization.py', 'fetch_llm_classification', namespace)
        self.assertEqual(classify('sensor-id', '雷达'), ('sensor-id', 'Active Microwave Sensor', None))

    def test_standard_factor_mapping_contains_only_english_vocabulary(self):
        """Ensure all standard names and related geographic types are English."""
        tree = ast.parse((ROOT / 'src/other/factor_map.py').read_text(encoding='utf-8-sig'))
        mapping = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign)
                                        and isinstance(node.targets[0], ast.Name)
                                        and node.targets[0].id == 'indicator_mapping'))
        self.assertEqual(set(mapping), set(STANDARD_FACTORS))
        self.assertFalse(any(language.contains_chinese(value)
                             for name, geos in mapping.items() for value in (name, *geos)))

    def test_fusion_validates_english_task_names_and_categories(self):
        """Keep English wrapper task names compatible with cluster validation."""
        namespace = dict(english_term=language.english_term,
                         get_setting=lambda key: 1, ThreadPoolExecutor=ThreadPoolExecutor,
                         as_completed=as_completed, tqdm=lambda iterable, **kwargs: iterable)
        cluster = load_function('src/fusion.py', 'generic_cluster_identify', namespace)
        cases = (
            ('GeoObject', SimpleNamespace(is_all_cluster=True, cluster_geo_name='道路',
                                           cluster_geo_type='道路', cluster_geometry='linestring'),
             'cluster_geo_name', lambda result: [result.cluster_geo_name, result.cluster_geo_type,
                                                result.cluster_geometry], ['Highway', 'Highway', 'linestring']),
            ('Sensor', SimpleNamespace(is_all_cluster=True, cluster_sensor_name='radar',
                                        cluster_sensor_type='主动微波传感器', cluster_platform='航天观测平台'),
             'cluster_sensor_name', lambda result: [result.cluster_sensor_name, result.cluster_sensor_type,
                                                   result.cluster_platform],
             ['radar', 'Active Microwave Sensor', 'Spaceborne Observation Platform']),
            ('ObsFactor', SimpleNamespace(is_all_cluster=True, cluster_factor_name='raw factor',
                                           cluster_factor_type='feature_factors'),
             'cluster_factor_name', lambda result: [result.cluster_factor_name, result.cluster_factor_type],
             ['raw factor', 'feature_factors']),
        )
        for task, response, name_field, construct, expected in cases:
            with self.subTest(task=task):
                result = cluster({0: [['source name']]}, task, lambda nodes: response,
                                 lambda node: node[0], construct,
                                 {'excluded': 'excluded_names', 'cluster_name': name_field})
                if task == 'ObsFactor':
                    result, deleted = result
                    self.assertEqual(deleted, [])
                self.assertEqual(next(iter(result.values()))['clustered_node'], expected)

    def test_graph_construction_uses_the_ontology_impact_label(self):
        """Prevent insertion from targeting a different class than argument edges accept."""
        from ontology.neo4jmodel import DisasterEvent, GeoObject, GlobalDisasterEvent, Impact
        for manager in (DisasterEvent().impact_argument, GeoObject().happen,
                        GlobalDisasterEvent().impact_argument):
            self.assertIs(manager.definition['node_class'], Impact)
        for path in ('main/main_extraction.py', 'src/generalization.py', 'src/fusion.py'):
            tree = ast.parse((ROOT / path).read_text(encoding='utf-8-sig'))
            imports = [alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
                       and node.module == 'ontology.neo4jmodel' for alias in node.names]
            self.assertIn('Impact', imports)
            self.assertNotIn('DisPhen', imports)

    def test_computed_topology_uses_english_keys_and_values(self):
        """Keep computed topology compatible with English instantiation matching."""
        topology = load_function('src/other/geo_encoding.py', 'get_topological_relation', {})
        self.assertEqual(topology({'Intersect': True, 'Contains': True}), 'Contains')
        self.assertEqual(topology({'Intersect': True, 'Touches': True}), 'Adjacent')
        self.assertEqual(topology({}), 'Disjoint')

    def test_ontology_relationship_defaults_are_english(self):
        """Check that new structural edges have no Chinese default labels."""
        tree = ast.parse((ROOT / 'ontology/neo4jmodel.py').read_text(encoding='utf-8-sig'))
        defaults = []
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef):
                continue
            for node in cls.body:
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                                                        and target.id == 'relation_name'
                                                        for target in node.targets):
                    defaults.extend(ast.literal_eval(keyword.value) for keyword in node.value.keywords
                                    if keyword.arg == 'default')
        self.assertGreater(len(defaults), 10)
        self.assertFalse(any(language.contains_chinese(value) for value in defaults))


class SpatialConnectionTests(unittest.TestCase):
    """Check explicit database connection passing and restoration during spatial computation."""

    def test_both_spatial_pipelines_restore_database_before_writing(self):
        """Exercise both real pipelines without a module-global connector or external services."""
        for name in ('main_calculate_spatial_relations', 'main_calculate_spatial_relations_only'):
            with self.subTest(pipeline=name):
                namespace = graph_namespace()
                active_database = ['selected-cli-database']
                switches = []

                def connect(database):
                    """Record the active database selected by the pipeline."""
                    switches.append(database)
                    active_database[0] = database

                connector = SimpleNamespace(database_name=active_database[0], connect_database=connect)
                cause = namespace['DisasterEvent'](event_description='cause').save()
                result = namespace['DisasterEvent'](event_description='result').save()
                cause.geo_argument.connect(namespace['GeoObject'](geo_object_name='A', geo_type='River'))
                result.geo_argument.connect(namespace['GeoObject'](geo_object_name='B', geo_type='Bridge'))
                cause.cause.connect(result)
                relation = cause.spatial_to.connect(result, {'relations': {'distance_computation': 10.0}})
                writes = []
                relation.save = lambda: writes.append(active_database[0])

                def find_path(cause_type, result_type):
                    """Require type queries to run on the geographic type database."""
                    self.assertEqual(active_database[0], 'geos-type-1')
                    self.assertEqual((cause_type, result_type), ('River', 'Bridge'))
                    return [['River', 'Intersect', 'Bridge']]

                namespace.update(find_geo_type_path=find_path, is_geo_location=lambda: {},
                                 get_setting=lambda key: {'type_database': 'geos-type-1'},
                                 event_pair_geo_encoding_parallel=lambda *args: [],
                                 find_geo_semantics_path=lambda *args: [['A', 'Near', 'B']])
                load_function('src/geo_extraction.py', '_populate_geo_type_paths', namespace)
                pipeline = load_function('src/geo_extraction.py', name, namespace)
                pipeline(connector)
                self.assertEqual(switches, ['geos-type-1', 'selected-cli-database'])
                self.assertEqual(active_database[0], 'selected-cli-database')
                self.assertEqual(relation.relations['type_path'], [['River', 'Intersect', 'Bridge']])
                self.assertTrue(writes)
                self.assertEqual(set(writes), {'selected-cli-database'})
                self.assertEqual(relation.relations['distance_computation'], 10.0)

    def test_failed_type_query_always_restores_selected_database(self):
        """Restore the caller's database when a type-path query raises an exception."""
        connector = SimpleNamespace(database_name='selected-cli-database', connect_database=Mock())
        namespace = {'find_geo_type_path': Mock(side_effect=RuntimeError('Type query failed')),
                     'get_setting': lambda key: {'type_database': 'geos-type-1'}}
        populate = load_function('src/geo_extraction.py', '_populate_geo_type_paths', namespace)
        pair = dict(geos_type=['River', 'Bridge'], events_desc=['cause', 'result'])
        with self.assertRaisesRegex(RuntimeError, 'Type query failed'):
            populate([pair], connector)
        self.assertEqual([call.args[0] for call in connector.connect_database.call_args_list],
                         ['geos-type-1', 'selected-cli-database'])

    def test_empty_graph_does_not_switch_databases(self):
        """Avoid an unnecessary geographic type connection when no causal pairs exist."""
        connector = SimpleNamespace(database_name='selected-cli-database', connect_database=Mock())
        populate = load_function('src/geo_extraction.py', '_populate_geo_type_paths', {})
        populate([], connector)
        connector.connect_database.assert_not_called()

    def test_spatial_entry_point_passes_its_connector(self):
        """Forward the active connector from the entry point to spatial computation."""
        connector = object()
        locations = object()
        calculate = Mock()
        namespace = dict(time=SimpleNamespace(time=lambda: 0),
                         print=Mock(),
                         is_geo_location_parallel=Mock(return_value=locations),
                         event_pair_geo_encoding_parallel=Mock(), get_setting=lambda key: True,
                         main_calculate_spatial_relations=calculate)
        run = load_function('main/main_extraction.py', 'run_spatial_computation', namespace)
        self.assertTrue(run(3, connector))
        calculate.assert_called_once_with(connector)
