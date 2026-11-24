"""Verify geocoding recovery and invalid-geometry handling without graph writes."""

import concurrent.futures
import contextlib
from copy import deepcopy
import io
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from other import geo_encoding as encoding
from instantiation_validation.tests.test_graph_language import graph_namespace, load_function


def response(data=None, status=200):
    """Create a response that implements HTTP validation and JSON decoding."""
    result = Mock(status_code=status)
    result.json.return_value = data
    if status >= 400:
        result.raise_for_status.side_effect = requests.HTTPError('HTTP failure', response=result)
    return result


def point(osm_type='relation', osm_id=1, x=113.6):
    """Return one valid Nominatim geometry result with a typed OSM identifier."""
    return {'osm_type': osm_type, 'osm_id': osm_id,
            'geojson': {'type': 'Point', 'coordinates': [x, 34.7]}}


class GeocodingRequestTests(unittest.TestCase):
    """Cover temporary failures, invalid payloads, batching, and missing geometries."""

    def setUp(self):
        """Use isolated endpoint settings and suppress expected recovery messages."""
        self.settings = {'search_url': 'http://example.test/search',
                         'lookup_url': 'http://example.test/lookup',
                         'reverse_url': 'http://example.test/reverse',
                         'http': {'connect_timeout_seconds': 1, 'read_timeout_seconds': 2,
                                  'max_attempts': 3, 'retry_delay_seconds': 1, 'lookup_batch_size': 50}}
        setting = patch.object(encoding, 'get_setting',
                               side_effect=lambda key: self.settings if key == 'geocoding'
                               else self.settings[key.split('.', 1)[1]])
        setting.start()
        self.addCleanup(setting.stop)
        sleep = patch.object(encoding.time, 'sleep')
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        redirect = contextlib.redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def test_timeout_recovers_with_finite_limits(self):
        """Retry a timeout and preserve the configured endpoint and lookup parameters."""
        with patch.object(encoding.requests, 'get', side_effect=[requests.Timeout('timeout'), response([point()])]) as get:
            self.assertEqual(encoding.get_address_lookup_local('R1'), [point()])
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args.args[0], self.settings['lookup_url'])
        self.assertEqual(get.call_args.kwargs['timeout'], (1.0, 2.0))
        self.assertEqual(get.call_args.kwargs['params']['polygon_geojson'], 1)

    def test_exhausted_timeout_returns_list(self):
        """Return an empty list after bounded retries instead of exposing None to callers."""
        with patch.object(encoding.requests, 'get', side_effect=requests.ConnectTimeout('timeout')) as get:
            self.assertEqual(encoding.get_address_lookup_local('R1'), [])
        self.assertEqual(get.call_count, 3)

    def test_server_error_recovers(self):
        """Retry a temporary server failure before decoding the successful response."""
        with patch.object(encoding.requests, 'get', side_effect=[response(status=503), response([point()])]) as get:
            self.assertEqual(encoding.get_address_lookup_local('R1'), [point()])
        self.assertEqual(get.call_count, 2)

    def test_client_error_is_not_retried(self):
        """Report a permanent HTTP failure without repeated requests."""
        with patch.object(encoding.requests, 'get', return_value=response(status=404)) as get:
            self.assertEqual(encoding.get_address_lookup_local('R1'), [])
        get.assert_called_once()

    def test_invalid_responses_do_not_escape(self):
        """Reject malformed JSON, API errors, and unexpected successful payloads."""
        malformed = response()
        malformed.json.side_effect = ValueError('invalid JSON')
        for item in [malformed, response({'error': 'not found'}), response({'unexpected': 'object'}), response([None])]:
            with self.subTest(payload=item), patch.object(encoding.requests, 'get', return_value=item):
                self.assertEqual(encoding.get_address_lookup_local('R1'), [])

    def test_search_and_reverse_have_stable_failure_types(self):
        """Keep search and reverse callers safe when the HTTP service is unavailable."""
        with patch.object(encoding.requests, 'get', side_effect=requests.ConnectionError('offline')):
            self.assertEqual(encoding.get_geo_encoding_local('site'), [])
            self.assertEqual(encoding.get_geo_reverse_local(34.7, 113.6), '')
            self.assertEqual(encoding.get_location.invoke({'geo_name': 'site'}), '找不到位置')

    def test_invalid_osm_ids_do_not_trigger_requests(self):
        """Reject empty and malformed IDs before contacting the service."""
        with patch.object(encoding.requests, 'get') as get:
            for value in ['', None, 'R0', 'R1,invalid', '123', [], 'N-1']:
                with self.subTest(value=value):
                    self.assertEqual(encoding.get_address_lookup_local(value), [])
        get.assert_not_called()
        self.assertEqual(encoding.normalize_osm_ids('r1-W2,N3,r1'), 'R1,W2,N3')

    def test_lookup_uses_batches_of_at_most_fifty(self):
        """Split large lookup lists according to the Nominatim API limit."""
        with patch.object(encoding.requests, 'get', side_effect=[response([point()]), response([point(osm_id=51)])]) as get:
            self.assertEqual(len(encoding.get_address_lookup_local([f'R{i}' for i in range(1, 52)])), 2)
        sizes = [len(call.kwargs['params']['osm_ids'].split(',')) for call in get.call_args_list]
        self.assertEqual(sizes, [50, 1])

    def test_failed_batch_discards_partial_results(self):
        """Avoid computing an entity extent from only its successful lookup batches."""
        self.settings['http']['lookup_batch_size'] = 1
        self.settings['http']['max_attempts'] = 1
        with patch.object(encoding.requests, 'get', side_effect=[response([point()]), requests.Timeout('timeout')]):
            self.assertEqual(encoding.get_address_lookup_local('R1,W2'), [])

    def test_geometry_handles_missing_lookup(self):
        """Handle the original None-returning failure without iterating over it."""
        for value in [None, [], {'error': 'offline'}]:
            with self.subTest(value=value), patch.object(encoding, 'get_address_lookup_local', return_value=value):
                self.assertIsNone(encoding.get_entity_combined_geometry('R1'))

    def test_incomplete_and_invalid_geometries_are_rejected(self):
        """Reject missing components and malformed GeoJSON rather than fabricating a full extent."""
        with patch.object(encoding, 'get_address_lookup_local', return_value=[point()]):
            self.assertIsNone(encoding.get_entity_combined_geometry('R1,W2'))
        for geojson in [None, {}, {'type': 'Point', 'coordinates': []}, {'type': 'invalid'}]:
            item = point()
            item['geojson'] = geojson
            with self.subTest(geojson=geojson), patch.object(encoding, 'get_address_lookup_local', return_value=[item]):
                self.assertIsNone(encoding.get_entity_combined_geometry('R1'))

    def test_valid_geometries_are_merged(self):
        """Preserve full valid geometry unions for multiple typed OSM IDs."""
        with patch.object(encoding, 'get_address_lookup_local', return_value=[point(), point('way', 2, 114.0)]):
            geometry = encoding.get_entity_combined_geometry('R1,W2')
        self.assertEqual(geometry.geom_type, 'MultiPoint')
        self.assertEqual(len(geometry.geoms), 2)

    def test_missing_geometry_skips_spatial_computation(self):
        """Avoid basin loading and a redundant second lookup when the first entity is unavailable."""
        with patch.object(encoding, 'get_entity_combined_geometry', return_value=None) as lookup, \
                patch.object(encoding.gpd, 'read_file') as read:
            self.assertEqual(encoding.analyze_spatial_relation('R1', 'W2'), {})
        lookup.assert_called_once_with('R1')
        read.assert_not_called()


class GeometryWriteTests(unittest.TestCase):
    """Prevent invalid location results and old empty caches from being accepted as geometry."""

    def setUp(self):
        """Load production save functions with isolated graph dependencies."""
        self.namespace = dict(math=math, re=re, normalize_osm_ids=encoding.normalize_osm_ids, Geometry=Mock())
        for name in ('_validated_location_geometry', '_save_geometry_to_db', '_get_cached_osm_ids', 'get_save_geo_geometry'):
            load_function('src/geo_extraction.py', name, self.namespace)
        self.geo = Mock(is_location=True, geo_object_name='site')
        self.geo.has_geometry.all.return_value = []
        self.info = SimpleNamespace(is_location=True, coordinates=[[113.6, 34.7]], osm_type=['relation'], osm_id=[1])
        redirect = contextlib.redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def save(self, info):
        """Call the actual save function using a completed worker result."""
        return self.namespace['_save_geometry_to_db'](self.geo, dict(
            geo_name='site', is_location_updated=False, new_is_location=True, location_info=info))

    def test_invalid_locations_never_create_geometry(self):
        """Reject empty IDs, unsupported types, mismatched arrays, and invalid coordinates."""
        for changes in [dict(osm_type=['country']), dict(osm_id=[0]), dict(osm_id=[]),
                        dict(osm_type=['relation', 'way']), dict(coordinates=[]),
                        dict(coordinates=[[181, 34]]), dict(coordinates=[[113, 91]]),
                        dict(coordinates=[[float('nan'), 34]]), dict(coordinates=[[113]]),
                        dict(is_location=False)]:
            info = deepcopy(self.info)
            info.__dict__.update(changes)
            with self.subTest(changes=changes):
                self.assertEqual(self.save(info), '')
        self.assertEqual(self.save(None), '')
        self.namespace['Geometry'].assert_not_called()
        self.namespace['Geometry'].nodes.first_or_none.assert_not_called()
        self.geo.has_geometry.connect.assert_not_called()

    def test_valid_location_is_written(self):
        """Save validated IDs with the original flattened coordinate representation."""
        self.namespace['Geometry'].nodes.first_or_none.return_value = None
        self.geo.has_geometry.is_connected.return_value = False
        self.assertEqual(self.save(self.info), 'R1')
        self.namespace['Geometry'].assert_called_once_with(location=[113.6, 34.7], osm_id='R1')
        self.geo.has_geometry.connect.assert_called_once()

    def test_cache_ignores_empty_geometry(self):
        """Use a valid later geometry instead of treating the first empty ID as cached success."""
        self.geo.has_geometry.all.return_value = [SimpleNamespace(osm_id=''), SimpleNamespace(osm_id='w2')]
        self.assertEqual(self.namespace['_get_cached_osm_ids'](self.geo), 'W2')

    def test_empty_cache_is_requeued_for_encoding(self):
        """Retry an old empty-ID cache and include the recovered entity in causal pairs."""
        other = Mock(is_location=True, geo_object_name='other', geo_type='Bridge')
        other.has_geometry.all.return_value = [SimpleNamespace(osm_id='W2')]
        self.geo.geo_type = 'River'
        self.geo.has_geometry.all.return_value = [SimpleNamespace(osm_id='')]
        self.geo.origin.all.return_value = []
        cause = Mock(node_id='cause')
        result = Mock(node_id='result')
        cause.geo_argument.all.return_value = [self.geo]
        result.geo_argument.all.return_value = [other]
        cause.cause.all.return_value = [result]
        self.namespace.update(DisasterEvent=Mock(nodes=Mock(all=Mock(return_value=[cause]))),
            get_setting=lambda key: 2, ThreadPoolExecutor=concurrent.futures.ThreadPoolExecutor,
            as_completed=concurrent.futures.as_completed,
            _fetch_geo_info_worker=Mock(return_value={}), _save_geometry_to_db=Mock(return_value='R1'))
        run = load_function('src/geo_extraction.py', 'event_pair_geo_encoding_parallel', self.namespace)
        pairs = run(['site', 'other'], do_encoding=True)
        self.namespace['_fetch_geo_info_worker'].assert_called_once()
        self.namespace['_save_geometry_to_db'].assert_called_once()
        self.assertEqual(pairs[0]['osm_ids_pair'], ['R1', 'W2'])

    def test_legacy_save_path_uses_shared_validation(self):
        """Route serial geocoding through the same validated writer as parallel geocoding."""
        self.geo.origin.all.return_value = []
        self.namespace['_fetch_geo_info_worker'] = Mock(return_value={'worker': 'result'})
        save = self.namespace['_save_geometry_to_db'] = Mock(return_value='R1')
        self.assertEqual(self.namespace['get_save_geo_geometry'](self.geo, True), 'R1')
        save.assert_called_once_with(self.geo, {'worker': 'result'})


class SpatialSummaryTests(unittest.TestCase):
    """Check that unavailable geometry is counted without erasing existing spatial metrics."""

    def test_skipped_lookup_is_counted_and_preserves_previous_metrics(self):
        """Continue semantic/type processing while reporting the failed geometry lookup."""
        namespace = graph_namespace()
        cause = namespace['DisasterEvent'](event_description='cause').save()
        result = namespace['DisasterEvent'](event_description='result').save()
        cause.geo_argument.connect(namespace['GeoObject'](geo_object_name='A', geo_type='River'))
        result.geo_argument.connect(namespace['GeoObject'](geo_object_name='B', geo_type='Bridge'))
        cause.cause.connect(result)
        relation = cause.spatial_to.connect(result, {'relations': {'distance_computation': 10.0}})
        connector = SimpleNamespace(database_name='selected', connect_database=Mock())
        pair = {'event_pair': [cause.node_id, result.node_id], 'osm_ids_pair': ['R1', 'W2']}
        namespace.update(is_geo_location=lambda: ['A', 'B'],
                         get_setting=lambda key: {'type_database': 'geos-type-1'},
                         event_pair_geo_encoding_parallel=lambda *args: [pair],
                         analyze_spatial_relation=Mock(return_value={}),
                         find_geo_semantics_path=lambda *args: [], find_geo_type_path=lambda *args: [])
        load_function('src/geo_extraction.py', '_populate_geo_type_paths', namespace)
        run = load_function('src/geo_extraction.py', 'main_calculate_spatial_relations', namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run(connector)
        self.assertEqual(summary, {'total_pairs': 1, 'eligible_pairs': 1, 'computed_pairs': 0,
                                   'skipped_pairs': 1, 'missing_coordinate_pairs': 0})
        self.assertEqual(relation.relations['distance_computation'], 10.0)
        self.assertIn('type_path', relation.relations)

    def test_successful_lookup_is_counted(self):
        """Count and save a successful computation without changing the semantic pipeline."""
        namespace = graph_namespace()
        cause = namespace['DisasterEvent'](event_description='cause').save()
        result = namespace['DisasterEvent'](event_description='result').save()
        cause.geo_argument.connect(namespace['GeoObject'](geo_object_name='A', geo_type='River'))
        result.geo_argument.connect(namespace['GeoObject'](geo_object_name='B', geo_type='Bridge'))
        cause.cause.connect(result)
        connector = SimpleNamespace(database_name='selected', connect_database=Mock())
        pair = {'event_pair': [cause.node_id, result.node_id], 'osm_ids_pair': ['R1', 'W2']}
        namespace.update(is_geo_location=lambda: ['A', 'B'],
                         get_setting=lambda key: {'type_database': 'geos-type-1'},
                         event_pair_geo_encoding_parallel=lambda *args: [pair],
                         analyze_spatial_relation=lambda *args: {'topological_relations': {},
                                                               'distance_meters': 12.0, 'basin_relations': {}},
                         get_topological_relation=lambda value: 'Adjacent',
                         find_geo_semantics_path=lambda *args: [], find_geo_type_path=lambda *args: [])
        load_function('src/geo_extraction.py', '_populate_geo_type_paths', namespace)
        run = load_function('src/geo_extraction.py', 'main_calculate_spatial_relations', namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run(connector)
        self.assertEqual(summary['computed_pairs'], 1)
        self.assertEqual(summary['skipped_pairs'], 0)
        self.assertEqual(cause.spatial_to.relationship(result).relations['distance_computation'], 12.0)

    def test_entry_point_reports_incomplete_computation(self):
        """Return failure rather than printing full success when an eligible pair was skipped."""
        namespace = dict(time=SimpleNamespace(time=lambda: 0), print=Mock(),
                         is_geo_location_parallel=Mock(return_value=[]),
                         event_pair_geo_encoding_parallel=Mock(), get_setting=lambda key: True,
                         main_calculate_spatial_relations=Mock(return_value={'skipped_pairs': 1}))
        run = load_function('main/main_extraction.py', 'run_spatial_computation', namespace)
        self.assertFalse(run(2, object()))
        self.assertFalse(any('completed in:' in str(call) for call in namespace['print'].call_args_list))


if __name__ == '__main__':
    unittest.main()
