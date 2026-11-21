"""离线空间测试：python -m unittest test_instantiation_oelg.py -v"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from shapely.geometry import box, mapping
from instantiation_oelg import (
    OELGInstantiationBuilder, screen_causal_graph, uid, load_instantiation_knowledge,
)


class OELGInstantiationTests(unittest.TestCase):
    def test_only_direct_standard_factor_query_and_canonical_names(self):
        from instantiation_oelg import OELGKnowledgeRepository
        from standard_observation_factors import STANDARD_FACTORS, canonical_factors
        self.assertEqual(len(STANDARD_FACTORS),25)
        self.assertEqual(canonical_factors(['Crack   Width','crack width','Tilt   Angle']),['Crack Width','Tilt Angle'])
        with self.assertRaises(ValueError):
            canonical_factors(['Subsidence'])
        with self.assertRaises(ValueError):
            canonical_factors(['水位'])
        repository=OELGKnowledgeRepository.__new__(OELGKnowledgeRepository)
        queries=[]
        def query(database,cypher):
            queries.append(cypher)
            return [dict(impact_type='Flood',node_id='obs',description='observe',
                         factors=['Water   Level','Water Level'],geo_types=['Road'],candidate_sensors=[]),
                    dict(impact_type='Flood',node_id='empty',description='no standard',
                         factors=[],geo_types=['Road'],candidate_sensors=[])]
        repository.query=query
        observations=repository.read_observation_knowledge('unused')
        self.assertEqual(observations['Flood'][0]['factors'],['Water Level'])
        self.assertEqual(len(observations['Flood']),1)
        self.assertIn('(factor:StandardFactor)',queries[0])
        self.assertIn('(impact:Impact)',queries[0])
        self.assertIn('(obs:GeneralizedObservationEvent)',queries[0])
        self.assertIn('impact.impact_type',queries[0])
        self.assertNotIn('event.event_description',queries[0])
        self.assertNotIn('MAPPING',queries[0])
        self.assertNotIn('GeoObject',queries[0])
        self.assertEqual(observations['Flood'][0]['source_geo_types'],[])
        self.assertEqual(observations['Flood'][0]['geo_types'],[])

    def test_latest_oelg_argument_types_are_used_from_one_database(self):
        from instantiation_oelg import OELGKnowledgeRepository
        repository = OELGKnowledgeRepository.__new__(OELGKnowledgeRepository)
        dataset = 'predefined_causal_temporal_v1'
        ids = {name: uid('predefined-causal-matrix', dataset, name) for name in ('Rain','Flood')}
        repository.query = Mock(side_effect=[
            [dict(node_id=ids['Rain'],name='Rain',geo_types=[]),
             dict(node_id=ids['Flood'],name='Flood',geo_types=['Road','New Appendix Type']),
             dict(node_id='unrelated',name='Other',geo_types=['Foreign Type'])],
            [dict(source_id=ids['Rain'],target_id=ids['Flood'],source='Rain',
                  target='Flood',relation_type='immediate')]])
        config = dict(self.config, abstract_database='oelg-1', dataset=dataset)
        graph = load_instantiation_knowledge(repository, config)
        self.assertEqual(graph['nodes']['Flood']['geo_types'],['New Appendix Type','Road'])
        self.assertNotIn('Other',graph['nodes'])
        self.assertEqual([call.args[0] for call in repository.query.call_args_list],['oelg-1','oelg-1'])
        self.assertIn('(event)-[mapping:ARGUMENT]->(geo:GeoObject)',repository.query.call_args_list[0].args[1])
        self.assertIn('coalesce(mapping.oelg_dbbody_enabled, true)',repository.query.call_args_list[0].args[1])

    def test_reviewed_out_phenomenon_is_reported_without_geographic_fallback(self):
        self.knowledge['nodes']['Flood'].update(geo_types=[],dbbody_review_complete=True)
        builder = OELGInstantiationBuilder(self.config,self.knowledge)
        self.assertEqual(len(builder.instantiate()),1)
        self.assertEqual(builder.pruned[0]['reason'],'no_reviewed_dbbody_mapping')

    def test_cached_geography_is_replaced_by_current_oelg_mapping(self):
        cached = json.loads(json.dumps(self.knowledge))
        cached['nodes']['Flood']['geo_types'] = ['Lake']
        path = self.folder/'stale_knowledge.json'
        path.write_text(json.dumps({'knowledge':cached}))
        config = dict(self.config, knowledge_json=str(path), abstract_database='oelg-1',dataset='current')
        repository = Mock()
        repository.read_abstract_graph.return_value = self.knowledge
        result = load_instantiation_knowledge(repository, config)
        self.assertEqual(result['nodes']['Flood']['geo_types'],['Road'])
        self.assertEqual(result['edges'],cached['edges'])
        repository.read_abstract_graph.assert_called_once_with('oelg-1','current')
        repository.read_abstract_graph.return_value = {'nodes':{'Rain':{'geo_types':[]}},'edges':[]}
        with self.assertRaisesRegex(ValueError,'OELG 图谱缺少'):
            load_instantiation_knowledge(repository,config)

    def test_observation_geotypes_do_not_change_geography_or_factor_matching(self):
        from instantiation_oelg import build_factor_to_disaster
        builder = OELGInstantiationBuilder(self.config,self.knowledge,{'Flood':[
            dict(node_id='lake-observation',geo_types=['Lake'],factors=['Water Level'])]})
        nodes = builder.instantiate()
        self.assertEqual({f['geo_type'] for f in nodes[1]['affect_features']},{'Road'})
        self.assertEqual(build_factor_to_disaster(nodes),{'Water Level':['Road,Flood']})

    def test_missing_oelg_geotypes_do_not_fall_back_to_observation_graph(self):
        self.knowledge['nodes']['Flood']['geo_types'] = []
        builder = OELGInstantiationBuilder(self.config,self.knowledge,{'Flood':[
            dict(node_id='other-graph',geo_types=['Road'],factors=['Water Level'])]})
        with self.assertRaisesRegex(ValueError,'OELG 库 oelg-1.*ARGUMENT'):
            builder.instantiate()

    def test_all_impact_tasks_retained_despite_unselected_source_geography(self):
        builder=OELGInstantiationBuilder(self.config,self.knowledge,{'Flood':[
            dict(node_id='from-lake',source_geo_type='Lake',geo_types=[],factors=['Water Level']),
            dict(node_id='from-mountain',source_geo_type='Mountain',geo_types=[],factors=['Tilt Angle'])]})
        node=builder.instantiate()[1]
        self.assertEqual({f for task in node['observation_tasks'] for f in task['factors']},
                         {'Water Level','Tilt Angle'})

    def test_raw_factor_snapshot_rejected_and_standard_tasks_preserved(self):
        with self.assertRaises(ValueError):
            OELGInstantiationBuilder(self.config,self.knowledge,{'Flood':[{'factors':['水位']}]})
        builder=OELGInstantiationBuilder(self.config,self.knowledge,{'Flood':[dict(
            node_id='obs',source_geo_type='Road',geo_types=['Road'],factors=['Water   Level'])]})
        node=builder.instantiate()[1]
        self.assertEqual(node['observation_tasks'][0]['factors'],['Water Level'])
        self.assertEqual(node['observation_tasks'][0]['factor_schema'],'StandardFactor')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.region = box(113.5, 34.7, 113.6, 34.8)
        self.rain = box(113.5, 34.7, 113.55, 34.8)
        self.write('region.geojson', self.region)
        self.write('rain.geojson', self.rain)
        self.write('mask.geojson', box(113.5, 34.7, 113.6, 34.75))
        features = [
            {'type': 'Feature', 'geometry': mapping(box(115, 36, 116, 37)), 'properties': {'slope': 1}},
            {'type': 'Feature', 'geometry': mapping(box(113.51, 34.71, 113.53, 34.73)), 'properties': {'slope': 1}},
            {'type': 'Feature', 'geometry': mapping(box(113.51, 34.76, 113.53, 34.78)), 'properties': {'slope': 1}},
            {'type': 'Feature', 'geometry': mapping(box(113.56, 34.71, 113.58, 34.73)), 'properties': {'slope': 1}},
        ]
        (self.folder / 'Road.json').write_text(json.dumps({'type': 'FeatureCollection', 'features': features}))
        self.config = dict(region=str(self.folder / 'region.geojson'), rainfall=str(self.folder / 'rain.geojson'),
                           features_folder=str(self.folder), projected_crs='EPSG:3857', feature_buffer_m=0,
                           environment_mode='strict', environment_rules={'Flood': {'mask': str(self.folder / 'mask.geojson')}},
                           root_phenomenon='Rain', max_depth=2, snapshot='test')
        self.knowledge = {'nodes': {'Rain': {'geo_types': []}, 'Flood': {'geo_types': ['Road']}},
                          'edges': [{'source': 'Rain', 'target': 'Flood', 'relation_type': 'immediate'}]}

    def write(self, filename, geometry):
        (self.folder / filename).write_text(json.dumps(mapping(geometry)))

    def test_parent_environment_intersection_and_original_id(self):
        builder = OELGInstantiationBuilder(self.config, self.knowledge)
        nodes = builder.instantiate()
        self.assertEqual(len(nodes), 2)
        self.assertEqual([f['osm_id'] for f in nodes[1]['affect_features']], ['Road_1'])
        self.assertTrue(nodes[1]['affect_extent'].difference(builder.rain).area < 1e-6)
        self.assertEqual(nodes[1]['priority'], 2)
        self.assertNotIn('probability', nodes[1])

    def test_strict_missing_environment_fails(self):
        self.config['environment_rules'] = {}
        with self.assertRaisesRegex(ValueError, 'DPEnvironment'):
            OELGInstantiationBuilder(self.config, self.knowledge).instantiate()

    def test_proxy_is_explicit(self):
        self.config.update(environment_mode='proxy', environment_rules={})
        builder = OELGInstantiationBuilder(self.config, self.knowledge)
        nodes = builder.instantiate()
        self.assertEqual({f['osm_id'] for f in nodes[1]['affect_features']}, {'Road_1', 'Road_2'})
        self.assertEqual(builder.proxy_phenomena, {'Flood'})

    def test_environment_attribute_missing_fails(self):
        self.config['environment_rules'] = {'Flood': {'all': [{'field': 'missing', 'op': 'lt', 'value': 3}]}}
        with self.assertRaisesRegex(ValueError, '缺少地物属性'):
            OELGInstantiationBuilder(self.config, self.knowledge).instantiate()

    def test_empty_environment_prunes_branch(self):
        self.config['environment_rules'] = {'Flood': {'all': [{'field': 'slope', 'op': 'gt', 'value': 3}]}}
        builder = OELGInstantiationBuilder(self.config, self.knowledge)
        self.assertEqual(len(builder.instantiate()), 1)
        self.assertEqual(builder.pruned[0]['reason'], 'no_occurring_conditions_in_study_area')

    def test_siblings_and_cycles(self):
        graph = {'nodes': {name: {} for name in ['Rain', 'A', 'B', 'C']}, 'edges': [
            {'source': a, 'target': b, 'relation_type': 'Altering'}
            for a, b in [('Rain', 'A'), ('Rain', 'B'), ('A', 'B'), ('B', 'A'), ('A', 'A'), ('A', 'C')]]}
        edges, removed = screen_causal_graph(graph, 'Rain', 2)
        self.assertEqual({(e['source'], e['target']) for e in edges}, {('Rain', 'A'), ('Rain', 'B'), ('A', 'C')})
        self.assertEqual(len(removed), 3)

    def test_stable_matrix_id(self):
        self.assertEqual(uid('predefined-causal-matrix', 'predefined_causal_temporal_v1', 'Water Level Rise'),
                         '0412d41fe8185ee1893bb255fe9cf31d')

    def test_importance_geotiff_preserves_values_zero_and_mask(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds
        from instantiation_oelg import export_importance_geotiff
        grid = np.array([[100.0, 0.0], [27.5, 999.0]])
        valid = np.array([[True, True], [True, False]])
        affine = from_bounds(*self.region.bounds, 2, 2)
        path = export_importance_geotiff(grid, valid, affine, 'Subsidence',
                                        self.folder/'events', 'phenomenon', 5)
        with rasterio.open(path) as src:
            self.assertEqual(src.crs.to_epsg(), 4326)
            self.assertEqual(src.transform, affine)
            self.assertEqual(src.nodata, -9999)
            np.testing.assert_array_equal(src.read(1), [[100,0],[27.5,-9999]])
            self.assertFalse(src.read(1,masked=True).mask[0,1])
            self.assertEqual(src.tags()['blur_sigma'], '5')
        np.testing.assert_array_equal(grid, [[100,0],[27.5,999]])

    def test_offline_event_and_factor_paths_without_png(self):
        import rasterio
        from instantiation_oelg import plot_disaster_risk_maps, plot_observation_factor_maps
        self.config.update(risk_grid_size=20, risk_blur_sigma=1, risk_render_png=False,
                           risk_maps_folder=str(self.folder/'event_maps'),
                           factor_maps_folder=str(self.folder/'factor_maps'))
        features = {'Road': {'Road_1': [{'Road,Subsidence':100.0}]}}
        event = plot_disaster_risk_maps(features,self.config)[0]
        factor = plot_observation_factor_maps(features,{'Surface Deformation':['Road,Subsidence']},self.config)[0]
        self.assertEqual(Path(event),self.folder/'event_maps/geotiff/Subsidence_importance.tif')
        self.assertEqual(Path(factor),self.folder/'factor_maps_geotiff/Surface Deformation_importance_field.tif')
        with rasterio.open(event) as left, rasterio.open(factor) as right:
            self.assertTrue((left.read(1)==right.read(1)).all())
            self.assertGreater(left.read(1).max(),0)
        self.assertFalse(list(self.folder.rglob('*.png')))

    def test_identical_reference_raster_has_unit_iou(self):
        import rasterio
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        from shapely.ops import transform
        from instantiation_oelg import export_rasters_and_compare
        import matplotlib
        matplotlib.use('Agg')
        builder = OELGInstantiationBuilder(self.config, self.knowledge)
        nodes = builder.instantiate()
        references = self.folder / 'reference'
        references.mkdir()
        affine = from_bounds(*self.region.bounds, 20, 20)
        geometry = transform(builder.features.to_wgs84, nodes[1]['affect_features'][0]['geometry'])
        expected = rasterize([(geometry, 10)], out_shape=(20, 20), transform=affine, dtype='float32')
        with rasterio.open(references / 'Flood_importance.tif', 'w', driver='GTiff', count=1,
                           width=20, height=20, dtype='float32', crs='EPSG:4326',
                           transform=affine, nodata=-9999) as dst:
            dst.write(expected, 1)
        self.config.update(grid_resolution_m=50, all_touched=False, doelg_rasters=str(references),
                           doelg_threshold=0, comparison_grid='doelg')
        result = export_rasters_and_compare(builder, {n['phenomenon']: n['affect_extent'] for n in nodes},
                                           self.folder, self.config)
        row = next(r for r in result if r['phenomenon'] == 'Flood')
        self.assertEqual(row['iou'], 1)
        self.assertGreater(row['intersection_km2'], 0)
        self.assertEqual(row['oelg_only_km2'], 0)
        self.assertEqual(row['reference_valid_fraction'], 1)


if __name__ == '__main__':
    unittest.main()
