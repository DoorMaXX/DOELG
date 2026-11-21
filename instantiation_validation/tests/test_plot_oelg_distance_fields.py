import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from shapely.geometry import box,mapping

import distance_fields as fields
from plot_oelg_distance_fields import score_events,build_snapshot,run


class OELGDistanceTests(unittest.TestCase):
    def setUp(self):
        self.events = {'type':'FeatureCollection','features':[
            self.node('root','Rain',0),
            self.node('a','Flood',1,'root','immediate'),
            self.node('b','Flood',1,'root','Altering'),
            self.node('aa','Subsidence',2,'a','immediate'),
            self.node('ab','Subsidence',2,'a','Altering'),
            self.node('ba','Subsidence',2,'b','immediate'),
            self.node('bb','Subsidence',2,'b','Altering')]}

    def node(self,key,name,depth,parent=None,relation=None):
        return dict(type='Feature',geometry=None,properties=dict(id=key,phenomenon=name,
                    current_depth=depth,parent_id=parent,relation_type=relation))

    def row(self,event,name,depth,key='Road_0',geo='Road'):
        return dict(event_id=event,phenomenon=name,depth=depth,osm_id=key,geo_type=geo,eligible=1)

    def test_depth_scores_ignore_timing(self):
        scored = score_events(self.events)
        for event in scored.values():
            self.assertEqual(event['score'],100*.5**max(0,event['depth']-1))
        self.assertEqual(scored['ba']['delayed_edges'],1)
        self.assertEqual(scored['ba']['immediate_edges'],1)
        self.assertEqual(scored['bb']['delayed_edges'],2)
        reverse = score_events(dict(self.events,features=list(reversed(self.events['features']))))
        self.assertEqual(scored,reverse)

    def test_configurable_depth_decay_and_invalid_values(self):
        extended = dict(self.events,features=self.events['features']+[
            self.node('aaa','Damage',3,'aa','Altering')])
        scored = score_events(extended,.6)
        self.assertEqual(scored['a']['score'],100)
        self.assertEqual(scored['b']['score'],100)
        for key in ('aa','ab','ba','bb'):
            self.assertEqual(scored[key]['score'],60)
        self.assertEqual(scored['aaa']['score'],36)
        for value in (0,1,-.1,1.1,float('nan'),float('inf')):
            with self.assertRaisesRegex(ValueError,'depth_decay'):
                score_events(self.events,value)

    def test_shallow_path_wins_but_factor_keeps_eligible_depth(self):
        events = dict(self.events,features=self.events['features']+[
            self.node('deep','Flood',2,'b','Altering')])
        scored = score_events(events)
        rows = [self.row('deep','Flood',2),self.row('a','Flood',1)]
        tasks = [dict(disaster_id='deep',factors=['Subsidence Depth'],geo_types=['Road'])]
        snapshot = build_snapshot(scored,rows,tasks)
        record = snapshot['records'][0]
        self.assertEqual(record['score'],100)
        self.assertEqual(record['best_depth'],1)
        self.assertEqual(record['factor_scores'],{'Subsidence Depth':50})

    def test_invalid_tree_fails(self):
        for bad in (self.node('bad','Bad',1,'missing','immediate'),
                    self.node('bad','Bad',2,'root','immediate'),
                    self.node('bad','Bad',1,'root','unknown'),
                    self.node('root','Rain',0)):
            with self.assertRaises(ValueError):
                score_events(dict(self.events,features=self.events['features']+[bad]))

    def test_paths_deduplicated_and_factor_scores_remain_instance_specific(self):
        scored = score_events(self.events)
        rows = [self.row('a','Flood',1),self.row('b','Flood',1),self.row('b','Flood',1)]
        tasks = [dict(disaster_id='b',factors=['Water Level','Water Level'],geo_types=['Road']),
                 dict(disaster_id='a',factors=['Soil Moisture'],geo_types=['Lake'])]
        snapshot = build_snapshot(scored,rows,tasks)
        record = snapshot['records'][0]
        self.assertEqual(record['score'],100)
        self.assertEqual(record['n_paths'],2)
        self.assertEqual(record['factor_scores'],{'Water Level':100})
        region = box(113.59,34.74,113.61,34.76)
        config = dict(default_cutoff_m=1000,distance_crs='EPSG:32649',grid_size=3,event_hazard_weights={})
        grid = fields.setup_grid(config,region)
        event,factor,_ = fields.construct_fields(snapshot,{'Road_0':region},grid,config)
        np.testing.assert_array_equal(event['Flood'],100)
        np.testing.assert_array_equal(factor['Water Level'],100)
        self.assertNotIn('Soil Moisture',factor)

    def test_unknown_references_and_feature_mismatch_fail(self):
        scored = score_events(self.events)
        for row in (self.row('missing','Flood',1),self.row('a','Flood',2),self.row('a','Wrong',1)):
            with self.assertRaises(ValueError):
                build_snapshot(scored,[row],[])
        with self.assertRaises(ValueError):
            build_snapshot(scored,[self.row('a','Flood',1)],[dict(disaster_id='missing')])

    def test_graph_end_to_end_without_local_instance_or_geometry_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'
            source.mkdir()
            region = box(113.59,34.74,113.61,34.76)
            def save(name,value):
                (root/name).write_text(json.dumps(value),encoding='utf-8')
            save('region.geojson',mapping(region))
            tasks=[dict(disaster_id='a',factors=['Water Level'],geo_types=['Road'])]
            rows = [self.row('a','Flood',1),self.row('aa','Subsidence',2)]
            save('oelg.json',dict(instance_database='test-oelg',snapshot='example',region='region.geojson'))
            save('maps.json',dict(default_cutoff_m=1000,distance_crs='EPSG:32649',grid_size=3,
                                  render_png=False,spatial_aggregation='log_overlap',overlap_strength=.25))
            save('config.json',dict(oelg_config='oelg.json',map_config='maps.json',output_dir='output'))
            provenance=dict(database='test-oelg',snapshot='example',run_signature='test-signature')
            with patch('plot_oelg_distance_fields.load_graph',return_value=(
                    self.events,rows,tasks,{'Road_0':region},provenance)) as graph:
                manifest = run(root/'config.json')
                graph.assert_called_once()
            self.assertEqual(manifest['graph_source'],provenance)
            self.assertEqual(len(manifest['outputs']),3)
            self.assertEqual(manifest['display_scale']['maximum_by_kind'],{'event':100,'factor':100})
            self.assertEqual(manifest['map_config']['plot_vmax_by_kind'],{'event':100,'factor':100})
            self.assertEqual(manifest['map_config']['geo_type_weight_by_type'],{})
            for item in manifest['outputs']:
                expected = 50 if item['field']=='Subsidence' else 100
                with rasterio.open(item['geotiff']) as src:
                    np.testing.assert_array_equal(src.read(1),expected)
                    self.assertEqual(src.tags()['aggregation'],'log_overlap')
                    self.assertEqual(src.tags()['source_method'],'oelg_depth_only')
                    self.assertEqual(src.tags()['depth_decay'],'0.5')
                    self.assertEqual(src.tags()['temporal_weighting'],'none')
                    self.assertEqual(src.tags()['source_database'],'test-oelg')
                    self.assertEqual(src.crs.to_epsg(),4326)
            self.assertEqual(fields.read_json(root/'output/run_status.json')['status'],'complete')
            with self.assertRaises(FileExistsError):
                run(root/'config.json')

    def test_database_error_propagates_without_export_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'oelg.json').write_text(json.dumps(dict(region='region.geojson',instance_database='missing',snapshot='s')))
            (root/'maps.json').write_text(json.dumps(dict(default_cutoff_m=1000)))
            (root/'config.json').write_text(json.dumps(dict(oelg_config='oelg.json',map_config='maps.json',output_dir='output')))
            with patch('plot_oelg_distance_fields.load_graph',side_effect=RuntimeError('database unavailable')):
                with self.assertRaisesRegex(RuntimeError,'database unavailable'):
                    run(root/'config.json')


if __name__=='__main__':
    unittest.main()

