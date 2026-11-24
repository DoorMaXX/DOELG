import unittest
from copy import deepcopy
from compare_importance_fields import pair_results, METRICS, filter_results, selected_factor_fields
from evaluate_importance_fields import field_point_label, normalize_point_set_config


def result(run,field='Water Level',domain='domain',sample='points'):
    return dict(run=run,kind='factor',field=field,point_set='waterlogging',used_points=22,
                result_dir=run,domain_hash=domain,sample_hash=sample,area_mode='geodesic',
                **{key:.7 if run=='OELG' else .6 for key in METRICS})


class ComparisonTests(unittest.TestCase):
    def test_hazard_specific_selection_and_whitespace(self):
        config={'point_sets':{'collapse':{},'waterlogging':{}}}
        rows=[dict(result('DOELG',field),point_set=point) for field,point in [
            ('Subsidence Depth','collapse'),('Crack   Width','collapse'),('Surface Deformation','collapse'),
            ('Tilt   Angle','collapse'),('Structural Characteristics','collapse'),('Water Level','collapse'),('Subsidence','collapse'),
            ('Precipitation','waterlogging'),('Soil Moisture','waterlogging'),
            ('Water Body Area','waterlogging'),('Water Level','waterlogging'),('Tilt Angle','waterlogging')]]
        rows.append(dict(result('DOELG','Subsidence'),kind='event',point_set='collapse'))
        selected=filter_results(rows,config)
        self.assertEqual(len(selected),10)
        self.assertFalse(any(r['kind']=='factor' and r['field']=='Subsidence' for r in selected))
        self.assertFalse(any(r['point_set']=='Subsidence' and r['field']=='Water Level' for r in selected))
        self.assertNotIn('collapse',{r['point_set'] for r in selected})
        self.assertEqual(len(selected_factor_fields(config)['Subsidence']),5)

    def test_legacy_config_normalization_preserves_paths_and_selection(self):
        config = {'point_sets':{'collapse':{'path':'custom/points.shp'}},
                  'event_pairs':[{'file':'Subsidence_importance.tif','points':'collapse'}],
                  'factor_fields_by_point_set':{'collapse':['Crack Width']}}
        original = deepcopy(config)
        normalized = normalize_point_set_config(config)
        self.assertEqual(config,original)
        self.assertEqual(normalized['point_sets'],{'Subsidence':{'path':'custom/points.shp'}})
        self.assertEqual(normalized['event_pairs'][0]['points'],'Subsidence')
        self.assertEqual(selected_factor_fields(normalized),{'Subsidence':['Crack Width']})

    def test_duplicate_point_set_aliases_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate point-set aliases'):
            normalize_point_set_config({'point_sets':{'collapse':{},'Subsidence':{}}})

    def test_legacy_and_canonical_results_pair(self):
        rows = [dict(result('DOELG','Crack Width'),point_set='collapse'),
                dict(result('OELG','Crack Width'),point_set='Subsidence')]
        pair = pair_results(rows,'DOELG','OELG')[0]
        self.assertEqual(pair['status'],'matched')
        self.assertEqual(pair['point_set'],'Subsidence')

    def test_display_labels_distinguish_hazards_without_repetition(self):
        self.assertEqual(field_point_label('Subsidence','collapse'),'Subsidence')
        self.assertEqual(field_point_label('Crack Width','Subsidence'),'Crack Width | Subsidence')
        self.assertEqual(field_point_label('Structural Collapse','Subsidence'),
                         'Structural Collapse | Subsidence')

    def test_normalized_names_and_delta_direction(self):
        pair = pair_results([result('DOELG'),result('OELG','WaterLevel')],'DOELG','OELG')[0]
        self.assertEqual(pair['status'],'matched')
        self.assertAlmostEqual(pair['delta_AUC_PA'],.1)
        self.assertAlmostEqual(pair['delta_Area_at_80pct_hit'],.1)

    def test_unmatched_fields_are_not_zero_filled(self):
        rows = pair_results([result('DOELG'),result('OELG','Precipitation')],'DOELG','OELG')
        self.assertEqual({r['status'] for r in rows},{'reference_only','candidate_only'})
        self.assertTrue(all(r['delta_AUC_PA'] is None for r in rows))

    def test_different_samples_or_domains_are_not_compared(self):
        for row in [result('OELG',domain='another'),result('OELG',sample='another')]:
            pair = pair_results([result('DOELG'),row],'DOELG','OELG')[0]
            self.assertEqual(pair['status'],'incompatible_domain_or_samples')
            self.assertIsNone(pair['delta_AUC_PA'])

    def test_ambiguous_normalized_names_rejected(self):
        with self.assertRaises(ValueError):
            pair_results([result('DOELG'),result('DOELG','WaterLevel'),result('OELG')],'DOELG','OELG')


if __name__=='__main__':
    unittest.main()
