from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock
from shapely.geometry import box
from shapely import wkt
from oelg_graph_store import prepare_rows, read_graph, write_builder, resolve_snapshot


class GraphStoreTests(unittest.TestCase):
    def test_geometry_is_stored_once_with_all_event_links_and_task_provenance(self):
        geometry=box(113.5,34.7,113.6,34.8)
        feature=dict(osm_id='Road_0',geo_type='Road',geometry=geometry)
        event=dict(id='event',event_description='Flood',phenomenon='Flood',current_depth=1,
                   priority=2,is_happen=False,parent_id='root',path=['Rain','Flood'],
                   relation_type='immediate',affect_extent=geometry,affect_features=[feature],
                   observation_tasks=[dict(node_id='knowledge-obs',description='observe',
                     factors=['Water Level'],source_event_ids=['global1','global2'],
                     source_geo_types=['Lake'],geo_types=[])])
        second=dict(event,id='event2',current_depth=2,parent_id='event',priority=3)
        builder=SimpleNamespace(config={'snapshot':'one'},nodes=[event,second],
                                features=SimpleNamespace(to_wgs84=lambda x,y:(x,y)))
        events,features,links,tasks=prepare_rows(builder)
        self.assertEqual((len(events),len(features),len(links),len(tasks)),(2,1,2,2))
        self.assertEqual(tasks[0]['source_event_ids'],['global1','global2'])
        self.assertEqual(tasks[0]['geo_types'],[])
        self.assertTrue(wkt.loads(features[0]['geometry']).equals(geometry))
        builder.config['snapshot']='two'
        other=prepare_rows(builder)[1][0]
        self.assertNotEqual(features[0]['id'],other['id'])
        self.assertNotEqual(features[0]['geometry_id'],other['geometry_id'])

    def test_refuses_writing_to_knowledge_database_before_connecting(self):
        builder=SimpleNamespace(config={'abstract_database':'abstract','observation_database':'knowledge'})
        for database in ('abstract','knowledge'):
            with self.assertRaisesRegex(ValueError,'Instance database'):
                write_builder(None,builder,database)

    def test_missing_or_incomplete_graph_never_falls_back_to_files(self):
        class Result:
            def __init__(self,record): self.record=record
            def single(self): return self.record
        class Session:
            def __init__(self,record): self.record=record
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def execute_read(self,fn): return fn(self)
            def run(self,*args,**kwargs): return Result(self.record)
        for record in (None, {'s':{'status':'writing','schema_version':2}},
                       {'s':{'status':'complete','schema_version':1}}):
            driver=SimpleNamespace(session=lambda **kw:Session(record))
            with self.assertRaisesRegex(ValueError,'No complete OELG graph snapshot'):
                read_graph(driver,'oelg','snapshot')

    def test_automatic_snapshot_uses_latest_complete_current_schema_result(self):
        driver = MagicMock()
        session = driver.session.return_value.__enter__.return_value
        tx = MagicMock()
        tx.run.return_value.single.return_value = {'s': dict(name='completed-with-suffix', status='complete',
            schema_version=2, event_count=20, feature_count=200)}
        session.execute_read.side_effect = lambda callback: callback(tx)
        self.assertEqual(resolve_snapshot(driver, 'selected-instance-db'), 'completed-with-suffix')
        driver.session.assert_called_once_with(database='selected-instance-db', default_access_mode='READ')
        self.assertIsNone(tx.run.call_args.kwargs['snapshot'])
        self.assertEqual(tx.run.call_args.kwargs['schema'], 2)
        self.assertIn('s.completed_at', tx.run.call_args.args[0])
        self.assertIn('DESC', tx.run.call_args.args[0])
        session.execute_write.assert_not_called()

    def test_no_usable_result_identifies_database_and_required_task(self):
        for meta in (None, {'s': {}}, {'s': dict(name='old', status='complete', schema_version=1,
                     event_count=10, feature_count=20)}, {'s': dict(name='empty', status='complete',
                     schema_version=2, event_count=0, feature_count=20)}):
            driver = MagicMock()
            session = driver.session.return_value.__enter__.return_value
            tx = MagicMock()
            tx.run.return_value.single.return_value = meta
            session.execute_read.side_effect = lambda callback: callback(tx)
            with self.subTest(meta=meta), self.assertRaisesRegex(ValueError, "selected-instance-db.*5.1"):
                resolve_snapshot(driver, 'selected-instance-db')

    def test_explicit_advanced_name_is_validated_without_falling_back(self):
        driver = MagicMock()
        session = driver.session.return_value.__enter__.return_value
        tx = MagicMock()
        tx.run.return_value.single.return_value = None
        session.execute_read.side_effect = lambda callback: callback(tx)
        with self.assertRaisesRegex(ValueError, 'missing-explicit-result'):
            resolve_snapshot(driver, 'selected-instance-db', 'missing-explicit-result')
        self.assertEqual(tx.run.call_args.kwargs['snapshot'], 'missing-explicit-result')
        tx.run.assert_called_once()


if __name__=='__main__':
    unittest.main()
