import copy
from types import SimpleNamespace
import unittest
from shapely.geometry import box
from shapely import wkt
from oelg_graph_store import prepare_rows, read_graph, write_builder


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


if __name__=='__main__':
    unittest.main()
