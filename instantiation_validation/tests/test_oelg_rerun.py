import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import instantiation_oelg as module


class RepeatRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / 'audit'
        self.config = dict(instance_database='instances', abstract_database='oelg',
                           observation_database='observations', snapshot='original',
                           output_folder=str(self.folder))
        self.repository = Mock()
        self.repository.query.return_value = [dict(snapshots=0, events=0)]

    def test_first_run_keeps_requested_name_and_empty_directory(self):
        self.folder.mkdir()
        self.assertEqual(module.prepare_run_destination(self.config, self.repository), self.config)

    def test_nonempty_audit_gets_new_matching_snapshot_and_folder(self):
        self.folder.mkdir()
        (self.folder/'old.json').write_text('old')
        with patch.object(module, 'run_suffix', return_value='new-run'):
            result = module.prepare_run_destination(self.config, self.repository)
        self.assertEqual(result['snapshot'], 'original-new-run')
        self.assertEqual(Path(result['output_folder']), self.root/'audit-new-run')
        self.assertEqual((self.folder/'old.json').read_text(), 'old')
        self.assertEqual(self.config['snapshot'], 'original')

    def test_database_snapshot_or_legacy_event_collision_with_new_folder(self):
        for existing in (dict(snapshots=1, events=0), dict(snapshots=0, events=3)):
            self.repository.query.side_effect = [[existing], [dict(snapshots=0, events=0)]]
            with self.subTest(existing=existing), patch.object(module, 'run_suffix', return_value='retry'):
                result = module.prepare_run_destination(self.config, self.repository)
            self.assertEqual(result['snapshot'], 'original-retry')
            self.assertEqual(Path(result['output_folder']), self.root/'audit-retry')

    def test_cannot_check_database_never_assumes_name_is_available(self):
        self.repository.query.side_effect = RuntimeError('connection failed')
        with self.assertRaisesRegex(RuntimeError, 'connection failed'):
            module.prepare_run_destination(self.config, self.repository)
        self.assertFalse(self.folder.exists())

    def test_map_directory_collision_keeps_files_and_retries(self):
        self.folder.mkdir()
        (self.folder/'old.png').write_bytes(b'old')
        (self.root/'audit-first').mkdir()
        (self.root/'audit-first/file').write_text('occupied')
        with patch.object(module, 'run_suffix', side_effect=['first', 'second']):
            chosen = module.available_output_folder(self.folder)
        self.assertEqual(chosen, self.root/'audit-second')
        self.assertEqual((self.folder/'old.png').read_bytes(), b'old')

    def test_main_uses_new_destination_for_builder_database_manifest_and_maps(self):
        self.folder.mkdir()
        (self.folder/'old.json').write_text('old')
        maps = self.root/'maps'
        maps.mkdir()
        (maps/'old.png').write_bytes(b'old')
        map_config = self.root/'maps.json'
        map_config.write_text(json.dumps(dict(output_dir='maps')))
        for name in ('rain.json', 'region.json'):
            (self.root/name).write_text('{}')
        self.config.update(rainfall=str(self.root/'rain.json'), region=str(self.root/'region.json'),
                           snapshot_time='2021-07-20T14:00:00+08:00', distance_map_config='maps.json')
        graph = dict(nodes={}, edges=[])
        self.repository.read_observation_knowledge.return_value = {}
        builder = SimpleNamespace(features=SimpleNamespace(input_hashes={}), instantiate=Mock())
        def create_builder(config, knowledge, observations):
            builder.config = config
            return builder
        def export(folder):
            folder.mkdir(parents=True)
            return {}
        builder.export_results = Mock(side_effect=export)
        plot = Mock()
        with (patch.object(sys, 'argv', ['instantiation_oelg.py', '--config', str(self.root/'config.json')]),
              patch.object(module, 'load_config', return_value=self.config),
              patch.object(module, 'OELGKnowledgeRepository', return_value=self.repository),
              patch.object(module, 'load_instantiation_knowledge', return_value=graph),
              patch.object(module, 'OELGInstantiationBuilder', side_effect=create_builder),
              patch.object(module, 'save_instance_database', return_value={'event_count': 1}) as write,
              patch.object(module, 'run_suffix', side_effect=['new-audit', 'new-map']),
              patch.dict(sys.modules, {'plot_oelg_distance_fields': SimpleNamespace(run=plot)})):
            module.main()
        self.assertEqual(builder.config['snapshot'], 'original-new-audit')
        self.assertEqual(Path(builder.config['output_folder']), self.root/'audit-new-audit')
        write.assert_called_once_with(self.repository, builder, 'instances')
        manifest = json.loads((self.root/'audit-new-audit/run_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['snapshot'], builder.config['snapshot'])
        self.assertEqual(plot.call_args.args[1], self.root/'maps-new-map')
        self.assertEqual(plot.call_args.kwargs['oelg_override']['snapshot'], 'original-new-audit')
        self.repository.driver.close.assert_called_once()
        self.assertEqual((self.folder/'old.json').read_text(), 'old')
        self.assertEqual((maps/'old.png').read_bytes(), b'old')


if __name__ == '__main__':
    unittest.main()
