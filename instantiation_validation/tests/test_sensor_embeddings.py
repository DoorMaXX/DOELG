"""Exercise tokenizer recovery and sensor-index initialization without external services."""

import ast
import concurrent.futures
import contextlib
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
from threading import Lock
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from langchain_core.documents import Document
import pandas as pd
from ollama import ResponseError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from other.sensor_embeddings import SensorEmbeddings


def load_functions(path, names, namespace):
    """Load selected production functions without initializing unrelated LLM clients."""
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(body) != len(names):
        raise AssertionError('A requested production function was not found.')
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class SensorEmbeddingTests(unittest.TestCase):
    """Check transient retries, permanent failures, ordering, and request serialization."""

    def setUp(self):
        """Suppress retry messages and create a reusable tokenizer connection error."""
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.error = ResponseError(
            'Post "http://127.0.0.1:8309/tokenize": dial tcp 127.0.0.1:8309: '
            'connectex: No connection could be made because the target machine actively refused it.',
            status_code=400,
        )

    def test_tokenizer_failure_recovers(self):
        """Retry the reported HTTP 400 tokenizer refusal and return the recovered vector."""
        client = Mock()
        client.embed_documents.side_effect = [self.error, [[1.0]]]
        with patch('other.sensor_embeddings.time.sleep') as sleep:
            result = SensorEmbeddings(client).embed_query('sensor')
        self.assertEqual(result, [1.0])
        self.assertEqual(client.embed_documents.call_count, 2)
        sleep.assert_called_once_with(2.0)

    def test_retry_limit_is_enforced(self):
        """Stop after the configured attempts and preserve the original failure as the cause."""
        client = Mock()
        client.embed_documents.side_effect = self.error
        with patch('other.sensor_embeddings.time.sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'after 3 attempts') as raised:
                SensorEmbeddings(client).embed_query('sensor')
        self.assertEqual(client.embed_documents.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2.0, 4.0])
        self.assertIs(raised.exception.__cause__, self.error)

    def test_permanent_request_errors_are_not_retried(self):
        """Propagate invalid inputs and missing models without retrying them."""
        for message, status in [('tokenize error: invalid input', 400), ('model not found', 404)]:
            with self.subTest(status=status):
                client = Mock()
                client.embed_documents.side_effect = ResponseError(message, status_code=status)
                with self.assertRaises(ResponseError):
                    SensorEmbeddings(client).embed_query('sensor')
                self.assertEqual(client.embed_documents.call_count, 1)

    def test_transport_timeout_recovers(self):
        """Treat a transport timeout as transient without retrying any graph writes."""
        client = Mock()
        client.embed_documents.side_effect = [httpx.ReadTimeout('timeout'), [[1.0]]]
        with patch('other.sensor_embeddings.time.sleep'):
            self.assertEqual(SensorEmbeddings(client).embed_query('sensor'), [1.0])

    def test_only_failed_batch_is_retried(self):
        """Keep vector ordering and retain earlier successful batches during recovery."""
        client = Mock()
        client.embed_documents.side_effect = [[[1], [2]], self.error, [[3], [4]]]
        with patch('other.sensor_embeddings.time.sleep'):
            vectors = SensorEmbeddings(client, batch_size=2).embed_documents(['a', 'b', 'c', 'd'])
        self.assertEqual(vectors, [[1], [2], [3], [4]])
        self.assertEqual([call.args[0] for call in client.embed_documents.call_args_list],
                         [['a', 'b'], ['c', 'd'], ['c', 'd']])

    def test_requests_are_serialized(self):
        """Prevent concurrent retrieval workers from overlapping embedding requests."""
        active = 0
        maximum = 0
        state_lock = Lock()

        def embed(texts):
            """Record overlapping calls while simulating a short model request."""
            nonlocal active, maximum
            with state_lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.005)
            with state_lock:
                active -= 1
            return [[float(text)] for text in texts]

        client = Mock()
        client.embed_documents.side_effect = embed
        embeddings = SensorEmbeddings(client)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            vectors = list(executor.map(embeddings.embed_query, map(str, range(16))))
        self.assertEqual(maximum, 1)
        self.assertEqual(vectors, [[float(i)] for i in range(16)])


class SensorIndexTests(unittest.TestCase):
    """Check single initialization and invalidation of persisted manual indexes."""

    def setUp(self):
        """Prepare a tiny local workbook and a fake FAISS store with real cache files."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.workbook = self.folder / 'manual.xlsx'
        self.workbook.write_bytes(b'workbook-v1')
        self.index = self.folder / 'index'
        self.store = Mock()
        self.store.save_local.side_effect = self.save_index
        self.faiss = Mock()
        self.faiss.from_documents.return_value = self.store
        self.faiss.load_local.return_value = self.store
        row = dict(instrument_name='sensor', full_name='Sensor', satellite_name='platform',
                   type_of_sensor='optical', resolution=1, coverage_cycle=2, swatch_width=3)
        self.namespace = load_functions(ROOT / 'src/other/observation_pattern.py',
            {'_file_sha256', '_load_cached_sensor_index', 'build_vector_store', 'get_vector_store'},
            dict(hashlib=hashlib, json=json, pickle=pickle, Path=Path, Lock=Lock,
                 sensor_embedding_settings={'model': 'bge-m3', 'base_url': None},
                 embeddings=Mock(), FAISS=self.faiss, pd=Mock(read_excel=Mock(return_value=pd.DataFrame([row]))),
                 Document=Document, vector_db=None, retriever=None, _VECTOR_STORE_LOCK=Lock(),
                 get_setting=lambda key: 3,
                 get_configured_path=lambda key: str(self.index if key.endswith('index_dir') else self.workbook)))
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def save_index(self, folder):
        """Write deterministic fake index files used to verify manifest hashes."""
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'index.faiss').write_bytes(b'index-data')
        (path / 'index.pkl').write_bytes(b'local-document-data')

    def test_index_initializes_once_across_threads(self):
        """Publish one complete index and retriever to concurrent callers."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            stores = list(executor.map(lambda _: self.namespace['get_vector_store'](), range(16)))
        self.assertTrue(all(store is self.store for store in stores))
        self.faiss.from_documents.assert_called_once()
        self.store.as_retriever.assert_called_once()

    def test_verified_index_is_reused(self):
        """Reload an unchanged local cache without recomputing document embeddings."""
        build = self.namespace['build_vector_store']
        build(str(self.workbook))
        self.assertIs(build(str(self.workbook)), self.store)
        self.faiss.from_documents.assert_called_once()
        self.faiss.load_local.assert_called_once()

    def test_changed_workbook_and_model_invalidate_cache(self):
        """Rebuild the index when either source data or the embedding model changes."""
        build = self.namespace['build_vector_store']
        build(str(self.workbook))
        self.workbook.write_bytes(b'workbook-v2')
        build(str(self.workbook))
        self.namespace['sensor_embedding_settings']['model'] = 'another-model'
        build(str(self.workbook))
        self.assertEqual(self.faiss.from_documents.call_count, 3)
        self.faiss.load_local.assert_not_called()

    def test_corrupt_index_is_rebuilt(self):
        """Reject a modified cache file before loading the persisted document store."""
        build = self.namespace['build_vector_store']
        build(str(self.workbook))
        (self.index / 'index.pkl').write_bytes(b'corrupt')
        build(str(self.workbook))
        self.assertEqual(self.faiss.from_documents.call_count, 2)
        self.faiss.load_local.assert_not_called()

    def test_failed_initialization_is_not_published(self):
        """Keep globals unset after initialization fails, allowing a later retry."""
        self.store.as_retriever.side_effect = RuntimeError('retriever unavailable')
        with self.assertRaises(RuntimeError):
            self.namespace['get_vector_store']()
        self.assertIsNone(self.namespace['vector_db'])
        self.assertIsNone(self.namespace['retriever'])
        self.store.as_retriever.side_effect = None
        self.assertIs(self.namespace['get_vector_store'](), self.store)


class ObservationStageTests(unittest.TestCase):
    """Ensure failed observation tasks are reported as incomplete."""

    def setUp(self):
        """Provide one pending task and mocked graph operations for the actual stage function."""
        disaster = Mock(node_id='disaster')
        observation = Mock(node_id='observation')
        observation.obsPat_argument.all.return_value = []
        observation.obsPat_argument.is_connected.return_value = False
        text = Mock(text_content='source text')
        text.exist.is_connected.return_value = True
        observation.origin.all.return_value = [text]
        disaster.trigger.all.return_value = [observation]
        pattern_node = Mock(disaster_ids=[])
        pattern = SimpleNamespace(space_res=1, time_res=2, weather_effect='weather', obs_position='position')
        self.observation = observation
        self.namespace = load_functions(ROOT / 'main/main_extraction.py', {'run_obs_pattern'},
            dict(time=time, concurrent=concurrent, tqdm=lambda values, **kwargs: values,
                 DisasterEvent=Mock(nodes=Mock(all=Mock(return_value=[disaster]), get=Mock(return_value=disaster))),
                 ObservationEvent=Mock(nodes=Mock(get=Mock(return_value=observation))),
                 ObservationPattern=Mock(nodes=Mock(first_or_none=Mock(return_value=pattern_node))),
                 english_term=lambda value, **kwargs: value, get_vector_store=Mock(),
                 _worker_obs_pattern=Mock(return_value={'obs_id': 'observation', 'dis_id': 'disaster', 'pattern': pattern})))
        self.output = io.StringIO()
        redirect = contextlib.redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def test_index_failure_prevents_worker_submission(self):
        """Fail before starting LLM workers if the shared sensor index cannot be initialized."""
        self.namespace['get_vector_store'].side_effect = RuntimeError('tokenizer unavailable')
        self.assertFalse(self.namespace['run_obs_pattern'](4))
        self.namespace['_worker_obs_pattern'].assert_not_called()

    def test_worker_failure_is_not_reported_as_success(self):
        """Return failure and report missing patterns when inference returns no result."""
        self.namespace['_worker_obs_pattern'].return_value = None
        self.assertFalse(self.namespace['run_obs_pattern'](4))
        self.assertIn('0/1 tasks saved; 1 failed', self.output.getvalue())
        self.observation.obsPat_argument.connect.assert_not_called()

    def test_successful_task_is_saved(self):
        """Save a successful pattern and report complete coverage of pending tasks."""
        self.assertTrue(self.namespace['run_obs_pattern'](4))
        self.observation.obsPat_argument.connect.assert_called_once()
        self.assertIn('1/1 tasks saved; 0 failed', self.output.getvalue())

    def test_write_failure_is_not_reported_as_success(self):
        """Return failure when a successful inference cannot be linked in the graph."""
        self.observation.obsPat_argument.connect.side_effect = RuntimeError('write failed')
        self.assertFalse(self.namespace['run_obs_pattern'](4))
        self.assertIn('0/1 tasks saved; 1 failed', self.output.getvalue())


if __name__ == '__main__':
    unittest.main()
