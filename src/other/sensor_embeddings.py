"""Bounded, serialized embedding requests for sensor manual retrieval."""

from threading import Lock
import time

import httpx
from langchain_core.embeddings import Embeddings


def is_transient_embedding_error(error):
    """Identify transport failures and unavailable Ollama tokenizer processes."""
    if isinstance(error, (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout,
                          httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError)):
        return True
    message = str(error).lower()
    if 'model runner has unexpectedly stopped' in message:
        return True
    return '/tokenize' in message and any(marker in message for marker in (
        'connection refused', 'actively refused', 'connectex', 'connection reset',
        'econnrefused', 'not responding', 'unexpected eof',
    ))


class SensorEmbeddings(Embeddings):
    """Wrap an embedding client with batching, serialization, and transient retries."""

    def __init__(self, client, batch_size=16, max_attempts=3, retry_delay_seconds=2.0):
        """Set the batch size and total attempts used for each embedding request."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Sensor embedding batch_size must be a positive integer.')
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError('Sensor embedding max_attempts must be a positive integer.')
        if retry_delay_seconds < 0:
            raise ValueError('Sensor embedding retry_delay_seconds must be nonnegative.')
        self.client = client
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self._lock = Lock()

    def embed_documents(self, texts):
        """Embed ordered batches, retrying only transient failures under one request lock."""
        vectors = []
        with self._lock:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                for attempt in range(1, self.max_attempts + 1):
                    try:
                        result = self.client.embed_documents(batch)
                        if len(result) != len(batch):
                            raise ValueError('Sensor embedding response has an unexpected vector count.')
                        vectors.extend(result)
                        break
                    except Exception as error:
                        if not is_transient_embedding_error(error):
                            raise
                        if attempt == self.max_attempts:
                            raise RuntimeError(
                                f'Sensor embedding service unavailable after {attempt} attempts '
                                f'for document batch {start + 1}-{start + len(batch)}: {error}'
                            ) from error
                        delay = self.retry_delay_seconds * attempt
                        print(f'[Sensor embeddings] Temporary Ollama connection failure; '
                              f'retrying attempt {attempt + 1}/{self.max_attempts} in {delay:g}s.', flush=True)
                        time.sleep(delay)
        return vectors

    def embed_query(self, text):
        """Embed a retrieval query using the same request lock and retry policy."""
        return self.embed_documents([text])[0]
