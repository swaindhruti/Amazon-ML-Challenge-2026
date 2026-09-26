"""
Optional semantic (embedding-based) candidate augmentation and similarity.

Why this exists: cross-script name matching (see preprocessing.py's
transliterate_to_ascii and candidate_generation.py's Soundex keys) narrows
but doesn't close the gap between Source 1 (~100% Latin-script) and the ~40%
of India's Source 2/3 pool written in native script (Devanagari, Tamil,
Telugu, Kannada, Gujarati, Bengali, Malayalam, ...). A multilingual
sentence-embedding model places the same name close together regardless of
script, which attacks that gap directly instead of relying on transliteration
quality alone.

This is a SUPPLEMENT, not the main scoring engine: the trained classifier over
lexical/address/IDF features (see training.py) carries the score. Embeddings
are off by default (--use_embeddings) because encoding is the single most
expensive step in the pipeline (a transformer forward pass per name).

Model credit: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
Apache-2.0, ~118M parameters (far under the competition's 8B limit), from the
UKP Lab / sentence-transformers project (Reimers & Gurevych, "Making
Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
EMNLP 2020). Supports 50+ languages including Hindi, Bengali, Gujarati, Tamil,
Marathi and Urdu. Used as-is, no fine-tuning.

History: this module originally targeted Graphlet-AI/eridu; that repository
could not be loaded on the team's SageMaker instance, so it was replaced with
the public model above (which eridu itself was fine-tuned from).

Offline / locked-down instances: if the instance cannot reach huggingface.co,
download the model once somewhere that can and point at the folder:
    python -c "from sentence_transformers import SentenceTransformer as S; \\
               S('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').save('models/hf/minilm')"
    python -m src.pipeline ... --use_embeddings --embedding_model models/hf/minilm
A local directory is loaded directly with no network check.

Nothing here has been run end-to-end against the real weights in the dev
sandbox; every failure path degrades to "embeddings unavailable" (lexical
pipeline continues) rather than crashing.
"""
import os
import numpy as np
from typing import List, Optional, Tuple

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_model_cache = {}


def _hub_reachable(host: str = "huggingface.co", timeout: float = 3.0) -> bool:
    """
    Cheap plain-socket reachability check before attempting a Hub download.
    Exists because a hard crash (segfault, not a catchable exception) was seen
    inside sentence-transformers' fallback path when the model could not be
    found AND the network was unreachable; skipping the load when the Hub is
    unreachable avoids ever entering that path.
    """
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(host)
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except OSError:
        return False


def load_embedding_model(model_name: Optional[str] = None, n_threads: Optional[int] = None):
    """
    Loads (and caches) the sentence-transformers model. Returns None on ANY
    failure -- no internet, model not cached, libraries missing, out of memory
    -- so callers degrade to lexical-only instead of crashing the run.
    model_name may be a Hub id or a local directory (see module docstring).
    """
    model_name = model_name or os.environ.get('EMBEDDING_MODEL') or EMBEDDING_MODEL_NAME
    if model_name in _model_cache:
        return _model_cache[model_name]

    is_local = os.path.isdir(model_name)
    offline = os.environ.get('HF_HUB_OFFLINE') == '1'
    if not is_local and not offline and not _hub_reachable():
        print(f"  [embeddings] huggingface.co not reachable; skipping '{model_name}' -- "
              f"continuing with lexical-only blocking and scoring. (Pre-download the model and "
              f"pass --embedding_model <local dir> to use it offline.)")
        _model_cache[model_name] = None
        return None

    try:
        from src.parallel import configure_native_threads
        configure_native_threads(n_threads)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device='cpu')
    except Exception as e:
        print(f"  [embeddings] Could not load '{model_name}' ({e}); "
              f"continuing with lexical-only blocking and scoring.")
        model = None
    _model_cache[model_name] = model
    return model


def encode_texts(model, texts: List[str], batch_size: int = 256) -> Optional[np.ndarray]:
    """
    Batch-encodes texts to L2-normalized float32 embeddings (cosine similarity
    == dot product). Returns None if model is None so callers can fall back.
    """
    if model is None:
        return None
    if len(texts) == 0:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


class EmbeddingRows:
    """
    Embeddings for only a SUBSET of a target pool (e.g. the rows whose raw name
    was in a non-Latin script -- the only rows lexical matching struggles
    with), addressable by the full pool's row index. Rows that were never
    encoded report `present=False`; their pair similarity is NaN (XGBoost's
    'missing'), not a fake 0.0, so the model can tell "not encoded" from
    "encoded and dissimilar".
    """

    def __init__(self, n_total: int, encoded_rows: np.ndarray, embeddings: np.ndarray):
        self.encoded_rows = np.asarray(encoded_rows, dtype=np.int64)
        self.embeddings = embeddings
        self.row_of = np.full(n_total, -1, dtype=np.int32)
        self.row_of[self.encoded_rows] = np.arange(len(self.encoded_rows), dtype=np.int32)

    def pair_sim(self, query_embeddings: np.ndarray, q_idx: np.ndarray, t_idx: np.ndarray,
                 chunk: int = 200_000) -> np.ndarray:
        n = len(q_idx)
        sim = np.full(n, np.nan, dtype=np.float32)
        rows = self.row_of[t_idx]
        present = np.nonzero(rows >= 0)[0]
        for s in range(0, len(present), chunk):
            sel = present[s:s + chunk]
            sim[sel] = np.einsum('ij,ij->i', query_embeddings[q_idx[sel]], self.embeddings[rows[sel]])
        return sim


class EmbeddingCandidateIndex:
    """
    FAISS nearest-neighbour index over (a subset of) one country's target
    embeddings. Proposes candidates lexical/phonetic blocking can miss
    entirely (a name in a script sharing no prefix, word, or Soundex code with
    its Latin counterpart, but semantically close).

    Small pools use an exact flat inner-product index. Large pools use HNSW
    (approximate, ~99% recall at these settings): an exact search of a 50k
    query batch against ~1M+ 384-dim vectors costs tens of trillions of
    floating-point operations per batch, which is not tractable, while HNSW
    answers the same queries in seconds after a one-off build.

    row_ids (optional): maps index positions back to full-pool target rows,
    for an index built over a subset.
    """

    HNSW_THRESHOLD = 200_000

    def __init__(self, target_embeddings: Optional[np.ndarray], row_ids: Optional[np.ndarray] = None):
        self.available = target_embeddings is not None and len(target_embeddings) > 0
        self.index = None
        self.row_ids = None if row_ids is None else np.asarray(row_ids, dtype=np.int64)
        if self.available:
            import faiss
            dim = target_embeddings.shape[1]
            if len(target_embeddings) >= self.HNSW_THRESHOLD:
                self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
                self.index.hnsw.efConstruction = 80
                self.index.hnsw.efSearch = 64
            else:
                self.index = faiss.IndexFlatIP(dim)
            self.index.add(np.ascontiguousarray(target_embeddings))

    def query(self, query_embeddings: Optional[np.ndarray], k: int) -> Tuple[np.ndarray, np.ndarray]:
        """(query_indices, target_indices) flattened like CompactInvertedIndex.query_candidates."""
        if not self.available or query_embeddings is None or len(query_embeddings) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.uint32)

        k = min(k, self.index.ntotal)
        _, neighbor_idx = self.index.search(np.ascontiguousarray(query_embeddings), k)

        n_queries = query_embeddings.shape[0]
        q_idx = np.repeat(np.arange(n_queries, dtype=np.int32), k)
        flat = neighbor_idx.reshape(-1)
        valid = flat >= 0  # FAISS pads with -1 when fewer than k neighbours exist
        flat = flat[valid]
        if self.row_ids is not None:
            flat = self.row_ids[flat]
        return q_idx[valid], flat.astype(np.uint32)
