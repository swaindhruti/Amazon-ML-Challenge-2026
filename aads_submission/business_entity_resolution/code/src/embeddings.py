"""
Semantic (embedding-based) candidate augmentation and similarity scoring.

Why this exists: cross-script name matching (see preprocessing.py's
transliterate_to_ascii and candidate_generation.py's Soundex keys) narrows
but doesn't close the gap between Source 1 (~100% Latin-script) and the
~40% of India's Source 2/3 pool written in native script (Devanagari,
Tamil, Telugu, Kannada, Gujarati, Bengali, Malayalam, Oriya, Gurmukhi --
see README roadmap #1). A sentence-embedding model trained to place
semantically equivalent business names close together *regardless of
script* attacks that gap directly, instead of relying on transliteration
quality.

Model credit: Graphlet-AI/eridu
(https://huggingface.co/Graphlet-AI/eridu), Apache-2.0 licensed, ~118M
parameters (well under the competition's 8B-parameter limit) -- created by
Russell Jurney / Graphlet AI in collaboration with the OpenSanctions
community. It's a fine-tune of
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, trained with
contrastive learning on 2M+ labeled matching/non-matching person and
company name pairs specifically for cross-language, cross-script name
matching -- chosen over generic multilingual embedding models (e5, LaBSE)
because it's fine-tuned for exactly this task rather than general-purpose
sentence similarity. Used here as-is (no further fine-tuning yet); nothing
about this codebase prevents fine-tuning it further on this competition's
own ground truth pairs, which is a natural next step, not something this
module currently does.

Honesty note -- read this before a real run: this module was written and
reviewed for correctness (API usage, shapes, graceful-fallback behavior)
without ever successfully loading the actual model weights. The dev
sandbox this was built in has unreliable/likely-proxied network access to
huggingface.co: one attempt failed with a DNS resolution error, another
segfaulted inside sentence-transformers' fallback model-construction path
(not a catchable Python exception -- see _hub_reachable()'s docstring),
and a direct API check returned "Invalid username or password" on a
public, unauthenticated endpoint, which looks like anti-bot/rate-limit
interference rather than a real 404. None of that is consistent evidence
that the model doesn't exist -- independent web search results
consistently named "Graphlet-AI/eridu" with specific real-looking sub-paths
(/tree/main, /commits/main/README.md) -- but it also means THE MODEL NAME
ITSELF HAS NOT BEEN DIRECTLY CONFIRMED FROM THIS ENVIRONMENT.

Before relying on this for a real run: on SageMaker (real internet access),
run `python -c "from sentence_transformers import SentenceTransformer;
SentenceTransformer('Graphlet-AI/eridu')"` in isolation FIRST. If the name
is wrong, has moved, or is gated, that single line will tell you in
seconds, rather than discovering it after a long pipeline run. Everything
in this module degrades gracefully (see load_embedding_model) if that load
fails, so a wrong name costs recall on cross-script cases, not a crash --
but you should know the actual load status rather than assume it worked.
"""
import numpy as np
from typing import List, Optional, Tuple

EMBEDDING_MODEL_NAME = "Graphlet-AI/eridu"

_model_cache = {}


def _hub_reachable(host: str = "huggingface.co", timeout: float = 3.0) -> bool:
    """
    Cheap, low-level (plain socket, not requests/urllib3) reachability check
    before attempting SentenceTransformer(...). Exists because a real
    failure was observed during development: when the model can't be found
    locally AND the network is unreachable, sentence-transformers falls
    back to constructing a generic Transformer+Pooling model from scratch,
    and that fallback path segfaulted rather than raising a catchable
    Python exception -- try/except cannot protect against that. Checking
    reachability first and skipping the load entirely when unreachable
    avoids ever entering that code path. (The crash was reproduced on an
    old macOS system Python built against LibreSSL rather than OpenSSL, a
    known source of unrelated low-level SSL/networking issues -- plausibly
    an artifact of that specific environment rather than a general risk,
    but this check costs nothing and removes the failure mode either way.)
    """
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(host)
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except OSError:
        return False


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """
    Loads (and caches) the sentence-transformers model. Returns None on ANY
    failure -- no internet, model not cached, sentence-transformers/torch
    not installed, out of memory -- so every caller in this module degrades
    to "embeddings unavailable" instead of crashing the whole pipeline.
    Lexical blocking (candidate_generation.py) and the 17 hand-engineered
    features (features.py) work completely independently of this, so a
    missing embedding model costs recall on the cross-script cases, not a
    failed run.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    if not _hub_reachable():
        print(f"  [embeddings] huggingface.co not reachable; skipping '{model_name}' -- "
              f"continuing with lexical-only blocking and scoring.")
        _model_cache[model_name] = None
        return None

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"  [embeddings] Could not load '{model_name}' ({e}); "
              f"continuing with lexical-only blocking and scoring.")
        model = None
    _model_cache[model_name] = model
    return model


def encode_texts(model, texts: List[str], batch_size: int = 256) -> Optional[np.ndarray]:
    """
    Batch-encodes texts to L2-normalized embeddings, so cosine similarity
    between any two rows is a plain dot product (used by both
    EmbeddingCandidateIndex and the semantic_sim feature in features.py).
    Returns None if model is None, propagating "not available" rather than
    raising, so callers can fall back cleanly.
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


class EmbeddingCandidateIndex:
    """
    FAISS-backed nearest-neighbor index over one country partition's
    target-pool embeddings. Proposes candidates that lexical/phonetic
    blocking (CompactInvertedIndex) can miss entirely -- e.g. a name
    written in a script with no shared prefix, word, or Soundex code
    against its Source 1 counterpart, but whose semantic embedding IS close
    if the model generalizes across scripts.

    Deliberately NOT a brute-force numpy similarity matrix: an exact
    query-batch x target-pool matrix is infeasible at real scale (a few
    million target rows x a few hundred embedding dims x a 50k query batch
    is tens of billions of floats for one batch alone). FAISS's
    IndexFlatIP still computes an EXACT inner-product search, just with a
    far smaller memory footprint and a heavily optimized inner loop --
    that's what makes this tractable, not an approximation trade-off.
    """

    def __init__(self, target_embeddings: Optional[np.ndarray]):
        self.available = target_embeddings is not None and len(target_embeddings) > 0
        self.index = None
        if self.available:
            import faiss
            dim = target_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # inner product == cosine sim on normalized vectors
            self.index.add(np.ascontiguousarray(target_embeddings))

    def query(self, query_embeddings: Optional[np.ndarray], k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (query_indices, target_indices) flattened in the same shape
        CompactInvertedIndex.query_candidates uses, so the two candidate
        sets can be concatenated and deduplicated directly by the caller.
        """
        if not self.available or query_embeddings is None or len(query_embeddings) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.uint32)

        k = min(k, self.index.ntotal)
        _, neighbor_idx = self.index.search(np.ascontiguousarray(query_embeddings), k)

        n_queries = query_embeddings.shape[0]
        q_idx = np.repeat(np.arange(n_queries, dtype=np.int32), k)
        flat_targets = neighbor_idx.reshape(-1)
        # FAISS returns -1 for a slot when fewer than k neighbors exist.
        valid = flat_targets >= 0
        return q_idx[valid], flat_targets[valid].astype(np.uint32)
