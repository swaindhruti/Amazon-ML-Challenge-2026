import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Tuple, Sequence, Dict, Set
import gc

from src.preprocessing import ADDRESS_LANDMARK_WORDS

# Words too common to be useful as a blocking key on their own -- e.g. keying
# on "private" or "limited" alone would pull in a large fraction of the whole
# India target pool as "candidates" for every query, defeating the point of
# blocking. Deliberately conservative (hand-curated, not learned): filtering
# too aggressively would blind blocking to real distinguishing tokens, so
# this only removes words that are near-universal across unrelated businesses.
GENERIC_STOP_WORDS: Set[str] = {
    'inc', 'llc', 'ltd', 'corp', 'company', 'corporation', 'limited',
    'pvt', 'private', 'the', 'and', 'street', 'road', 'avenue', 'drive',
    'lane', 'dr', 'st', 'rd', 'ave', 'blvd', 'suite', 'floor', 'unit',
    'center', 'services', 'service', 'enterprise', 'enterprises',
    'solutions', 'group', 'holdings', 'international', 'industries',
    'store', 'shop', 'market', 'restaurant', 'cafe', 'hotel'
}

def get_first_token(text: str) -> str:
    """Kept for backward compatibility -- re-exported via blocking.py's public API."""
    if not text:
        return ""
    tokens = text.split()
    return tokens[0] if tokens else ""

_SOUNDEX_CODES = {
    'B': '1', 'F': '1', 'P': '1', 'V': '1',
    'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
    'D': '3', 'T': '3',
    'L': '4',
    'M': '5', 'N': '5',
    'R': '6',
}

def soundex(word: str) -> str:
    """
    Classic Soundex phonetic code. Used as a blocking key so words that reach
    a similar pronunciation via different spelling paths -- e.g. "private"
    vs. "praaivett" (the latter being what a Devanagari-script "प्राइवेट"
    turns into after transliterate_to_ascii) -- still land in the same
    candidate block, even though their literal prefixes differ ("pri" vs
    "pra"), which the existing p3/p4 prefix keys alone would miss.
    """
    word = ''.join(ch for ch in word.upper() if ch.isalpha())
    if not word:
        return "0000"
    first_letter = word[0]
    encoded = [_SOUNDEX_CODES.get(first_letter, '')]
    prev_code = encoded[0]
    for ch in word[1:]:
        if ch in 'HW':
            continue
        code = _SOUNDEX_CODES.get(ch, '')
        if code and code != prev_code:
            encoded.append(code)
        prev_code = code
    result = (first_letter + ''.join(encoded[1:]))[:4]
    return result + '0' * (4 - len(result))

class CompactInvertedIndex:
    """
    Memory-compact inverted index for candidate generation in entity resolution.
    Stores posting lists as 32-bit unsigned integer arrays (np.uint32) to minimize
    memory footprint on large datasets (10M+ rows).
    
    Automatically prunes high-frequency keys (> max_block_size) to avoid
    quadratic candidate blowups and OOM.
    """
    def __init__(self, max_block_size: int = 10000, max_candidates: int = 30):
        self.max_block_size = max_block_size
        self.max_candidates = max_candidates
        self.index: Dict[str, np.ndarray] = {}
        self.num_records = 0

    @staticmethod
    def extract_keys(name: str, addr: str, nums: str) -> List[str]:
        """
        Builds the set of inverted-index keys for one record. A candidate pair
        is proposed whenever a query and a target record share ANY key, so
        this list is a deliberate blend of tight and loose keys: tight keys
        (exact, prefix) buy precision/small candidate sets for clean records,
        looser keys (soundex, single significant word) buy recall for noisy
        ones -- typos, transliteration, word-order swaps, missing suffixes.
        """
        keys = []
        if name:
            name_len = len(name)
            # Prefix keys: catches near-matches with a typo/suffix later in the
            # name (e.g. "acme corp" vs "acme corporation" share 'p4:acme').
            if name_len >= 4:
                keys.append('p4:' + name[:4])
            if name_len >= 3:
                keys.append('p3:' + name[:3])
            # Exact short-name key: only for names <=25 chars, since a long
            # exact string is unlikely to repeat verbatim and isn't worth the
            # index entry -- this key exists for terse names prefix keys alone
            # would leave under-blocked (e.g. a 3-4 char business name).
            if 3 <= name_len <= 25:
                keys.append('exact:' + name)

            # Significant-word keys: first/second/last non-stopword tokens,
            # tolerant of word reordering (a full name match isn't required).
            words = [w for w in name.split() if w not in GENERIC_STOP_WORDS and len(w) >= 3]
            if words:
                keys.append('w1:' + words[0])
                # Soundex sibling of w1/w2: catches the same word spelled
                # differently -- notably the output of transliterate_to_ascii()
                # on a non-Latin script, which won't share a literal prefix
                # with its Latin-script counterpart (see README architecture
                # diagram for a worked example: "private" vs "praaivett").
                keys.append('sdx1:' + soundex(words[0]))
                if len(words) > 1:
                    keys.append('w2:' + words[1])
                    keys.append('sdx2:' + soundex(words[1]))
                if len(words) > 2:
                    keys.append('wlast:' + words[-1])

        num_list = nums.split() if nums else []
        addr_words = [
            w for w in addr.split()
            if w not in GENERIC_STOP_WORDS and w not in ADDRESS_LANDMARK_WORDS and len(w) >= 4
        ] if addr else []

        if num_list:
            # Numeric tokens (building/PIN/zip numbers) are high-precision on
            # their own -- two unrelated businesses rarely share a 4+ digit
            # number, so this key alone often pins down the right country
            # block even when the name has heavy noise.
            for num in num_list:
                if len(num) >= 4:
                    keys.append('num:' + num)
            # Composite keys: number + a same-record token, so two records
            # that share ONLY a number (common, e.g. shared building/PIN in a
            # dense area) or ONLY a name-prefix don't collide as often as they
            # would on either signal alone.
            if addr_words:
                keys.append('num_w:' + num_list[0] + '_' + addr_words[0])
            if name and len(name) >= 3:
                keys.append('num_p3:' + num_list[0] + '_' + name[:3])

        if addr_words:
            # Locality/street-name tokens, landmark filler words already
            # excluded (see ADDRESS_LANDMARK_WORDS) so this doesn't key on
            # "near"/"opposite" instead of the actual place name.
            keys.append('aw1:' + addr_words[0])
            if len(addr_words) > 1:
                keys.append('aw2:' + addr_words[1])

        return keys

    def build(self, names: Sequence[str], addrs: Sequence[str], nums: Sequence[str]):
        """
        Builds the inverted index from target pool arrays.

        A key's final size can't be known until every record has been
        indexed, so this necessarily accumulates the full posting list for
        every key first (as plain Python lists -- a real but transient
        memory cost for very common keys) before pruning and compacting to
        np.uint32. There's no way to decide "this key is too common, skip
        it" any earlier without a separate frequency-counting pass.
        """
        self.num_records = len(names)
        raw_index = defaultdict(list)

        for i in range(self.num_records):
            keys = self.extract_keys(names[i], addrs[i], nums[i])
            for k in keys:
                raw_index[k].append(i)

        # Compaction: drop blocks > max_block_size (prevents any single key,
        # e.g. a generic name prefix, from turning every query into an
        # O(target pool size) comparison), convert retained posting lists to
        # np.uint32 (a plain Python int list is ~4-8x the memory for the same
        # data on 10M+ row target pools).
        self.index = {}
        for k, v in raw_index.items():
            if len(v) <= self.max_block_size:
                self.index[k] = np.array(v, dtype=np.uint32)

        del raw_index
        gc.collect()

    def query_candidates(
        self, 
        query_names: Sequence[str], 
        query_addrs: Sequence[str], 
        query_nums: Sequence[str],
        max_candidates: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Queries the inverted index for a batch of query records.
        Returns:
            query_indices: 1D array of row indices in the query batch
            target_indices: 1D array of row indices in the target index
        """
        if max_candidates is None:
            max_candidates = self.max_candidates

        all_query_idx = []
        all_target_idx = []

        index_map = self.index
        n_queries = len(query_names)

        for q_i in range(n_queries):
            keys = self.extract_keys(query_names[q_i], query_addrs[q_i], query_nums[q_i])
            if not keys:
                continue

            # Counter, not a plain set union: a target record matching on
            # MORE of the query's keys is a stronger candidate than one
            # matching on just one, so key-overlap count doubles as a cheap
            # relevance ranking used below to pick which candidates survive
            # the max_candidates cap.
            cand_counts = Counter()
            for k in keys:
                if k in index_map:
                    cand_counts.update(index_map[k])

            if not cand_counts:
                continue

            if len(cand_counts) > max_candidates:
                top_cands = [c for c, _ in cand_counts.most_common(max_candidates)]
            else:
                top_cands = list(cand_counts.keys())
                
            for t_i in top_cands:
                all_query_idx.append(q_i)
                all_target_idx.append(t_i)
                
        return np.array(all_query_idx, dtype=np.int32), np.array(all_target_idx, dtype=np.uint32)

def generate_candidates(
    df_s1: pd.DataFrame, 
    df_s2_s3: pd.DataFrame, 
    top_k: int = 30, 
    batch_size: int = 50000
) -> pd.DataFrame:
    """
    Generates candidate pairs using country partitioning and memory-compact inverted index.
    Backward-compatible with original API returning DataFrame of:
    ['source1_entity_id', 'candidate_entity_ids']
    """
    candidate_records = []
    
    # Ensure country column exists
    s1_countries = df_s1['country_clean'].values if 'country_clean' in df_s1.columns else df_s1['country'].str.strip().str.lower().values
    s23_countries = df_s2_s3['country_clean'].values if 'country_clean' in df_s2_s3.columns else df_s2_s3['country'].str.strip().str.lower().values
    
    unique_countries = set(s1_countries).union(set(s23_countries))
    
    for country in unique_countries:
        s1_mask = (s1_countries == country)
        s23_mask = (s23_countries == country)
        
        s1_country = df_s1[s1_mask].reset_index(drop=True)
        s23_country = df_s2_s3[s23_mask].reset_index(drop=True)
        
        if s1_country.empty:
            continue
            
        if s23_country.empty:
            for s1_id in s1_country['entity_id'].values:
                candidate_records.append({'source1_entity_id': s1_id, 'candidate_entity_ids': ''})
            continue
            
        print(f"Candidate generation for country: {country} | S1: {len(s1_country)} | S2+S3: {len(s23_country)}")
        
        # Prepare target text arrays
        t_names = s23_country['business_name_norm'].values if 'business_name_norm' in s23_country.columns else s23_country['business_name'].values
        t_addrs = s23_country['business_address_norm'].values if 'business_address_norm' in s23_country.columns else s23_country['business_address'].values
        t_nums = s23_country['address_numbers'].values if 'address_numbers' in s23_country.columns else [""] * len(s23_country)
        t_ids = s23_country['entity_id'].values
        
        # Build inverted index for this country
        index = CompactInvertedIndex(max_block_size=10000, max_candidates=top_k)
        index.build(t_names, t_addrs, t_nums)
        
        # Query in batches
        num_batches = int(np.ceil(len(s1_country) / batch_size))
        for b in range(num_batches):
            b_start = b * batch_size
            b_end = min((b + 1) * batch_size, len(s1_country))
            s1_batch = s1_country.iloc[b_start:b_end]
            
            q_names = s1_batch['business_name_norm'].values if 'business_name_norm' in s1_batch.columns else s1_batch['business_name'].values
            q_addrs = s1_batch['business_address_norm'].values if 'business_address_norm' in s1_batch.columns else s1_batch['business_address'].values
            q_nums = s1_batch['address_numbers'].values if 'address_numbers' in s1_batch.columns else [""] * len(s1_batch)
            q_ids = s1_batch['entity_id'].values
            
            q_idx, t_idx = index.query_candidates(q_names, q_addrs, q_nums, max_candidates=top_k)
            
            # Group candidates by query index
            batch_cand_map = defaultdict(list)
            for qi, ti in zip(q_idx, t_idx):
                batch_cand_map[qi].append(t_ids[ti])
                
            for qi in range(len(s1_batch)):
                cands = batch_cand_map.get(qi, [])
                cand_str = ",".join(cands) if cands else ""
                candidate_records.append({'source1_entity_id': q_ids[qi], 'candidate_entity_ids': cand_str})
                
            del q_idx, t_idx, batch_cand_map
            gc.collect()
            
        del index, s23_country, s1_country
        gc.collect()
        
    return pd.DataFrame(candidate_records)
