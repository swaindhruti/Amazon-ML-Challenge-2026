import pandas as pd
import re
import string
from text_unidecode import unidecode as _to_ascii

# ─────────────────────────────────────────────────────────────────────────────
# Legal suffix removal patterns (international, sorted longest-first)
# ─────────────────────────────────────────────────────────────────────────────

LEGAL_SUFFIXES_RAW = [
    # Multi-word (must come first in alternation)
    'private limited', 'pvt limited', 'pvt ltd',
    'limited liability company', 'limited liability partnership',
    'societe anonyme', 'societe a responsabilite limitee',
    'societe par actions simplifiee',
    'societe civile immobiliere',
    'gesellschaft mit beschrankter haftung',
    # Single-word English
    'incorporated', 'inc', 'corporation', 'corp', 'company', 'co',
    'limited', 'ltd', 'llc', 'llp', 'lp', 'plc',
    'holdings', 'holding',
    # French
    'sa', 'sas', 'sarl', 'sasu', 'eurl', 'sci', 'snc', 'sem',
    # German
    'gmbh', 'ag', 'kg', 'ohg', 'ug', 'ev',
    # Indian
    'pvt', 'private', 'nidhi', 'opc',
    # Other
    'pty', 'bv', 'nv', 'ab', 'as', 'oy', 'srl', 'spa',
    'pte',
]
# Sort longest-first so multi-word suffixes match before their fragments
LEGAL_SUFFIXES_RAW.sort(key=len, reverse=True)

LEGAL_SUFFIX_PATTERN = re.compile(
    r'(?:^|\s)(?:' + '|'.join(re.escape(s) for s in LEGAL_SUFFIXES_RAW) + r')(?:\s*\.?\s*,?\s*$|\s)',
    re.IGNORECASE
)

# normalize_abbreviations() below EXPANDS abbreviations ("pvt" -> "private")
# so a fully-spelled name on one source still matches its abbreviated
# counterpart on another. This is deliberately kept separate from
# strip_legal_suffixes(), which instead REMOVES legal-suffix words entirely
# to produce the bare "core" name -- different features want different
# normalization strength, so both normalized forms are kept side by side
# rather than picking one.
ABBR_MAP = {
    'pvt': 'private',
    'ltd': 'limited',
    'corp': 'corporation',
    'inc': 'incorporated',
    'rd': 'road',
    'st': 'street',
    'ave': 'avenue',
    'llc': 'limited liability company',
    'co': 'company'
}
ABBR_PATTERN = re.compile(r'\b(' + '|'.join(ABBR_MAP.keys()) + r')\b')
NUM_PATTERN = re.compile(r'\d+')
WHITESPACE_PATTERN = re.compile(r'\s+')
PUNCT_TRANS = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

# Ampersand normalization
AMPERSAND_PATTERN = re.compile(r'\s*&\s*')

# Address landmark filler words -- verified against the real dataset: e.g.
# "near" alone appears in ~5.5% of India business_address rows, "opposite"
# in ~3.4%. These don't identify a specific location, so left in they dilute
# address similarity scores and blocking keys with tokens that coincidentally
# match across genuinely different addresses.
ADDRESS_LANDMARK_WORDS = {
    'near', 'opp', 'opposite', 'behind', 'beside', 'backside',
    'above', 'below', 'next', 'front', 'infront', 'landmark',
}
LANDMARK_PATTERN = re.compile(r'\b(?:' + '|'.join(ADDRESS_LANDMARK_WORDS) + r')\b')


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads TSV data explicitly with string dtypes.
    """
    return pd.read_csv(filepath, sep="\t", dtype=str, keep_default_na=False)


def transliterate_to_ascii(text: str) -> str:
    """
    Converts non-Latin script text (e.g. Devanagari, Tamil, Telugu, Kannada,
    Gujarati, Bengali, Malayalam, Oriya, Gurmukhi) and accented Latin noise
    (e.g. "Nétwork", "Président") to a phonetic ASCII approximation, so
    business names/addresses written in different scripts or with injected
    accent noise become comparable to their Source 1 (Latin-script) counterparts.

    Source 1 is ~100% Latin-script even for India, while a large share of
    Source 2/3 India records use native scripts directly -- without this step
    those records are invisible to both blocking and similarity scoring.
    Skips already-ASCII strings so the majority of records pay no extra cost.
    """
    if not text:
        return ""
    if all(ord(ch) < 128 for ch in text):
        return text
    return _to_ascii(text)


def clean_text(text: str) -> str:
    """
    Normalizes uppercase/lowercase, removes punctuation, and cleans whitespace.
    """
    if not isinstance(text, str) or not text:
        return ""
    text = transliterate_to_ascii(text)
    text = text.lower()
    text = text.translate(PUNCT_TRANS)
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    return text


def strip_legal_suffixes(text: str) -> str:
    """
    Removes legal/business suffixes from a cleaned business name.
    Produces the 'core' business identity for high-precision matching.
    E.g., "abc solutions pvt ltd" → "abc"
    """
    if not text:
        return ""
    # Iteratively strip suffixes (may need multiple passes for compound suffixes)
    prev = ""
    result = text
    for _ in range(3):
        if result == prev:
            break
        prev = result
        result = LEGAL_SUFFIX_PATTERN.sub(' ', result).strip()
    return WHITESPACE_PATTERN.sub(' ', result).strip()


def normalize_ampersand(text: str) -> str:
    """Normalizes '&' to 'and'."""
    if not text:
        return ""
    return AMPERSAND_PATTERN.sub(' and ', text).strip()


def normalize_abbreviations(text: str) -> str:
    """
    Expands legal/business and address abbreviations using compiled regex.
    """
    if not text:
        return ""
    return ABBR_PATTERN.sub(lambda m: ABBR_MAP[m.group(0)], text)


def extract_numerical_tokens(text: str) -> str:
    """
    Extracts numerical tokens (e.g. zip codes, building numbers) from text.
    Returns them as a space-separated string.
    """
    if not text:
        return ""
    numbers = NUM_PATTERN.findall(text)
    return " ".join(numbers)


def strip_landmark_words(text: str) -> str:
    """
    Removes address landmark filler words (e.g. "Near", "Opposite", "Behind")
    from an already-cleaned address string. See ADDRESS_LANDMARK_WORDS for
    why -- these are common but not locality-identifying.
    """
    if not text:
        return ""
    result = LANDMARK_PATTERN.sub(' ', text)
    return WHITESPACE_PATTERN.sub(' ', result).strip()


def normalize_name_for_matching(raw_name: str) -> str:
    """
    Full normalization pipeline for business name matching:
    1. Lowercase + remove punctuation
    2. Normalize ampersand
    3. Strip legal suffixes
    Returns the 'core' business name for matching.
    """
    cleaned = clean_text(raw_name)
    cleaned = normalize_ampersand(cleaned)
    stripped = strip_legal_suffixes(cleaned)
    return stripped if stripped else cleaned


def preprocess_dataframe(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Applies text cleaning and normalization to dataframe columns.
    """
    if not inplace:
        df = df.copy()

    if 'business_name' in df.columns:
        df['business_name_clean'] = [clean_text(t) for t in df['business_name'].values]
        df['business_name_norm'] = [normalize_abbreviations(t) for t in df['business_name_clean'].values]
        df['business_name_stripped'] = [strip_legal_suffixes(t) for t in df['business_name_clean'].values]

    if 'business_address' in df.columns:
        df['business_address_clean'] = [clean_text(t) for t in df['business_address'].values]
        df['business_address_norm'] = [
            strip_landmark_words(normalize_abbreviations(t)) for t in df['business_address_clean'].values
        ]
        df['address_numbers'] = [extract_numerical_tokens(t) for t in df['business_address_clean'].values]

    if 'country' in df.columns:
        df['country_clean'] = [clean_text(t) for t in df['country'].values]

    return df
