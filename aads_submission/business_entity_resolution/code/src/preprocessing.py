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
    # What text_unidecode turns the native-script spellings of "private
    # limited" / "LLP" into (Devanagari, Gujarati, Telugu, Kannada, Bengali,
    # Tamil...): without these, a Hindi-script "... प्राइवेट लिमिटेड" keeps
    # "praaivett limittedd" as if it were part of the business name, which
    # can never line up with the Latin-script counterpart's stripped name.
    'praaivett', 'praiveett', 'praaibhett', 'limittedd', 'limittett', 'elelpii',
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


# Domain-style tokens that appear when a name is really a website
# ("wilfordhancock.com" -> "wilfordhancock com"). Dropped ONLY when building
# the space-free "compact" name below, never from the normal name columns.
DOMAIN_TOKENS = {'www', 'com', 'net', 'org', 'biz', 'info'}


def compact_name(stripped_name: str) -> str:
    """
    Space-free form of a (legal-suffix-stripped) name, e.g. "wilford hancock
    associates" -> "wilfordhancockassociates". Some records are written as a
    single glued domain-style token ("wilfordhancock.com") while their true
    counterpart is space-separated; every token-based feature scores such a
    pair near zero, but a character-level comparison of the glued forms still
    lines up.
    """
    if not stripped_name:
        return ""
    tokens = [t for t in stripped_name.split() if t not in DOMAIN_TOKENS]
    return ''.join(tokens) if tokens else stripped_name.replace(' ', '')


# ─────────────────────────────────────────────────────────────────────────────
# Round-2 noise normalizations (see README "Round 2"). Every one of these is
# applied SYMMETRICALLY to Source 1 and to the target pool, so two strings that
# were already identical stay identical; they only remove differences that
# aren't real. Prevalence figures are from the real training files.
# ─────────────────────────────────────────────────────────────────────────────

# "L.L.C." / "P.V.T." / "L.T.D": ~2.9% of S2/S3 names. clean_text turns the dots
# into spaces -> "l l c", which never matches the legal-suffix list or the
# other source's "LLC".
_ACRONYM_DOTS = re.compile(r'(?<![a-z0-9])(?:[a-z]\.){2,}[a-z]?(?![a-z0-9])')

# "Quigley Usa | www.quigleyus.com", "www.colonialm.com Colonial-Meshflow, |":
# a website glued onto the name (~0.3% of S2/S3).
_URL_LIKE = re.compile(r'(?:https?://)?(?:www\.)?[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.(?:com|net|org|biz|info|in|co|io|us)\b[^\s|]*',
                       re.IGNORECASE)

# Digit-for-letter typos ("F0nes", "Ava1anche", "8rewing"): ~1.6% of S2/S3 names,
# ~0 in S1, so mapping them back is safe. Only unambiguous look-alikes, only in
# tokens with >=3 letters and <=2 digits that are ALL mappable (leaves "3M",
# "A1", "H2O", "Studio54", "24hr", "Route66" alone).
_LEET = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '5': 's', '8': 'b'})
_LEET_OK = re.compile(r'^[0-9]?(?:[a-z]+[0-9]?)+$')


def _fix_leet_token(tok: str) -> str:
    digits = [c for c in tok if c.isdigit()]
    letters = len(tok) - len(digits)
    if not digits or letters < 3 or len(digits) > 2:
        return tok
    if any(d not in '01358' for d in digits):
        return tok
    return tok.translate(_LEET)


def clean_name(text: str) -> str:
    """
    clean_text() plus the name-specific noise fixes above: website tails,
    dotted acronyms, digit-for-letter typos, a leading "The", and the Indian
    "M/s" prefix. Used for business names only (addresses legitimately contain
    digit/letter mixes like "A-20" or "Syno38" that must stay intact).
    """
    if not isinstance(text, str) or not text:
        return ""
    raw = text
    if '|' in raw:
        parts = [p.strip() for p in raw.split('|')]
        keep = [p for p in parts if p and not _URL_LIKE.fullmatch(p.strip().lower())]
        raw = keep[0] if keep else parts[0]
    lowered = raw.lower()
    stripped = _URL_LIKE.sub(' ', lowered)
    if stripped.strip():          # only drop URL tokens if a real name remains
        lowered = stripped
    lowered = _ACRONYM_DOTS.sub(lambda m: m.group(0).replace('.', ''), lowered)
    out = clean_text(lowered)
    if not out:
        return out
    toks = [_fix_leet_token(t) for t in out.split(' ')]
    if len(toks) > 2 and toks[0] == 'm' and toks[1] == 's':
        toks = toks[2:]
    if len(toks) > 1 and toks[0] == 'the':
        toks = toks[1:]
    return ' '.join(toks)


# --- Address normalization ---------------------------------------------------
# Real address formats differ SYSTEMATICALLY by source (measured on the real
# files): US state is a 2-letter code in S1/S2 but a full name in S3 ("TX" vs
# "Texas"); India's state is a full name in S1/S2 but a 2-letter code in S3
# ("Maharashtra" vs "MH"), and S2/S3 India also carry native-script states.
# Without this, every S1<->S3 pair starts with a phantom state mismatch.
_US_STATES = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
    'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'district of columbia': 'dc',
    'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il',
    'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la',
    'maine': 'me', 'maryland': 'md', 'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn',
    'mississippi': 'ms', 'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv',
    'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny',
    'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or',
    'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc', 'south dakota': 'sd',
    'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut', 'vermont': 'vt', 'virginia': 'va',
    'washington': 'wa', 'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy', 'puerto rico': 'pr',
}
_INDIA_STATES = {
    'andhra pradesh': 'ap', 'arunachal pradesh': 'ar', 'assam': 'as', 'bihar': 'br',
    'chhattisgarh': 'cg', 'chattisgarh': 'cg', 'goa': 'ga', 'gujarat': 'gj', 'haryana': 'hr',
    'himachal pradesh': 'hp', 'jharkhand': 'jh', 'karnataka': 'ka', 'kerala': 'kl',
    'madhya pradesh': 'mp', 'maharashtra': 'mh', 'manipur': 'mn', 'meghalaya': 'ml',
    'mizoram': 'mz', 'nagaland': 'nl', 'odisha': 'od', 'orissa': 'od', 'punjab': 'pb',
    'rajasthan': 'rj', 'sikkim': 'sk', 'tamil nadu': 'tn', 'telangana': 'tg', 'tripura': 'tr',
    'uttar pradesh': 'up', 'uttarakhand': 'uk', 'uttaranchal': 'uk', 'west bengal': 'wb',
    'delhi': 'dl', 'new delhi': 'dl', 'nct of delhi': 'dl', 'jammu and kashmir': 'jk',
    'ladakh': 'la', 'chandigarh': 'ch', 'puducherry': 'py', 'pondicherry': 'py',
    'andaman and nicobar islands': 'an', 'lakshadweep': 'ld',
}
# Native-script (Devanagari) state names seen as the last address component
# (e.g. "महाराष्ट्र" ~4.8% of S2 India addresses). Keys are matched AFTER
# transliteration, so they share clean_text()'s output space.
_INDIA_NATIVE = {
    'महाराष्ट्र': 'mh', 'दिल्ली': 'dl', 'उत्तर प्रदेश': 'up', 'कर्नाटक': 'ka', 'तमिलनाडु': 'tn',
    'गुजरात': 'gj', 'पश्चिम बंगाल': 'wb', 'तेलंगाना': 'tg', 'हरियाणा': 'hr', 'केरल': 'kl',
    'राजस्थान': 'rj', 'मध्य प्रदेश': 'mp', 'बिहार': 'br', 'आंध्र प्रदेश': 'ap', 'पंजाब': 'pb',
    'ओडिशा': 'od', 'झारखंड': 'jh', 'छत्तीसगढ़': 'cg', 'उत्तराखंड': 'uk', 'असम': 'as',
    'गोवा': 'ga', 'हिमाचल प्रदेश': 'hp', 'जम्मू और कश्मीर': 'jk', 'चंडीगढ़': 'ch',
}
# Alternative 2-letter codes used for the same India state.
_INDIA_CODE_ALIAS = {'ut': 'uk', 'or': 'od', 'ts': 'tg', 'ct': 'cg', 'uc': 'uk', 'dd': 'dl'}


def _state_table(country: str):
    """(name/native -> code, valid codes, code aliases) for a cleaned country name, or None."""
    if country == 'us':
        return _US_TABLE
    if country == 'india':
        return _INDIA_TABLE
    return None


def _build_tables():
    us = ({k: v for k, v in _US_STATES.items()}, set(_US_STATES.values()), {})
    india_map = {k: v for k, v in _INDIA_STATES.items()}
    for native, code in _INDIA_NATIVE.items():
        india_map[clean_text(native)] = code
    india = (india_map, set(india_map.values()), _INDIA_CODE_ALIAS)
    return us, india


_US_TABLE, _INDIA_TABLE = _build_tables()

_NULLISH = re.compile(r'<\s*null\s*>|\bn\s*/\s*a\b|\bnull\b|\bnone\b|\bnan\b', re.IGNORECASE)

_ORDINALS = {
    'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th', 'fifth': '5th',
    'sixth': '6th', 'seventh': '7th', 'eighth': '8th', 'ninth': '9th', 'tenth': '10th',
    'eleventh': '11th', 'twelfth': '12th', 'thirteenth': '13th', 'fourteenth': '14th',
    'fifteenth': '15th', 'sixteenth': '16th', 'seventeenth': '17th', 'eighteenth': '18th',
    'nineteenth': '19th', 'twentieth': '20th',
}
_STREET_TYPES = {
    'drive': 'dr', 'court': 'ct', 'place': 'pl', 'lane': 'ln', 'boulevard': 'blvd',
    'highway': 'hwy', 'parkway': 'pkwy', 'circle': 'cir', 'terrace': 'ter', 'square': 'sq',
    'trail': 'trl',
}
# Unit designators and place-type filler that appear on one side and not the
# other ("Unit B" vs "# B", "Freeport CDP" vs "Freeport", "Town Of Utica").
_ADDR_DROP = {'unit', 'suite', 'ste', 'apt', 'apartment', 'floor', 'fl', 'room', 'rm',
              'cdp', 'city', 'town', 'borough', 'county'}
_LEADING_ZEROS = re.compile(r'\b0+(\d)')


def clean_address(raw: str, country: str = ''):
    """
    Returns (address_clean, state_code). The state is read from the raw
    address's last comma-separated component BEFORE commas are erased, mapped
    to a canonical 2-letter code, and re-appended as that code -- so "Texas"
    (S3) and "TX" (S1/S2) become the same token. Also: drops null-ish
    placeholders ("<NULL>", "N/A"), maps ordinal words to digits ("Fourth" ->
    "4th"), strips leading zeros ("0658" -> "658"), shortens street types
    ("Drive"/"Dr" -> "dr"), and drops unit/place-type filler.
    """
    if not isinstance(raw, str) or not raw:
        return "", ""
    text = _NULLISH.sub(' ', raw)
    table = _state_table(country)
    state = ""
    if table is not None:
        names, codes, alias = table
        comps = [c for c in text.split(',')]
        # Scan the last (up to) 3 non-empty components from the end: usually
        # the state is last, but a locality can trail it ("..., GJ, Sachin").
        idxs = [i for i in range(len(comps) - 1, -1, -1) if comps[i].strip()][:3]
        for i in idxs:
            last = clean_text(comps[i])
            code = names.get(last) or (last if last in codes else alias.get(last))
            if code:
                state = code
                text = ','.join(comps[:i] + comps[i + 1:])
                break
    out = clean_text(text)
    if out:
        toks = []
        for t in out.split(' '):
            t = _ORDINALS.get(t, _STREET_TYPES.get(t, t))
            if t in _ADDR_DROP:
                continue
            toks.append(t)
        out = _LEADING_ZEROS.sub(r'\1', ' '.join(toks))
    if state:
        out = (out + ' ' + state).strip()
    return out, state


def prepare_text_chunk(names, addrs, countries=None):
    """
    Cleans one chunk of raw names/addresses into every derived text column the
    pipeline needs. Module-level (picklable) and free of shared state so
    src.parallel can run chunks on separate cores; also used serially.
    Returns 9 parallel lists:
      names_clean, names_norm, names_stripped, names_compact,
      addrs_clean, addrs_norm, nums, name_is_native (raw name had non-ASCII),
      states (canonical state code or '').
    """
    if countries is None:
        countries = [''] * len(names)
    names_clean = [clean_name(t) for t in names]
    names_norm = [normalize_abbreviations(t) for t in names_clean]
    names_stripped = [strip_legal_suffixes(t) for t in names_clean]
    names_compact = [compact_name(t) for t in names_stripped]
    parsed = [clean_address(a, clean_text(c)) for a, c in zip(addrs, countries)]
    addrs_clean = [p[0] for p in parsed]
    states = [p[1] for p in parsed]
    addrs_norm = [strip_landmark_words(normalize_abbreviations(t)) for t in addrs_clean]
    nums = [extract_numerical_tokens(t) for t in addrs_clean]
    native = [not (isinstance(t, str) and t.isascii()) for t in names]
    return names_clean, names_norm, names_stripped, names_compact, addrs_clean, addrs_norm, nums, native, states
