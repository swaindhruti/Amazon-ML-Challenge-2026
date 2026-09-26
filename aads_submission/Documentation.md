# ML Challenge 2026: Business Entity Resolution — Solution Documentation

**Team Name:** aads
**Track:** Business Entity Resolution

> **Before submitting:** the fields marked `«FILL»` in §7.2 are numbers that only a real run on the
> full dataset can produce (`python3 -m src.pipeline --data_dir student_resource/dataset --is_train --validate`
> prints all of them). They are deliberately left blank rather than estimated.

---

## 1. Executive Summary

The task is to find, for every Source 1 (S1) business, all matching records in Source 2 and Source 3 (≈10.3M
target records, three countries), scored by macro-averaged F0.5 with singletons counted.

The solution is a **country-partitioned, three-stage pipeline** whose design follows directly from measurements
of the real training data:

1. **Blocking** with a compact inverted index, then a cheap TF-IDF re-rank of a larger candidate pool down to the
   30 best candidates per S1 entity (§4).
2. **Pair scoring** with a gradient-boosted classifier (XGBoost) over ~29 name / address / location features,
   trained on the *same blocked candidate pairs it later scores* (§5, §6.1).
3. **Decision** by a *one-owner* rule — every matched target record belongs to exactly one S1 entity in the real
   ground truth — followed by a calibrated threshold and per-source caps (§6.2–6.3).

Score history, stated plainly: the first version of this pipeline (an untrained hand-weighted scorer with a guessed
threshold of 0.62) scored **0.616** on the platform. Earlier local figures of 0.88–0.99 were tuned and reported on
the same data and are **not** comparable to that; they are not used as results anywhere in this document. The
changes described below were developed to close the gap between those two numbers; their effect on the real test
set is reported in §7.2 only once measured.

---

## 2. Problem Analysis

### 2.1 Metric
Macro F0.5 over all S1 entities: `F0.5 = 1.25·P·R / (0.25·P + R)` per entity, which for one entity with `k`
predictions, `tp` correct and `T` true matches equals `1.25·tp / (k + 0.25·T)`. A singleton (T = 0) scores 1.0 for
an empty prediction and 0.0 for any prediction.

### 2.2 Structure of the real training ground truth
Values below were computed from `train_ground_truth.tsv` and the source files, except where marked; `scripts/gt_stats.py` recomputes all of them.

| Property | Value | Consequence for the design |
|---|---|---|
| S1 entities | 2,206,821 | |
| Singletons | **5.6%** | An empty prediction is almost always wrong; a global threshold that leaves entities empty is costly |
| Matches per non-singleton | mean 3.67, median 4, max 11 | Recall matters; caps of 5 (S2) / 6 (S3) lose nothing |
| Both S2 and S3 matched | ~85% of non-singletons | |
| Matched targets | 7,638,365, **all distinct** | **Each matched record has exactly one owner** → one-owner rule |
| Target records owned by some S1 | ~74% | ~26% are pure distractors |
| Cross-country matches | *not re-measured this round* — earlier analysis found ~0%, but a spot check turned up at least one apparent cross-country pair; measure with `scripts/gt_stats.py` | Strict per-country blocking; any cross-country ground-truth pair is an unrecoverable recall loss |

Per-source match counts (all S1 entities, singletons included in the "0" row):

| S2 matches | Share | S3 matches | Share |
|---|---|---|---|
| 0 | 13.0% | 0 | 12.1% |
| 1 | 35.8% | 1 | 32.5% |
| 2 | 29.6% | 2 | 30.3% |
| 3 | 15.1% | 3 | 16.9% |
| 4 | 5.4% | 4 | 6.6% |
| 5 | 1.1% | 5 | 1.6% |
| >5 | 0% | 6 | 0.1% |

### 2.3 Noise inventory (measured on the real files)
| Noise | Where | Prevalence |
|---|---|---|
| Native-script names (Devanagari, Tamil, Telugu, …) | India S2/S3 | ~40% of India targets |
| Dotted acronyms (`L.L.C.`, `P.V.T.`) | all sources | ~2.9% of names |
| Digit-for-letter typos (`F0nes`, `Ava1anche`) | S2/S3 only (≈0 in S1) | ~1.6% |
| Leading "The" | S2/S3 | ~1% |
| Website glued to the name (`X \| www.x.com`) | S2/S3 | ~0.3% |
| Domain-style names (`techzib.com`) | S3 mostly | ~4% of targets |
| `<NULL>` placeholders / empty addresses | S2/S3 | ~2.6% / ~3.4% |
| Address format differs by source: US state `TX` (S1/S2) vs `Texas` (S3); India `Maharashtra` (S1/S2) vs `MH` (S3) | | systematic |
| Truncated names, address typos, alias/DBA names that share only the address | | common |

---

## 3. Preprocessing and Normalization (`src/preprocessing.py`)

Every rule is applied **symmetrically** to S1 and to the target pool, so strings that were already identical stay
identical and only spurious differences disappear.

- Case, punctuation, whitespace; **transliteration to ASCII** (`text-unidecode`, a bundled table — not an
  external lookup) for non-Latin scripts and accent noise.
- **Names** (`clean_name`): dotted-acronym collapse, website-tail/URL removal, digit-for-letter repair (only
  unambiguous look-alikes, only in tokens with ≥3 letters and ≤2 digits so `3M`, `A1`, `Studio54` are untouched),
  leading `The` and Indian `M/s` removal, then international legal-suffix stripping (US/Global, India incl.
  transliterated native-script forms, France, Germany/EU).
- **Addresses** (`clean_address`): null placeholders removed; ordinals to digits (`Fourth`→`4th`); leading zeros
  stripped (`0658`→`658`); street types shortened; unit/place-type filler dropped; **state parsed from the last
  address components and mapped to a canonical 2-letter code** (US and India, including Devanagari state names),
  so `Texas` and `TX` become the same token. The state is also exposed separately for features.
- Space-free "compact" name (drops `com/net/org`) for glued domain-style names.

---

## 4. Candidate Generation (`src/candidate_generation.py`, `src/country_pipeline.py`)

1. **Country isolation.** Blocking and scoring run per cleaned country value; any unseen country (France in the
   test set) is handled by the same code with no country-specific parameters.
2. **Compact inverted index** (`np.uint32` posting lists). Keys held by more than 10,000 records are dropped.
3. **Key hierarchy:** name prefixes (`p3/p4`), exact short names, significant words (`w1/w2/wlast`) and their
   Soundex codes (`sdx1/sdx2`, which link transliterated native-script words to their Latin counterparts),
   address numbers (`num`), number+street and number+name-prefix composites, locality words (`aw1/aw2`), and
   **compound keys** (name word + state, name prefix + state, name word + street). The compound keys exist for
   scale: on the ~6M-record US pool many common single keys exceed the 10,000 block limit and would be dropped
   entirely; the compound keys stay small.
4. **Pool + re-rank.** A pool of 100 candidates per S1 entity is pulled by *rarity-weighted* key overlap (each
   key contributes `1/log2(posting size + 2)`), then cut to the best **30** by
   `0.25·cos(name TF-IDF) + 0.5·cos(address TF-IDF) + 0.4·key weight/6`. Address is weighted about twice the name
   because many real matches share an address but not a name. The weights were fixed on a training world only.
   The lexical keys reach ~99% (US) / ~96% (India) recall uncapped on the development sub-world, while a plain
   30-candidate cap keeps ~95% / ~93%; the re-rank recovers most of that difference at the same candidate-set size.
5. **`candidate_pairs.tsv`** is written from exactly this final 30-candidate set — the set the model scores.
6. *Optional* (`--use_embeddings`): multilingual sentence embeddings add up to 5 semantic neighbours per query for
   native-script targets (§9).

---

## 5. Features (`src/features.py`)

Computed per (S1, candidate) pair; fuzzy scorers run on RapidFuzz's multi-threaded pairwise API, token features on
a process pool.

| Family | Features |
|---|---|
| Name (normalized) | Levenshtein ratio, Jaro-Winkler, token sort, token set, partial ratio |
| Name (legal-suffix-stripped) | Jaro-Winkler, token sort, exact match, first-token match, common-token fraction |
| Name structure | token-length ratio, character-length ratio, shared-token count, whole-token **prefix** match (truncated names), space-free "compact" ratio / gated partial ratio (glued domain-style names) |
| Address | token Jaccard, character 3-gram cosine, token-set ratio, plain ratio, `addr_missing` flag |
| Numbers | postal/number overlap, exact match, first-number match, **`nums_conflict`** (each side has a number the other lacks, e.g. `15/383` vs `15/404`), `first_num_conflict` |
| Location | **`state_match`**, **`state_conflict`** (canonical state codes both known and equal / different) |
| IDF-weighted | name and address TF-IDF cosine (common words such as *pizza*, *road* count for little) |
| Optional | `semantic_sim` (embedding cosine; NaN when the target was not encoded) |

---

## 6. Model and Decision

### 6.1 Classifier (`src/training.py`, `src/model.py`)
XGBoost (`tree_method=hist`, all cores, early stopping on validation logloss). It is trained on candidate pairs
produced by **the same blocking and feature code used at inference** for a country-stratified sample of S1
entities, labelled from the ground truth — so its negatives are the genuinely confusable near-misses blocking
surfaces, not random records. Train and validation are split **by S1 entity**, never by pair. In `--validate`
mode the held-out val/test entities are excluded from training.

### 6.2 One-owner normalization (`src/decision.py`)
Because every matched record has exactly one owner (§2.2), a target that several S1 entities all find plausible can
belong to at most one of them. With `w = p/(1−p)`, the probability that S1 entity *i* owns target *t* is
`w_i / (1 + Σ_j w_j)` over the S1 entities competing for *t*; the `1` is the "owned by none of them" outcome
(~26% of targets are distractors). A target with a single competitor keeps its score; a runner-up is pushed down.
Competition is computed over **all** S1 entities of the country after the whole country has been scored.

### 6.3 Threshold and caps
A threshold tuned on the **validation fold** against the real metric (macro F0.5 with the caps applied), then at
most the 5 best S2 and 6 best S3 matches per entity (the empirical maxima, §2.2). Validation compares the raw
probability and the one-owner score on the same pairs and keeps the rule with the better **validation** F0.5; the
threshold, rule and metadata are stored beside the model (`*_calibration.json`) and read by inference.

---

## 7. Validation and Results

### 7.1 Protocol
S1 entities are split by country-stratified random draw into train / val / test. The threshold and decision rule
are chosen on **val**; the reported figure is on **test**, which influenced neither. All S1 entities are scored so
that target competition is complete; only val/test pairs are evaluated. Candidate quality (recall ceiling, average
candidates per entity, reduction ratio) is reported with it.

### 7.2 Results

| Quantity | Value |
|---|---|
| Platform score of the first version (untrained heuristic, threshold 0.62) | **0.616** |
| Held-out test-fold macro F0.5, current pipeline, full train split | «FILL» |
| — US / India per-country test F0.5 | «FILL» / «FILL» |
| Blocking recall ceiling / average candidates per entity | «FILL» / «FILL» |
| Raw-probability vs one-owner test F0.5 (same pairs) | «FILL» / «FILL» |
| Platform score, current pipeline | «FILL» |

**Development proxy (indicative only).** On a small *closed sub-world* of the real training data (every S1 entity
whose name begins with `z, y, q, x` — 44,736 entities — plus the targets they own and the same slice's
distractors; disjoint train and test worlds; test world 15,658 entities / 54,215 true pairs; built by
`scripts/make_subworld.py`, scored by `scripts/score_submission.py`), the successive changes scored:

| Pipeline | Macro F0.5 (proxy test world) |
|---|---|
| First version (untrained heuristic, 0.62) | 0.789 |
| Trained classifier on blocking-derived pairs | 0.952 |
| + candidate re-rank | 0.959 |
| + noise normalization + state / number-conflict features | 0.966 |
| + one-owner decision | 0.974 |

This proxy keeps the real ownership structure, singleton rate and noise but is ~50× smaller than the real pool, so
it is optimistic: the first version scored 0.789 on it against 0.616 on the platform. These figures show the
*direction and relative size* of each change, not the expected platform score. The proxy used a scikit-learn
gradient-boosting stand-in for XGBoost. The compound blocking keys and France (absent from training data) are not
reflected in it.

---

## 8. Reproduction

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu   # only needed for --use_embeddings
pip install -r aads_submission/business_entity_resolution/code/requirements.txt
export PYTHONPATH=aads_submission/business_entity_resolution/code

# train + calibrate + held-out evaluation (writes models/entity_model.json + *_calibration.json)
python3 -m src.pipeline --data_dir student_resource/dataset --is_train --validate

# submission files (output/matching_results.tsv, output/candidate_pairs.tsv)
python3 -m src.pipeline --data_dir student_resource/dataset

python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
A Dockerfile at the repository root runs the same commands (`python:3.11-slim`, CPU-only PyTorch, dataset / models /
output bind-mounted). Peak memory is bounded by the largest single country; the pipeline logs `[mem]` checkpoints.

---

## 9. Compliance

- **Final model:** XGBoost gradient-boosted trees — Apache-2.0, a few hundred small trees, far below the 8B-parameter limit.
- **Optional semantic component** (off by default): `sentence-transformers/LaBSE` (Apache-2.0, ~0.5B parameters;
  109 languages), or `BAAI/bge-m3` (MIT), or `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, ~118M), used
  unmodified; FAISS (MIT) for nearest-neighbour search. Model licences, sizes and language coverage were read from
  each model's Hugging Face page.
- **No external lookups:** no APIs, registries or geocoders. The only bundled reference data are the transliteration
  table (`text-unidecode`) and small hand-written dictionaries (legal suffixes, US/India state names, street types).
- All libraries are open-source and pinned in `requirements.txt`.

---

## 10. Limitations

- **France** has no training data. It is processed by the same code (no state features, TF-IDF fitted on its own
  pool), but the classifier and threshold were fitted on US/India, so its accuracy is unverified.
- **Domain-style target names** (~4% of targets) have noticeably lower blocking recall; a dedicated compact-name key
  is a natural next step.
- **Scale:** the compound keys and the pool size (`--pool_k`) address blocking crowding that a small development
  sub-world cannot exhibit; they should be tuned against the `recall_ceiling` printed by the full validation run.
- Some ground-truth matches share neither name nor address with their S1 entity and are unrecoverable by any
  pairwise method.
