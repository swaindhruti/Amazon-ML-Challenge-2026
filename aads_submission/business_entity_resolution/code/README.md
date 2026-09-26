# Business Entity Resolution — Team `aads`
Amazon ML Challenge 2026

## Overview
This repository contains the high-performance, memory-optimized Business Entity Resolution pipeline developed by team `aads` for the Amazon ML Challenge 2026.

The pipeline resolves entities from a deduplicated reference source (Source 1) against large, noisy target pools (Source 2 and Source 3) totaling over 11 million records across dynamic open-set countries (US, India, France).

### Key Optimizations for F₀.₅
- **Calibrated threshold** — tuned against entity-level macro F₀.₅ on held-out entities during training and saved beside the model (F₀.₅ weights precision 2× over recall); an explicit `--threshold` overrides it
- **Per-source capping** — at most top-5 S2 + top-6 S3 per S1 entity (data-driven from ground truth)
- **24 lexical/address features** (+ optional IDF cosine and `semantic_sim`): suffix-stripped name similarity, address token-set/ratio, truncation and glued-name features
- **International legal suffix removal** — LLC, Inc, Ltd, SA, SAS, SARL, GmbH, etc.
- **Training on blocking-derived hard pairs** (same code path as inference), entity-level train/val split, early stopping
- **Multi-core** feature building (`--n_jobs`, default all CPUs)
- **One-owner decision layer** (`src/decision.py`): each matched S2/S3 record belongs to exactly one S1 entity (verified on the real ground truth), so competing candidates are pushed down before the threshold
- **Candidate re-ranking**: pool of 100 by key overlap → best 30 by TF-IDF similarity (`--pool_k`, `--top_k`)
- **Symmetric noise normalization + canonical state codes** in `preprocessing.py`

### 32 GB RAM Architecture Optimizations
1. **S1 Batch Chunking**: Slices Source 1 into batches (default: `batch_size = 50,000`). Blocking, feature extraction, scoring, and output writing are executed per batch, followed by immediate `gc.collect()`.
2. **Compact Inverted Index (`np.uint32`)**: Candidate blocking maps store 32-bit unsigned integer arrays, reducing index memory by >85%.
3. **High-Frequency Pruning**: Keys indexing >10,000 target records are dropped to eliminate quadratic blowups.
4. **Candidate Capping**: Candidates capped at 30 per S1 query record by blocking key overlap count.
5. **Streaming Output Writing**: Predictions flushed incrementally to disk.

---

## Project Structure
```
code/
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── candidate_generation.py        # CompactInvertedIndex (np.uint32), pruning, candidate capping
    ├── blocking.py                    # Backward-compatibility alias
    ├── features.py                    # 24 features + optional IDF/semantic; multi-threaded RapidFuzz (cpdist)
    ├── country_pipeline.py            # Shared per-country blocking + TF-IDF + embeddings + features
    ├── training.py                    # Hard-pair training, threshold calibration, calibration JSON
    ├── parallel.py                    # Process pool + native thread config
    ├── embeddings.py                  # Optional multilingual embeddings (--use_embeddings)
    ├── model.py                       # XGBoost (500 trees) + high-precision heuristic fallback
    ├── preprocessing.py               # Text cleaning, legal suffix stripping, abbreviation normalization
    ├── evaluate.py                    # Macro F_0.5 with per-source capping and threshold tuning
    └── pipeline.py                    # Country partitioning, batching, validation, and streaming

optimize_submission.py                 # Standalone post-processing script (at repo root)
```

---

## Installation & Setup

```bash
pip install -r aads_submission/business_entity_resolution/code/requirements.txt
```

---

## Running the Pipeline

### 1. Train Model + Tune Threshold (Recommended First Step)
```bash
export PYTHONPATH=aads_submission/business_entity_resolution/code
python3 -m src.pipeline \
    --data_dir student_resource/dataset \
    --is_train --validate \
    --model_path models/entity_model_v2.json \
    --matching_out output/matching_results_train.tsv \
    --candidate_out output/candidate_pairs_train.tsv \
    --max_s2 5 --max_s3 6
```
Writes the model plus `models/entity_model_v2_calibration.json` (the threshold tuned on held-out entities).

### 2. Test Inference (Full Submission Run)
```bash
export PYTHONPATH=aads_submission/business_entity_resolution/code
python3 -m src.pipeline \
    --data_dir student_resource/dataset \
    --model_path models/entity_model_v2.json \
    --matching_out output/matching_results.tsv \
    --candidate_out output/candidate_pairs.tsv \
    --max_s2 5 --max_s3 6
```
The calibrated threshold is read automatically from the calibration file next to the model (pass `--threshold` only to override it).

### 3. Fast Subset Run (Local Testing)
```bash
export PYTHONPATH=aads_submission/business_entity_resolution/code
python3 -m src.pipeline \
    --data_dir student_resource/dataset \
    --is_train --subset 2000 \
    --matching_out output/matching_results.tsv \
    --candidate_out output/candidate_pairs.tsv \
    --threshold 0.80
```

### 4. Quick Post-Processing (Rescore Existing Candidates)
```bash
python3 optimize_submission.py \
    --data_dir student_resource/dataset \
    --candidates output/candidate_pairs.tsv \
    --output output/matching_results.tsv \
    --threshold 0.80 --max_s2 5 --max_s3 6
```

---

## Command-Line Arguments

### pipeline.py
| Argument | Default | Description |
|---|---|---|
| `--data_dir` | *(required)* | Path to dataset directory (containing `train/` and `test/`) |
| `--is_train` | `False` | Run in training mode with ground truth pairs |
| `--validate` | `False` | Run validation threshold tuning (requires `--is_train`) |
| `--matching_out` | `output/matching_results.tsv` | Final prediction results output path |
| `--candidate_out` | `output/candidate_pairs.tsv` | Candidate pairs output path |
| `--model_path` | `models/entity_model.json` | Load/save XGBoost model path |
| `--batch_size` | `50000` | S1 records per batch |
| `--top_k` | `30` | Max candidates per S1 record |
| `--threshold` | `0.80` | Decision threshold for match acceptance |
| `--max_s2` | `5` | Max S2 matches per S1 entity |
| `--max_s3` | `6` | Max S3 matches per S1 entity |
| `--subset` | `0` | Process only first N S1 records (0 = all) |

### optimize_submission.py
| Argument | Default | Description |
|---|---|---|
| `--data_dir` | *(required)* | Dataset directory |
| `--candidates` | `output/candidate_pairs.tsv` | Input candidate pairs |
| `--output` | `output/matching_results.tsv` | Output matching results |
| `--threshold` | `0.80` | Decision threshold |
| `--max_s2` | `5` | Max S2 per S1 |
| `--max_s3` | `6` | Max S3 per S1 |
| `--validate` | `False` | Evaluate against ground truth |
| `--train` | `False` | Train a new model |
| `--is_train` | `False` | Use training data |
| `--subset` | `0` | Subset size |

---

## Output Schema

The final `output/matching_results.tsv` conforms to the competition format:
- Tab-separated values (`.tsv`)
- Header: `source1_entity_id\tmatched_entity_ids`
- One row for every entity in Source 1
- Matched IDs from Source 2 and Source 3 separated by commas
- Singletons have an empty `matched_entity_ids`

```tsv
source1_entity_id	matched_entity_ids
S1-156285671	S2-611995673,S2-522132855
S1-717749279	
S1-913506265	S3-867068018,S2-813184149
```
