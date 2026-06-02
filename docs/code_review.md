# Code Review: Noonchi Translator

Reviewed 2026-05-11. Covers all Python modules, the Colab notebook, frontend, tests, and project
infrastructure. Organized from highest to lowest severity.

---

## 1. Critical Issues

### 1.1 API Key Committed to Repository

**File:** `.env`

A live `ANTHROPIC_API_KEY` is committed and readable in the repo. This is a security incident
regardless of whether the key is still active.

**Fix:**
1. Rotate the key immediately at console.anthropic.com.
2. Add `.env` to `.gitignore` (check that it's actually excluded, not just listed).
3. Remove from git history: `git filter-branch` or `git filter-repo`.
4. Use `.env.example` (already present) as the committed template.

---

### 1.2 mBART Inference Code Is Complete; Checkpoint Is Not Local

**Files:** `backend/model/inference.py`, `backend/api/agents/mbart_agent.py`

Both files are fully implemented and correct. `MBartInference` loads the tokenizer and model
from a directory via `from_pretrained`, runs beam search, and decodes. `mbart_agent.py` wires
it correctly to the same `TranslationResponse` schema as the Claude agent.

The mBART backend cannot run locally because the trained checkpoint lives on **Google Drive**
(saved from Colab training) and is not in the repo. The code assumes `models/noonchi-mbart/`
exists locally.

**This is not a code bug — it is a workflow gap.** To run the mBART backend:
1. Download the checkpoint from Drive to `models/noonchi-mbart/`
2. Start the API with `BACKEND=mbart MODEL_DIR=models/noonchi-mbart`

Optional improvement: add a `scripts/download_checkpoint.py` using `gdown` to automate the
Drive download so the setup step is one command rather than manual.

---

### 1.3 Data Leakage: Augmentation Happens Before Split — FIXED

**Files:** `backend/data_pipeline/pipeline.py`, `backend/data_pipeline/split.py`

**Fixed.** The pipeline now splits first, then augments the training portion only:
- `split.py` gained a `split_rows()` function (in-memory, importable)
- `pipeline.py` imports `split_rows`, performs the stratified split on `filtered` data
  immediately after Stage 4, then runs all augmentation (substitution, LLM, triplets) on the
  training split only
- Val and test TSVs contain only original-corpus sentences
- `--output` replaced by `--output-dir`; `--train`, `--val`, `--split-seed` added
- `split.py` is unchanged as a standalone tool for splitting pre-built datasets

---

### 1.4 Test Import Path Broken — FIXED

**File:** `tests/test_formality_resolver.py`, line 12

**Fixed.** `FormalityResolver` and all associated types were moved to `backend/formality/resolver.py`.
The test's existing import path now resolves correctly. All 32 parameterized unit tests pass.

---

## 2. Data Pipeline Issues

### 2.1 Korean Length Filtering Is Meaningless — FIXED

**File:** `backend/data_pipeline/filter.py`

**Fixed.** `ko_len` now uses `len(get_mecab().morphs(ko))` (morpheme count) instead of
whitespace tokenization. Per-class bounds were re-derived from the morpheme-count distribution
of the 338K-row training set (formal KO: 3–30, polite/casual KO: 3–35).

---

### 2.2 Empty and Whitespace-Only Sentences Pass the Filter

**File:** `backend/data_pipeline/filter.py`, lines 34–42

If both `en` and `ko` contain only whitespace, `.split()` returns `[]`, so `en_len = 0`
and `ko_len = 0`. Both fall below `MIN_TOKENS = 3` and the pair is correctly rejected
by the length check. However, if `en` has three or more non-whitespace words but `ko` is
purely whitespace (`ko_len = 0`), the Korean bounds check passes (0 < 3 is True → rejected).
So the filter is correct for truly empty strings but would admit a Korean sentence that is
a single long whitespace-padded word (ko_len = 1 ≥ 3 is False → rejected).

**Status:** This is actually handled correctly. No code change needed, but the guard at line
42 (`max(min(en_len, ko_len), 1)`) means a (0, 0) pair would produce ratio 0 which passes
the 4.0 cap. Worth adding an explicit `if en_len == 0 or ko_len == 0: return False` check
to make the intent clear.

---

### 2.3 LLM Augmentation Does Not Deduplicate Against Original Corpus

**File:** `backend/data_pipeline/augment.py`, `augment_by_llm()`, line 304

`seen_ko` is seeded from pairs already collected in the checkpoint, but not from the full
`filtered` corpus. If the LLM generates a Korean sentence that already appears in the
original corpus (possible for common phrases like "감사합니다"), that sentence enters
the dataset twice — once from the original with one English source, and once as a synthetic
pair with a different English source.

This doesn't corrupt labels (both are verified by `label_sentence`), but it introduces
exact-Korean-duplicate pairs that skew model training toward common completions.

**Fix:** Seed `seen_ko` from the full `pairs` argument in addition to the checkpoint.

---

### 2.4 Triplet Counter Can Drift If LLM Output Is Unusual

**File:** `backend/data_pipeline/augment.py`, line 496

```python
collected_triplet_count += len(new_triplets) // 3
```

`_parse_and_verify_triplets` always adds multiples of 3 (it's all-or-nothing per triplet),
so `len(new_triplets) % 3 == 0` should always hold. But if a future bug causes partial
triplets to slip through, the counter would undercount, causing `augment_by_triplets` to
over-generate. Safe now; worth adding an assertion:

```python
assert len(new_triplets) % 3 == 0, "triplet verifier returned partial triplet"
collected_triplet_count += len(new_triplets) // 3
```

---

### 2.5 merge_llm.py Is Not Atomic

**File:** `backend/data_pipeline/merge_llm.py`, lines 72–74

The script opens `train.tsv` in append mode and writes directly. Running it twice appends
duplicate rows. There is no idempotency guard. If the process crashes mid-append, the file
is corrupted (partial row at end).

**Fix:** Write to a temp file, then `os.replace(tmp, train_tsv)` for atomic swap. Add a
check that the target row count increases by exactly the expected delta before replacing.

---

### 2.6 extract.py — Fully Implemented; Corpus Data Files Required

**File:** `backend/data_pipeline/extract.py`

Both `load_tatoeba` and `load_opus_opensubtitles` are fully implemented. Tatoeba supports
dual-path loading (per-language `.tsv` files or combined `sentences.csv`). OPUS uses
streaming `iterparse` with `elem.clear()` for the ~500MB gzipped TMX. Both raise
`FileNotFoundError` with download URLs when corpus files are absent.

**Status:** No code changes needed. Running the pipeline without corpus data fails with a
descriptive error that includes the download URL — this is correct behavior for large
external datasets that aren't committed to the repo.

---

## 3. Model Layer Issues

### 3.1 max_new_tokens Mismatch: Training=200, Inference=128 — FIXED

**Files:** `backend/model/train.py`, `backend/model/inference.py`, `backend/model/evaluate.py`

**Fixed.** Both `inference.py` and `evaluate.py` now use `max_new_tokens=200`, matching
the `generation_max_length: 200` set in training.

---

### 3.2 Stratified Sample Silently Loses Up to 2 Rows

**File:** `backend/model/dataset.py`, `_stratified_sample()`, around line 99

```python
per_class = n // len(by_class)  # e.g., 50000 // 3 = 16666
```

`16666 × 3 = 49998`, not 50000. The user passes `--max-rows 50000` and gets 49998 rows.
For validation: `2000 // 3 = 666` → 1998 rows evaluated. Minor data loss, but the reported
counts are off from what users expect.

**Fix:** Distribute the remainder rows across classes:

```python
remainder = n % len(by_class)
for i, (cls, rows) in enumerate(sorted_classes):
    take = per_class + (1 if i < remainder else 0)
    sampled.extend(rows[:take])
```

---

### 3.3 Formality Accuracy Metric Conflates Two Failure Modes — FIXED

**File:** `backend/evaluation/metrics.py`

**Fixed.** `formality_accuracy()` now tracks `none_count` separately and reports accuracy as
`correct / classifiable` (excluding unclassifiable predictions). `evaluate()` surfaces
`fa_none_count` in its results dict.

---

### 3.4 No Per-Class Evaluation Breakdown — FIXED

**File:** `backend/model/evaluate.py`, `backend/evaluation/metrics.py`

**Fixed.** `evaluate_by_class()` added to `metrics.py`; `evaluate.py` calls it and prints
per-class chrF, formality accuracy, unclassifiable count, and n for formal/polite/casual.

---

### 3.5 Inconsistent num_beams Across Pipeline Stages

**Files:** `train.py` line 75 (`num_beams=1`), `evaluate.py` line 91 (`num_beams=4`),
`inference.py` line 50 (`num_beams=4`)

Mid-training validation uses greedy decoding; final eval and production use beam-4. The
reported mid-training chrF values (in training logs) are not directly comparable to the
final chrF=28.64 from Cell 8. This is intentional (documented) but should be prominently
noted when reporting results, since the training curves understate model quality.

---

## 4. API and Backend Issues

### 4.1 Session Cleanup Is Never Called — FIXED

**File:** `backend/api/app.py`

**Fixed.** A `_cleanup_loop` coroutine runs every 300 seconds inside the FastAPI `lifespan`
context manager. It is cancelled cleanly on shutdown. Uses the modern lifespan pattern,
not the deprecated `@app.on_event`.

---

### 4.2 RAG Retriever Catches All Exceptions Silently

**File:** `backend/api/rag/retriever.py`, lines 59–68

```python
except Exception:
    logger.warning("RAG retrieval failed; continuing without augmentation", exc_info=True)
    return []
```

A bare `except Exception` catches ChromaDB failures, type errors, and programming mistakes
alike. If the ChromaDB collection is mis-configured or the metadata filter key is wrong,
the RAG system silently returns no results for every request without any observable failure.

**Fix:** Catch only expected exceptions (`chromadb.errors.ChromaError`) and let programming
errors propagate. Add a startup check that verifies the collection is reachable.

---

### 4.3 relationship_formality_map.json Is Out of Sync with Code

**File:** `data/relationship_formality_map.json` vs. `backend/api/agents/claude_agent.py`

The JSON file maps relationships to all 7 Korean speech levels and includes relationship
types (`customer`, `teacher`, `younger_friend`, `younger_sibling`, `child`) that have
no corresponding `RelationshipType` enum value in the Python code. The JSON is not imported
or used at runtime — the resolver hardcodes its rules directly in Python.

**Impact:** The JSON is described in `CLAUDE.md` as authoritative, but it isn't. If someone
updates the JSON expecting the code to change, nothing happens. If the Python rules change,
the JSON goes stale.

**Fix:** Either (a) load the JSON at startup and derive resolver rules from it, making it
the actual source of truth, or (b) delete the JSON and update `CLAUDE.md` to acknowledge
the rules are code-only.

---

### 4.4 FormalityResolver Is Defined Twice — FIXED

**Files:** `backend/formality/resolver.py`, `backend/api/models/schemas.py`

**Fixed.** `FormalityResolver`, `SocialContext`, `RelationshipType`, `SettingType`, and
`FormalityToken` now live exclusively in `backend/formality/resolver.py`. `schemas.py`
imports the three enum types from there instead of re-defining them. `app.py` adds the
project root to `sys.path` at startup so the import resolves from any module in the package.

---

### 4.5 Claude Model for Context Parsing Is Hardcoded

**File:** `backend/api/agents/claude_agent.py`, line 39

The `parse_situation()` method hardcodes `"claude-haiku-4-5-20251001"` rather than reading
from `config.CLAUDE_MODEL`. The translation step correctly uses the config value; the parsing
step does not. If you want to upgrade the parser to Sonnet for better accuracy, you'd need to
edit code rather than config.

---

### 4.6 formality_override Not Returned in API Response

**File:** `backend/api/app.py`, `set_context()` response

When a user provides `formality_override` in the request, the final `formality_token` in the
response reflects the override — but the response includes no field indicating that an override
was applied. The frontend has no way to show "You requested casual regardless of context" vs.
"The system inferred casual from your context."

**Fix:** Add `override_applied: bool` to `ContextResponse`.

---

## 5. Project Structure Issues

### 5.1 Test Suite Cannot Run in CI

**File:** `tests/test_formality_resolver.py`

Beyond the broken import path (issue 1.4), the API integration tests (`test_translate`,
`test_set_context`, etc.) require a running FastAPI server at `localhost:8000`. They are
written as `requests`-based integration tests, not `pytest`-compatible unit or
`TestClient`-based tests. Running `pytest tests/` fails on both import errors and connection
refused errors.

**Fix:** Use FastAPI's `TestClient` for API tests:
```python
from fastapi.testclient import TestClient
from backend.api.app import app
client = TestClient(app)
```

---

### 5.2 Data Pipeline Has Zero Test Coverage — FIXED

**File:** `tests/test_data_pipeline.py`

**Fixed.** 22 tests added covering:
- `_to_formal`: 10 parametrized cases (regular, ㄹ-irregular, ㄷ-irregular, ㅅ-irregular,
  multi-word stems)
- `label_sentence`: 6 parametrized cases across formal/polite/casual + None for non-Korean
- `is_valid_pair`: 5 cases (valid pair, EN too short, formality=None, unknown formality,
  EN too long)

Note: `label_sentence` tests use full-sentence inputs rather than bare verb forms — Mecab
tags short standalone verbs as EC (connective) rather than EF (sentence-final), so bare
`먹습니다` returns None. The test cases reflect actual system behavior.

---

### 5.3 Orphaned and Duplicate Files

Files that should be deleted or archived:

| File | Reason |
|---|---|
| `Untitled-1.ipynb` | Empty junk notebook (255 bytes) |
| `romanize.ipynb` | Abandoned romanization experiment |
| `noonchi_translator_documentation (1).docx` | Superseded by CLAUDE.md |
| `notebooks/train_colab copy.ipynb` | Backup of training notebook |
| `notebooks/train_colab copy 2.ipynb` | Second backup |
| `backend/data_pipeline/merge_llm.py` | Never imported; logic subsumed by pipeline.py |

---

### 5.4 Requirements Are Split Across Two Backends Inconsistently

`backend/requirements.txt` (ML stack) and `backend/api/requirements.txt` (Claude API stack)
exist separately but are not clearly documented as alternatives. `anthropic==0.18.0` is pinned
in the API requirements but `anthropic>=0.40.0` in the ML requirements — if both are installed
in the same environment, the lower pin wins and API calls using newer SDK features would fail.

**Fix:** Consolidate into one `requirements.txt` at the project root (production) and a separate
`requirements-dev.txt` for test/lint tools.

---

## 6. Gaps Worth Tracking

These are not bugs but genuine missing pieces for a complete system:

| Gap | Impact | Effort |
|---|---|---|
| mBART inference not implemented | Can't demo trained model | Medium |
| No per-class eval metrics (chrF@formal, etc.) | Can't diagnose formality accuracy gap | Low |
| extract.py corpus loading not implemented | Can't re-run data pipeline from scratch | High |
| Korean length measured in whitespace tokens | Length filter bounds are approximate | Medium |
| No request length validation on translation text | Edge case: very long inputs not handled | Low |
| Session expiry not communicated to user | UX gap: sessions expire silently after 30 min | Low |
| RAG priority weighting absent | Formal notes may surface for casual queries | Medium |

---

## 7. What Is Working Well

To be balanced: the following are implemented correctly and represent real quality:

- **FormalityResolver**: Sociolinguistically sound rule encoding, 38-test coverage, clean
  dataclass interface. The best piece of the project.
- **mBART tokenizer setup**: Special token injection, embedding resize, and decoder target
  language handling are all correct. The `generation_config` / `model.config` split for
  `forced_bos_token_id` avoids the transformers ≥4.46 `ValueError`.
- **Irregular verb handling in augment.py**: Unicode Hangul arithmetic for ㄹ-irregular,
  lookup tables for ㄷ and ㅅ, with correct disambiguation of ambiguous stems. This is
  sophisticated and well-documented.
- **Checkpoint/resume in LLM augmentation**: Adaptive batch sizing, consecutive failure
  detection, and graceful KeyboardInterrupt handling are all production-grade patterns.
- **API schema validation**: Pydantic models correctly validate formality enums, enforce
  the either-or path requirement, and handle optional fields consistently.
- **Two-step UX design**: The separation of context resolution from translation is the right
  call — it surfaces the formality decision explicitly rather than burying it in the prompt.

---

## Priority Fix Order

1. **Rotate API key** — immediate, before anything else ← **ACTION REQUIRED**
2. ~~**Fix test import path**~~ — done; `backend/formality/resolver.py` created; 32 tests pass
3. ~~**Download mBART checkpoint locally**~~ — done; checkpoint at `models/noonchi-mbart/`
4. ~~**Fix data pipeline ordering**~~ — done; pipeline now splits before augmenting
5. ~~**Add per-class eval metrics**~~ — done; `evaluate_by_class()` in metrics.py + evaluate.py
6. ~~**Fix max_new_tokens mismatch**~~ — done; inference.py and evaluate.py both use 200
7. ~~**Add Korean morpheme-based length counting**~~ — done; filter.py uses `get_mecab().morphs()`
8. ~~**Consolidate FormalityResolver**~~ — done; canonical location is `backend/formality/resolver.py`; schemas.py imports from there
9. ~~**Schedule session cleanup**~~ — done; `_cleanup_loop` in app.py lifespan, runs every 300s
10. ~~**Add data pipeline unit tests**~~ — done; 22 tests in `tests/test_data_pipeline.py`
