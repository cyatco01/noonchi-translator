# Data Improvements

Tracking potential and completed improvements to the mBART training data and preprocessing pipeline.

---

## Completed

### Irregular verb handling in suffix substitution augmenter
**File:** `backend/data_pipeline/augment.py`

The suffix substitution augmenter generates formal Korean variants from polite-labeled training pairs
by stripping terminal morphemes (아요/어요/여요) and appending 습니다. This works for regular
consonant-final verb stems but silently produced wrong formal forms for three classes of irregular verb.

**Fixed:** `_to_formal()` now handles three irregular patterns before falling back to regular 습니다:

| Class | Example | Before | After |
|---|---|---|---|
| ㄹ-irregular | 알아요 → 알 | 알습니다 ✗ | 압니다 ✓ |
| ㄷ-irregular | 음악을 들어요 → 들 | 들습니다 ✗ | 음악을 듣습니다 ✓ |
| ㅅ-irregular | 집을 지어요 → 지 | 지습니다 ✗ | 집을 짓습니다 ✓ |

**How each fix works:**
- **ㄹ-irregular** (알다, 살다, 만들다, 팔다, 울다…): uses Unicode Hangul syllable arithmetic to
  swap the ㄹ 종성 (index 8) for ㅂ 종성 (index 17) in the final syllable, then appends 니다.
  알 → 압니다, 만들 → 만듭니다.
- **ㄷ-irregular** (듣다, 걷다, 묻다, 싣다): lookup table mapping the extracted polite stem to the
  correct formal stem. Matched against the full last eojeol (word) to avoid confusing 들 (듣다/listen)
  with 들 at the end of 만들 (만들다/make, which is ㄹ-regular).
- **ㅅ-irregular** (낫다, 짓다, 붓다): same last-word lookup approach, restoring ㅅ before 습니다.
  Only three stems added — 이→잇 was excluded because 이어요 collides with the copula 이다, and
  그→긋 was excluded because 긋다 (draw a line) is rare in subtitle data.

**Known ambiguities (unfixable without morphological analysis):**
- 들어요 could be 듣다 (listen, ㄷ-irregular → 듣습니다) or 들다 (lift/enter, ㄹ-regular → 듭니다)
- 걸어요 could be 걷다 (walk, ㄷ-irregular → 걷습니다) or 걸다 (hang, ㄹ-regular → 겁니다)
- 물어요 could be 묻다 (ask, ㄷ-irregular → 묻습니다) or 물다 (bite, ㄹ-regular → 뭅니다)

The current code defaults to the ㄷ-irregular reading for these, which is the more likely
interpretation for subtitle/conversational data but will be wrong some fraction of the time.
Resolving these would require a morphological parser (KoNLPy/Mecab is already used in the labeling
step and could be integrated here if precision becomes a priority).

---

### Hybrid triplet set
**File:** `backend/data_pipeline/augment.py`, `backend/data_pipeline/pipeline.py`

Added `augment_by_triplets()` — generates the same English sentence paired with all three
formality-level Korean translations. Each complete triplet is only accepted if all three Korean
sentences pass morphological verification via `label_sentence()`.

**How to use:**
```bash
python -m backend.data_pipeline.pipeline --data-dir data/ --output data/train.tsv --triplets 5000
```
This adds 5,000 triplets (15,000 rows) to the dataset. Checkpoint/resume at
`data/checkpoints/triplets.json`. Target 5–10K triplets to mix in at ~10–15% of total data.

---

### Improve suffix substitution naturalness
**File:** `backend/data_pipeline/check_naturalness.py`

Added `check_naturalness.py` — a diagnostic script that re-runs `augment_by_substitution` on a
polite sample, then asks Claude Haiku to rate whether each synthetic formal/casual sentence sounds
natural. Uses forced tool use for structured JSON output. Prints a report with pass rate and flagged
examples; saves full results to `data/check_naturalness_results.json`.

**How to use:**
```bash
python -m backend.data_pipeline.check_naturalness --data data/train.tsv --sample 100
```

---

### Reduce Mecab label noise
**File:** `backend/data_pipeline/check_labels.py`

Added `check_labels.py` — a diagnostic script that stratified-samples rows from `train.tsv` and
asks Claude Haiku to independently label each Korean sentence's formality. Compares Claude's label
to the pipeline label, reports mismatch rate per class, and flags the specific sentence-final
endings that are mismatched most often. Full results saved to `data/check_labels_results.json`.

**How to use:**
```bash
python -m backend.data_pipeline.check_labels --data data/train.tsv --sample 500
```

---

### Length filtering by formality class
**File:** `backend/data_pipeline/filter.py`

Replaced the global `MIN_TOKENS=3 / MAX_TOKENS=150` bounds with per-class bounds derived from
the p99 of the actual training distribution plus ~50% headroom:

| Class | EN bounds | KO bounds | p99 EN | p99 KO |
|---|---|---|---|---|
| formal | [3, 30] | [3, 20] | 21 | 14 |
| polite | [3, 35] | [3, 25] | 24 | 16 |
| casual | [3, 35] | [3, 25] | 24 | 16 |

Formal Korean is structurally tighter at the upper tail because honorific verb endings are compact;
long sentences in the formal class are more likely misaligned or corrupted pairs than valid data.
The new bounds remove 920 rows (0.27%) from the existing training set — mostly from formal (645),
which had outlier sentences reaching EN=52, KO=42 tokens under the old 150-token cap.

---

### Larger validation set for more reliable metrics
**File:** `backend/model/train.py`

Added `--val-max-rows` CLI argument to the training script. The 2,000-row eval cap is now a
configurable default rather than a hardcode — pass `--val-max-rows 0` to evaluate on the full
50K-row `val.tsv` when running on a better GPU.

```bash
# T4 Colab (default — fast):
python -m backend.model.train --data data/train.tsv --output models/noonchi-mbart

# A100 / university cluster (full validation set):
python -m backend.model.train --data data/train.tsv --output models/noonchi-mbart --val-max-rows 0
```

---

## Planned

### Train on the full dataset (423K rows vs. 50K)

Current training used a 50K-row stratified sample due to Colab T4 time limits. The full dataset
(423K rows: 32% formal / 33% polite / 35% casual) already exists at `data/train.tsv`. Training on
the full dataset would improve translation quality and register coverage, particularly for edge-case
sentence structures and rare vocabulary.

**Effort:** Run the training script with `--max-rows` removed or set to the full count. Requires
either a longer Colab session or a better GPU (A100 Colab, university cluster, or cloud instance).
