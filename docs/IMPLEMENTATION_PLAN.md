# Noonchi Translator — Middle Path Implementation Plan

## Context

The project has two competing visions:
- **The docx**: A full mBART-50 fine-tuning pipeline with custom formality tokens — technically impressive for grad school portfolios but a 2-3 month project requiring GPU infrastructure
- **The 2-week curriculum**: A Papago API wrapper + rule-based engine — completable but weak for a portfolio

This plan follows a **middle path** that is completable without GPU compute, demonstrates genuine research depth, and hits the portfolio goals. It uses the curriculum's pedagogical structure (Week 1 rule-based engine) but replaces the Papago plan with a FormalityResolver + Claude API approach and adds a standalone data pipeline that showcases the core research contribution.

---

## Phases

### Phase 1: Complete the Traditional NLP Engine (Week 1 curriculum — Days 2–7)

These files are designed but not yet implemented. Pseudocode/class outlines exist in `docs/ARCHITECTURE_DECISION.md`.

**Files to create:**

**`backend/nlp/traditional/english_analyzer.py`**
- `EnglishAnalyzer` class using spaCy (`en_core_web_sm`)
- Methods: `analyze(text)` → returns subject, verb, object, sentence type (declarative/interrogative)
- Reuses spaCy loading pattern from `backend/nlp/day1_tokenization_lab.py`

**`backend/nlp/traditional/dictionary.py`**
- `BilingualDictionary` class with ~50 EN→KR word entries
- Formality-aware: each word stores formal/polite/casual Korean variants
- Methods: `lookup(word, formality)` → Korean word

**`backend/nlp/traditional/reorderer.py`**
- `SyntaxReorderer` class
- Transforms spaCy dependency parse from SVO (English) → SOV (Korean)
- Handles auxiliaries and negation

**`backend/nlp/traditional/korean_generator.py`**
- `KoreanGenerator` class using `korean-conjugator`
- Methods: `conjugate(verb_stem, formality, sentence_type)` → conjugated verb ending
- Maps the three formality levels (formal/polite/casual) to appropriate endings (합쇼체/-ㅂ니다, 해요체/-아요, 해체/-아)

**`backend/nlp/traditional/translator.py`**
- `TraditionalTranslator` class that chains all four components
- Method: `translate(english_text, formality_token)` → Korean string

---

### Phase 2: FormalityResolver (Core Research Component)

**`backend/formality/social_context.py`**
- `SocialContext` dataclass: `relationship` (enum: boss, peer, subordinate, professor, friend, stranger), `age_differential` (int), `setting` (enum: workplace, academic, social, public, intimate), `formality_override` (optional)

**`backend/formality/resolver.py`**
- `FormalityResolver` class
- `resolve(context: SocialContext) → str` returns `"<formal>"`, `"<polite>"`, or `"<casual>"`
- Rule-based inference encoding sociolinguistic rules, e.g.:
  - `(workplace + superior) → <formal>`
  - `(friend + similar_age + intimate) → <casual>`
  - `(acquaintance/default) → <polite>`

---

### Phase 3: LLM API Production Engine (replaces Papago)

**`backend/translation/llm_engine.py`**
- `LLMTranslator` class using Claude API (via `anthropic` SDK)
- `translate(english_text, formality_token)` → Korean string
- Formality-conditioned system prompt that instructs Claude to output in the specified speech level, referencing the exact sentence-final endings (e.g., `-ㅂ니다` for `<formal>`, `-아요` for `<polite>`, `-야/-아` for `<casual>`)
- This mirrors the doc's token-prepending approach but through prompting rather than model weights

**Updated `backend/requirements.txt`**
- Add: `anthropic>=0.20.0`, `sacrebleu>=2.0.0`

---

### Phase 4: Data Pipeline (Standalone Research Component)

Demonstrates the core research skill without requiring model training. Portfolio-worthy as a standalone artifact.

**`backend/data_pipeline/corpus_loader.py`**
- Downloads/loads Tatoeba EN-KR sentence pairs (freely available, no login required)
- Outputs raw `(en, ko)` pairs as a list

**`backend/data_pipeline/morphological_labeler.py`**
- `MorphologicalLabeler` class using KoNLPy (Mecab with Okt fallback — same pattern as `day1_tokenization_lab.py`)
- `extract_ef_morpheme(korean_sentence)` → final EF morpheme string
- `label(ef_morpheme)` → `"<formal>"`, `"<polite>"`, `"<casual>"`, or `None` (unknown)
- Suffix rule table (from doc):
  - `-습니다/-ㅂ니다/-습니까` → `<formal>`
  - `-아요/-어요/-여요` → `<polite>`
  - `-아/-어/-냐/-지/-구나` → `<casual>`

**`backend/data_pipeline/pipeline.py`**
- `run_pipeline(raw_pairs)` → labeled TSV
- Stages: morphological analysis → confidence filtering (removes unknowns, length outliers) → class balance check → output TSV (`en`, `ko`, `formality` columns)

---

### Phase 5: Formality Accuracy Metric

**`backend/evaluation/formality_accuracy.py`**
- Reuses `MorphologicalLabeler` from Phase 4
- `formality_accuracy(predictions: list[str], requested_labels: list[str]) → float`
- Runs each predicted Korean string through morphological analysis, compares detected label to requested label

**`backend/evaluation/evaluate.py`**
- Evaluation script: takes a test set TSV, runs both the traditional engine and LLM engine, reports chrF (via sacrebleu), BLEU, and Formality Accuracy for each

---

### Phase 6: FastAPI Backend

**`backend/api/main.py`**
- `POST /translate` — accepts `{ text: str, social_context: SocialContext }`, returns `{ korean: str, formality_token: str, engine: str }`
- `POST /translate/compare` — runs both engines and returns side-by-side results
- `GET /formality/resolve` — returns resolved formality token for a given social context

---

### Phase 7: React Frontend

**`frontend/` (new directory)**
- Two-phase UI matching the doc's design:
  - **Phase 1**: Social context form — dropdowns for relationship, setting, age differential slider, optional formality override
  - **Phase 2**: Text input + translation display showing Korean output with speech level label
- Side-by-side comparison view: Traditional Engine vs. LLM Engine outputs
- Built with Vite + plain React (no heavy UI framework needed)

---

## File Structure After Completion

```
noonchi-translator/
├── backend/
│   ├── nlp/
│   │   ├── day1_tokenization_lab.py   ✅ exists
│   │   └── traditional/
│   │       ├── english_analyzer.py    🔨 build
│   │       ├── dictionary.py          🔨 build
│   │       ├── reorderer.py           🔨 build
│   │       ├── korean_generator.py    🔨 build
│   │       └── translator.py          🔨 build
│   ├── formality/
│   │   ├── social_context.py          🔨 build
│   │   └── resolver.py                🔨 build
│   ├── translation/
│   │   └── llm_engine.py             🔨 build
│   ├── data_pipeline/
│   │   ├── corpus_loader.py           🔨 build
│   │   ├── morphological_labeler.py   🔨 build
│   │   └── pipeline.py               🔨 build
│   ├── evaluation/
│   │   ├── formality_accuracy.py      🔨 build
│   │   └── evaluate.py               🔨 build
│   ├── api/
│   │   └── main.py                   🔨 build
│   ├── check_installation.py          ✅ exists
│   └── requirements.txt               ✅ update
└── frontend/                          🔨 build
    └── src/
        ├── App.jsx
        ├── components/
        │   ├── SocialContextForm.jsx
        │   ├── TranslationInput.jsx
        │   └── ComparisonView.jsx
        └── api/client.js
```

---

## Critical Files to Reuse

- `backend/nlp/day1_tokenization_lab.py` — Mecab/Okt fallback pattern, KoNLPy usage
- `backend/check_installation.py` — dependency verification pattern
- `docs/ARCHITECTURE_DECISION.md` — class outlines and pseudocode for Phase 1 components

---

## Verification

1. **Phase 1**: `python backend/nlp/traditional/translator.py` → translates "Do you want to eat?" in all three formality levels
2. **Phase 2**: `python -c "from backend.formality.resolver import FormalityResolver; ..."` → resolves (boss, workplace) → `<formal>`
3. **Phase 3**: `python backend/translation/llm_engine.py` → returns Korean output with correct ending for each formality
4. **Phase 4**: `python backend/data_pipeline/pipeline.py` → outputs labeled TSV with balanced class distribution
5. **Phase 5**: `python backend/evaluation/evaluate.py` → prints chrF, BLEU, FormAcc for both engines on test set
6. **Phase 6**: `uvicorn backend.api.main:app --reload` → API running at localhost:8000
7. **Phase 7**: `npm run dev` in `frontend/` → UI accessible at localhost:5173, translates a sentence end-to-end through both engines
