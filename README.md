# Noonchi Translator

A full-stack English-to-Korean translation system that addresses a critical gap in existing tools: **context-aware selection of Korean honorifics**. Unlike commercial tools that offer only a binary formal/informal toggle, Noonchi Translator models the Korean speech level system and automatically determines the appropriate register from the interpersonal relationship between speaker and addressee.

**"눈치" (noonchi)** — the Korean concept of social awareness; reading a room and responding accordingly.

---

## The Problem

Korean grammaticalizes social relationships directly into verb morphology through a system of speech levels (경어법). Existing NMT systems underspecify this pragmatic dimension, producing translations that are grammatically correct but socially inappropriate.

| Tool | Honorific Handling | Limitation |
|---|---|---|
| Google Translate | Binary formal/informal | No social context modeling |
| Papago | Basic toggle | No relationship inference |
| NLLB / mBART (baseline) | Implicit from training data | Cannot be conditioned at inference |
| **Noonchi Translator** | **3-tier, context-inferred** | **This project's contribution** |

---

## How It Works

```
Social Context Input
(relationship, age differential, setting)
        ↓
FormalityResolver
(rule-based pragmatic inference)
        ↓
Conditioning Token  ←──  <formal> | <polite> | <casual>
        ↓
Fine-tuned mBART-50
(formality-conditioned translation)
        ↓
Korean Output
```

1. **User provides** English text + social context (speaker-addressee relationship, age differential, situational setting)
2. **FormalityResolver** applies sociolinguistic rules to select one of three speech levels
3. **Conditioning token** is prepended to the source sentence before encoding
4. **mBART-50** (fine-tuned on formality-labeled parallel data) generates Korean at the target speech level
5. **Evaluation** reports chrF, BLEU, and a custom Formality Accuracy metric

---

## Korean Speech Levels

The system consolidates the seven-level Korean speech system (경어법) into three operative tiers for training and inference:

| Token | Covers | Example Endings | Context |
|---|---|---|---|
| `<formal>` | 합쇼체 / 하십시오체 | -습니다 / -ㅂ니다 | Boss, professor, public speech, interview |
| `<polite>` | 해요체 | -아요 / -어요 | Colleagues, service staff, neutral adult interaction |
| `<casual>` | 해라체 / 해체 | -아 / -어 / -냐 | Close friends, peers, family of similar age |

---

## Project Structure

```
noonchi-translator/
├── backend/
│   ├── api/                        # Working prototype (Claude API + FastAPI)
│   │   ├── app.py                  # FastAPI server
│   │   ├── agents/claude_agent.py  # Translation agent + FormalityResolver
│   │   ├── models/schemas.py       # Pydantic request/response models
│   │   ├── session_manager.py
│   │   ├── config.py
│   │   └── requirements.txt
│   ├── formality/                  # Standalone FormalityResolver module
│   │   └── resolver.py
│   ├── data_pipeline/              # Corpus labeling pipeline
│   │   ├── extract.py              # Corpus extraction (OpenSubtitles, Tatoeba)
│   │   ├── label.py                # Morphological analysis + suffix labeling
│   │   ├── filter.py               # Confidence filtering
│   │   ├── augment.py              # Class balancing via suffix substitution + LLM
│   │   └── pipeline.py             # End-to-end orchestration
│   ├── model/
│   │   └── train.py                # mBART-50 fine-tuning
│   ├── evaluation/
│   │   └── metrics.py              # chrF, BLEU, Formality Accuracy
│   └── requirements.txt
├── frontend/                       # React + Vite UI
│   └── src/
│       ├── App.jsx
│       ├── components/
│       └── services/
├── data/
│   └── relationship_formality_map.json
├── tests/
│   └── test_api.py
├── docs/
│   ├── ARCHITECTURE_DECISION.md
│   └── IMPLEMENTATION_PLAN.md
├── .env.example
└── pyproject.toml
```

---

## Components

### FormalityResolver

Rule-based pragmatic inference engine. Maps structured social context to one of three formality tokens by encoding Korean sociolinguistic norms:

```python
context = SocialContext(
    relationship="boss",
    age_differential=-10,   # speaker is younger
    setting="workplace"
)
resolver.resolve(context)  # → "<formal>"
```

Social context fields:

| Field | Type | Example Values |
|---|---|---|
| `relationship` | enum | boss, peer, subordinate, professor, friend, stranger |
| `age_differential` | int | -10 (speaker younger), 0 (same), +10 (speaker older) |
| `setting` | enum | workplace, academic, social, public, intimate |
| `formality_override` | optional | formal, polite, casual |

### Data Pipeline

Six-stage pipeline transforming raw Korean–English parallel corpora into formality-labeled training data:

1. **Corpus extraction** — OPUS OpenSubtitles + Tatoeba EN–KR pairs
2. **Morphological analysis** — KoNLPy/Mecab POS tagging; extract sentence-final EF morpheme
3. **Suffix labeling** — pattern rules map EF endings to formality token:
   - `-습니다 / -ㅂ니다 / -습니까` → `<formal>`
   - `-아요 / -어요 / -여요` → `<polite>`
   - `-아 / -어 / -냐 / -지 / -구나` → `<casual>`
4. **Confidence filtering** — remove fragments, ambiguous endings, length outliers (~40–60% removed)
5. **Class balancing** — suffix substitution + LLM-assisted generation for underrepresented registers
6. **Dataset output** — TSV with `(en, ko, formality)` columns

### mBART-50 Fine-tuning

Base model: `facebook/mbart-large-50-many-to-many-mmt`

- Tokenizer expanded with `<formal>`, `<polite>`, `<casual>` conditioning tokens
- Embedding layers resized to accommodate new vocabulary
- Formality token prepended to source sentence at training and inference time:
  ```
  "<formal> Can you help me with this?" → "도와주시겠습니까?"
  ```
- Fine-tuned with AdamW, lr=5e-5, linear warmup, early stopping on validation loss

### Evaluation

- **chrF** (primary) — character n-gram F-score; more sensitive to Korean morphological variation than word-level BLEU
- **BLEU** (secondary) — for comparability with prior work
- **Formality Accuracy (FA)** — custom metric; applies the morphological labeling pipeline to model outputs and checks predicted endings against the requested formality token

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | React + Vite | Context elicitation UI + translation display |
| Backend | Python / FastAPI | API, formality resolver, inference orchestration |
| Korean NLP | KoNLPy + Mecab | Morphological analysis, POS tagging, EF extraction |
| English NLP | spaCy | English preprocessing |
| Translation Model | mBART-50 (HuggingFace) | Neural machine translation backbone |
| Training Data | OPUS OpenSubtitles + Tatoeba | Parallel EN–KR corpora |
| Evaluation | sacrebleu + custom FA | Translation and formality quality metrics |
| Prototype Backend | Claude API | Baseline comparison before mBART is trained |

---

## Status

| Component | Status |
|---|---|
| Working prototype (Claude API backend) | Done |
| React frontend | Done |
| FormalityResolver | Done |
| Data pipeline (corpus labeling) | In progress |
| LLM augmentation for class balance | Planned |
| mBART-50 fine-tuning | Planned |
| Evaluation (chrF, BLEU, formality accuracy) | Planned |

---

## Success Criteria

- Formality Accuracy > 80% on held-out test set across all three registers
- chrF score competitive with unconditioned mBART baseline
- System correctly varies output morphology for identical English input across all three formality levels
- Frontend usable by a non-technical Korean learner

---

## Setup

### Backend prototype
```bash
cp .env.example .env          # add your ANTHROPIC_API_KEY
cd backend/api
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd frontend
npm install && npm run dev
```
