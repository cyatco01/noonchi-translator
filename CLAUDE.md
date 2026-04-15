# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

**Noonchi Translator**: A context-aware English-to-Korean translation system. Users provide social context (relationship, ages, setting) and the system infers the appropriate Korean speech level, prepends a conditioning token, and produces formality-conditioned output via a fine-tuned mBART-50 model.

The name comes from **눈치** — the Korean concept of social awareness and reading a situation.

---

## Architecture

### System Flow

```
Social Context (relationship, age_differential, setting)
        ↓
FormalityResolver  →  <formal> | <polite> | <casual>
        ↓
"<formal> Do you want to eat?"
        ↓
mBART-50 (fine-tuned, formality-conditioned)
        ↓
"드시고 싶으십니까?"
```

### Components

**FormalityResolver** (`backend/formality/`)
- `SocialContext` dataclass: relationship, age_differential, setting, optional override
- Rule-based inference encoding sociolinguistic rules:
  - `(workplace + superior)` → `<formal>`
  - `(friend + similar_age + intimate)` → `<casual>`
  - default → `<polite>`

**Data Pipeline** (`backend/data_pipeline/`)
- Sources: Korean OpenSubtitles, Tatoeba EN-KR
- KoNLPy (Mecab) morphological analysis → extract sentence-final EF morpheme
- Label by suffix pattern:
  - `-습니다/-ㅂ니다/-습니까` → `<formal>`
  - `-아요/-어요/-여요` → `<polite>`
  - `-아/-어/-냐/-지/-구나` → `<casual>`
- LLM augmentation to balance class distribution
- Output: TSV with `(en, ko, formality)` columns

**mBART-50 Fine-tuning** (`backend/model/`)
- Expand tokenizer with `<formal>`, `<polite>`, `<casual>` tokens
- Resize embedding layers
- Train on formality-labeled parallel data
- Hyperparameter tuning

**Evaluation** (`backend/evaluation/`)
- BLEU and chrF via sacrebleu
- Formality Accuracy: morphological analysis of predicted endings vs. requested token

**Working Prototype** (`backend-agent/`)
- FastAPI backend using Claude API for translation (not fine-tuned mBART)
- Demonstrates the 2-step UX: set context → translate
- Useful for frontend development and end-to-end testing before model is ready

**Frontend** (`frontend/`)
- React + Vite
- Social context form (relationship, setting, age differential)
- Translation display with formality badge

---

## Korean Speech Levels (Cultural Context)

Modern Korean primarily uses three levels:

| Token | Name | Endings | Context |
|---|---|---|---|
| `<formal>` | 하십시오체 | -습니다, -ㅂ니다 | Boss, customers, formal/public settings |
| `<polite>` | 해요체 | -아요, -어요 | Colleagues, everyday |
| `<casual>` | 해체 | -아, -어 | Close friends, family, same-age peers |

Korean is **agglutinative** — speech level is encoded in sentence-final verb endings (EF morphemes), not separate words. This is why morphological analysis is essential for both labeling and evaluation.

---

## Project Structure

```
noonchi-translator/
├── backend/
│   ├── nlp/                    # NLP labs (tokenization, morphology exploration)
│   ├── check_installation.py
│   └── requirements.txt
├── backend-agent/              # Working Claude API prototype
│   ├── app.py
│   ├── agents/claude_agent.py
│   ├── session_manager.py
│   └── models/schemas.py
├── data/
│   └── relationship_formality_map.json
├── docs/
│   ├── ARCHITECTURE_DECISION.md    # Traditional NLP engine class outlines
│   └── IMPLEMENTATION_PLAN.md      # Full mBART pipeline plan
├── frontend/
│   └── src/
├── tests/
└── noonchi-venv/
```

---

## Development Notes

- The `backend-agent/` prototype uses Claude API and is fully functional — good for frontend work
- The `backend/nlp/` labs are exploration/reference code, not production modules
- The mBART pipeline (`data_pipeline/`, `model/`, `evaluation/`) needs to be built
- GPU access is required for mBART fine-tuning (Colab, university cluster, or cloud)
- `docs/IMPLEMENTATION_PLAN.md` has the detailed phase-by-phase build plan
- `data/relationship_formality_map.json` has all 7 Korean speech levels + relationship mappings

---

## Key References

- KoNLPy docs: https://konlpy.org/en/latest/
- mBART-50: `facebook/mbart-large-50-many-to-many-mmt` on HuggingFace
- sacrebleu: https://github.com/mjpost/sacrebleu
- FastAPI: https://fastapi.tiangolo.com/
