# Noonchi Translator

A context-aware English-to-Korean translation system that infers the appropriate Korean speech level from social context and produces formality-conditioned translations via a fine-tuned mBART-50 model.

**"눈치" (noonchi)** — social awareness in Korean; the ability to read a situation and respond accordingly.

---

## How It Works

```
Social Context Input
(relationship, age, setting)
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

1. **User provides** English text + social context (speaker-addressee relationship, ages, setting formality)
2. **FormalityResolver** applies sociolinguistic rules to select one of three speech levels
3. **Conditioning token** is prepended to the input before passing to the translation model
4. **mBART-50** (fine-tuned on formality-labeled parallel data) generates Korean at the target speech level
5. **Evaluation** reports BLEU, chrF, and a custom formality accuracy metric

---

## Korean Speech Levels

| Token | Level | Example Ending | Usage |
|---|---|---|---|
| `<formal>` | 하십시오체 | -습니다 / -ㅂ니다 | Boss, customers, formal settings |
| `<polite>` | 해요체 | -아요 / -어요 | Colleagues, everyday polite |
| `<casual>` | 해체 | -아 / -어 | Close friends, family |

---

## Project Structure

```
noonchi-translator/
├── backend/
│   ├── nlp/                        # NLP exploration and labs
│   │   ├── day1_tokenization_lab.py
│   │   ├── day4_lab.py
│   │   ├── day5_en-kr.py
│   │   └── day5_lab.py
│   ├── check_installation.py
│   └── requirements.txt
├── backend-agent/                  # Working prototype (Claude API)
│   ├── app.py                      # FastAPI server
│   ├── agents/claude_agent.py      # Translation agent
│   ├── session_manager.py
│   ├── models/schemas.py
│   └── requirements.txt
├── data/
│   └── relationship_formality_map.json
├── docs/
│   ├── ARCHITECTURE_DECISION.md    # Traditional NLP engine design
│   └── IMPLEMENTATION_PLAN.md      # mBART pipeline plan
├── frontend/                       # React UI
│   └── src/
│       ├── App.jsx
│       ├── components/
│       └── services/
└── tests/
```

---

## Components

### FormalityResolver
Rule-based pragmatic inference mapping social context to a formality token.

```python
context = SocialContext(
    relationship="boss",
    age_differential=15,
    setting="workplace"
)
resolver.resolve(context)  # → "<formal>"
```

### Data Pipeline
Uses KoNLPy (Mecab) morphological analysis to automatically label formality from sentence-final verb endings in Korean OpenSubtitles and Tatoeba corpora. LLM prompting augments the dataset to balance class distribution across formal/polite/casual.

Suffix rules:
- `-습니다 / -ㅂ니다 / -습니까` → `<formal>`
- `-아요 / -어요 / -여요` → `<polite>`
- `-아 / -어 / -냐 / -지 / -구나` → `<casual>`

### mBART-50 Fine-tuning
Tokenizer expanded with `<formal>`, `<polite>`, `<casual>` conditioning tokens. Embedding layers resized and trained on formality-labeled parallel data with hyperparameter tuning.

### Evaluation
- **BLEU** and **chrF** on held-out test set
- **Formality Accuracy**: morphological analysis of predicted endings vs. requested speech level

---

## Tech Stack

- **Python 3.9+**, FastAPI
- **KoNLPy / Mecab** — Korean morphological analysis
- **spaCy** — English linguistic analysis
- **Hugging Face Transformers** — mBART-50 fine-tuning
- **React** — frontend UI

---

## Status

| Component | Status |
|---|---|
| NLP labs (tokenization, morphology) | Done |
| Working prototype (Claude API backend) | Done |
| React frontend | Done |
| FormalityResolver | Done |
| Data pipeline (corpus labeling) | In progress |
| LLM augmentation for class balance | Planned |
| mBART-50 fine-tuning | Planned |
| Evaluation (BLEU, chrF, formality accuracy) | Planned |

---

## Setup

### Backend prototype
```bash
cd backend-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env  # add ANTHROPIC_API_KEY
python app.py
```

### Frontend
```bash
cd frontend
npm install && npm run dev
```
