# Noonchi Translator — Development Plan

## Context

This is a long-term computational linguistics portfolio project targeting grad school applications. The strategy is to build in demonstrable layers: first get a working, polished demo using the Claude API, then build the research layer (labeled dataset + mBART fine-tuning) on top of something already presentable. By the time mBART training starts, you have a live demo, a complete dataset, and training is the final research layer — not a prerequisite for anything.

**Ordering principle:** Each phase produces something independently valuable. You never need the next phase to have a working project.

---

## Current State

| Component | File | Status |
|---|---|---|
| FastAPI server | `backend/api/app.py` | Code complete — not yet confirmed running |
| Claude translation agent | `backend/api/agents/claude_agent.py` | Complete |
| FormalityResolver | `backend/formality/resolver.py` | Complete |
| Pydantic schemas | `backend/api/models/schemas.py` | Complete |
| Session management | `backend/api/session_manager.py` | Complete |
| Config | `backend/api/config.py` | Complete |
| React frontend | `frontend/src/` | Code complete — not yet confirmed running |
| Morphological labeling | `backend/data_pipeline/label.py` | Complete |
| Confidence filtering | `backend/data_pipeline/filter.py` | Complete |
| Suffix augmentation | `backend/data_pipeline/augment.py` | Partial — `augment_by_llm` is stub |
| Pipeline orchestration | `backend/data_pipeline/pipeline.py` | Complete (blocked by extract.py stub) |
| Evaluation metrics | `backend/evaluation/metrics.py` | Complete |
| mBART tokenizer setup | `backend/model/train.py` | Partial — `load_model_and_tokenizer()` works, `train()` is stub |
| Corpus extraction | `backend/data_pipeline/extract.py` | Stub only |
| Dataset class | `backend/model/dataset.py` | Does not exist |
| Model inference | `backend/model/inference.py` | Does not exist |
| mBART API agent | `backend/api/agents/mbart_agent.py` | Does not exist |
| Traditional NLP engine | `backend/nlp/traditional/` | Does not exist |

---

## Phase 1: Get the Claude Prototype Running End-to-End

**Goal:** A fully working demo — frontend talking to backend, context set, translations returned, formality badges showing. This is your portfolio demo before any ML work.  
**No GPU. Estimated time: half a day.**

### Step 1a: Fix the Frontend ↔ Backend Data Contract

There are two mismatches between the frontend and backend that will cause validation errors:

**Problem 1 — Missing required fields.** The backend's `ContextRequest` schema (which extends `SocialContext`) requires three fields: `relationship`, `age_differential` (int), and `setting` (enum). The frontend currently only sends `relationship` and `situation` (a free-text optional field). The backend will return a 422 validation error.

**Fix:** Update the frontend form in `frontend/src/App.jsx` to collect `age_differential` and `setting` as structured fields. Add:
- An `age_differential` number input (range -50 to 50, label: "Age difference — negative means you are younger")
- A `setting` dropdown with options: `workplace`, `academic`, `social`, `public`, `intimate`

Update the API call in `frontend/src/services/api.js`:
```js
// Replace: { relationship, situation }
// With:
{ relationship, age_differential: parseInt(ageDifferential), setting }
```

**Problem 2 — Relationship enum mismatch.** The frontend's relationship options include values the backend doesn't recognize (`customer`, `teacher`, `younger_friend`, `younger_sibling`, `child`), and the backend has values the frontend doesn't have (`professor`, `peer`, `subordinate`, `stranger`). The backend will reject unrecognized enum values.

**Fix — Option A (preferred):** Align both sides to the same set. Update the frontend dropdown to exactly match the backend's `RelationshipType` enum:
`boss, elder, professor, colleague, peer, subordinate, friend, acquaintance, stranger`

**Fix — Option B:** Extend the backend enum to include all frontend values and add rules for the new ones in `FormalityResolver`. More work but richer social modeling.

Recommendation: Start with Option A (simpler, gets you running), revisit Option B later.

### Step 1b: Environment Setup

```bash
# 1. Copy .env.example and add your key
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY=sk-ant-...

# 2. Set up backend venv and install deps
cd backend/api
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Install frontend deps
cd ../../frontend
npm install
```

### Step 1c: Run Both Servers

```bash
# Terminal 1 — backend (from backend/api/)
source venv/bin/activate
python app.py
# Should print: Uvicorn running on http://0.0.0.0:8000

# Terminal 2 — frontend (from frontend/)
npm run dev
# Should print: Local: http://localhost:5173
```

### Step 1d: Verify the Full Flow

Walk through this manually in the browser:

1. Open `http://localhost:5173`
2. Select relationship: `boss`, age differential: `-10`, setting: `workplace`
3. Click "Set Context" — should show a blue `<formal>` badge
4. Type: "Do you want to eat?"
5. Click "Translate" — should return Korean output ending in `습니다` or `ㅂ니다`
6. Reset context. Select `friend`, `0`, `intimate`
7. Translate same phrase — should return casual endings (`아` / `어`)
8. Test `formality_override`: set boss/workplace context but manually override to casual — output should still be casual

**Also test the health endpoint:**
```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","message":"API is operational"}
```

**Check for session expiry handling:** Wait for session to expire (or send a bad session_id manually) — frontend should show a friendly error suggesting you re-run set-context.

### Step 1e: Deliverable

At the end of Phase 1 you have a working, demo-able application. This is your baseline portfolio artifact regardless of how much ML work follows.

---

## Phase 2: Implement Corpus Extraction

**Goal:** Implement `backend/data_pipeline/extract.py` so the data pipeline can load real EN–KR parallel data.  
**No GPU. Estimated time: 1 day.**

### Step 2a: Download the Raw Corpora

Create `data/raw/` directory (already in `.gitignore` — do not commit raw data).

**Tatoeba:**
- URL: https://tatoeba.org/en/downloads
- Download: `sentences.csv` and `links.csv`
- `sentences.csv` format: `id \t lang \t text` (tab-separated, no header)
- `links.csv` format: `sentence_id \t translation_id` (tab-separated, no header)
- Save to: `data/raw/tatoeba/`

**OPUS OpenSubtitles:**
- URL: https://opus.nlpl.eu/OpenSubtitles/corpus/version/OpenSubtitles
- Download: `OpenSubtitles.en-ko.tmx.gz` (EN–KO TMX file)
- TMX is XML — each `<tu>` element is one translation unit with `<tuv xml:lang="en">` and `<tuv xml:lang="ko">` children containing a `<seg>` text node
- Save to: `data/raw/opus/`
- Warning: this file is large (~500MB compressed). Consider downloading only a sample if bandwidth is a concern — the pipeline's filtering step will reduce it dramatically anyway.

### Step 2b: Implement `load_tatoeba`

```python
def load_tatoeba(data_dir: str) -> list[tuple[str, str]]:
    sentences_path = Path(data_dir) / "raw" / "tatoeba" / "sentences.csv"
    links_path = Path(data_dir) / "raw" / "tatoeba" / "links.csv"

    # Step 1: Load all sentences into a dict: id → (lang, text)
    sentences = {}
    with open(sentences_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                sid, lang, text = parts
                sentences[sid] = (lang, text.strip())

    # Step 2: Build sets of English and Korean sentence IDs
    eng_ids = {sid for sid, (lang, _) in sentences.items() if lang == "eng"}
    kor_ids = {sid for sid, (lang, _) in sentences.items() if lang == "kor"}

    # Step 3: Walk links, yield aligned pairs
    pairs = []
    with open(links_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                a, b = parts
                if a in eng_ids and b in kor_ids:
                    pairs.append((sentences[a][1], sentences[b][1]))
                elif a in kor_ids and b in eng_ids:
                    pairs.append((sentences[b][1], sentences[a][1]))

    logger.info(f"Tatoeba: loaded {len(pairs):,} EN–KR pairs")
    return pairs
```

### Step 2c: Implement `load_opus_opensubtitles`

```python
def load_opus_opensubtitles(data_dir: str) -> list[tuple[str, str]]:
    import gzip
    tmx_path = Path(data_dir) / "raw" / "opus" / "OpenSubtitles.en-ko.tmx.gz"

    pairs = []
    open_fn = gzip.open if str(tmx_path).endswith(".gz") else open

    with open_fn(tmx_path, "rt", encoding="utf-8") as f:
        # Use iterparse to avoid loading the whole XML into memory
        for event, elem in ET.iterparse(f, events=("end",)):
            if elem.tag == "tu":
                en_text = ko_text = None
                for tuv in elem:
                    lang = tuv.get("{http://www.w3.org/XML/1998/namespace}lang", "")
                    seg = tuv.find("seg")
                    if seg is not None and seg.text:
                        if lang.startswith("en"):
                            en_text = seg.text.strip()
                        elif lang.startswith("ko"):
                            ko_text = seg.text.strip()
                if en_text and ko_text:
                    pairs.append((en_text, ko_text))
                elem.clear()  # free memory

    logger.info(f"OPUS OpenSubtitles: loaded {len(pairs):,} EN–KR pairs")
    return pairs
```

Note: `elem.clear()` is important — without it, `iterparse` accumulates all elements in memory and will OOM on large files.

### Step 2d: Update `load_corpus` and add `.gitignore` entry

Make sure `data/raw/` is in `.gitignore` (it already is from the restructuring). Also add `data/train.tsv` and `data/test.tsv` to `.gitignore` since these are derived artifacts, not source files.

---

## Phase 3: Implement LLM Augmentation

**Goal:** Implement `augment_by_llm()` to generate synthetic EN–KR pairs in underrepresented registers using the Claude API you already have set up.  
**No GPU. Estimated time: half a day.**

### Why This Is Needed

Natural subtitle data skews heavily toward polite register (해요체 dominates casual dialogue). After corpus extraction and suffix substitution, `<formal>` examples will be significantly underrepresented. `augment_by_llm` fills the gap with synthetically generated pairs that are morphologically verified before inclusion.

### Step 3a: Implement `augment_by_llm` in `backend/data_pipeline/augment.py`

```python
def augment_by_llm(target_label: str, target_count: int) -> list[tuple[str, str, str]]:
    from anthropic import Anthropic
    from backend.api.config import get_settings
    from backend.data_pipeline.label import label_sentence

    settings = get_settings()
    client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    LEVEL_GUIDE = {
        "formal": {
            "name": "하십시오체 (formal polite)",
            "endings": "-습니다, -ㅂ니다, -습니까",
            "example": "도와주시겠습니까?"
        },
        "polite": {
            "name": "해요체 (informal polite)",
            "endings": "-아요, -어요, -여요",
            "example": "도와주세요."
        },
        "casual": {
            "name": "해체 / banmal (casual)",
            "endings": "-아, -어, -냐, -지",
            "example": "도와줘."
        },
    }

    guide = LEVEL_GUIDE[target_label]
    verified = []

    while len(verified) < target_count:
        needed = target_count - len(verified)
        batch_size = min(20, needed * 3)  # request extra to account for verification failures

        prompt = f"""Generate {batch_size} natural English–Korean sentence pairs.

The Korean MUST be in {guide['name']} register.
Required sentence-final endings: {guide['endings']}
Example of correct output: "{guide['example']}"

Use a variety of everyday situations: workplace, social, academic, everyday requests.
Vary sentence length and complexity.

Return ONLY a JSON array, no other text:
[{{"en": "English sentence here", "ko": "Korean sentence here"}}, ...]"""

        response = client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            raw = response.content[0].text.strip()
            # Handle markdown-wrapped JSON
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            pairs = json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse LLM response, retrying...")
            continue

        for item in pairs:
            en = item.get("en", "").strip()
            ko = item.get("ko", "").strip()
            if not en or not ko:
                continue
            # Morphological verification — only keep pairs where the Korean
            # ending actually matches the requested label
            detected = label_sentence(ko)
            if detected == target_label:
                verified.append((en, ko, target_label))
                if len(verified) >= target_count:
                    break

        logger.info(f"augment_by_llm({target_label}): {len(verified)}/{target_count} verified")

    return verified[:target_count]
```

### Step 3b: Update `pipeline.py` to Auto-Balance Classes

After the filtering stage, check class distribution and call `augment_by_llm` for any class below the target minimum. Add a `--augment-to` CLI flag (default: 3000 pairs per class):

```python
# After filter stage in pipeline.py run():
from collections import Counter
counts = Counter(label for _, _, label in filtered)
for label in ["formal", "polite", "casual"]:
    deficit = args.augment_to - counts.get(label, 0)
    if deficit > 0:
        logger.info(f"Augmenting {label}: need {deficit} more pairs")
        synthetic = augment_by_llm(label, deficit)
        all_pairs.extend(synthetic)
```

---

## Phase 4: Run the Full Data Pipeline → `data/train.tsv`

**Goal:** Execute the complete pipeline end-to-end and produce the labeled training dataset.  
**No GPU. Estimated time: a few hours (mostly I/O and API calls).**

### Step 4a: Run the Pipeline

```bash
cd /path/to/noonchi-translator
source backend/api/venv/bin/activate
pip install -r backend/requirements.txt  # adds konlpy, pandas, etc.

python -m backend.data_pipeline.pipeline \
    --data-dir data/ \
    --output data/train.tsv \
    --augment-to 3000
```

Expected console output:
```
Stage 1: Extracting corpora...
  Tatoeba: loaded 8,432 EN–KR pairs
  OPUS OpenSubtitles: loaded 1,247,891 EN–KR pairs
  Total: 1,256,323 raw pairs loaded
Stage 2–3: Morphological analysis and formality labeling...
Stage 4: Confidence filtering...
  489,201 pairs retained (767,122 removed)
Stage 5: Augmenting underrepresented registers...
  Augmenting formal: need 1,847 more pairs
  augment_by_llm(formal): 1847/1847 verified
  2,311 synthetic pairs added → 491,512 total
Stage 6: Writing dataset...
  Dataset written to data/train.tsv
  Class distribution: {'polite': 301,244, 'casual': 187,421, 'formal': 3,000}
```

Note: Numbers above are illustrative — actual distribution will vary.

### Step 4b: Validate the Dataset

```bash
# Inspect first 50 rows
head -50 data/train.tsv | column -t -s $'\t'

# Check class distribution
cut -f3 data/train.tsv | sort | uniq -c

# Spot-check formal examples (should end in 습니다/ㅂ니다)
awk -F'\t' '$3=="formal"' data/train.tsv | head -20

# Spot-check casual examples (should end in 아/어/냐)
awk -F'\t' '$3=="casual"' data/train.tsv | head -20
```

**Quality bar:** Formal rows end in `습니다`/`ㅂ니다`/`습니까`. Polite rows end in `아요`/`어요`. Casual rows end in `아`/`어`/`냐`/`지`. If a spot-check reveals systematic mislabeling, debug `label.py`'s suffix rules.

### Step 4c: Split into Train / Val / Test

Create a script `backend/model/prepare_data.py` (or add to pipeline.py) to split the full TSV:
- 90% train → `data/train_split.tsv`
- 5% validation → `data/val.tsv`
- 5% test → `data/test.tsv`

Use a fixed seed (42) so the split is reproducible. Stratify by formality label so all three classes appear in all splits. Commit the split indices (or the split files if small enough) so training is reproducible.

### Deliverable

At the end of Phase 4 you have:
- A working live demo (Claude API backend + frontend)
- A fully labeled, filtered, class-balanced training dataset
- This is a complete portfolio artifact on its own — the data pipeline is a research contribution

---

## Phase 5: mBART-50 Fine-tuning

**Goal:** Train a formality-conditioned EN→KR translation model on the labeled data from Phase 4.  
**Requires GPU. Use Google Colab (T4 free tier is sufficient for ~10K rows; rent an A100 instance for larger datasets). Estimated time: 1 day setup + 2–4 hours training.**

### Step 5a: Create `backend/model/dataset.py`

HuggingFace-compatible PyTorch Dataset class.

**`NoonchiDataset(Dataset)`:**

```python
class NoonchiDataset(Dataset):
    def __init__(self, rows: list[tuple[str, str, str]], tokenizer, max_length: int = 128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        en, ko, formality = self.rows[idx]
        src = f"<{formality}> {en}"  # e.g. "<formal> Can you help me with this?"

        self.tokenizer.src_lang = "en_XX"
        model_inputs = self.tokenizer(
            src, max_length=self.max_length, truncation=True, padding="max_length",
            return_tensors="pt"
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                ko, max_length=self.max_length, truncation=True, padding="max_length",
                return_tensors="pt"
            )

        label_ids = labels["input_ids"].squeeze().tolist()
        # Replace padding token id with -100 so loss ignores padding positions
        label_ids = [-100 if t == self.tokenizer.pad_token_id else t for t in label_ids]
        model_inputs["labels"] = label_ids
        return {k: v.squeeze() if hasattr(v, "squeeze") else v
                for k, v in model_inputs.items()}
```

**`split_dataset(tsv_path, tokenizer, train=0.90, val=0.05, test=0.05, seed=42)`:**
- Read TSV
- Shuffle with fixed seed
- Split proportionally by formality class (stratified)
- Return three `NoonchiDataset` instances: `(train_ds, val_ds, test_ds)`

### Step 5b: Implement `train()` in `backend/model/train.py`

```python
def train(data_path: str, output_dir: str) -> None:
    model, tokenizer = load_model_and_tokenizer()
    train_ds, val_ds, _ = split_dataset(data_path, tokenizer)

    def compute_metrics(eval_preds):
        preds, label_ids = eval_preds
        # Replace -100 with pad_token_id before decoding
        label_ids = [
            [t if t != -100 else tokenizer.pad_token_id for t in row]
            for row in label_ids
        ]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"chrf": compute_chrf(decoded_preds, decoded_labels)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        predict_with_generate=True,
        generation_max_length=128,
        fp16=True,                    # T4 supports fp16
        **TRAINING_ARGS               # lr, warmup, batch_size, epochs, early stopping
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
```

**Also update `load_model_and_tokenizer(model_dir=None)`:** When `model_dir` is provided, load from that path instead of from HuggingFace hub. This is used by inference and evaluate.

### Step 5c: Google Colab Setup

Create a Colab notebook at `notebooks/train_colab.ipynb`:

```python
# Cell 1 — install deps
!pip install transformers datasets sentencepiece sacrebleu konlpy torch

# Cell 2 — clone repo and upload data
!git clone https://github.com/cyatco01/noonchi-translator.git
%cd noonchi-translator
# Upload data/train.tsv via Colab sidebar or mount Drive:
# from google.colab import drive; drive.mount('/content/drive')
# !cp /content/drive/MyDrive/noonchi/train.tsv data/train.tsv

# Cell 3 — train
!python -m backend.model.train --data data/train.tsv --output models/noonchi-mbart

# Cell 4 — save model to Drive
!cp -r models/noonchi-mbart /content/drive/MyDrive/noonchi/
```

Expected training time on T4: ~2–4 hours for 10K examples, 5 epochs.

### Step 5d: Create `backend/model/evaluate.py`

After training, run evaluation on the held-out test split.

```python
def evaluate_model(model_dir: str, test_tsv: str) -> dict[str, float]:
    model, tokenizer = load_model_and_tokenizer(model_dir)
    model.eval()

    _, _, test_ds = split_dataset(test_tsv, tokenizer)
    
    hypotheses, references, labels = [], [], []
    for row in test_ds.rows:
        en, ko_ref, formality = row
        src = f"<{formality}> {en}"
        tokenizer.src_lang = "en_XX"
        inputs = tokenizer(src, return_tensors="pt", max_length=128, truncation=True)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"],
                num_beams=4,
                max_length=128,
            )
        hypotheses.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
        references.append(ko_ref)
        labels.append(formality)

    results = evaluate(hypotheses, references, labels)  # backend/evaluation/metrics.py
    print(results)
    return results
```

**Target metrics:** Formality Accuracy > 80%, chrF competitive with unconditioned mBART baseline.

---

## Phase 6: mBART Inference API + Frontend Integration

**Goal:** Connect the trained model to the app so users can switch between Claude and mBART.  
**No GPU required for inference at demo scale (CPU is slow but works for single requests).  
Estimated time: half a day.**

### Step 6a: Create `backend/model/inference.py`

```python
class MBartInference:
    def __init__(self, model_dir: str):
        self.model, self.tokenizer = load_model_and_tokenizer(model_dir)
        self.model.eval()

    def translate(self, text: str, formality_token: str) -> str:
        """formality_token: 'formal' | 'polite' | 'casual'"""
        src = f"<{formality_token}> {text}"
        self.tokenizer.src_lang = "en_XX"
        inputs = self.tokenizer(src, return_tensors="pt", max_length=128, truncation=True)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["ko_KR"],
                num_beams=4,
                max_length=128,
                early_stopping=True,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### Step 6b: Create `backend/api/agents/mbart_agent.py`

Mirrors `ClaudeTranslationAgent` exactly so `app.py` can swap agents without changing endpoint logic.

```python
from backend.formality.resolver import FormalityResolver   # reuse — do not duplicate
from backend.model.inference import MBartInference
from backend.api.models.schemas import TranslationResponse, SocialContext, FormalityToken

class MBartTranslationAgent:
    def __init__(self, model_dir: str):
        self.inference = MBartInference(model_dir)
        self.resolver = FormalityResolver()

    def resolve_formality(self, context: SocialContext) -> FormalityToken:
        return self.resolver.resolve(context)

    async def translate(self, context: SocialContext, text: str) -> TranslationResponse:
        formality_token = self.resolver.resolve(context)
        conditioned_input = f"{formality_token.as_token()} {text}"
        korean = self.inference.translate(text, formality_token.value)

        return TranslationResponse(
            original_text=text,
            conditioned_input=conditioned_input,
            translated_text=korean,
            formality_token=formality_token,
            relationship=context.relationship,
            explanation=None,
            romanization=None,
        )
```

### Step 6c: Add Backend Routing to `app.py` + `config.py`

In `backend/api/config.py`, add:
```python
BACKEND: str = os.getenv("BACKEND", "claude")
MBART_MODEL_DIR: str = os.getenv("MBART_MODEL_DIR", "models/noonchi-mbart")
```

In `backend/api/app.py` lifespan handler:
```python
settings = get_settings()
if settings.BACKEND == "mbart":
    translation_agent = MBartTranslationAgent(settings.MBART_MODEL_DIR)
else:
    translation_agent = ClaudeTranslationAgent()
```

Run the mBART server on port 8002:
```bash
BACKEND=mbart API_PORT=8002 python backend/api/app.py
```

### Step 6d: Enable in Frontend

In `frontend/src/config/backends.js`:
```js
ml_model: {
    available: true,          // was: false
    baseURL: 'http://localhost:8002',
    name: 'Fine-tuned mBART',
    ...
}
```

No other frontend changes needed — the method selector and API layer already handle this backend.

### Step 6e: End-to-End Comparison Test

```bash
# Terminal 1 — Claude backend
BACKEND=claude API_PORT=8000 python backend/api/app.py

# Terminal 2 — mBART backend
BACKEND=mbart API_PORT=8002 python backend/api/app.py

# Terminal 3 — frontend
cd frontend && npm run dev
```

In the browser: translate the same phrase ("Do you want to eat?") with the same context (boss / workplace / -10) using both backends. Both should produce output ending in `습니다`/`ㅂ니다`. Switch to casual context (friend / intimate / 0) — both should produce `아`/`어` endings. The difference you'll see is that Claude also produces romanization and explanation; mBART produces only the Korean text.

---

## Phase 7: Traditional NLP Engine (Optional)

**Goal:** Build a rule-based MT engine from scratch to demonstrate understanding of pre-neural translation fundamentals alongside the neural approach.  
**No GPU. Estimated time: 2–3 days.**  
**Portfolio pitch:** Shows you understand *why* neural MT is an advance, not just how to use it. Lets you narrate the whole arc of MT history in an interview.

### Architecture

```
English Input
     ↓
english_analyzer.py  —  spaCy POS + dependency parse → structured ParsedSentence
     ↓
reorderer.py         —  SVO → SOV word order + Korean case markers
     ↓
dictionary.py        —  EN→KR lexical lookup (curated ~500 word dictionary)
     ↓
korean_generator.py  —  formality-conditioned verb morpheme attachment
     ↓
Korean Output
```

### Step 7a: `backend/nlp/traditional/english_analyzer.py`

Uses spaCy (`en_core_web_sm`). Load model once at module level.

```python
@dataclass
class ParsedSentence:
    subject: str | None       # "I", "she", "the manager"
    verb: str | None          # lemma form: "eat", "want"
    object_: str | None       # "food", "the report"
    auxiliaries: list[str]    # ["want", "to"] for "want to eat"
    tense: str                # "present" | "past" | "future"
    negated: bool
    raw_tokens: list[str]
```

Walk the dependency tree:
- ROOT → main verb
- `nsubj` arc → subject
- `dobj` / `obj` arc → direct object
- Tense from `token.morph.get("Tense")`
- Negation from any token with `dep_ == "neg"` under the main verb

### Step 7b: `backend/nlp/traditional/dictionary.py`

Curated bilingual dictionary. Seed it using the 500 most frequent content words from the Tatoeba corpus (mine it in Phase 2). Structure:

```python
NOUNS: dict[str, str] = {"food": "음식", "water": "물", "time": "시간", ...}
VERBS: dict[str, str] = {"eat": "먹", "go": "가", "want": "원하", "know": "알", ...}
PRONOUNS: dict[str, dict[str, str]] = {
    "i":   {"subject": "저", "object": "저를"},
    "you": {"subject": "당신", "object": "당신을"},
    "he":  {"subject": "그", "object": "그를"},
    ...
}
ADJECTIVES: dict[str, str] = {"good": "좋은", "big": "큰", ...}
```

### Step 7c: `backend/nlp/traditional/reorderer.py`

Transform English SVO to Korean SOV and attach case markers.

Korean case markers depend on whether the preceding syllable has a final consonant (받침):
- Topic marker: 은 (after consonant) / 는 (after vowel)
- Subject marker: 이 (after consonant) / 가 (after vowel)
- Object marker: 을 (after consonant) / 를 (after vowel)

```python
def has_final_consonant(syllable: str) -> bool:
    """Check if the last syllable of a Korean word ends in a consonant."""
    if not syllable:
        return False
    last_char = syllable[-1]
    code = ord(last_char)
    if 0xAC00 <= code <= 0xD7A3:   # Korean syllable block range
        return (code - 0xAC00) % 28 != 0
    return False

def attach_topic_marker(word: str) -> str:
    return word + ("은" if has_final_consonant(word) else "는")

def attach_object_marker(word: str) -> str:
    return word + ("을" if has_final_consonant(word) else "를")
```

### Step 7d: `backend/nlp/traditional/korean_generator.py`

Attach formality-conditioned verb endings. Handle vowel harmony (stems ending in ㅏ/ㅗ use 아-class; all others use 어-class).

```python
ENDINGS = {
    "formal": {"present": "습니다", "past": "었습니다", "question": "습니까?"},
    "polite": {"present": "어요",   "past": "었어요",   "question": "어요?"},
    "casual": {"present": "어",     "past": "었어",     "question": "어?"},
}

def get_last_vowel(stem: str) -> str | None:
    """Extract the vowel of the last syllable for vowel harmony."""
    for char in reversed(stem):
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            jungseong_idx = ((code - 0xAC00) // 28) % 21
            # ㅏ=0, ㅗ=8 → 아-class; everything else → 어-class
            return "아" if jungseong_idx in (0, 8) else "어"
    return None
```

### Step 7e: `backend/nlp/traditional/translator.py`

```python
def translate(text: str, formality: str) -> str:
    parsed = analyze(text)
    tokens = reorder(parsed)
    korean = generate(tokens, formality, parsed.tense, parsed.negated)
    return korean
```

### Step 7f: `backend/nlp/traditional/server.py`

Minimal FastAPI app on port 8001. Implement the same `/health`, `/api/set-context`, `/api/translate` endpoints. Reuse:
- `FormalityResolver` from `backend/formality/resolver.py`
- `SessionManager` from `backend/api/session_manager.py`
- All schemas from `backend/api/models/schemas.py`

In `frontend/src/config/backends.js`:
```js
papago_rules: {
    name: 'Traditional NLP Engine',
    available: true,             // was: false
    baseURL: 'http://localhost:8001',
    ...
}
```

---

## Key Files Reference

### Files to Create (in order)

| File | Phase | Purpose |
|---|---|---|
| `backend/model/dataset.py` | 5 | PyTorch Dataset + train/val/test split |
| `backend/model/evaluate.py` | 5 | Run chrF, BLEU, FA on test split |
| `backend/model/inference.py` | 6 | Load fine-tuned model, beam search |
| `backend/api/agents/mbart_agent.py` | 6 | mBART agent mirroring Claude agent interface |
| `notebooks/train_colab.ipynb` | 5 | Colab training notebook |
| `backend/nlp/traditional/english_analyzer.py` | 7 | spaCy dependency parse |
| `backend/nlp/traditional/dictionary.py` | 7 | EN→KR lexical mappings |
| `backend/nlp/traditional/reorderer.py` | 7 | SVO→SOV + case markers |
| `backend/nlp/traditional/korean_generator.py` | 7 | Morpheme generation |
| `backend/nlp/traditional/translator.py` | 7 | Orchestration |
| `backend/nlp/traditional/server.py` | 7 | FastAPI server port 8001 |

### Files to Modify (in order)

| File | Phase | Change |
|---|---|---|
| `frontend/src/App.jsx` | 1 | Add `age_differential` + `setting` fields |
| `frontend/src/services/api.js` | 1 | Update payload to include all three context fields |
| `backend/data_pipeline/extract.py` | 2 | Implement Tatoeba + OPUS loaders |
| `backend/data_pipeline/augment.py` | 3 | Implement `augment_by_llm()` |
| `backend/data_pipeline/pipeline.py` | 3 | Add `--augment-to` flag + auto class balancing |
| `backend/model/train.py` | 5 | Implement `train()` + update `load_model_and_tokenizer(model_dir=None)` |
| `backend/api/app.py` | 6 | Add BACKEND routing in lifespan |
| `backend/api/config.py` | 6 | Add `BACKEND` + `MBART_MODEL_DIR` settings |
| `frontend/src/config/backends.js` | 6, 7 | Enable ml_model + papago_rules |

### Existing Code to Reuse (Never Duplicate)

| File | Reused by |
|---|---|
| `backend/formality/resolver.py` | `mbart_agent.py`, `traditional/server.py` |
| `backend/data_pipeline/label.py` `label_sentence()` | `augment_by_llm()` verification, `evaluate.py` |
| `backend/evaluation/metrics.py` `evaluate()` | `model/evaluate.py` |
| `backend/api/models/schemas.py` | `mbart_agent.py`, `traditional/server.py` |
| `backend/api/session_manager.py` | `traditional/server.py` |

---

## Verification Checkpoints

### After Phase 1
```bash
curl http://localhost:8000/health
# {"status":"healthy","message":"API is operational"}

# In browser: boss/workplace/-10 context → translate "Do you want to eat?"
# Expected Korean output ends in: 습니다 or ㅂ니다
# friend/intimate/0 context → same phrase
# Expected Korean output ends in: 아 or 어
```

### After Phase 4
```bash
head -50 data/train.tsv | column -t -s $'\t'
cut -f3 data/train.tsv | sort | uniq -c
# Target: ≥3000 of each label, ≤2:1 ratio across classes
```

### After Phase 5
```bash
python -m backend.model.evaluate --model models/noonchi-mbart --data data/test.tsv
# Target: formality_accuracy > 0.80, chrF > baseline unconditioned mBART
```

### After Phase 6
```bash
# Both backends running (ports 8000 and 8002)
# In browser: switch method selector from "AI Agent (Claude)" → "Fine-tuned mBART"
# Same context and phrase → both should produce correct formality endings
# Claude output includes explanation + romanization; mBART output is Korean text only
```

### After Phase 7
```bash
python backend/nlp/traditional/server.py
# In browser: switch to "Traditional NLP Engine"
# Translate "I eat food" with boss/workplace/-10 context
# Expected: 저는 음식을 먹습니다  (simple, dictionary-bounded — this is the point)
# Compare to Claude output: richer, more natural
# This contrast IS the portfolio demonstration
```
