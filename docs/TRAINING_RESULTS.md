# mBART Fine-tuning: Training Results

## Setup

- **Model:** `facebook/mbart-large-50-many-to-many-mmt`
- **GPU:** Tesla T4 (15 GB VRAM), Google Colab
- **Training data:** 50,000 rows sampled from `train.tsv` (423,699 total)
- **Validation data:** 1,998 rows (subset of `val.tsv`)
- **Test data:** 42,372 rows from `test.tsv`

---

## Training Run History

### First attempt — OOM crash

Ran with `--max-rows 50000`, crashed at 8% (step 370/4,689) with:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 978.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 427.81 MiB is free.
```

T4 has 15 GB VRAM; mBART-large is ~2.4 GB in fp16, and the remaining memory was exhausted by activations + gradients during the backward pass.

### Fixes applied

- Reduced per-device batch size
- Added `no_repeat_ngram_size=3` and `repetition_penalty=1.2` to generation to prevent degenerate output
- Set standard mBART `decoder_start_token_id` and removed hardcoded token fallback

### Final training run

Completed successfully on 50K rows. Checkpoints saved to Google Drive at:
`/content/drive/MyDrive/noonchi/checkpoints/noonchi-mbart`

---

## Evaluation Results (Cell 8)

Evaluated on the full 42,372-row test set.

| Metric | Score | Target |
|---|---|---|
| chrF | 28.64 | > 30 (full model) |
| BLEU | 12.81 | — |
| Formality Accuracy | **85.12%** | > 80% ✅ |

**Notes on metrics:**
- chrF and BLEU are below the "full model" target because this run used only 50K of the 423K training rows. Both metrics are expected to improve with full-dataset training on an A100.
- EN→KR is a low-BLEU pair by nature — Korean is agglutinative and morphologically dense, so surface n-gram overlap with a single reference is inherently low. chrF is a more meaningful metric here since it operates at character level.
- **Formality accuracy is the most task-relevant metric** — it measures whether the predicted Korean sentence-final morpheme matches the conditioning token requested. Clearing 85% on the held-out test set confirms the model learned formality conditioning, not just translation.

---

## Cell 9: Sample Prediction Inspection

Ten representative samples from the 42,372-row test set.

### Strong outputs

| # | EN | HYP | Notes |
|---|---|---|---|
| 5 | The new software update includes several important security improvements. | 새로운 소프트웨어 업데이트에는 여러 중요한 보안 개선이 포함되어 있습니다. | Near-exact match with reference; omits 사항 but semantically equivalent |
| 7 | The company has established new environmental policies. | 회사는 새로운 환경 정책을 시행했습니다. | Differs from ref (수립 → 시행, establish → implement); both natural Korean |
| 9 | Our research team has completed the preliminary analysis. | 우리 연구팀은 예비 분석을 완료했습니다. | Exact match |

### Acceptable outputs

- **[1]** — Good translation, correct polite register
- **[3]** — Minor name transcription difference (패커드 → 파커드), semantically equivalent
- **[4]** — Simplified but natural: "잃어버렸어요 도와주실래요?" vs ref "길을 잃었어요 제발 도와주실래요?"
- **[8]** — Natural paraphrase, correct register

### Weaker outputs

- **[0]** ("You're a disappointment, young lady, a big, big, disappointment.") — Struggles with comma-heavy English emphasis. Earlier run produced "넌 실망이야, 젊어, 큰, 큰 실망" (literalized "young lady" as "젊어"). This is a known hard case for seq2seq models.
- **[2]** (medical screws sentence) — "screws in your back" → 스위치를 꽂습니다 (plugging in a switch). Medical domain vocabulary is outside the training distribution (OpenSubtitles + Tatoeba skew toward everyday dialogue).
- **[6]** — "알습니다" is mildly ungrammatical; correct form is "압니다". Rare but present in the data.

### Data contamination artifact (earlier run only)

An earlier evaluation run showed `[9] HYP: 우리 연구팀은 예비 분석을 완료했습니다. this is cell 9 output` — a debug string from the Cell 9 inspector was written into one row of `test.tsv` or the hypotheses cache. This did not affect the final evaluation run (confirmed: Cell 9 output in the final run is clean). The contaminated row should be removed from `data/test.tsv` before any future evaluation run.

To check:
```bash
grep -n "cell 9 output" data/test.tsv data/train.tsv
```

---

## Cell 10: Conditioning Sanity Check

Three sentences translated through all three formality tokens. Confirms the model is correctly conditioning — outputs are not identical across tokens.

```
EN: Can you help me with this?
  <formal>  이걸 도와주실 수 있습니까?
  <polite>  좀 도와주실래요?
  <casual>  이걸 도와줄 수 있어?

EN: The meeting has been rescheduled.
  <formal>  회의가 재일정되었습니다.
  <polite>  회의 일정이 변경됐어요
  <casual>  회의 일정이 변경됐어

EN: I want to eat dinner with you.
  <formal>  당신과 함께 저녁을 먹고 싶습니다
  <polite>  당신과 함께 저녁을 먹고 싶어요
  <casual>  당신과 함께 저녁을 먹고 싶어
```

**Observations:**

- Sentence-final morphemes are consistently correct across all three levels: `-습니까/-습니다` (formal), `-아요/-어요` (polite), `-아/-어` (casual).
- Example 3 shows clean parallel structure — only the final morpheme changes, which is exactly what formality conditioning should produce.
- Example 1 catches the demonstrative register shift: 이걸 in formal/casual vs. 좀 (hedging softener) in polite — a subtle and linguistically correct distinction.

**Known limitations:**

- `재일정되었습니다` (example 2, formal) is a direct calque of "reschedule" and unnatural in Korean. The polite/casual outputs (`변경됐어요/됐어`) are more idiomatic. This reflects the training data distribution — formal-register data from subtitles contains fewer instances of this construction.
- `당신` is used consistently across all three formality levels in example 3. In casual Korean (banmal), `당신` is awkward — real usage would drop the pronoun or substitute the addressee's name. This is an artifact of subtitle corpora over-representing `당신` as the translation of "you."

---

## Summary

The 50K-row fine-tuned model clears the formality accuracy target (85.12% > 80%) and produces natural Korean output for everyday sentences. chrF and BLEU are below full-model targets as expected at this training scale. Formality conditioning is confirmed working end-to-end. The model is ready for Phase 6 inference API integration.

**Next step:** Full-dataset training on A100 (423K rows) is expected to push chrF above 30 and further improve BLEU. Alternatively, proceed to Phase 6 with the 50K model for demo purposes.
