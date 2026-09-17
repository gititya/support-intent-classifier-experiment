# Experiments Log

## Setup

All new experiments run on branch `encoder-setfit-experiments`. Python venv: `/Users/aditya/venvs/pytorch_practice` (torch 2.12.1, MPS, sklearn 1.6.1, setfit 1.1.3, transformers 4.57.6, sentence-transformers 3.3.1).

Existing Qwen LoRA work is untouched — see `adapters/` and `adapters_v1_original/`.

---

## Experiment A — ModernBERT fine-tune

**Model:** `answerdotai/ModernBERT-base` (149M encoder)  
**Script:** `scripts/modernbert_train.py`  
**Config:** lr=2e-5, epochs=5, batch=16, early stopping patience=2, macro-F1 checkpoint metric  
**Saved to:** `models/modernbert/v1` and `models/modernbert/v2`

Two runs: v1 on `train_original.jsonl` (576 examples), v2 on `train.jsonl` (2,104 augmented examples).

## Experiment B — SetFit

**Backbone:** `sentence-transformers/all-MiniLM-L6-v2`  
**Script:** `scripts/setfit_train.py`  
**Config:** batch=16, 1 epoch, 20 contrastive iterations, logistic regression head  
**Saved to:** `models/setfit/`

Three variants: v1 full (576 examples), v1 few-shot (16/class = 128 examples), v2 full (2,104 examples).

---

## Results

Eval script: `scripts/eval_compare.py` — runs synthetic val (144 examples) and natural test (10 hand-written messages) on each model.

| Model | Synthetic acc | Natural acc | Macro-F1 (nat) | Gap | % <0.8 conf |
|---|---|---|---|---|---|
| Qwen LoRA v1 (w/ label hint) | 99.3% | 60% | — | 39.3% | — |
| Qwen LoRA v2 (w/ label hint) | 97.2% | 60% | — | 37.2% | — |
| ModernBERT v1 | 90.3% | 50% | 0.350 | 40.3% | 60% |
| ModernBERT v2 | 95.1% | 40% | 0.208 | 55.1% | 70% |
| SetFit v1 (full) | 96.5% | 50% | 0.362 | 46.5% | 60% |
| SetFit v1 (16-shot) | 89.6% | 20% | 0.125 | 69.6% | 100% |
| SetFit v2 (full) | 95.8% | 50% | 0.333 | 45.8% | 30% |

**Comparability notes:**
- Qwen receives an explicit label list in its inference prompt; ModernBERT and SetFit do not (structurally correct for discriminative classifiers). Qwen's 60% natural score is achieved *with* this hint.
- Synthetic val contains 36.5% Bitext template tokens (`{{Order Number}}` etc.); natural test has zero. Val and natural scores measure different things and should not be averaged.
- 3.5% train/val leakage (5 of 144 val examples appear in train) affects all models equally.
- Natural test n=10: binomial 95% CI ≈ ±30 points at 60%. Results are directional only — a difference of 1–2 correct answers is noise.

---

## Notable per-model failure patterns

**ModernBERT v1 (50% natural):** Collapses billing and delivery onto `bug_report` for keyword-free messages. "Monthly statement numbers don't add up" → bug_report. "Order supposed to arrive Monday" → bug_report. High confidence on wrong answers (0.81, 0.95).

**ModernBERT v2 (40% natural):** The v2 run scored lower than v1 on the same ten messages. "Cancellation without the word cancel" → account_access (0.97 confidence, dead wrong).

**SetFit v1 full (50% natural):** Cancellation failure ("I don't want to keep paying") → other. Refund failure ("sent item back two weeks ago") → delivery. Billing failure ("monthly statement") → other. Different failure modes than ModernBERT, but same overall count.

**SetFit 16-shot (20% natural):** Most non-obvious inputs were assigned to `bug_report` or `other`. All ten predictions were below the 0.8 confidence threshold in this run. This small result does not validate a confidence rule or establish a minimum number of examples per class.

**SetFit v2 full (50% natural):** Three of the ten predictions were below the 0.8 confidence threshold. It still fails on the same hard cases: keyword-free cancellation, implicit billing/refund.

---

## Conclusion

**The saved runs show a gap between synthetic validation and ordinary customer wording.** The natural-language development check ranged from 20% to 60% across the six variants, while synthetic validation ranged from 89.6% to 99.3%. The ten natural messages were reused during development, so these results are directional and not a final held-out estimate.

**Changing model family did not remove the observed failures.** ModernBERT v1→v2 moved from 50% to 40% on the same ten messages, while SetFit variants ranged from 20% to 50%. These comparisons describe this data and these runs; they do not identify whether data, labels, prompts, or architecture was the main cause.

**The portfolio lesson is bounded:** validation examples that share template patterns can overstate readiness for ordinary customer language. Stop the experiment here. Any future work would need a separately designed, held-out support corpus and label review rather than more runs on these reused messages.
