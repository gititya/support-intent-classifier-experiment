# [EXPERIMENT] Intent Classifier

I wanted to test a basic support use case with LLMs: can a small model put a customer’s message into the right support category, even when customers do not use obvious keywords?

I hand-wrote 10 customer requests in ordinary language. Across the saved runs, the models got between 2 and 6 of them right.
I used the same messages while changing the training data, prompts and model choices. Those results are part of this experiment, not a test on a fresh set of customer requests.

Fine-tuned `Qwen2.5-1.5B-Instruct` to classify customer support messages into 8 labels: `billing`, `account_access`, `refund`, `product_how_to`, `bug_report`, `cancellation`, `delivery`, `other`.

## The process
1. **Data prep**:  Used 26K Bitext rows (from Hugging Face), mapped 27 intents to 8 labels, capped at 100 per label to balance, then produced 576 training and 144 validation examples.
2. **LoRA config**: Rank 8, alpha 16, lr 1e-4, batch 4, 720 iters (~5 passes through the data).
3. **Training** : `mlx-lm` on M3 MacBook Air 16GB. It ran for ~20 mins with peak memory: ~2GB & Loss gradually improved from: 1.28 → 0.15.
4. **Eval** : 99.3% on 144 synthetic validation examples. The one miss was an `other` message predicted as `account_access`.
5. **Baseline** : The same 144-example evaluation on the untouched base model scored 50.7%. The fine-tuned run scored 99.3% on that validation set. This comparison measures output accuracy on the supplied labels; it does not show why the scores differed.

**Then the check on ten hand-written customer requests scored 60%.**
## TL;DR
1. The fine-tuned model hit 99.3% accuracy on 144 synthetic validation examples.
2. On ten ordinary messages without the Bitext template patterns, the same run scored 60%.
3. The test included failures such as "I don't want to keep paying for this" → `other`.

That was the useful lesson: doing well on tidy, familiar examples did not mean the classifier could handle ordinary customer wording. I reused the same ten messages for the later checks, so I am keeping the conclusion limited to the failures I actually saw.

## The augmentation round

Used Claude Haiku to paraphrase every training example without the obvious keyword. For `other`, generated fresh diverse examples instead — compliments, rants, off-topic questions — since paraphrasing Bitext `other` rows just produces more newsletter-unsubscribe variations. Dataset grew from 576 → 2,104 examples. Retrained (~75 minutes).

Result: still 60% on the same ten reused messages, but with different failures.

**Fixed:** `cancellation` ✓ — paraphrases without "cancel" worked. `other` ✓ — diverse generation worked.

**New failures:** `billing` and two `account_access` messages all predicted as `bug_report`. The saved comparison shows a different error pattern after augmentation; it does not establish why the class boundary changed.

This points to a label-definition question: the `bug_report` examples did not cleanly separate technical failures from billing and account complaints. The run does not tell us whether label definitions, data, prompts, or model choice was the main cause.

## The encoder round — was it the model?

The obvious next question was whether another model family would change the result. I built two more classifiers on the *same* data and tested them on the *same* ten reused messages.

- **ModernBERT** (`answerdotai/ModernBERT-base`, 149M encoder) — a different model family. Full fine-tune, v1 (576) and v2 (2,104).
- **SetFit** (`all-MiniLM-L6-v2` + logistic head) — contrastive few-shot, chosen because the bottleneck looked like small/overlapping-label data, not model size. Full, 16-shot, and v2.

The saved runs did not show another model family removing the failures. One different answer changes a score by ten points, so differences between adjacent results are noisy.

| Model | Synthetic val (n=144) | Natural (n=10) | Macro-F1 (n=10) | Below 0.8 confidence (n=10) |
|---|---|---|---|---|
| Qwen LoRA v1 (w/ label hint) | 99.3% | 60% | — | — |
| Qwen LoRA v2 (w/ label hint) | 97.2% | 60% | — | — |
| ModernBERT v1 | 90.3% | 50% | 0.350 | 60% |
| ModernBERT v2 | 95.1% | 40% | 0.208 | 70% |
| SetFit v1 (full) | 96.5% | 50% | 0.362 | 60% |
| SetFit v1 (16-shot) | 89.6% | 20% | 0.125 | 100% |
| SetFit v2 (full) | 95.8% | 50% | 0.333 | 30% |

Two things fell out of this:

1. **The augmentation run did not fix the reused natural-language cases.** ModernBERT scored 50% on v1 and 40% on v2 in these saved runs.
2. **Confidence values varied across runs.** They are recorded for inspection, but this ten-message development check does not validate a confidence threshold or a routing policy.

**The conclusion:** this experiment shows a gap between template-like validation and ordinary customer wording in a small development set. Changing model family did not remove the observed failures. The runs do not establish that data, labels, prompts, or architecture is the general cause.
Full tables and per-model examples are in `EXPERIMENTS.md`.

(I also checked CFPB's 49K real consumer complaints as a real-language source — rejected: 77% credit-reporting, a taxonomy with no delivery/how-to/bug/cancellation, and `XXXX` redaction tokens that would just swap one template artifact for another.)

## What this is not

1. NOT a production classifier. Bitext is synthetic. The labels are generic e-commerce. The saved natural-language results are 20–60% on the reused ten-message development set.
2. NOT a benchmark of Qwen2.5-1.5B, ModernBERT, or SetFit. A different dataset or label set could produce different results.

## Stack

- Model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- Training: `mlx-lm` (LoRA, rank 8)
- Data: Bitext customer support dataset + Claude Haiku augmentation
- Hardware: Apple M3, 16GB unified memory
- No TypeScript, no server, no web UI

## Files

```
scripts/prepare_data.py     — download, map, balance, split
scripts/augment_data.py     — Haiku paraphrase augmentation
scripts/eval.py             — accuracy + confusion matrix (Qwen)
scripts/natural_test.py     — 10 hand-written messages reused during development
scripts/modernbert_train.py — ModernBERT full fine-tune (v1 + v2)
scripts/setfit_train.py     — SetFit contrastive (full, 16-shot, v2)
scripts/eval_compare.py     — unified harness: synthetic + natural, all models
config/lora_config.yaml     — training hyperparameters
EXPERIMENTS.md              — full comparison table + per-model failure patterns
```

## Results

Qwen LoRA round (the original experiment):

| Model | Synthetic val (n=144) | Natural language (n=10) |
|---|---|---|
| Baseline | 50.7% | — |
| v1 fine-tune (576 examples) | 99.3% | 60% |
| v2 fine-tune (2,104 examples, augmented) | 97.2% | 60% |

Full three-way comparison (Qwen vs ModernBERT vs SetFit) is in the table above and in `EXPERIMENTS.md`. The saved results range from 20% to 60% on the same ten reused natural-language messages; treat them as directional evidence from this experiment.
