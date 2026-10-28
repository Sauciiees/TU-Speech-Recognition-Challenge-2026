# DSI443 TU Speech Recognition Challenge 2026: Individual Plan

Deadline: Kaggle submission **Wed 4 Nov 2026, 1:00 PM** (no late entries). Presentation at 1:30 PM. Colab notebook due 4:00 PM.

## Progress (updated 28 Oct)

| # | Step | Status |
|---|---|---|
| 0 | Repo, Kaggle access, data CSVs downloaded | ✅ Done |
| 1 | Understand the data and the metric | ✅ Done (see findings) |
| 2 | Zero-shot comparison of the 6 leaderboard models on dev | ⏳ Running on Kaggle ([run](https://www.kaggle.com/code/sauciiees/dsi443-zeroshot-dev)) |
| 3 | Pick the base model, first Kaggle submission | ⬜ Next |
| 4 | Fine-tune (train, then train + dev) | ⬜ |
| 5 | Error analysis, text clean-up and decoding tuning | ⬜ |
| 6 | Final submission, notebook and slides | ⬜ |

## Rules that shape the plan

- **Models:** only the 6 zero-shot models on the leaderboard.
- **You must beat your team baseline**, or you get 0 leaderboard points. *Still need: your team's assigned model and its score.*
- **Choose by dev WER, not by public leaderboard score.** The public score covers only 23% of test (~43 clips); the private one covers 77%.
- **Never use test audio or test transcripts for training or tuning.** Dev and test look like the public FLEURS Thai set. Its test transcripts are almost certainly the answers here, so never touch them.
- **External data is unclear.** The Data tab says train only; the Overview says public data is fine if cited. Ask the instructor before using any.

## Findings so far

**1. Train comes from different sources than dev and test.**

| Split | Source | Clips | What it is |
|---|---|---|---|
| Train `a-…` | real speech | 256 | Unscripted talk and news shows. Has fillers ("เอ่อ") and numbers spelled out as words. |
| Train `b-…` | likely computer-generated | 1,380 | Made-up sentences full of names, often in English letters. |
| Dev `c-validation-…` | read news (FLEURS-like) | 240 | Wikipedia-style sentences with digits ("ปี 1978"), English names and punctuation. |
| Test `c-test-…` | same as dev | 186 | Same style as dev. |

What this means:
- Dev is the only data that looks like test.
- Fine-tuning on train alone may teach the wrong style, such as spelling out numbers. Always check dev WER before and after.
- Dev should go into the final training run.
- No dev sentence also appears in train.

**2. Scoring.**
- The spaces in the references are natural Thai phrase breaks, not word boundaries.
- The correct local metric runs newmm on the text as written: [experiments/score_dev.py](experiments/score_dev.py).
- With this metric, word spacing barely matters: removing every space costs only 1.3% WER.

**3. Leaderboard on 28 Oct.**
- The host's zero-shot scores: Pathumma Fastcon 0.25, Typhoon Whisper Large V3 0.26, Whisper Large V3 0.27, Typhoon Whisper Turbo 0.31, Qwen3-ASR 0.6B 0.32, Typhoon ASR Realtime 0.37.
- The best student so far: 0.18.

## Plan, day by day

### Wed 28 – Thu 29 Oct: pick the base model
1. Score all 6 models on dev with `score_dev.py`. Report WER, WER without punctuation, and character error rate (CER).
2. Look at the best 2–3 models' mistakes:
   - Numbers as digits vs words.
   - English names in Thai vs Latin letters.
   - Punctuation.
   - Repeated output (hallucination).
3. Pick the base model. It needs good dev WER and must be possible to fine-tune on a free GPU:
   - **Whisper models** (Typhoon large-v3, Typhoon turbo): fine-tune with LoRA (a small trainable add-on) in transformers. We know this method well.
   - **FastConformer models** (Pathumma, Typhoon realtime): small (~115M) and fast to fine-tune with NeMo. NeMo setup is harder.
   - **Qwen3-ASR**: newest and least-known fine-tuning route. Only pick it if it clearly wins.
4. Submit the best zero-shot model's test predictions once. This checks the pipeline end to end and shows whether it already beats your team baseline.

### Fri 30 Oct – Sun 1 Nov: fine-tune
5. Fine-tune on train and check dev after every epoch. Keep the best checkpoint.
6. **If train alone hurts dev, train on part of dev.** Split dev into folds: train on train + 4/5 of dev, score the remaining 1/5, and rotate through all folds. This uses in-domain data and still gives an honest score.
7. Augmentation: try speed changes (0.9× / 1.1×), background noise and SpecAugment (masking parts of the audio). Keep only what lowers dev WER.

### Mon 2 – Tue 3 Nov: clean-up and decoding
8. Rule-based text clean-up from the error analysis. Examples: number format, ๆ, punctuation, cutting off repeated output.
9. Decoding settings: beam size, length penalty, no-repeat limit.
10. Final model: retrain on **train + all of dev** with the best settings. Predict on test.

### Wed 4 Nov (before 1 PM)
11. Final Kaggle submission, choosing by dev WER. Select two submissions as finals.
12. Build the **Colab notebook** to hand in (experiments run on Kaggle, but the deliverable is Colab). It reproduces the final pipeline end to end:
    - Download the data with the Kaggle API, which needs your API token in Colab.
    - Fine-tune the model.
    - Predict on test and write the submission CSV.
    - Run it once on Colab yourself so the saved notebook shows its outputs.
13. Build the presentation from [experiments/LOG.md](experiments/LOG.md) and [experiments/results.csv](experiments/results.csv). Planned slides:
    - The task and the rules.
    - Data findings.
    - The scoring pitfall.
    - Zero-shot comparison (chart).
    - Fine-tuning experiments, including what failed.
    - Error examples before and after.
    - Final result vs team baseline.
    - Lessons learned.

**Logging rule:** every experiment gets an entry in `experiments/LOG.md` (what, why, result, decision) and a row in `experiments/results.csv`. Failed runs are logged too, since they make good presentation material. Results table: zero-shot → fine-tuned → plus clean-up and decoding. Add error examples and what helped or didn't.

## Files

- `experiments/zeroshot_dev.py`: runs the 6 models on dev (Kaggle GPU) and saves `pred_<model>.csv`.
- `experiments/score_dev.py`: scores prediction files locally.
- `experiments/kernel-metadata.json`: Kaggle run settings.
- `data/datasets/*.csv`: competition CSVs (audio is on Kaggle only).
