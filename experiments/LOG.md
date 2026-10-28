# Experiment log

One entry per step, newest at the bottom. Each entry says **what** we did, **why**, the **result** and **what we decided**. Numbers also go in [results.csv](results.csv). This log is the source for the final presentation.

---

## E00: Read the brief (28 Oct)
- **What:** read the launch slides and the Kaggle overview, data and leaderboard pages.
- **Result:**
  - Task: Thai speech-to-text. The score is WER after PyThaiNLP `newmm` word splitting.
  - Splits: 1,636 train / 240 dev / 186 test clips.
  - Private leaderboard (77% of test) sets your rank. You must beat your team baseline to get points.
  - Leaderboard zero-shot scores: Pathumma Fastcon 0.25, Typhoon Whisper Large V3 0.26, Whisper Large V3 0.27, Typhoon Whisper Turbo 0.31, Qwen3-ASR 0.6B 0.32, Typhoon ASR Realtime 0.37.
- **Decision:** limit ourselves to these 6 models. Choose by dev WER, since the public leaderboard covers only ~43 clips.

## E01: Data analysis (28 Oct)
- **What:** compared train, dev and test by file name, sentence content and length.
- **Result:**
  - Train has two sources: `a-` (256 clips, real unscripted speech, numbers spelled out, filler words) and `b-` (1,380 clips, likely computer-generated, lots of names in English letters).
  - Dev and test are both `c-`: read news sentences in the style of FLEURS, with digits, English names and punctuation.
  - Average words per sentence: a 19.3, b 27.0, dev 23.5.
  - Share of sentences with digits: train 0.1%, dev 13.3%.
  - No dev sentence also appears in train.
- **Decision:**
  - Dev is the only data that looks like test, so it guides every choice and goes into final training.
  - Fine-tuning on train alone risks teaching the wrong style, such as spelling out numbers.

## E02: Building the scoring function (28 Oct)
- **What:** built a local WER that matches Kaggle's description.
- **Result:**
  - Splitting references on their own spaces is wrong: they are phrase breaks, and a perfect transcript scored 600% WER.
  - Running newmm on the text as written gives 0% for a perfect transcript and 12.5% on the slide example.
  - Removing every space costs only 1.3% WER, so spacing barely matters.
- **Decision:** use [score_dev.py](score_dev.py) (newmm on the text as written). Also report WER without punctuation and character error rate (CER).

## E03: Zero-shot comparison on dev (28 Oct, running)
- **What:** ran all 6 leaderboard models on the 240 dev clips, with no training. Whisper uses `language=th` and beam size 5.
- **Where:** Kaggle GPU, [run](https://www.kaggle.com/code/sauciiees/dsi443-zeroshot-dev), script [zeroshot_dev.py](zeroshot_dev.py).
- **Problems on the way:**
  - Version 1 had no internet (account not phone-verified).
  - Version 2 had the wrong model list.
  - Version 3 is the correct run.
- **Result:** *pending*
- **Decision:** *pending*. Plan: take the top 2 from different families and give each a short fine-tune before committing.
