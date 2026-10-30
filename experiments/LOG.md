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
  - Version 3: both Whisper large models **ran out of GPU memory** (16 clips at once × beam 5 on a 15 GB T4).
  - Version 4 reruns them with 4 clips at once and also writes test predictions.
- **Version 3 results** (dev, 240 clips, 46 min of audio; T4 time includes model loading):

  | Model | Dev WER | WER no punct. | CER | Public LB (host) | T4 time |
  |---|---|---|---|---|---|
  | Pathumma Fastcon | **0.111** | 0.110 | 0.049 | 0.25 | 65 s |
  | Qwen3-ASR 0.6B | 0.193 | 0.186 | 0.077 | 0.32 | 761 s |
  | Typhoon Whisper Turbo | 0.195 | 0.194 | 0.085 | 0.31 | 270 s |
  | Typhoon ASR Realtime | 0.270 | 0.269 | 0.105 | 0.37 | 47 s |

- **Our dev scores are about 0.12 lower than the host's leaderboard scores, for every model.**
  - The ranking is the same, so it looks like a constant offset.
  - Possible causes: the host's runs skipped newmm splitting, or Kaggle scores differently, or dev and test differ.
  - A real submission will tell us.
- **Error analysis of Pathumma Fastcon** (325 substitutions, 212 insertions, 88 deletions):
  - Writes numbers as digits, like the references (32 reference clips have digits; its output has digits in 35).
  - **Main error: English names written in Thai letters.** Examples: "Jamie Siminoff" → "เจมี่ ซีมินอฟ", "Carter" → "คาร์เตอร์", "CEP" → "ซีอีพี". Each name costs several words.
  - Small spelling slips: "เส้นชัย" → "เส้นชาย", "super-g" → "superg".
- **Error analysis of Typhoon Whisper Turbo:**
  - **Never writes digits.** "1978" comes out as "หนึ่งพันเก้าร้อยเจ็ดสิบแปด". That is one reference word turned into about 7 errors, and the most common insertions are พัน, หนึ่ง, ร้อย, เก้า, สิบ.
  - **Drops the repetition mark ๆ:** "ต่าง ๆ" → "ต่าง" (18 times).
  - **Repeats itself:** in one clip it repeated a whole phrase (hallucination).
  - Never outputs punctuation.
- **Version 4: all 6 models** (Whisper large at batch 4; dev and test transcribed; T4 time covers dev + test):

  | Model | Dev WER | WER no punct. | CER | Public LB (host) | T4 time |
  |---|---|---|---|---|---|
  | **Pathumma Fastcon** | **0.111** | 0.110 | 0.049 | 0.25 | 86 s |
  | Typhoon Whisper Large V3 | 0.151 | 0.150 | 0.072 | 0.26 | 2,432 s |
  | Qwen3-ASR 0.6B | 0.193 | 0.186 | 0.077 | 0.32 | 1,382 s |
  | Typhoon Whisper Turbo | 0.195 | 0.194 | 0.085 | 0.31 | 521 s |
  | Whisper Large V3 | 0.220 | 0.219 | 0.079 | 0.27 | 2,275 s |
  | Typhoon ASR Realtime | 0.270 | 0.269 | 0.105 | 0.37 | 56 s |

  - Pathumma Fastcon gave exactly 0.1110 again, so results are reproducible.
  - Typhoon Whisper Large V3 also writes **no digits** (0 clips), the same problem as Turbo.
  - OpenAI Whisper Large V3 does write digits (45 clips). Its CER (0.079) is close to Typhoon large's, so most of its extra WER comes from word choice and spelling, not from mishearing.
  - Whisper large is about 28× slower than Pathumma Fastcon.
- **Decision:** fine-tune **Pathumma Fastcon** (E04).
  - Runner-up: **Typhoon Whisper Large V3** (0.151). It could gain a lot from a "number words → digits" clean-up, but it would need LoRA and is about 28× slower.
  - Submission file built from Pathumma Fastcon's zero-shot test predictions, to check our dev scoring against Kaggle's.
- **Earlier provisional decision (from version 3):**
  - Pathumma Fastcon is the clear base-model candidate: 0.111, nearly half the next model's WER, and it's small (~115M) and fast, so full fine-tuning fits on a T4.
  - For the Whisper models, the numbers-as-words problem could be fixed with a "Thai number words → digits" clean-up step.

## E04: Fine-tuning Pathumma Fastcon (28 Oct, running)
- **Why:** best zero-shot model by far (0.111). Its ~115M parameters allow full fine-tuning on a T4, so no LoRA needed. Its main weakness is English names, which train has plenty of.
- **What:** [finetune_fastcon/finetune_fastcon.py](finetune_fastcon/finetune_fastcon.py) on Kaggle ([run](https://www.kaggle.com/code/sauciiees/dsi443-finetune-fastcon)). Both experiments start from the pretrained model:
  - **A_train:** fine-tune on train (1,636 clips) and score on all of dev after epochs 0 (no training), 2, 4, 6, 8 and 10. Question: does mismatched training data help or hurt?
  - **B_train_dev80:** fine-tune on train + dev folds 1–4 (190 clips), and score on dev fold 0 (50 clips) after epochs 2–10. Question: how much does test-like data add?
  - Dev folds = clip number mod 5.
- **Settings:**
  - Optimizer: AdamW, learning rate 1e-4, cosine schedule with 10% warmup, weight decay 1e-3, gradient clipping 1.0.
  - Batch: 8 with accumulation of 2 (effective 16), mixed precision.
  - Built-in SpecAugment. Clips over 40 s are dropped.
  - Characters the tokenizer can't encode are removed from the training text.
- **Bug caught before running:** the clip-number pattern matched the "3" in ".mp3", which put all of dev in one fold. Fixed; folds are now 50/50/46/48/46.
- **Data stats (from this run):**
  - Train: 4.26 h, mean 9.4 s per clip.
  - Dev: 0.77 h, mean 11.6 s.
  - Test: 0.65 h, mean 12.6 s.
  - No clip is over 40 s.
  - Only one character can't be encoded: "ń".
- **Result A_train** (22 min of training on a T4):

  | Epoch | 0 (no training) | 2 | 4 | 6 | 8 | 10 |
  |---|---|---|---|---|---|---|
  | Dev WER | **0.111** | 0.123 | 0.130 | 0.133 | 0.133 | 0.132 |
  | Dev CER | 0.049 | 0.057 | 0.059 | 0.061 | 0.060 | 0.060 |
  | Dev clips with digits in output | 35 | 11 | — | — | — | 6 |
  | Dev clips with Thai number words | 30 | 40 | — | — | — | 45 |
  | Dev clips with punctuation | 8 | 3 | — | — | — | 0 |

- **Fine-tuning on train alone made dev worse, from the first checkpoint on.** The model picked up train's writing style:
  - Numbers spelled out as words instead of digits.
  - No punctuation.
  - It does start writing some English names in English letters ("Gornan John"), but not well enough to make up for the losses.

  This confirms the domain-mismatch risk from E01. The training data teaches a different transcription style from what dev and test use.
- **Result B_train_dev80:** crashed with **GPU out of memory**.
  - Dev clips are longer (up to 35 s) with longer transcripts.
  - RNN-T's joint network builds a table of audio frames × text pieces × vocabulary for each clip, and those tables got too big at batch size 8.
  - It needed 4.1 GB more with only 3.3 GB free.
- **Decision:**
  - Do not fine-tune on train as-is.
  - Next run (E05): retry B with batch 4 × accumulation 4 (same effective batch 16).
  - Add a dev-only experiment with a lower learning rate, to test whether in-domain data alone helps.

## E05: Fine-tuning runs B and C (28 Oct, running)
- **What:** Kaggle notebook `dsi443-finetune-fastcon`, version 2. Batch 4 × accumulation 4 (effective 16), to fit dev's longer clips in memory.
  - **B_train_dev80:** train + dev folds 1–4, learning rate 1e-4.
  - **C_dev80:** only dev folds 1–4 (190 clips), learning rate 3e-5 (gentler).
  - Both are scored on dev fold 0 (50 clips) after epochs 2–10. C also scores epoch 0, the untouched model on fold 0.
- **Result** (dev fold 0, 50 clips; zero-shot = **0.105 WER / 0.043 CER**). Training took 27.4 min (B) and 2.8 min (C).

  | Epoch | 2 | 4 | 6 | 8 | 10 |
  |---|---|---|---|---|---|
  | **B** full FT, train + dev 80%: WER | 0.112 | 0.105 | 0.104 | 0.101 | **0.100** |
  | B: CER | 0.049 | 0.041 | 0.043 | 0.040 | **0.040** |
  | **C** full FT, dev 80% only, lr 3e-5: WER | 0.106 | 0.104 | **0.098** | 0.099 | 0.099 |
  | C: CER | 0.045 | 0.045 | 0.042 | 0.042 | 0.042 |
  | **E** LoRA, train + dev 80%: WER (E06) | 0.113 | 0.106 | 0.103 | 0.105 | 0.101 |
  | E: CER | 0.049 | 0.046 | 0.044 | 0.047 | 0.045 |

- **Every run that includes dev data ends better than zero-shot**, at about 0.098–0.101 vs 0.105 (5–7% fewer word errors).
  - Three different setups agree, which is more convincing than any one of them alone on 50 clips.
  - **B (full fine-tuning, train + dev) is the only one that also clearly lowers CER** (0.040 vs 0.043).
  - C gets the best WER with only 190 clips and under 3 minutes of training.
- **Compared with A and D (train only, both worse than zero-shot):** test-like dev data is what makes fine-tuning help.
- **Decision:** fine-tuning with dev data helps a little.
  - Candidates: B epoch 10 and C epoch 6.
  - Fold 0 is only 50 clips, so confirm on Kaggle before building the final model (train + all of dev).

## S02: Kaggle check of fine-tuning with dev data (28 Oct)

| Submission | Dev fold 0 WER | **Kaggle public** |
|---|---|---|
| E03 zero-shot | 0.105 | 0.26 |
| E04 A: full FT, train only, epoch 10 | — | 0.29 |
| **E05 B: full FT, train + dev 80%, epoch 10** | 0.100 | **0.24** ✅ best |
| E05 C: full FT, dev 80% only, epoch 6 | 0.098 | 0.25 |

- **Confirmed on Kaggle:** fine-tuning with dev data beats zero-shot (0.24 vs 0.26). Fine-tuning on train alone is worse (0.29).
- **B beats C on Kaggle** even though C had the slightly better fold 0 WER. This matches B's better CER, and reminds us that a 50-clip check can't separate small differences.
- **Decision:** the final recipe is B's setup (full fine-tuning, lr 1e-4, about 10 epochs), retrained on **train + all of dev**.

## A01: What the best model (B epoch 10) still gets wrong (28 Oct)
- **What:** aligned the newmm words of each reference and prediction on dev fold 0 (50 clips, 1,263 reference words). Each error is sorted into a category. Compared with zero-shot on the same clips.

  | Error category | Zero-shot | B epoch 10 |
  |---|---|---|
  | Extra Thai word (insertion) | 44 | 44 |
  | Wrong Thai word (substitution) | 37 | 36 |
  | Missing Thai word (deletion) | 18 | 18 |
  | **English word / name** | **17** | **13** |
  | Word split / merge (one word contains the other) | 13 | 13 |
  | **Number** | **3** | **1** |
  | Punctuation | 1 | 1 |
  | **Total (WER)** | 133 (0.105) | 126 (0.100) |

- **Where fine-tuning helped:** fewer errors on English names (17 → 13) and numbers (3 → 1). The Thai-word errors barely changed.
- **Main error sources:**
  1. **Foreign names, written inconsistently in the references.**
     - Some references use English letters ("Jamie Siminoff", "ring", "super-g").
     - Others use Thai spelling for the same kind of name ("อาร์ลี เวลาซเกซ", "แจแปนิสโคสเตอร์").
     - The model often picks the other form: "Jamie Siminoff" → "เจมมี่ ซิมินอฟ", "แจแปนิสโคสเตอร์" → "Japanese Coaster".
     - One name can cost several errors when newmm cuts an unfamiliar Thai spelling into fragments ("เจ", "ม", "มี่").
     - **Correction after checking:** this explains only a small part of the Thai-word errors. Of the "extra Thai word" errors, 42 of 44 (95%) are real dictionary words; of the wrong-word substitutions, 44 of 49 (90%). The check isn't perfect, since the dictionary also contains short syllables.
     - **So the largest error source is genuine Thai-word errors (items 2–4 below). Names come second**, but they cause the worst individual clips.
     - **But the clip-by-clip examples show many "extra Thai words" come from a single misspelled word cut into real short words:**
       - "แรงเสียดทาน" → "แรงเสียทาน" (missing ด) is scored as + แรง, + เสีย, แรงเสียดทาน → ทาน: **3 errors for one letter**.
       - "แคโรทีน" → "แคโรจีน" is scored as + แค, + โร, → จีน: **3 errors**.
       - Correct speech in a different written form also costs: "ฯลฯ" → "และอื่น ๆ" is 3 errors.
       - The dictionary check can't catch this, because the fragments are real words.
  2. **Similar-sounding Thai words:** "เส้นชัย" → "เส้นชาย" (twice), "บริหาร" → "บริษัท", "ราชอาณาจักร" → "โลชนาจาก", "แล้ว" → "และ".
  3. **Small function words added or dropped:** "ที่" (4 extra), "ใช้", "และ".
  4. **Compound words split differently:** "กล่าวว่า" → "ว่า", "จาก" → "ออกจาก", "ระเบียบวิธี" → "ระเบียบ".
  5. **Numbers, rare but inconsistent:** in one clip "15" stays a digit but "9" becomes "เก้า".
  6. **Errors in the references themselves:** e.g. "อนุสรณ์สถานใของ" has a stray "ใ". The model can't match a typo, so some errors are unavoidable.
- **Worst clips:** the ones packed with foreign names (WER 0.42–0.48), e.g. c-validation-0005 (Jamie Siminoff / ring) and c-validation-0030 (skiers' names + super-g).
- **Takeaway:** most of what's left is either ambiguous (Thai vs English spelling of names, which the references themselves don't do consistently) or acoustic confusion between similar Thai words. Fine-tuning fixed part of the name and number problem but can't resolve the inconsistency in the references.

## E09: LLM spelling correction on zero-shot Pathumma (28 Oct, notebook ready, run on Colab)
- **Why:** A01 showed one wrong letter often costs about 3 word errors (newmm cuts the non-word into pieces). An LLM knows which Thai words exist.
- **Setup:** zero-shot Pathumma, so no retraining is needed (fine-tuned weights weren't kept). The LLM's effect can be measured on **all 240 dev clips**.
- **Notebook:** [notebooks/pathumma_llm_correction.ipynb](../notebooks/pathumma_llm_correction.ipynb) (Colab, T4).
  - Part 1 (NeMo): transcribe dev and test with zero-shot Pathumma.
  - **Restart the session**, because NeMo pins an older transformers.
  - Part 2: LLM correction.
  - Part 3: write the test submission.
- **LLM:** `typhoon-ai/typhoon2.1-gemma3-4b` (Thai, 3.9B, open, Gemma license), loaded in bf16 because Gemma 3 overflows in fp16. Greedy decoding.
  - Alternative setting: `Qwen/Qwen3-4B-Instruct-2507`.
- **Method: "detect and fix with a constraint".**
  - The LLM returns a JSON list of `{"wrong", "correct"}` pairs for misspelled or non-existent words.
  - The prompt has 3 made-up few-shot examples (not from dev or test).
  - **A fix is applied only if** it changes at most 2 characters and at most 34% of the word, the word really occurs in the sentence, and it doesn't touch English letters or digits. So the LLM can't rephrase, add words, or change names and numbers.
- **Filter checked locally** with made-up LLM answers on real dev sentences:
  - Accepted: แรงเสียทาน → แรงเสียดทาน (that sentence goes from 0.25 to 0.00 WER), โอซีลอก → โอซีล็อต.
  - Rejected: a 9-character rewrite, เจมี่ → Jamie, 1978 → number words, a word not in the sentence.
- **Rules:** open-source LLM run locally on Colab, no paid API. The instructor should confirm an extra LLM is allowed.
- **Result:** *pending (run on Colab)*. Compare dev WER/CER before and after, and Kaggle against zero-shot 0.26 and B 0.24.

## E11: ROVER, combining several models by word voting (28 Oct)
- **What:** [rover.py](rover.py) splits each system's transcript into newmm words and aligns the systems one by one into a word network (slots of candidate words, "" meaning "no word here"). Each slot keeps its most-voted word. The first system is the backbone and wins ties.
  - Checked on toy examples: majority wins, extra words are outvoted, missing words are restored.
  - No GPU or training needed; it uses predictions we already had.
- **Data:** the 50 held-out clips (dev fold 0). Every input system was trained without them.

  | System | WER | CER |
  |---|---|---|
  | P0 Pathumma zero-shot | 0.1053 | 0.0433 |
  | PB Pathumma full FT B, epoch 10 | 0.0998 | 0.0397 |
  | PC Pathumma full FT C, epoch 6 | 0.0982 | 0.0421 |
  | PE Pathumma LoRA E, epoch 10 | 0.1006 | 0.0447 |
  | TL Typhoon Large LoRA B, epoch 3 | 0.1235 | 0.0451 |
  | TT Typhoon Turbo LoRA B, epoch 2 | 0.1615 | 0.0505 |
  | **ROVER PB + PC + TL** | **0.0895** | 0.0390 |
  | **ROVER PB + PC + PE + TL + TT (equal votes)** | 0.0911 | **0.0379** |
  | ROVER same 5, weighted 1.2/1.1/1.0/0.8/0.6 | 0.0934 | 0.0376 |
  | ROVER all 6 including zero-shot | 0.0942 | 0.0397 |
  | ROVER PB + PC + PE (Pathumma only) | 0.0966 | 0.0390 |
  | ROVER PB + TL + TT (one per family) | 0.1045 | 0.0425 |

- **Findings:**
  - **ROVER beats every single system:** 0.0998 → 0.0895 (−10% vs the best single model).
  - **Diversity matters:** Pathumma-only voting gains less (0.0966) because the three Pathumma models make similar mistakes. Adding Typhoon Large, a different family, helps most.
  - **Quality matters too:** one strong Pathumma plus two weaker Typhoons (PB + TL + TT) is worse than PB alone. Weak systems outvote the strong one.
  - **Caveat:** 6 combinations were tried on the same 50 clips, so the very best number is a little optimistic. The two top combinations agree, which makes the gain believable.
- **Test submissions built** (not submitted):
  - `submissions/e11_rover_PB_PC_TL.csv`: differs from Pathumma B on 52 of 186 test clips.
  - `submissions/e11_rover_5sys.csv`: differs on 82 of 186.
- **Kaggle (S05): both ROVER submissions = 0.24 public**, the same as Pathumma B alone.
  - The public score covers only ~43 test clips and shows 2 decimals, so a gain under about 0.005 can't show.
  - The dev gain (−10%) did not show up on Kaggle. It may be smaller on test, or hidden by rounding; the private set (~143 clips) will tell.
  - **Decision:** ROVER is no worse than B on Kaggle and clearly better on dev, so the PB + PC + TL ROVER is a good candidate for one of the two final selected submissions, next to B.
- **Follow-up: more voters (zero-shot models) don't help** (50 held-out clips):

  | ROVER combination | WER |
  |---|---|
  | PB + PC + TL (best) | **0.0895** |
  | + Qwen3 zero-shot | 0.0911 |
  | + PE + OpenAI Whisper Large zero-shot | 0.0942 |
  | + OpenAI Whisper Large zero-shot | 0.0966 |
  | + Typhoon Large zero-shot + OpenAI Whisper | 0.0966 |
  | + OpenAI Whisper + Qwen3 | 0.1006 |

  Weak voters (0.13–0.27 WER alone) outvote correct words more often than they fix wrong ones. ROVER needs voters that are both **strong** and **different**.
- **Approach agreed for further improvement:** judge every idea on dev first and submit only clear gains. No fishing on the public leaderboard (~43 clips, 2 decimals), since tuning on test is against the rules and tends to hurt the private score.

## E16: Model soup (29 Oct, done: not submitted)
- **Why:** recipe step 5. Averaging the weights of several runs that differ only in random seed gives one model, so there's no ROVER-style mismatch on test.
- **What:** same H recipe (train + dev ×3, 10-epoch schedule; each run's ingredient = its own average of epochs 6, 8 and 10), with `pl.seed_everything(seed)` per run.
  - **`dsi443-soup-val`:** 3 seeds (11, 12, 13) on train + dev 80% ×3. Scores each seed and the 3-model soup (greedy and beam) on fold 0. Compare with H_val average + beam (0.0800).
  - **`dsi443-soup-final`:** 2 seeds (12, 13) on train + all dev ×3, plus the saved `H_final_avg6-8-10.nemo` from E15 as the third ingredient (attached via `kernel_sources`). Writes test predictions (greedy and beam) and saves the soup model.
- **Rule:** submit `soup-final` only if the soup beats a single model on fold 0.
- **Result** (fold 0, 50 clips):

  | Model | Greedy | Beam 4 |
  |---|---|---|
  | Seed 11, own avg 6-8-10 | 0.0879 | |
  | Seed 12, own avg 6-8-10 | 0.0934 | |
  | Seed 13, own avg 6-8-10 | 0.0863 | |
  | **3-seed soup** | 0.0895 | 0.0823 |
  | H_val avg 6-8-10 (E15, default seed) | 0.0816 | **0.0800** |

  - **The soup is no better than an average single seed** (0.0895 vs 0.086-0.093), and worse than H_val with beam (0.0823 vs 0.0800).
  - **Seed noise is large:** the same recipe gives 0.086 to 0.093 on fold 0 (about ±4 words of 1,250). Differences smaller than that between experiments aren't real; the 0.0800 of H_val was partly a lucky seed.
  - Likely reason: full fine-tuning from the same start for 10 epochs with different seeds moves the weights into slightly different places; averaging them lands between them instead of in a better spot.
- **Decision:** soup-final **not submitted** (rule: only clear fold-0 gains). E15 (0.19) stays the best. The soup outputs are kept in `results/soup_final/` (not used).

## E17: Smarter LLM corrector: new system prompt + newmm gate (29 Oct, done: not submitted)
- **Why:** an audit of E10 (all 123 proposals vs the references) showed what the LLM got wrong:
  - copied the prompt's example word 29 times;
  - "fixed" correct words it couldn't hear (cricket รัน → เรน, +2 errors);
  - swapped in synonyms (ตัวอาคาร → ตัวบ้าน, ปลอดภาษี → ไม่เสียภาษี);
  - added words (เกาหลี → เกาหลีใต้), invented names, and translated English into Thai.
  - Without the filter it would have made dev worse: 0.1110 → 0.1140 (37 sentences worse, 15 better).
- **New prompt (P1):** tells the LLM the input is an ASR transcript of Thai read news speech, that it can't hear the audio, and that most sentences are already right. Errors look like "sounds right, spelled wrong". Strict rules: same sound only, no synonyms, leave real words alone, never add/remove/rephrase, never touch English/numbers or translate, don't invent names, `[]` when unsure. 7 invented examples (4 with no fix, one showing รัน is correct); `build.py` checks that no example word occurs in the real inputs.
- **New filter (tok):** on top of the E10 filter, a fix must help newmm: the corrected word splits into fewer newmm pieces, or turns a non-dictionary word into a dictionary word. On E10's old replies it keeps the 3 good fixes and blocks the 3 bad ones: zero-shot dev 0.1110 → **0.1097**.
- **Run:** Kaggle `dsi443-llm-prompt` ([llm_prompt/](llm_prompt/)), Typhoon 2.1 4B with the old prompt (P0) and the new one (P1) on 611 unique sentences: fold-0 output of H_val, G_over and B_long, all 240 dev clips from A (train only) and zero-shot, and the 186 test sentences of E15. The LLM only proposes; `score.py` filters and scores locally. 12B dropped (4B is enough, per the user).
- **Rule:** apply to E15's test output and submit only if P1 + filter lowers WER on the fine-tuned dev sets (hval, gover, blong, a10) without making any of them worse.
- **Result** (dev WER; in brackets: sentences changed, +better, -worse):

  | Set (dev) | No LLM | Old prompt + tok | **New prompt + tok** | New prompt, E10 filter | New prompt, no filter |
  |---|---|---|---|---|---|
  | hval (best recipe, 50) | **0.0800** | 0.0808 (1: +0 -1) | 0.0839 (2: +0 -2) | 0.0847 | 0.0911 |
  | gover (50) | **0.0839** | 0.0847 | 0.0871 (3: +1 -2) | 0.0895 | 0.0950 |
  | blong (50) | **0.0903** | 0.0911 | 0.0911 (1: +0 -1) | 0.0918 | 0.1006 |
  | a10 (train only, 240) | 0.1319 | **0.1312** (2: +2 -0) | 0.1314 (3: +2 -1) | 0.1344 | 0.1428 |
  | zero-shot (240) | 0.1110 | 0.1097 (3: +3 -0) | **0.1094** (6: +4 -2) | 0.1117 | 0.1190 |

  - **The new prompt made the LLM bolder, not smarter.** It proposed more fixes (93 vs 58 sentences on a10) and more of them were wrong. Without a filter it hurts every set badly.
  - **It still breaks real words it doesn't know:** เคลือบแคลง → เคลือบคลาง (+4), นิวเคลียร์ → นิวเคลียส (+2), กริ่ง → กรอง, เสียสมดุล → เสียสมดิท (+3). The rules in the prompt didn't stop that; only the filters did.
  - **"Correct" spelling can be wrong for the metric:** ไบเซนไทน์ → ไบแซนไทน์ is the dictionary spelling, but the reference writes ไบเซนไทน์, so the fix cost +4. The fine-tuned model has already learned the references' spelling style.
  - **On the fine-tuned model's output there is almost nothing left to fix:** every LLM variant makes hval, gover and blong worse. The LLM only helps the zero-shot model, which we don't submit.
  - The newmm gate (tok) is the best filter in every case: it blocks most bad fixes.
- **Decision:** not submitted (it makes our best model's held-out WER worse). **LLM correction is closed.** E15 (0.19) stays the best.

## A02: Why is dev 0.08 but Kaggle 0.19? and two free checks (29 Oct)
- **Is it the metric?** Re-scored the 7 dev runs with known Kaggle scores using other WER variants: lowercase, spaces removed, space tokens kept, mean per-sentence WER. **None matches Kaggle**; all stay close to our score. So the gap isn't a metric mismatch.
- **The test set is harder than dev:**
  - Pathumma and Typhoon Large disagree on 17% of words on test vs 12% on dev.
  - Test outputs have about twice the share of English words (1.4% vs 0.7%), and English names are the most inconsistently written part of the references (A01).
  - Test sentences are a bit longer (26 vs 24 words).
  - This also explains why Typhoon Large ties Pathumma on Kaggle (0.26) while it's clearly behind on dev.
- **Free check 1: ROVER of matched models** (all trained on train + dev80 ×3: H_val beam + seeds 11/12/13): 0.0871-0.0879 vs H_val beam 0.0800. No gain.
- **Free check 2: ROVER with fine-tuned Typhoon Large** (LoRA, dev80) added: 0.0831. No gain.
- **Noise level:** seeds of one recipe span 0.086-0.093 on fold 0 (E16), so gains under about 0.005 can't be detected on 50 clips.

## E18: Speed perturbation and a lower learning rate (29 Oct, done: neither submitted)
- **Why:** the two ideas from A02's research that fit our rules (same data, same model) and cost one run each.
  - **P (speed perturbation):** the 3 dev copies are played at 1.0×, 0.9× and 1.1× instead of 3 identical copies (tempo and pitch change together, Kaldi style). More acoustic variety from the same clips.
  - **L (lower learning rate):** 5e-5 instead of 1e-4, so the model forgets less of what it learned in pre-training.
- **Design:** each recipe is run with seeds 11 and 13. They are compared with S_val_s11 / S_val_s13 (E16: same recipe at 1e-4 without speed changes): average of epochs 6-8-10, greedy **0.0879 / 0.0863 (mean 0.0871)**. This takes out most of the seed noise (±0.004).
  - Kaggle `dsi443-speed-val` (P_val_s11, P_val_s13) and `dsi443-lowlr-val` (L_val_s11, L_val_s13).
- **Rule:** if a recipe beats the matched seeds in both seeds by a clear margin, run its final twin (train + all dev) and submit.
- **Result P (speed perturbation)**, fold 0 WER, same seed side by side (baseline = S_val from E16):

  | | Seed 11 baseline | Seed 11 **speed** | Seed 13 baseline | Seed 13 **speed** |
  |---|---|---|---|---|
  | epoch 6 | 0.0918 | 0.0918 | 0.0871 | 0.0911 |
  | epoch 8 | 0.0887 | **0.0839** | 0.0839 | **0.0831** |
  | epoch 10 | 0.0887 | **0.0871** | 0.0863 | **0.0839** |
  | avg 6-8-10 | **0.0879** | 0.0911 | 0.0863 | **0.0847** |
  | avg 6-8-10 + beam | - | 0.0800 | - | 0.0839 |

  - Mean over all 8 cells: baseline 0.0876, speed 0.0871. **No clear gain:** better in 5 of 8 cells, worse in 2, and the differences are within seed noise.
  - Epoch 8 is better in both seeds (0.0887 → 0.0839, 0.0839 → 0.0831), but the averaged model, which is what we submit, is mixed.
  - Speed perturbation doesn't hurt, but on 50 clips it can't be shown to help. **Not submitted.**
- **Result L (learning rate 5e-5):** **clearly worse in both seeds.**

  | | Seed 11 baseline (1e-4) | Seed 11 **5e-5** | Seed 13 baseline | Seed 13 **5e-5** |
  |---|---|---|---|---|
  | epoch 6 | 0.0918 | 0.0950 | 0.0871 | 0.0934 |
  | epoch 8 | 0.0887 | 0.0982 | 0.0839 | 0.0942 |
  | epoch 10 | 0.0887 | 0.0958 | 0.0863 | 0.0966 |
  | avg 6-8-10 | 0.0879 | 0.0958 | 0.0863 | 0.0950 |
  | avg 6-8-10 + beam | - | 0.0934 | - | 0.0911 |

  - Mean over the 8 cells: 1e-4 0.0876, 5e-5 **0.0955** (+0.008, about twice the seed noise, and worse in every cell).
  - So forgetting is not the problem: the model needs to move **further** from Pathumma toward the dev style, not less. If anything, a higher rate (2e-4) is the direction to try.
- **Also changed:** val runs no longer transcribe test (`VAL_TEST_PREDS = False`); this saves about 40% of a val notebook's time.

## E19: Higher learning rate, 2e-4 (29 Oct, running on Colab)
- **Why:** 5e-5 was clearly worse than 1e-4 (E18), so the model may need to move further toward the dev style.
- **What:** the H recipe at lr 2e-4, seeds 11 and 13 (U_val_s11 / U_val_s13), scored on fold 0.
- **Where:** Kaggle's weekly GPU quota (30 h) ran out, so it runs on Colab (free T4): [notebooks/highlr_2e-4_colab.ipynb](../notebooks/highlr_2e-4_colab.ipynb), the same training script embedded, with results saved to Google Drive.
- **Compare with:** same seeds, average of epochs 6-8-10, greedy: 1e-4 **0.0879 / 0.0863**, 5e-5 0.0958 / 0.0950.
- **Result:** *pending*

## E20: Add public FLEURS Thai train data (29 Oct, done: worse on Kaggle)
- **Rules (Kaggle Overview, "Rules"):** external data allowed if public and cited; pretrained models allowed; no external transcription APIs; **no test data in training**.
- **Why:** test-like data is our biggest lever (train only 0.29 → + dev 0.22 → + dev ×3 0.19). Dev and test are read Wikipedia-style sentences like Google FLEURS. FLEURS Thai **train**: **2,602 clips, 8.5 h, 1,408 unique sentences**, about 10× our 190 usable dev clips.
- **Leakage checks:**
  - Only the FLEURS train split is downloaded; dev/test (`dev.tar.gz`, `test.tar.gz`) never are.
  - Checked locally with the transcripts: only **1 of our 240 dev sentences** has a near-copy in FLEURS train (2 clips, dropped). So the splits line up with ours, and FLEURS train is separate from our dev (and so, almost surely, from our test). The script stops if more than 5% of dev overlaps.
- **Possible downside:** FLEURS writes foreign names in Thai script (ดิสนีย์), while our references often use English letters ("Jamie Siminoff"). It may push the model toward Thai spellings of names; stage 1 measures the net effect.
- **What:** E15 recipe + FLEURS train ×1 (seed 11). Stage 1 `FL_val` (train + dev80 ×3 + FLEURS) scored on fold 0; stage 2 `FL_final` (train + all dev ×3 + FLEURS) → test predictions → `submission_e20.csv`. Stage 2 runs only if stage 1's best is ≤ 0.0780 (our best: 0.0800; same seed without FLEURS: avg 0.0879).
- **Notebook:** [notebooks/fleurs_pathumma_colab.ipynb](../notebooks/fleurs_pathumma_colab.ipynb), about 2.5 h on a T4.
- **Citation:** A. Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech", IEEE SLT 2022 (`google/fleurs`, CC-BY-4.0).
- **Result, stage 1** (Colab T4, 54 min of training on 4,806 clips per epoch; fold 0, seed 11):

  | Weights | Without FLEURS (S_val_s11) | **With FLEURS (FL_val)** |
  |---|---|---|
  | epoch 6 | 0.0918 | **0.0816** |
  | epoch 8 | 0.0887 | **0.0823** |
  | epoch 10 | 0.0887 | **0.0855** |
  | avg 6-8-10 | 0.0879 | **0.0800** |
  | avg 6-8-10 + beam | - | 0.0816 |

  - **Better in every cell, by 0.003-0.010 (mean -0.007, about twice the seed noise).** The first idea since E15 that clearly helps.
  - It ties our best-ever fold-0 score (0.0800, H_val avg + beam, which had a lucky seed) with an ordinary seed and greedy decoding.
  - Beam search didn't help on top of FLEURS here (0.0816 vs 0.0800), so the submission uses avg 6-8-10 greedy.
  - The notebook's automatic go/no-go compared against the lucky 0.0800 (needed ≤ 0.0780), so stage 2 was skipped. **Changed the rule to the like-for-like one used in E18:** same seed with vs without FLEURS, the averaged model must win by > 0.005. It does (0.0800 vs 0.0879).
- **Stage 2 (final model):** first try ran out of T4 memory (a batch of long clips; the RNN-T loss tensor). Fixed by computing the joint + loss one clip at a time (`joint.set_fused_batch_size(1)`, same loss, about 1/4 of the peak memory) and `expandable_segments`. 56 min of training on 4,956 clips per epoch.
- **Kaggle: 0.23, worse than E15's 0.19** (submission `e20_FL_final_avg.csv`, 186 rows, checked).
- **Why (comparing the E15 and E20 test outputs):**
  - The FLEURS model writes **foreign names in English letters** much more often: 95 Latin-letter words on test vs 54 for E15 (Princeton, Cincinnati, Christina Fernandez de Canos, La Bata, Raju Kandelan, Rejeb Daef Aduan…), where E15 wrote Thai (พริ้นเซตัน, ราจู…).
  - FLEURS train writes names in Latin letters a bit more than our dev references do (1.6% vs 1.1% of words; 16% vs 12.5% of sentences), so the model learned "use English letters for names".
  - **The test set has many more names than dev** (A02: about twice the Latin share in model outputs). If the references write those names in Thai, every switched name costs several word errors (newmm splits a Thai name into pieces). Our 50 held-out clips contain only a few names, so stage 1 couldn't see this.
- **Lesson:** a gain on 50 held-out clips doesn't carry over when the change affects something those clips barely contain. Extra data must match the references' writing conventions, not just the speaking style.
- **Decision:** E20 not used. **E15 (0.19) stays the best.** Final Kaggle selection: E15 (0.19) and E14 (0.20).

## A03: Why dev doesn't predict test for Pathumma (29 Oct)
- **Dev-to-Kaggle ratio by model** (zero-shot dev WER on 240 clips → Kaggle public):

  | Model | Dev | Kaggle | Ratio |
  |---|---|---|---|
  | Pathumma Fastcon | 0.111 | 0.26 | **2.3×** |
  | Typhoon Whisper Large | 0.151 | 0.26 | 1.7× |
  | Typhoon Whisper Turbo | 0.195 | 0.31 | 1.6× |
  | Qwen3-ASR 0.6B | 0.193 | 0.32 | 1.7× |
  | Typhoon ASR Realtime | 0.270 | 0.37 | 1.4× |
  | Whisper Large V3 | 0.220 | 0.27 | 1.2× |

  Our fine-tuned Pathumma models keep the 2.3-2.4× ratio (B 0.100 → 0.24, E14 0.090 → 0.20, E15 0.080 → 0.19); E20 jumped to 2.9×.
- **Pathumma looks unusually good on dev:**
  - Zero-shot, it reproduces **56 of 240 dev references exactly, spaces and punctuation included** (Typhoon Large 20, Whisper 12), and 77% of the reference's space positions on clips it gets right.
  - Its model card says it was trained on "open-source and in-house datasets" without listing them. FLEURS is a common open-source Thai set, and dev is FLEURS-style (c-validation), so **Pathumma may have seen dev-like or even dev sentences in pre-training.** We can't prove it, but it fits every number above.
- **Consequences:**
  - Fold 0 overstates Pathumma models and rewards "more FLEURS-like" changes. That is likely why E20 (FLEURS train) gained on dev but lost on test.
  - Changes that helped both dev and Kaggle (dev ×3, epochs, averaging, beam) are still trustworthy, since Kaggle confirmed them.
  - Typhoon Whisper Large ties Pathumma on test (0.26) with far fewer English-letter names (13 vs 71 words on test).
- **Names on dev** (aligned newmm words): the references are inconsistent. The same name appears in Thai, in English, or both ("Newt Gingrich newt gingrich", "ริก เพอร์รี่ rick perry"). Name errors are about 12% of H_val's fold-0 errors.

## E21: FLEURS train without English-letter sentences (29-30 Oct, done: worse on Kaggle)
- **Why:** E20's test output wrote far more names in English letters (95 words vs 54 for E15) and scored 0.23. FLEURS train has English letters in 16% of sentences (dev references: 12.5%).
- **What:** the E20 final recipe (train + all dev ×3 + FLEURS, seed 11, avg 6-8-10 greedy) but only FLEURS sentences with **no** Latin letters (`fleurs_nolatin`, ~84% of FLEURS train).
- **No validation twin:** fold 0 favours FLEURS-like changes for Pathumma (A03), so the check is Kaggle against E15 (0.19), plus the English-letter word count on test (should drop back to about 54).
- **Notebook:** [notebooks/fleurs_nolatin_colab.ipynb](../notebooks/fleurs_nolatin_colab.ipynb), one stage, about 85 min on a T4.
- **Result:** 2,174 FLEURS clips (6.8 h) kept; 51 min of training on Colab.
  - **Names check worked:** only **19** English-letter words in the test output (E15: 54, E20: 95).
  - **Kaggle: 0.22.** Better than E20 (0.23) but still **worse than E15 (0.19)**.
  - So English names explain only a small part of E20's drop. **Adding FLEURS itself makes test worse** for Pathumma, which fits A03: Pathumma probably already knows FLEURS-style data, and more of it pulls the model toward FLEURS' own writing habits (spacing, spelling of names in Thai, numbers) instead of this competition's references.
- **Decision:** FLEURS line closed. **E15 (0.19) stays the final model.** Final Kaggle selection: E15 (0.19) and E14 (0.20).

## E22: Switch architecture: Typhoon Whisper Large with the full recipe (30 Oct, ready)
- **The search (alternatives considered and rejected):**
  - Training a new architecture from scratch: impossible with 6 h of labelled audio, a T4 and 5 days.
  - More Pathumma tweaks (soup, LLM, ROVER, speed, LR, FLEURS): exhausted; dev favours Pathumma (A03), so it can't steer them.
  - Extra data (FLEURS, Common Voice): FLEURS failed twice on Kaggle (E20, E21).
  - **Typhoon Whisper Large: under-explored, and the evidence points to it.**
- **Evidence for Typhoon (no new GPU needed):**
  - Zero-shot it ties Pathumma on Kaggle (0.26) **although it writes every number as Thai words**: 0 digit groups on dev and test, while the dev references have 69 digit groups. So acoustically it is the better test model.
  - Its Kaggle/dev ratio (about 1.7-2.1×) is lower than Pathumma's (2.3-2.4×), so its dev score is a more honest guide.
  - **Calibration submission (existing predictions, no GPU):** Typhoon LoRA B5 epoch 5 (train + dev80, 5 epochs, fold 0 0.1108) → **Kaggle 0.20**. Pathumma at the same stage (B: train + dev80) scored 0.24. Typhoon was still improving and had only partly learned digits (39 digit groups on test vs 78 for E15).
- **What (E22):** the recipe that took Pathumma from 0.24 to 0.19, applied to Typhoon: LoRA r32 (attention, encoder + decoder), lr 5e-4, train + dev ×3, 6 epochs, average of the epoch 4-5-6 LoRA weights, greedy decoding.
  - `TW_final` (train + all dev ×3) → test predictions + saved LoRA weights. `TW_val` (train + dev80 ×3) scored on fold 0, to check the epoch choice.
  - Script: [finetune_whisper/finetune_whisper.py](finetune_whisper/finetune_whisper.py) (oversampling, LoRA checkpoint averaging, val runs skip test). Colab: [notebooks/typhoon_final_colab.ipynb](../notebooks/typhoon_final_colab.ipynb), ~4 h on a T4. Kaggle: `bash finetune_whisper/push.sh typhoon-ai/typhoon-whisper-large-v3 dsi443-typhoon-final TW_final` after the weekly GPU reset.
- **Expected:** if Typhoon gains like Pathumma did from this recipe (about -20%), roughly 0.16-0.17. Then a two-architecture combination (E15 + E22) becomes worth testing.
- **Result:** *pending*

## R01: The best recipe, from all experiments plus research (28 Oct)

**What our experiments proved**

| Finding | Evidence |
|---|---|
| Pathumma FastConformer is the best base model | Zero-shot dev 0.111 vs 0.151+ for every other model; still best after fine-tuning |
| **Test-like data is the main lever** | Train only made it worse (A: 0.29 on Kaggle); + 80% dev 0.24; + 100% dev **0.22** |
| **A longer schedule helps** | 10-epoch schedule 0.100 → 14-epoch schedule 0.090 at epoch 12 (held-out clips) |
| Full fine-tuning > LoRA for Pathumma | B 0.100 vs E 0.101 (and B better CER); LoRA only reduced damage on train-only data |
| ROVER helps only when every voter saw the same data | +10% on dev, but on Kaggle ROVER with F (0.23) lost to F alone (0.22) |
| LLM spelling correction ≈ no effect | 0.1110 → 0.1106; most proposals made up |
| Post-processing rules: no effect | Pathumma's formatting already matches the references |

**Ranked recipe** (each step checked on held-out dev before it goes into the final run):

1. **Train + all dev, 14-epoch schedule, epoch 12.** Proven; submit first. File ready: `submissions/e14_F14_ep12.csv`.
2. **Oversample dev** (G_over, running). Upsampling small in-domain data is a common tactic: the STOP low-resource challenge upsampled small domains ×20 when fine-tuning Whisper ([arXiv 2305.01194](https://arxiv.org/pdf/2305.01194)). Closer-domain data beats distant data ([arXiv 2106.01227](https://arxiv.org/pdf/2106.01227)), which matches our A vs B vs F results.
3. **Beam search decoding** (G_over, running). Free at test time. NeMo also supports n-gram language-model fusion for RNN-T beam search (NGPU-LM; KenLM `.arpa` files) ([NeMo docs](https://docs.nvidia.com/nemo/speech/latest/asr/asr_language_modeling_and_customization.html)). An LM built from train + dev text is a possible next step, with medium effort.
4. **Checkpoint averaging** (average the weights of epochs 10, 12 and 14 of one run). Costs nothing extra. Last-k checkpoint averaging is reported to help Conformer ASR ([ApproBiVT, arXiv 2308.02870](https://arxiv.org/pdf/2308.02870)).
5. **Model soup:** average the weights of 2–3 runs trained on the same data with different random seeds. One model at test time, so there's no ROVER-style mismatch. Weight averaging beat the best single model in Model Soups ([Wortsman et al., ICML 2022](https://proceedings.mlr.press/v162/wortsman22a.html)), and averaging fine-tuned ASR experts gave lower WER than any single one ([arXiv 2305.09681](https://arxiv.org/pdf/2305.09681)). Costs about 40 min of GPU per extra run.

**Not worth more time:** LLM correction, ROVER with voters trained on different data, the Typhoon models, post-processing rules, a TPU.

**Final run design (one Kaggle notebook, about 2 h):** two twins with identical settings.
- **H_val:** train + dev 80% (oversampled if G_over wins), 14-epoch schedule. Scores on fold 0 for: epoch 12 greedy, epoch 12 beam, averaged 10–14 greedy, averaged 10–14 beam.
- **H_final:** train + all dev, same settings. Writes test predictions for the same four variants and saves the best model.
- **Submit the variant that wins on H_val.**

## E14: Chasing 1st place (0.17) (28 Oct, running)
- **Leaderboard on 28 Oct:** 1st 0.17, 2nd 0.18, 3rd 0.21, **us 4th at 0.22**.
- **Ideas, each checked on dev fold 0 before submitting:**
  1. **Longer training:** B was still improving at epoch 10.
     - `B_long`: train + dev 80%, 14 epochs, scored at 6–14.
     - Paired with `F_final`: train + all dev, same schedule, test predictions at the same epochs.
     - Kaggle notebook `dsi443-finetune-fastcon`, version 3. (This pair went out by accident while I was preparing G_over, but it answers the "longer training" question, so it was kept.)
  2. **Oversampling test-like data** (`G_over`): dev clips repeated 3× per epoch, 10 epochs.
  3. **Beam search** (4 candidates) vs greedy decoding, on the same `G_over` weights at epoch 10.
  - Ideas 2 and 3 are in notebook `dsi443-finetune-fastcon-g`, queued until a GPU slot frees up.
- The script now **saves the chosen model's weights** (`keep_epoch`), so a winner can be reused without retraining.
- **Result 1: longer training** (B_long; same recipe as B but a 14-epoch cosine schedule; 50 held-out clips):

  | Epoch | 6 | 8 | 10 | **12** | 14 |
  |---|---|---|---|---|---|
  | WER | 0.102 | 0.100 | 0.094 | **0.090** | 0.091 |
  | CER | 0.040 | 0.038 | 0.035 | **0.034** | 0.035 |
  | Old B (10-epoch schedule) | | | 0.100 | | |

  - **10% fewer errors than B at its best (epoch 12)**, and slightly worse again by 14, so 12 is the stopping point.
  - Epoch 10 here (0.094) also beats the old B epoch 10 (0.100). A longer schedule keeps the learning rate higher for longer.
  - Training took 35.4 min (B_long) and 28.2 min (F_final).
  - **Candidate: F_final epoch 12** (train + all dev, same schedule): `submissions/e14_F14_ep12.csv`.
  - **Not submitted yet: Kaggle's daily limit is 10 submissions, and all 10 were used on 28 Oct.** The limit resets at 00:00 UTC (07:00 Thai time).
- **Result 2: oversampling + beam search** (G_over; train + dev 80% with dev repeated 3× per epoch, 10-epoch schedule, 32 min of training, 50 held-out clips):

  | Variant | WER | CER |
  |---|---|---|
  | B_long epoch 12 (no oversampling) | 0.0903 | 0.0337 |
  | G_over epoch 6 | 0.0823 | 0.0325 |
  | **G_over epoch 8** | **0.0816** | 0.0323 |
  | G_over epoch 10, greedy | 0.0879 | 0.0327 |
  | G_over epoch 10, **beam 4** | 0.0839 | **0.0320** |

  - **Oversampling dev 3×: −10% WER** (0.0903 → 0.0816), best at epoch 8. By epoch 10 it starts to memorize.
  - **Beam search: −4.5% WER** on the same weights (0.0879 → 0.0839), and the lowest CER.
  - `G_over_ep10.nemo` was saved (462 MB, in `results/g_over/`).
- **Final run launched (E15).**

## E15: Final run (29 Oct, running)
- **Notebook:** `dsi443-finetune-fastcon`, version 4. Twins with identical settings: train + dev ×3, full fine-tuning, lr 1e-4, 10-epoch cosine schedule, effective batch 16.
  - **H_val:** dev folds 1–4 ×3; scored on fold 0.
  - **H_final:** all dev ×3; test predictions only. Saves `H_final_ep8.nemo` and the averaged model.
- **Variants:** epoch 6 / 8 / 10 (greedy), epoch 8 with beam 4, average of epochs 6 + 8 + 10 (greedy and beam).
- **Rule:** submit H_final's variant that wins on H_val.
- **Result** (H_val, 50 held-out clips; training took 31.3 min for H_val and 26.0 min for H_final):

  | Variant | WER | CER |
  |---|---|---|
  | Epoch 6 | 0.0855 | 0.0353 |
  | Epoch 8 | 0.0823 | 0.0331 |
  | Epoch 8 + beam 4 | 0.0816 | 0.0330 |
  | Epoch 10 | 0.0831 | 0.0328 |
  | Average of epochs 6 + 8 + 10 | 0.0816 | 0.0327 |
  | **Average + beam 4** | **0.0800** | 0.0331 |

  - **Best: checkpoint averaging + beam search, 0.0800.** That's 24% below zero-shot (0.105) and 20% below B (0.100, Kaggle 0.24).
  - The H_val epoch-8 result (0.0823) reproduces G_over epoch 8 (0.0816), so the oversampling gain is stable across runs.
  - Each step adds a little: averaging −0.0007, beam −0.0007 to −0.0016. Small on 50 clips, but consistently in the right direction.
  - **Saved models:** `results/final_h/H_final_ep8.nemo` and `H_final_avg6-8-10.nemo`.
- **Submission:** `submissions/e15_H_final_avg_beam.csv` (differs from F on 99 of 186 test clips).
  - The overnight scheduled job died (most likely the PC slept), so I submitted manually at 10:27 Thai time on 29 Oct.
- **Kaggle (S07):**

  | Submission | Held-out WER | Kaggle public |
  |---|---|---|
  | **E15 final: train + all dev ×3, average of epochs 6–10, beam 4** | 0.080 | **0.19** ✅ best |
  | E14 F: train + all dev, 14-epoch schedule, epoch 12 | 0.090 | 0.20 |
  | E10 F: train + all dev, 10-epoch schedule (previous best) | — | 0.22 |

  - Dev and Kaggle agree again: each held-out improvement showed up on Kaggle (0.22 → 0.20 → 0.19).
  - **Leaderboard on 29 Oct:** 1st 0.14, 2nd 0.17, 3rd 0.18, **us 4th at 0.19**. 1st place has moved down from 0.17 to 0.14.

## E13: Quick no-GPU ideas: text clean-up rules and spell checker (28 Oct)
- **Formatting comparison** (zero-shot Pathumma vs references, all 240 dev clips):

  | Item | References | Pathumma output |
  |---|---|---|
  | ๆ | 38 (34 with a space before) | 39 (all with a space before) |
  | digits | 69 | 71 |
  | commas | 8 | 8 |
  | periods | 9 | 0 |
  | hyphens | 7 | 0 |
  | ฯ | 5 | 0 |

  **Pathumma already writes almost exactly like the references.** Rule tests:
  - Adding a space before ๆ: 0.1110 → 0.1110 (no change).
  - Removing the space before ๆ: 0.1163 (worse).
  - Dropping punctuation: 0.1120 (worse).
  
  **No clean-up rule gives a gain.**
- **PyThaiNLP spell checker** (`pythainlp.spell.correct` on runs of 2–3 short newmm pieces, accepted only if 1 character away from a dictionary word): **stopped after more than 10 minutes without finishing 240 sentences.** It tries every possible edit against the whole dictionary, which is too slow to be a "quick" option. No result.
- **Conclusion:** there's no remaining quick win in post-processing. Further gains need GPU work (stronger ROVER voters, more test-like training data, the LLM run).

## E12: Typhoon Large LoRA trained longer, as a stronger ROVER voter (28 Oct, running)
- **Why:** Typhoon Large is the voter that makes ROVER work, since it errs differently from Pathumma. It was still improving at epoch 3 (0.135 → 0.130 → 0.124 on the 50 held-out clips).
- **What:** same LoRA settings and data (train + dev folds 1–4), **5 epochs**, scored on fold 0 after epochs 3, 4 and 5. Kaggle notebook `dsi443-lora-typhoon-large-b5`. The longer cosine/linear schedule means epoch 3 here won't exactly equal the earlier epoch 3.
- **Plan:** if it beats 0.124, swap it into ROVER (PB + PC + TL), check on the 50 held-out clips, and submit only if ROVER clearly improves.
- **Result** (50 held-out clips):

  | Epoch (5-epoch schedule) | 3 | 4 | 5 |
  |---|---|---|---|
  | WER | 0.113 | **0.111** | **0.111** |
  | CER | 0.048 | 0.043 | **0.041** |

  - Better than the 3-epoch run (0.124), and it levels off at epochs 4–5. **Still behind Pathumma** (B_long epoch 12: 0.090).
- **ROVER with the new best Pathumma** (BL = B_long epoch 12):

  | Combination | WER | CER |
  |---|---|---|
  | BL alone | 0.0903 | 0.0337 |
  | BL + PC + TL (3-epoch Typhoon) | **0.0863** | 0.0342 |
  | BL + PC + TL5 (5-epoch Typhoon) | 0.0871 | 0.0337 |
  | BL + BL epoch 10 + TL5 | 0.0918 | 0.0343 |

  - **ROVER gains about 4% WER over BL alone, with no gain in CER.** The stronger Typhoon doesn't add anything over the 3-epoch one.
  - **Caution from S06:** on test, F is trained on all of dev but the other voters only on 80%. Last time that made ROVER worse on Kaggle (0.23 vs 0.22). **Not a priority submission.**

## E10: Final model: Pathumma full fine-tuning on train + 100% dev, plus LLM correction (28 Oct, running)
- **What:** Kaggle notebook `dsi443-final-pathumma-llm`, script [final_llm/final_llm.py](final_llm/final_llm.py).
  1. Zero-shot Pathumma transcribes all 240 dev clips. **The LLM's effect is measured on these**, because the final model trains on all of dev and can't be scored on it.
  2. Full fine-tuning on **train + all of dev** (1,876 clips) with recipe B: AdamW, lr 1e-4, cosine with 10% warmup, effective batch 16, **10 epochs** (B's best).
  3. Transcribe test with the fine-tuned model.
  4. LLM correction (`typhoon-ai/typhoon2.1-gemma3-4b`, bf16) in a separate process with a newer transformers. Same prompt and the same "≤ 2 characters, no English or digits" filter as E09.
- **Outputs:** two test submissions, the fine-tuned model alone and fine-tuned + LLM. Submit the LLM version only if it lowers dev WER in step 1.
- **Result (fine-tuning):** worked. Zero-shot dev WER 0.111 (matches E03). Trained on 1,876/1,876 clips in 27.6 min. F's test transcripts differ from B's on 82 of 186 clips.
- **Result (LLM):** crashed. The LLM loaded, but the newer transformers returns a dictionary from `apply_chat_template(..., return_tensors="pt")`, so `ids.shape` raised an error. No LLM numbers yet.
- **Kaggle (S06):**

  | Submission | Kaggle public |
  |---|---|
  | **F: full FT on train + 100% dev, epoch 10** | **0.22** ✅ new best |
  | ROVER F + PC + TL | 0.23 |
  | B: full FT on train + 80% dev (previous best) | 0.24 |

  - **The 50 extra test-like dev clips gave the biggest Kaggle gain so far** (0.24 → 0.22). This fits every earlier finding: test-like data is what helps.
  - ROVER with F is slightly worse than F alone. The other voters (PC, TL) were trained on less dev data, so they now drag F down.
- **Decision:** F is the best submission. Final pair to select on Kaggle: **F (0.22)** and **ROVER F + PC + TL (0.23)**, or B as a safe backup.
- **LLM rerun:**
  - Fixed: `apply_chat_template(..., return_dict=True)`, and `generate(**inputs)`.
  - New `LLM_ONLY` mode: Kaggle notebook `dsi443-final-llm-only` is attached to the earlier run's output (`kernel_sources`) and reuses its zero-shot dev and fine-tuned test transcripts. No retraining.
  - **Result** (Typhoon 2.1 4B, bf16 on a T4; about 6.5 min per 240 sentences):

    | | Before LLM | After LLM |
    |---|---|---|
    | Dev WER (zero-shot Pathumma, 240 clips) | 0.1110 | **0.1106** |
    | Dev CER | 0.0491 | 0.0491 |

    - The LLM proposed **123 fixes in 95 of 240 sentences. Only 12 passed the filter.** Rejected because:
      - edit too big: 47
      - "wrong" word not in the sentence: 29
      - empty or unchanged: 21
      - touches English or numbers: 14
    - **Many proposals were made up.** It repeated the prompt's example "ดาวเคาระห์" in sentences that don't contain it, invented names (กอร์ดอน → "กอร์ดอน รัดกินสัน"), and added words ("คาบสมุทรเกาหลี" → "...ใต้").
    - **The 10 changed sentences: 3 better, 3 worse, 4 the same.**
      - Good: แคโรจี → แคโรทีน (−3), คอร์โมโรส → คอโมโรส.
      - Bad: รัน → เรน (+2), "…ออร์ด" → "…อร์ท" (+2).
    - **Net effect: about 1 word out of ~5,600.** Within noise.
  - **Conclusion:**
    - A 4B LLM doesn't know rare Thai spellings and foreign names well enough. It guesses as often as it fixes.
    - **The strict filter was essential:** without it, the 111 rejected proposals would have damaged many sentences.
    - **F + LLM was not submitted**, following the "submit only clear dev gains" rule (20 of 186 test sentences would change, for an expected effect of about zero).
    - **F (0.22) stays the final model.**

## E08: Final Pathumma model, choosing the best epoch (28 Oct, queued)
- **Rule (from you):** don't blindly train 10 epochs; use the best epoch and stop where WER starts getting worse.
- **Problem:** the final model trains on all of dev, so nothing is left to measure its WER.
- **Solution: two runs with identical settings and schedule** (14 epochs, cosine learning rate, so "epoch N" means the same in both):
  - **B_long:** train + dev folds 1–4, scored on fold 0 after epochs 6, 8, 10, 12 and 14. **This picks the best epoch.**
  - **F_final:** train + **all of dev** (1,876 clips). It writes test predictions at the same epochs, with no scoring. **We submit F at B_long's best epoch.**
  - B was still improving at epoch 10 (0.104 → 0.101 → 0.100), which is why the run goes to 14.
- **Status: cancelled before it ran (your call).** The expected gain over B (Kaggle 0.24) is small, and it would hold a GPU slot for about 2 hours. B epoch 10 stays our best submission. The B_long / F_final setup remains in the script if we want it later.

## E06: LoRA instead of full fine-tuning (28 Oct, running)
- **Why:** full fine-tuning (A) changes all ~115M weights and may overwrite skills the model already had. LoRA freezes them and trains only small add-on matrices.
- **Caveat:** A's main damage was learning train's style (numbers as words, no punctuation). LoRA can still learn that, just more gently. This experiment tests whether LoRA limits it.
- **What:** separate Kaggle notebook `dsi443-finetune-fastcon-lora`, same script ([finetune_fastcon.py](finetune_fastcon/finetune_fastcon.py)).
  - LoRA via PEFT `inject_adapter_in_model` on the encoder attention layers (`linear_q`, `linear_k`, `linear_v`, `linear_out`): r = 16, alpha = 32, dropout 0.05. All other weights frozen.
  - Learning rate 3e-4 (LoRA usually needs a higher rate than full fine-tuning). Effective batch 16.
  - **D_lora_train:** train only, scored on all of dev. Mirrors A (0.132; zero-shot 0.111).
  - **E_lora_train_dev80:** train + dev folds 1–4, scored on dev fold 0. Mirrors B.
- **Version 1 failed:** PEFT refused to load because Kaggle's `torchao` (0.10) is older than PEFT requires (above 0.16).
- **Fix (version 2):** LoRA written by hand in about 20 lines, with no extra packages. `LoRALinear` wraps each target layer: `output = W·x + (alpha/r)·B·(A·x)`, W frozen, B starting at zero.
- **Size:** 1,114,112 trainable weights out of 116,521,473 (**0.96%**). Training took 18.0 min (D) and 16.6 min (E), similar to full fine-tuning, since every layer still runs forward and backward.
- **Result D_lora_train** (train only, all of dev; zero-shot 0.111):

  | Epoch | 2 | 4 | 6 | 8 | 10 |
  |---|---|---|---|---|---|
  | D: LoRA, dev WER | 0.118 | 0.123 | 0.126 | 0.122 | 0.127 |
  | A: full fine-tuning, dev WER | 0.123 | 0.130 | 0.133 | 0.133 | 0.132 |
  | D: dev clips with digits (zero-shot: 35) | 17 | 16 | 11 | 11 | 10 |

  - **LoRA does less damage than full fine-tuning** (about 0.005–0.008 lower WER at every epoch), which supports the "it forgets less" idea.
  - **But it is still worse than no training.** It too drifts toward writing numbers as words. The training data's style is the main problem, not the method.
- **Result E_lora_train_dev80** (train + dev folds 1–4, scored on dev fold 0, 50 clips):

  | Epoch | 0 (zero-shot) | 2 | 4 | 6 | 8 | 10 |
  |---|---|---|---|---|---|---|
  | Fold 0 WER | 0.105 | 0.113 | 0.106 | 0.103 | 0.105 | **0.101** |
  | Fold 0 CER | 0.043 | 0.049 | 0.046 | 0.044 | 0.047 | 0.045 |

  - **First result that beats zero-shot on WER:** 0.101 vs 0.105 at epoch 10, about 4% fewer errors.
  - **But it is weak evidence:** only 50 clips (~1,100 words), so the gap is about 5 words. CER is slightly *worse* (0.045 vs 0.043).
  - Adding test-like dev data clearly helps compared with D (train only), and the trend improves with more epochs.
- **Decision:** LoRA + dev data is the most promising direction so far, but needs confirmation on more than 50 clips. Next: train on train + all of dev, scored by rotating the held-out fold (cross-validation), or check on Kaggle.

## E07: LoRA on the Whisper models (28 Oct, queued)
- **Why:** test LoRA on other strong models, keeping the data unchanged.
  - Typhoon Whisper Large V3 is the zero-shot runner-up (0.151, beam 5).
  - Typhoon Whisper Turbo is the same family and about 3× faster.
  - Qwen3-ASR skipped for now (chat-style training is more work).
- **What:** [finetune_whisper/finetune_whisper.py](finetune_whisper/finetune_whisper.py), one Kaggle notebook per model (`dsi443-lora-typhoon-large`, `dsi443-lora-typhoon-turbo`), pushed with [push.sh](finetune_whisper/push.sh).
  - Hand-written LoRA on all attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`) in encoder and decoder: r = 32, alpha = 64. Everything else frozen.
  - AdamW, learning rate 5e-4, 50 warmup steps then linear decay. Effective batch 16 (4 × accumulation 4). fp16 mixed precision. Gradient checkpointing (non-reentrant).
  - SpecAugment: time masking, probability 0.05.
  - Training clips over 30 s are dropped (Whisper's window).
  - Evaluation uses greedy decoding in 30 s chunks, so epoch 0 differs slightly from the beam-5 zero-shot scores in E03.
  - **A_train:** train only, scored on all of dev after epochs 0, 1, 2 and 3.
  - **B_train_dev80:** train + dev folds 1–4, scored on dev fold 0 after epochs 1–3.
- **Constraint found:** Kaggle allows at most **2 GPU runs at once** per account. Both notebooks wait in a queue that resends them every 2 minutes until a slot frees up.
- **Result: Typhoon Whisper Turbo** (LoRA trains 13.1M of 822M weights = 1.59%; about 21 min per epoch on a T4):

  | | Epoch 0 (zero-shot, greedy) | Epoch 1 | Epoch 2 | Epoch 3 |
  |---|---|---|---|---|
  | **A:** train only, all of dev, WER | 0.202 | 0.207 | 0.205 | 0.211 |
  | A: CER | 0.086 | 0.084 | — | 0.084 |
  | **B:** train + dev 80%, fold 0, WER | 0.211 | 0.178 | **0.162** | 0.163 |
  | B: CER | 0.075 | 0.053 | 0.051 | **0.049** |

  - **A (train only):** WER slightly worse, the same pattern as Pathumma. It still writes no digits (0 clips).
  - **B (with dev data):** a big relative gain, **0.211 → 0.162 (−23%)**. That's far more than Pathumma's −5%, because Turbo had more room to improve. Best around epoch 2–3.
  - **Still well behind Pathumma B** (0.100 on the same clips). Pathumma stays the best model.
  - Turbo B writes digits in only 2 of 50 clips, so it's still mostly spelling numbers out.
  - **Kaggle (S03): Turbo LoRA B epoch 2 = 0.29 public**, compared with the host's zero-shot Turbo entry at 0.31. A small confirmed gain, but still well behind Pathumma B (0.24).
- **Result: Typhoon Whisper Large V3** (LoRA trains 31.5M of 1,575M weights = 2.00%; about 35–40 min per epoch on a T4, about 3.5 h for the whole notebook):

  | | Epoch 0 (zero-shot, greedy) | Epoch 1 | Epoch 2 | Epoch 3 |
  |---|---|---|---|---|
  | **A:** train only, all of dev, WER | 0.152 | 0.155 | 0.151 | 0.153 |
  | A: CER | 0.073 | 0.072 | 0.071 | 0.073 |
  | **B:** train + dev 80%, fold 0, WER | 0.135 | 0.134 | 0.130 | **0.124** |
  | B: CER | 0.057 | 0.052 | 0.047 | **0.045** |

  - **A (train only):** flat, about ±0.003. LoRA on the big model barely moves on mismatched data; it neither helps nor hurts much.
  - **B (with dev data):** steady gain, **0.135 → 0.124 (−9%)**, still improving at epoch 3. More epochs might help, but each costs about 40 min.
  - **Still behind Pathumma B** (0.100 on the same clips). Writes digits in only 3 of 50 clips.
  - **Kaggle (S04): Typhoon Large LoRA B epoch 3 = 0.26 public**, compared with the host's zero-shot Typhoon Large entry at 0.26. No visible gain at 2 decimal places, and behind Pathumma B (0.24).
- **Summary across model families (setup B, dev fold 0, 50 clips):**

  | Model | Parameters | Zero-shot | Fine-tuned | Change |
  |---|---|---|---|---|
  | **Pathumma Fastcon** (full fine-tuning) | 116M | 0.105 | **0.100** | −5% |
  | Typhoon Whisper Large V3 (LoRA) | 1,575M | 0.135 | 0.124 | −9% |
  | Typhoon Whisper Turbo (LoRA) | 822M | 0.211 | 0.162 | −23% |

  **The order doesn't change after fine-tuning:** weaker models gain more, but none catches Pathumma.

## S01: First Kaggle submissions: is our dev score a good guide? (28 Oct)
- **What:** submitted two test prediction files, each split into words with default newmm by [make_submission.py](make_submission.py).

  | Submission | Our dev WER | Kaggle public WER |
  |---|---|---|
  | E03 Pathumma Fastcon zero-shot | 0.111 | **0.26** |
  | E04 fine-tuned on train, epoch 10 | 0.132 | **0.29** |

- **Result:**
  - **The ranking agrees.** Kaggle also says fine-tuning on train hurt (+0.03 on Kaggle, +0.021 on dev). So dev is a valid guide for **choosing between models**.
  - **The absolute numbers don't match.** Kaggle's are about 2.3× ours. Our zero-shot 0.26 is close to the host's own Pathumma Fastcon entry (0.25).
- **Tried on dev only** (no test data) to find what reproduces Kaggle's numbers:

  | Scoring method | Zero-shot | A ep10 | Matches Kaggle? |
  |---|---|---|---|
  | Reference split with `longest` | 0.150 | 0.171 | no |
  | Reference split with `mm` | 0.324 | 0.345 | no |
  | Same tokenizer on both sides | 0.117–0.129 | 0.142–0.154 | no |
  | Per-sentence average instead of pooled | 0.109 | 0.127 | no |
  | No word splitting at all | 0.649 | 0.864 | no |

  None reproduces 0.26 / 0.29.
- **Possible causes:**
  - Kaggle's references are split with a different tokenizer or version. Kaggle's sample file splits more finely, e.g. "เจ้า หน้า ที่".
  - Different text normalization.
  - Test is harder than dev.
  - Noise from only ~43 public clips.
- **Caution:** the 4 example rows in `sample_submission.csv` look like real test transcripts (they match our model's test output closely). We have not used them for anything and should not tune toward them; that would be test-set leakage.
- **Decision:**
  - Keep choosing models by dev WER, since the ranking agrees.
  - Compare against the team baseline **using Kaggle scores only**.
  - Ask the instructor exactly how the scorer splits words. If submissions are scored by splitting on spaces against pre-split references, matching that splitting could lower everyone's WER a lot.
- **Final call (28 Oct):** we follow Kaggle's stated method: PyThaiNLP `newmm`, default dictionary, on submissions and for local scoring. No guessing or tuning toward the scorer.
