"""Final model: Pathumma Fastcon fully fine-tuned on train + 100% of dev, then LLM spelling correction.

Runs as a Kaggle GPU script (competition data attached, internet on), or with --data-dir on Colab.

Steps
  1. Zero-shot Pathumma transcribes all of dev. These transcripts are what the LLM step
     is measured on, since the final model trains on all of dev and can't be scored on it.
  2. Fine-tune on train + all of dev with recipe B (best on Kaggle, 0.24): full fine-tuning,
     AdamW lr 1e-4, cosine with 10% warmup, effective batch 16, 10 epochs.
  3. Transcribe test with the fine-tuned model.
  4. In a separate Python process (Gemma 3 needs a newer transformers than NeMo pins), an LLM
     lists misspelled words; only small edits (<= 2 characters, no English or digits) are applied.

Outputs in --out-dir:
  llm_dev.csv            dev: audio, ref, hyp (zero-shot), fixed (after LLM)
  test_final.csv         test: audio, hyp (fine-tuned), fixed (after LLM)
  llm_log_dev.json, llm_log_test.json   every fix the LLM proposed and why it was accepted or rejected
  summary.json           dev WER/CER before and after the LLM, settings
"""
import argparse
import glob
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import time

ASR_ID = "nectec/Pathumma-stt-th-fastconformer-rnnt-large"
LLM_ID = "typhoon-ai/typhoon2.1-gemma3-4b"  # alternative: "Qwen/Qwen3-4B-Instruct-2507"
SR = 16_000
EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 4
ACCUMULATE = 4  # effective batch 16
MAX_SECONDS = 40.0
LLM_ONLY = False  # True: skip ASR, reuse llm_dev.csv / test_final.csv from an earlier run's output
MAX_EDIT = 2            # an LLM fix may change at most this many characters
MAX_EDIT_RATIO = 0.34   # ...and at most this share of the word


def pip(*packages):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *packages], check=False)


def find_data_dir(given):
    if given:
        return given
    hits = glob.glob("/kaggle/input/**/dev.csv", recursive=True)
    if not hits:
        sys.exit("dev.csv not found; pass --data-dir")
    return os.path.dirname(hits[0])


# ---------------------------------------------------------------- text and metric

def words(text):
    from pythainlp.tokenize import word_tokenize
    return [w for w in word_tokenize(str(text), engine="newmm", keep_whitespace=False) if w.strip()]


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer(refs, hyps):
    errors = total = 0
    for r, h in zip(refs, hyps):
        rw = words(r)
        errors += edit_distance(rw, words(h))
        total += len(rw)
    return errors / max(total, 1)


def cer(refs, hyps):
    strip = lambda s: re.sub(r"\s+", "", str(s))
    errors = total = 0
    for r, h in zip(refs, hyps):
        errors += edit_distance(list(strip(r)), list(strip(h)))
        total += len(strip(r))
    return errors / max(total, 1)


# ---------------------------------------------------------------- ASR (main process)

def load_split(data_dir, split, wav_dir, limit):
    import librosa
    import pandas as pd
    import soundfile as sf

    index = {os.path.basename(p): p for p in glob.glob(os.path.join(data_dir, "**", "*.mp3"), recursive=True)}
    df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
    if limit:
        df = df.head(limit)
    rows = []
    for audio, ref in zip(df["audio"], df["sentence"] if "sentence" in df else [""] * len(df)):
        src = os.path.join(data_dir, audio)
        src = src if os.path.exists(src) else index[os.path.basename(audio)]
        wav = os.path.join(wav_dir, os.path.splitext(os.path.basename(audio))[0] + ".wav")
        y, _ = librosa.load(src, sr=SR, mono=True)
        sf.write(wav, y, SR)
        rows.append({"audio": audio, "ref": str(ref), "wav": wav, "seconds": len(y) / SR})
    return rows


def transcribe(model, rows):
    model.eval()
    out = model.transcribe([r["wav"] for r in rows], batch_size=16, verbose=False)
    if isinstance(out, tuple):  # older NeMo RNNT returns (best, all_hypotheses)
        out = out[0]
    return [(o if isinstance(o, str) else o.text).strip() for o in out]


def fine_tune(model, rows, work_dir, epochs):
    from omegaconf import OmegaConf, open_dict
    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl

    manifest = os.path.join(work_dir, "train_final.json")
    kept = 0
    with open(manifest, "w", encoding="utf-8") as f:
        for r in rows:
            text = re.sub(r"\s+", " ", r["ref"].replace("ń", "")).strip()  # ń is the only character the tokenizer lacks
            if text and r["seconds"] <= MAX_SECONDS:
                f.write(json.dumps({"audio_filepath": r["wav"], "duration": r["seconds"], "text": text}, ensure_ascii=False) + "\n")
                kept += 1
    print(f"fine-tuning on {kept}/{len(rows)} clips for {epochs} epochs", flush=True)

    trainer = pl.Trainer(
        devices=1, accelerator="gpu", max_epochs=epochs, precision="16-mixed",
        accumulate_grad_batches=ACCUMULATE, gradient_clip_val=1.0,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        limit_val_batches=0, num_sanity_val_steps=0, log_every_n_steps=25,
    )
    model.set_trainer(trainer)
    cfg = model.cfg.train_ds
    with open_dict(cfg):
        cfg.manifest_filepath = manifest
        cfg.batch_size = BATCH_SIZE
        cfg.shuffle = True
        cfg.num_workers = 4
        cfg.max_duration = MAX_SECONDS
        cfg.is_tarred = False
        cfg.tarred_audio_filepaths = None
    model.setup_training_data(cfg)
    model.setup_optimization(optim_config=OmegaConf.create({
        "name": "adamw", "lr": LR, "betas": [0.9, 0.98], "weight_decay": 1e-3,
        "sched": {"name": "CosineAnnealing", "warmup_ratio": 0.1, "min_lr": 1e-6},
    }))
    t0 = time.time()
    model.train()
    trainer.fit(model)
    return round((time.time() - t0) / 60, 1)


# ---------------------------------------------------------------- LLM (worker process)

INSTRUCTIONS = (
    "You check Thai speech-recognition output for spelling mistakes.\n"
    "Find words that are misspelled or are not real Thai words, and give the correct spelling.\n"
    "Rules:\n"
    "- Only fix misspelled or non-existent words. Do not rephrase, add or remove words.\n"
    "- Do not change English words, names written in English, numbers or punctuation.\n"
    "- If nothing is wrong, answer [].\n"
    'Answer with JSON only: [{"wrong": "...", "correct": "..."}]'
)
# Made up for the prompt; not taken from dev or test.
EXAMPLES = [
    ("เมื่อวานฉันไปซื้อผลไม้ที่ตลาดและได้ทุเรีนมาสองลูก", [{"wrong": "ทุเรีน", "correct": "ทุเรียน"}]),
    ("การประชุมครั้งนี้มีผู้เข้าร่วมจากหลายประเทศในเอเชีย", []),
    ("นักดาราศาสตร์ศึกษาการเคลื่อนที่ของดาวเคาระห์รอบดวงอาทิตย์", [{"wrong": "ดาวเคาระห์", "correct": "ดาวเคราะห์"}]),
]
LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9]")


def accept(fix, sentence):
    """Keep only small, safe edits. Returns (ok, reason)."""
    wrong, correct = str(fix.get("wrong", "")).strip(), str(fix.get("correct", "")).strip()
    if not wrong or not correct or wrong == correct:
        return False, "empty or unchanged"
    if wrong not in sentence:
        return False, "wrong word not in sentence"
    if LATIN_OR_DIGIT.search(wrong + correct):
        return False, "touches English/number"
    d = edit_distance(list(wrong), list(correct))
    if d > MAX_EDIT or d > math.ceil(MAX_EDIT_RATIO * len(wrong)):
        return False, f"edit too big ({d} chars)"
    return True, f"{d} chars"


def llm_worker(out_dir):
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if "gemma" in LLM_ID else torch.float16  # Gemma 3 overflows in float16
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    llm = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=dtype, device_map="cuda").eval()
    print("LLM loaded:", LLM_ID, flush=True)

    @torch.no_grad()
    def ask(sentence):
        messages = [{"role": "system", "content": INSTRUCTIONS}]
        for text, fixes in EXAMPLES:
            messages += [{"role": "user", "content": text},
                         {"role": "assistant", "content": json.dumps(fixes, ensure_ascii=False)}]
        messages.append({"role": "user", "content": sentence})
        # return_dict: newer transformers return a dict here rather than a tensor
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                               return_dict=True).to(llm.device)
        out = llm.generate(**inputs, max_new_tokens=200, do_sample=False)
        reply = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        match = re.search(r"\[.*\]", reply, re.S)
        try:
            fixes = json.loads(match.group(0)) if match else []
        except json.JSONDecodeError:
            fixes = []
        return reply, [f for f in fixes if isinstance(f, dict)]

    def correct_all(df, name):
        t0, fixed_all, logs = time.time(), [], []
        for i, (audio, sentence) in enumerate(zip(df["audio"], df["hyp"].fillna(""))):
            reply, fixes = ask(sentence)
            fixed, log = sentence, []
            for f in fixes:
                ok, why = accept(f, fixed)
                log.append({"wrong": f.get("wrong"), "correct": f.get("correct"), "accepted": ok, "why": why})
                if ok:
                    fixed = fixed.replace(str(f["wrong"]).strip(), str(f["correct"]).strip(), 1)
            fixed_all.append(fixed)
            logs.append({"audio": audio, "reply": reply, "fixes": log})
            if (i + 1) % 25 == 0:
                print(f"LLM {name}: {i + 1}/{len(df)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        json.dump(logs, open(os.path.join(out_dir, f"llm_log_{name}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return fixed_all

    dev = pd.read_csv(os.path.join(out_dir, "llm_dev.csv")).fillna("")
    dev["fixed"] = correct_all(dev, "dev")
    dev.to_csv(os.path.join(out_dir, "llm_dev.csv"), index=False, encoding="utf-8")
    test = pd.read_csv(os.path.join(out_dir, "test_final.csv")).fillna("")
    test["fixed"] = correct_all(test, "test")
    test.to_csv(os.path.join(out_dir, "test_final.csv"), index=False, encoding="utf-8")

    summary_path = os.path.join(out_dir, "summary.json")
    summary = json.load(open(summary_path, encoding="utf-8"))
    summary["llm"] = {
        "model": LLM_ID, "max_edit": MAX_EDIT, "max_edit_ratio": MAX_EDIT_RATIO,
        "dev_wer_before": round(wer(dev["ref"], dev["hyp"]), 4), "dev_wer_after": round(wer(dev["ref"], dev["fixed"]), 4),
        "dev_cer_before": round(cer(dev["ref"], dev["hyp"]), 4), "dev_cer_after": round(cer(dev["ref"], dev["fixed"]), 4),
        "dev_sentences_changed": int((dev["hyp"] != dev["fixed"]).sum()),
        "test_sentences_changed": int((test["hyp"] != test["fixed"]).sum()),
    }
    json.dump(summary, open(summary_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary["llm"], indent=2), flush=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--out-dir", default="/kaggle/working" if os.path.isdir("/kaggle") else "results/final_llm")
    ap.add_argument("--limit", type=int, help="only the first N clips per split (smoke test)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--llm-worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.llm_worker:
        llm_worker(args.out_dir)
        return

    if LLM_ONLY:
        import shutil
        os.makedirs(args.out_dir, exist_ok=True)
        for name in ("llm_dev.csv", "test_final.csv", "summary.json"):
            src = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
            if not src:
                sys.exit(f"{name} not found in /kaggle/input (attach the earlier run's output)")
            shutil.copy(src[0], os.path.join(args.out_dir, name))
            print("reusing", src[0], flush=True)
        pip("transformers", "accelerate", "pythainlp")
        done = subprocess.run([sys.executable, __file__, "--llm-worker", "--out-dir", args.out_dir])
        print("LLM step", "finished" if done.returncode == 0 else f"FAILED (exit {done.returncode})", flush=True)
        return

    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError:
        sys.exit("No internet: enable it in the notebook settings (needs a phone-verified Kaggle account).")
    pip("nemo_toolkit[asr]", "pythainlp")
    import pandas as pd
    import torch
    if not torch.cuda.is_available():
        sys.exit("No GPU: switch on the accelerator.")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import HfApi, hf_hub_download

    os.makedirs(args.out_dir, exist_ok=True)
    work_dir = tempfile.mkdtemp()
    data_dir = find_data_dir(args.data_dir)
    os.makedirs(os.path.join(work_dir, "wav"), exist_ok=True)
    parts = {s: load_split(data_dir, s, os.path.join(work_dir, "wav"), args.limit) for s in ("train", "dev", "test")}
    print({s: len(r) for s, r in parts.items()}, flush=True)

    nemo_file = next(f for f in HfApi().list_repo_files(ASR_ID) if f.endswith(".nemo"))
    model = nemo_asr.models.ASRModel.restore_from(hf_hub_download(ASR_ID, nemo_file), map_location="cuda")

    # 1. zero-shot dev transcripts: the LLM step is measured on these
    dev = parts["dev"]
    dev_hyps = transcribe(model, dev)
    pd.DataFrame({"audio": [r["audio"] for r in dev], "ref": [r["ref"] for r in dev], "hyp": dev_hyps}).to_csv(
        os.path.join(args.out_dir, "llm_dev.csv"), index=False, encoding="utf-8")
    summary = {"asr_model": ASR_ID, "recipe": {"epochs": args.epochs, "lr": LR, "batch_size": BATCH_SIZE, "accumulate": ACCUMULATE},
               "zero_shot_dev_wer": round(wer([r["ref"] for r in dev], dev_hyps), 4)}
    print("zero-shot dev WER", summary["zero_shot_dev_wer"], flush=True)

    # 2-3. fine-tune on train + all of dev, transcribe test
    summary["train_minutes"] = fine_tune(model, parts["train"] + parts["dev"], work_dir, args.epochs)
    test = parts["test"]
    pd.DataFrame({"audio": [r["audio"] for r in test], "hyp": transcribe(model, test)}).to_csv(
        os.path.join(args.out_dir, "test_final.csv"), index=False, encoding="utf-8")
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("fine-tuned test transcripts written", flush=True)
    del model
    torch.cuda.empty_cache()

    # 4. LLM correction in a fresh process with a newer transformers
    pip("transformers", "accelerate")
    done = subprocess.run([sys.executable, __file__, "--llm-worker", "--out-dir", args.out_dir])
    print("LLM step", "finished" if done.returncode == 0 else f"FAILED (exit {done.returncode})", flush=True)


if __name__ == "__main__":
    main()
