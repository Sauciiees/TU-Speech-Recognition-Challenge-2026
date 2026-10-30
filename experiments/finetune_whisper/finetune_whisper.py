"""LoRA fine-tuning of a Whisper model (Typhoon Whisper Large V3 / Turbo) on the competition data.

Runs as a Kaggle GPU script (competition data attached, internet on), or with
--data-dir on Colab. One notebook per model: MODEL_ID below is swapped when the
notebook is pushed. Each experiment starts again from the pretrained model; after
each listed epoch it transcribes its eval clips and the test set (epoch 0 = the
untouched model, greedy decoding).

Outputs in --out-dir:
  pred_<exp>_ep<N>.csv   eval clips: audio, ref, hyp   (score with ../score_dev.py)
  test_<exp>_ep<N>.csv   test clips: audio, hyp        (submit with ../make_submission.py)
  summary.json           WER per experiment and epoch, settings
"""
import argparse
import glob
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import time

MODEL_ID = "typhoon-ai/typhoon-whisper-large-v3"
SR = 16_000
FOLDS = 5  # dev fold = clip number % 5; fold 0 is held out when dev is used for training

EXPERIMENTS = [
    # A: train only, judged on all of dev (mirrors Pathumma A / D)
    {"name": "A_train", "train": ["train"], "eval": "dev", "epochs": 3, "eval_epochs": [0, 1, 2, 3]},
    # B: train + 80% of dev, judged on the other 20% (mirrors Pathumma B / E)
    {"name": "B_train_dev80", "train": ["train", "dev_rest"], "eval": "dev_fold0", "epochs": 3, "eval_epochs": [1, 2, 3]},
    # B5: Typhoon Large was still improving at epoch 3 (0.135 -> 0.124); train longer as a ROVER voter
    {"name": "B5_train_dev80", "train": ["train", "dev_rest"], "eval": "dev_fold0", "epochs": 5, "eval_epochs": [3, 4, 5]},
]
# E22: B5 epoch 5 scored 0.20 on Kaggle (Pathumma at the same stage, B: 0.24), and it was still improving.
# Give Typhoon the recipe that took Pathumma from 0.24 to 0.19: all of dev, dev x3, 6 epochs, average of the
# last epochs' LoRA weights. Twins as before: TW_val scored on fold 0, TW_final writes test predictions.
EXPERIMENTS += [
    {"name": "TW_val", "train": ["train", "dev_rest*3"], "eval": "dev_fold0", "epochs": 6,
     "eval_epochs": [3, 4, 5, 6], "avg_epochs": [4, 5, 6]},
    {"name": "TW_final", "train": ["train", "dev*3"], "eval": None, "epochs": 6,
     "eval_epochs": [4, 5, 6], "avg_epochs": [4, 5, 6], "keep_lora": True},
]
DEFAULT_EXPERIMENTS = ["A_train", "B_train_dev80"]
LORA = {"r": 32, "alpha": 64, "dropout": 0.05, "targets": ("q_proj", "k_proj", "v_proj", "out_proj")}
LR = 5e-4
WARMUP_STEPS = 50
BATCH_SIZE = 4
ACCUMULATE = 4      # effective batch 16
MAX_TRAIN_SECONDS = 30.0  # Whisper sees 30 s windows; longer clips would lose audio but keep their full text
EVAL_BATCH = 8


def pip(*packages):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *packages], check=False)


def find_data_dir(given):
    # given (e.g. Colab's unzip folder) may hold the CSVs in a subfolder
    hits = glob.glob(os.path.join(given or "/kaggle/input", "**", "dev.csv"), recursive=True)
    if not hits:
        sys.exit("dev.csv not found; pass --data-dir")
    return os.path.dirname(hits[0])


def dev_fold(audio):
    stem = os.path.splitext(os.path.basename(audio))[0]
    return int(re.search(r"(\d+)$", stem).group(1)) % FOLDS


def load_splits(data_dir, limit=None, audio_root=None):
    """All clips decoded to 16 kHz mono float arrays. Parts: train, dev, dev_fold0, dev_rest, test.
    Audio files are looked up anywhere under audio_root (default: the CSV folder)."""
    import librosa
    import pandas as pd

    index = {os.path.basename(p): p for p in glob.glob(os.path.join(audio_root or data_dir, "**", "*.mp3"), recursive=True)}
    parts = {}
    for split in ("train", "dev", "test"):
        df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
        if limit:
            df = df.head(limit)
        refs = df["sentence"].tolist() if "sentence" in df else [""] * len(df)
        rows = []
        for audio, ref in zip(df["audio"], refs):
            src = os.path.join(data_dir, audio)
            src = src if os.path.exists(src) else index[os.path.basename(audio)]
            y, _ = librosa.load(src, sr=SR, mono=True)
            rows.append({"audio": audio, "ref": str(ref), "array": y, "seconds": len(y) / SR})
        parts[split] = rows
    parts["dev_fold0"] = [r for r in parts["dev"] if dev_fold(r["audio"]) == 0]
    parts["dev_rest"] = [r for r in parts["dev"] if dev_fold(r["audio"]) != 0]
    return parts


def add_lora(model):
    """Wrap the attention projections (encoder and decoder) with LoRA; freeze everything else."""
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        """output = W x + (alpha / r) * B (A x), with W frozen and B starting at zero."""

        def __init__(self, base, r, alpha, dropout):
            super().__init__()
            self.base = base
            self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            self.scale = alpha / r
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale

    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.split(".")[-1] in LORA["targets"]]
    for name, module in targets:
        parent_name, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), attr, LoRALinear(module, LORA["r"], LORA["alpha"], LORA["dropout"]))
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA on {len(targets)} layers: training {trainable:,} of {total:,} weights ({100 * trainable / total:.2f}%)", flush=True)
    return {"layers": len(targets), "trainable": trainable, "total": total}


def make_batches(rows, processor, shuffle):
    """Yield (input_features, labels) batches. Labels are the tokenized transcript with
    Whisper's prefix tokens; padding is -100 so the loss ignores it."""
    import torch

    order = list(range(len(rows)))
    if shuffle:
        random.shuffle(order)
    start_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    for i in range(0, len(order), BATCH_SIZE):
        batch = [rows[j] for j in order[i:i + BATCH_SIZE]]
        feats = processor.feature_extractor([r["array"] for r in batch], sampling_rate=SR, return_tensors="pt").input_features
        ids = [processor.tokenizer(r["ref"]).input_ids for r in batch]
        ids = [x[1:] if x and x[0] == start_id else x for x in ids]  # the model adds the start token itself
        width = max(len(x) for x in ids)
        labels = torch.full((len(ids), width), -100, dtype=torch.long)
        for k, x in enumerate(ids):
            labels[k, :len(x)] = torch.tensor(x)
        yield feats, labels


def transcribe(model, processor, rows):
    import torch
    from transformers import pipeline

    model.eval()
    asr = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                   feature_extractor=processor.feature_extractor, device=0, chunk_length_s=30)
    with torch.autocast("cuda", dtype=torch.float16):
        out = asr([{"raw": r["array"], "sampling_rate": SR} for r in rows], batch_size=EVAL_BATCH,
                  generate_kwargs={"language": "th", "task": "transcribe", "num_beams": 1})
    return [o["text"].strip() for o in out]


def wer(refs, hyps):
    from pythainlp.tokenize import word_tokenize

    def words(t):
        return [w for w in word_tokenize(str(t), engine="newmm", keep_whitespace=False) if w.strip()]

    def dist(a, b):
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i] + [0] * len(b)
            for j, y in enumerate(b, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
            prev = cur
        return prev[-1]

    errors = total = 0
    for r, h in zip(refs, hyps):
        rw = words(r)
        errors += dist(rw, words(h))
        total += len(rw)
    return errors / max(total, 1)


def evaluate(model, processor, exp, epoch, parts, out_dir, summary):
    """epoch is an epoch number, or a label such as "avg4-5-6" for averaged LoRA weights. Experiments with
    eval None (trained on all of dev) write test predictions only; scored runs whose name ends in _val
    skip test, since only their twin is submitted."""
    import pandas as pd

    tag = f"ep{epoch}" if isinstance(epoch, int) else epoch
    if exp["eval"] is None or not exp["name"].endswith("_val"):
        test = parts["test"]
        test_hyps = transcribe(model, processor, test)
        pd.DataFrame({"audio": [r["audio"] for r in test], "hyp": test_hyps}).to_csv(
            os.path.join(out_dir, f"test_{exp['name']}_{tag}.csv"), index=False, encoding="utf-8")
    if exp["eval"] is None:
        print(f"[{exp['name']}] {tag}: test predictions written", flush=True)
    else:
        rows = parts[exp["eval"]]
        hyps = transcribe(model, processor, rows)
        pd.DataFrame({"audio": [r["audio"] for r in rows], "ref": [r["ref"] for r in rows], "hyp": hyps}).to_csv(
            os.path.join(out_dir, f"pred_{exp['name']}_{tag}.csv"), index=False, encoding="utf-8")
        score = round(wer([r["ref"] for r in rows], hyps), 4)
        summary["results"].setdefault(exp["name"], {})[tag] = score
        print(f"[{exp['name']}] {tag}: {exp['eval']} WER {score}", flush=True)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def run_experiment(exp, parts, out_dir, summary):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="thai", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to("cuda")
    model.config.apply_spec_augment = True  # time/frequency masking while training
    model.config.mask_time_prob = 0.05
    model.config.use_cache = False
    model.generation_config.forced_decoder_ids = None
    summary.setdefault("lora_params", {})[exp["name"]] = add_lora(model)
    # Non-reentrant checkpointing: with the reentrant kind, LoRA inside frozen encoder
    # layers gets no gradient because none of those layers' inputs require one.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # "dev*3" = that part repeated 3 times per epoch (oversampling)
    train_rows = [r for p in exp["train"] for r in parts[p.partition("*")[0]] * int(p.partition("*")[2] or 1)
                  if r["seconds"] <= MAX_TRAIN_SECONDS and r["ref"].strip()]
    print(f"\n##### {exp['name']}: training on {len(train_rows)} clips, eval on {exp['eval']}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-2)
    steps_per_epoch = math.ceil(len(train_rows) / BATCH_SIZE / ACCUMULATE)
    total_steps = steps_per_epoch * exp["epochs"]
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / WARMUP_STEPS) * max(0.0, 1 - s / total_steps))
    scaler = torch.amp.GradScaler("cuda")

    if 0 in exp["eval_epochs"]:
        evaluate(model, processor, exp, 0, parts, out_dir, summary)
    lora = {n: p for n, p in model.named_parameters() if "lora_" in n}
    saved = {}  # epoch -> copy of the LoRA weights, for checkpoint averaging
    t0 = time.time()
    for epoch in range(1, exp["epochs"] + 1):
        model.train()
        losses = []
        for k, (feats, labels) in enumerate(make_batches(train_rows, processor, shuffle=True)):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_features=feats.to("cuda"), labels=labels.to("cuda")).loss / ACCUMULATE
            scaler.scale(loss).backward()
            losses.append(loss.item() * ACCUMULATE)
            if (k + 1) % ACCUMULATE == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                schedule.step()
        print(f"[{exp['name']}] epoch {epoch}: train loss {sum(losses) / len(losses):.3f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if epoch in exp.get("avg_epochs", []):
            saved[epoch] = {n: p.detach().cpu().clone() for n, p in lora.items()}
        if epoch in exp["eval_epochs"]:
            evaluate(model, processor, exp, epoch, parts, out_dir, summary)
    summary["train_minutes"][exp["name"]] = round((time.time() - t0) / 60, 1)

    if exp.get("avg_epochs"):  # checkpoint averaging of the LoRA weights (the frozen base is the same for all)
        label = "avg" + "-".join(str(e) for e in exp["avg_epochs"])
        with torch.no_grad():
            for n, p in lora.items():
                p.copy_(torch.stack([saved[e][n] for e in exp["avg_epochs"]]).mean(0).to(p.device))
        evaluate(model, processor, exp, label, parts, out_dir, summary)
        if exp.get("keep_lora"):  # small file (LoRA weights only) for the final Colab notebook
            torch.save({n: p.detach().cpu() for n, p in lora.items()}, os.path.join(out_dir, f"{exp['name']}_{label}_lora.pt"))

    del model, optimizer
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--out-dir", default="/kaggle/working" if os.path.isdir("/kaggle") else "results/finetune_whisper")
    ap.add_argument("--experiments", help="comma-separated experiment names (default: all)")
    ap.add_argument("--limit", type=int, help="only the first N clips per split (smoke test)")
    ap.add_argument("--epochs", type=int, help="override epochs for every experiment (smoke test)")
    args = ap.parse_args()

    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError:
        sys.exit("No internet: enable it in the notebook settings (needs a phone-verified Kaggle account).")
    pip("pythainlp")
    import torch
    if not torch.cuda.is_available():
        sys.exit("No GPU: switch on the accelerator.")
    print("GPU:", torch.cuda.get_device_name(0), "| model:", MODEL_ID, flush=True)
    random.seed(0)
    torch.manual_seed(0)

    os.makedirs(args.out_dir, exist_ok=True)
    parts = load_splits(find_data_dir(args.data_dir), args.limit, args.data_dir)
    wanted = args.experiments.split(",") if args.experiments else DEFAULT_EXPERIMENTS
    experiments = [e for e in EXPERIMENTS if e["name"] in wanted]
    if args.epochs:
        experiments = [{**e, "epochs": args.epochs, "eval_epochs": [x for x in e["eval_epochs"] if x <= args.epochs]} for e in experiments]

    summary = {"model": MODEL_ID, "lora": dict(LORA, targets=list(LORA["targets"])), "lr": LR, "batch_size": BATCH_SIZE,
               "accumulate": ACCUMULATE, "decoding": "greedy, language=th", "experiments": experiments,
               "results": {}, "train_minutes": {}}
    for exp in experiments:
        run_experiment(exp, parts, args.out_dir, summary)
    print(json.dumps(summary["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
