"""Fine-tune Pathumma FastConformer-RNNT (nectec/Pathumma-stt-th-fastconformer-rnnt-large).

Runs as a Kaggle GPU script (competition data attached, internet on), or with
--data-dir on Colab. Each experiment starts again from the pretrained model, saves
weights at the listed epochs, then transcribes its eval clips and the test set
with each saved epoch (epoch 0 = the untouched pretrained model).

Outputs in --out-dir:
  pred_<exp>_ep<N>.csv   eval clips: audio, ref, hyp   (score with ../score_dev.py)
  test_<exp>_ep<N>.csv   test clips: audio, hyp        (submit with ../make_submission.py)
  summary.json           WER per experiment and epoch, data stats, settings
"""
import argparse
import gc
import glob
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time

# less memory fragmentation; must be set before torch starts using CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HF_ID = "nectec/Pathumma-stt-th-fastconformer-rnnt-large"
SR = 16_000
FOLDS = 5  # dev is split into 5 folds by sorted file name; fold 0 is held out in B

EXPERIMENTS = [
    # E04 A: does the (mismatched) training set help on dev at all?  -> no, dev WER rose 0.111 -> 0.132
    {"name": "A_train", "train": ["train"], "eval": "dev", "epochs": 10, "lr": 1e-4, "eval_epochs": [0, 2, 4, 6, 8, 10]},
    # B: add 80% of dev (same source as test), judge on the other 20%
    {"name": "B_train_dev80", "train": ["train", "dev_rest"], "eval": "dev_fold0", "epochs": 10, "lr": 1e-4, "eval_epochs": [2, 4, 6, 8, 10]},
    # C: in-domain data only, gentler learning rate
    {"name": "C_dev80", "train": ["dev_rest"], "eval": "dev_fold0", "epochs": 10, "lr": 3e-5, "eval_epochs": [0, 2, 4, 6, 8, 10]},
    # D/E: LoRA on the encoder's attention, everything else frozen; mirror A and B
    {"name": "D_lora_train", "train": ["train"], "eval": "dev", "epochs": 10, "lr": 3e-4, "lora": True, "eval_epochs": [2, 4, 6, 8, 10]},
    {"name": "E_lora_train_dev80", "train": ["train", "dev_rest"], "eval": "dev_fold0", "epochs": 10, "lr": 3e-4, "lora": True, "eval_epochs": [2, 4, 6, 8, 10]},
    # Final recipe (B was best on Kaggle, 0.24). B_long finds the best epoch on held-out
    # fold 0; F_final trains on train + all of dev with the identical schedule and only
    # writes test predictions (nothing is left to score it on). Use F at B_long's best epoch.
    {"name": "B_long", "train": ["train", "dev_rest"], "eval": "dev_fold0", "epochs": 14, "lr": 1e-4, "eval_epochs": [6, 8, 10, 12, 14]},
    {"name": "F_final", "train": ["train", "dev"], "eval": None, "epochs": 14, "lr": 1e-4, "eval_epochs": [6, 8, 10, 12, 14]},
    # E14 G_over (chasing 1st place, 0.17): test-like data is what helps, so show the dev clips
    # 3 times per epoch. Beam search vs greedy is compared on the same weights at epoch 10.
    {"name": "G_over", "train": ["train", "dev_rest*3"], "eval": "dev_fold0", "epochs": 10, "lr": 1e-4,
     "eval_epochs": [6, 8, 10], "beam_epochs": [10], "keep_epoch": 10},
]
# Final run (R01): twins with identical settings. H_val is scored on held-out fold 0 for every
# variant (epoch 12 / average of epochs 10-14, greedy / beam); H_final trains on all of dev and
# writes test predictions for the same variants. Submit H_final's variant that wins on H_val.
# G_over showed: dev x3 beats no oversampling (0.0816 vs 0.0903), best at epoch 8 of a 10-epoch
# schedule; beam search beat greedy on the same weights (0.0839 vs 0.0879).
H_OVERSAMPLE = 3
EXPERIMENTS += [
    {"name": "H_val", "train": ["train", f"dev_rest*{H_OVERSAMPLE}"], "eval": "dev_fold0", "epochs": 10, "lr": 1e-4,
     "eval_epochs": [6, 8, 10], "beam_epochs": [8], "avg_epochs": [6, 8, 10], "avg_beam": True},
    {"name": "H_final", "train": ["train", f"dev*{H_OVERSAMPLE}"], "eval": None, "epochs": 10, "lr": 1e-4,
     "eval_epochs": [6, 8, 10], "beam_epochs": [8], "avg_epochs": [6, 8, 10], "avg_beam": True,
     "keep_epoch": 8, "keep_avg": True},
]
# Model soup (E16): the H recipe with different seeds; each run's averaged weights are an
# ingredient, and the soup is their element-wise mean. S_val checks it on held-out fold 0;
# S_final adds the saved H_final model from the previous run as a third ingredient.
EXPERIMENTS += [
    {"name": f"S_val_s{seed}", "seed": seed, "train": ["train", f"dev_rest*{H_OVERSAMPLE}"], "eval": "dev_fold0",
     "epochs": 10, "lr": 1e-4, "eval_epochs": [6, 8, 10], "avg_epochs": [6, 8, 10], "soup_ingredient": True}
    for seed in (11, 12, 13)
] + [
    {"name": f"S_final_s{seed}", "seed": seed, "train": ["train", f"dev*{H_OVERSAMPLE}"], "eval": None,
     "epochs": 10, "lr": 1e-4, "eval_epochs": [6, 8, 10], "avg_epochs": [6, 8, 10], "soup_ingredient": True}
    for seed in (12, 13)
]
# E18: two changes to the H recipe, each run with seeds 11 and 13 so they compare like-for-like
# with S_val_s11 / S_val_s13 (seed noise on fold 0 is about +-0.004).
#   P: speed perturbation - the 3 dev copies are played at 1.0x, 0.9x and 1.1x instead of 3 identical copies
#   L: lower learning rate, 5e-5 instead of 1e-4
EXPERIMENTS += [
    {"name": f"P_val_s{seed}", "seed": seed, "train": ["train", f"dev_rest*{H_OVERSAMPLE}"], "speed": [1.0, 0.9, 1.1],
     "eval": "dev_fold0", "epochs": 10, "lr": 1e-4, "eval_epochs": [6, 8, 10], "avg_epochs": [6, 8, 10], "avg_beam": True}
    for seed in (11, 13)
] + [
    {"name": f"L_val_s{seed}", "seed": seed, "train": ["train", f"dev_rest*{H_OVERSAMPLE}"],
     "eval": "dev_fold0", "epochs": 10, "lr": 5e-5, "eval_epochs": [6, 8, 10], "avg_epochs": [6, 8, 10], "avg_beam": True}
    for seed in (11, 13)
] + [
    # E19: 5e-5 was clearly worse than 1e-4, so try the other direction
    {"name": f"U_val_s{seed}", "seed": seed, "train": ["train", f"dev_rest*{H_OVERSAMPLE}"],
     "eval": "dev_fold0", "epochs": 10, "lr": 2e-4, "eval_epochs": [6, 8, 10], "avg_epochs": [6, 8, 10], "avg_beam": True}
    for seed in (11, 13)
]
# E20: add the public Google FLEURS Thai *train* split (read Wikipedia-style sentences, the same style
# as dev and test; ~10x more test-like speech than dev). Rules allow public external data if cited:
# Conneau et al. 2022, "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech".
# FLEURS dev/test are never downloaded (our test set is very likely FLEURS test). load_fleurs()
# also drops any FLEURS sentence that overlaps a dev sentence, and stops if the overlap is large.
EXPERIMENTS += [
    {"name": "FL_val", "seed": 11, "train": ["train", f"dev_rest*{H_OVERSAMPLE}", "fleurs"], "eval": "dev_fold0",
     "epochs": 10, "lr": 1e-4, "eval_epochs": [6, 8, 10], "beam_epochs": [8], "avg_epochs": [6, 8, 10], "avg_beam": True},
    {"name": "FL_final", "seed": 11, "train": ["train", f"dev*{H_OVERSAMPLE}", "fleurs"], "eval": None,
     "epochs": 10, "lr": 1e-4, "eval_epochs": [6, 8, 10], "beam_epochs": [8], "avg_epochs": [6, 8, 10], "avg_beam": True,
     "keep_avg": True},
    # E21: FL_final scored 0.23 on Kaggle (E15: 0.19) because it wrote far more names in English letters.
    # Same run without the 16% of FLEURS sentences that contain Latin letters. No val twin: fold 0 favours
    # FLEURS-like changes for Pathumma (A03), so Kaggle is the check.
    {"name": "FLN_final", "seed": 11, "train": ["train", f"dev*{H_OVERSAMPLE}", "fleurs_nolatin"], "eval": None,
     "epochs": 10, "lr": 1e-4, "eval_epochs": [6, 8, 10], "avg_epochs": [6, 8, 10], "avg_beam": True,
     "keep_avg": True},
]
SOUP = None  # e.g. {"name": "S_val", "eval": "dev_fold0", "extra_nemo": [], "keep": False}; set per notebook
BEAM_SIZE = 4
VAL_TEST_PREDS = False  # True: val runs also write test predictions (needed only to submit a val model itself)
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ("linear_q", "linear_k", "linear_v", "linear_out")}
DEFAULT_EXPERIMENTS = ["H_val", "H_final"]
# Batch 8 ran out of T4 memory on dev's longer clips (RNN-T joint tensor); same effective batch 16.
BATCH_SIZE = 4
ACCUMULATE = 4
MAX_SECONDS = 40.0


def pip(*packages):
    if os.environ.get("SKIP_PIP"):  # Colab notebook installs once, without upgrading Colab's CUDA build of torch
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *packages], check=False)


def find_data_dir(given):
    # given (e.g. Colab's unzip folder) may hold the CSVs in a subfolder
    hits = glob.glob(os.path.join(given or "/kaggle/input", "**", "dev.csv"), recursive=True)
    if not hits:
        sys.exit("dev.csv not found; pass --data-dir")
    return os.path.dirname(hits[0])


def dev_fold(audio):
    """Deterministic fold from the clip number, e.g. c-validation-0123.mp3 -> 123 % 5."""
    stem = os.path.splitext(os.path.basename(audio))[0]
    return int(re.search(r"(\d+)$", stem).group(1)) % FOLDS


def load_splits(data_dir, wav_dir, limit=None, audio_root=None):
    """Decode every clip once to 16 kHz mono WAV. Returns {part: [row, ...]} with parts
    train, dev, dev_fold0, dev_rest, test. Audio files are looked up anywhere under audio_root
    (default: the CSV folder)."""
    import librosa
    import pandas as pd
    import soundfile as sf

    index = {os.path.basename(p): p for p in glob.glob(os.path.join(audio_root or data_dir, "**", "*.mp3"), recursive=True)}
    os.makedirs(wav_dir, exist_ok=True)
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
            wav = os.path.join(wav_dir, os.path.splitext(os.path.basename(audio))[0] + ".wav")
            y, _ = librosa.load(src, sr=SR, mono=True)
            sf.write(wav, y, SR)
            rows.append({"audio": audio, "ref": str(ref), "wav": wav, "seconds": len(y) / SR})
        parts[split] = rows
    parts["dev_fold0"] = [r for r in parts["dev"] if dev_fold(r["audio"]) == 0]
    parts["dev_rest"] = [r for r in parts["dev"] if dev_fold(r["audio"]) != 0]
    return parts


def _norm(text):
    return re.sub(r"[\W_]+", "", str(text).lower())


def load_fleurs(fleurs_dir, dev_rows, max_overlap=0.05):
    """FLEURS Thai train: <fleurs_dir>/train.tsv plus its extracted 16 kHz WAVs (any subfolder).
    TSV columns: id, file name, raw transcription, normalized transcription, phonemes, num samples, gender.
    The raw transcription keeps digits, Latin-script names and punctuation, like our references.
    A FLEURS sentence is dropped if it shares a 20-character chunk with any dev sentence; if more than
    max_overlap of dev sentences have such a twin, the splits don't line up with ours and we stop."""
    import csv
    wavs = {os.path.basename(p): p for p in glob.glob(os.path.join(fleurs_dir, "**", "*.wav"), recursive=True)}
    rows = []
    with open(os.path.join(fleurs_dir, "train.tsv"), encoding="utf-8") as f:
        for rec in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(rec) >= 6 and rec[1] in wavs:
                rows.append({"audio": rec[1], "ref": rec[2].strip(), "wav": wavs[rec[1]], "seconds": int(rec[5]) / SR})
    chunks = lambda t: {t[i:i + 20] for i in range(max(1, len(t) - 19))}
    dev_chunks, dev_size = {}, {}
    for r in dev_rows:
        cs = chunks(_norm(r["ref"]))
        dev_size[r["audio"]] = len(cs)
        for c in cs:
            dev_chunks.setdefault(c, set()).add(r["audio"])
    kept, hit_dev = [], set()
    for r in rows:
        count = {}
        for c in chunks(_norm(r["ref"])):
            for a in dev_chunks.get(c, ()):
                count[a] = count.get(a, 0) + 1
        # a twin shares at least 30% of a dev sentence's chunks (one shared stock phrase is not enough)
        twins = {a for a, n in count.items() if n >= 0.3 * dev_size[a]}
        if twins:
            hit_dev |= twins
        else:
            kept.append(r)
    print(f"FLEURS train: {len(rows)} clips ({sum(r['seconds'] for r in rows) / 3600:.1f} h); dropped {len(rows) - len(kept)} "
          f"that overlap {len(hit_dev)} of {len(dev_rows)} dev sentences", flush=True)
    if len(hit_dev) > max_overlap * len(dev_rows):
        sys.exit("FLEURS train overlaps dev a lot: this dataset's splits differ from FLEURS, so test may be in "
                 "FLEURS train too. Not using it.")
    return kept


def unsupported_chars(model, texts):
    """Characters the model's tokenizer cannot encode (they would become <unk>)."""
    tok = model.tokenizer
    unk = tok.tokenizer.unk_id() if hasattr(tok, "tokenizer") and hasattr(tok.tokenizer, "unk_id") else None
    bad = set()
    for ch in sorted(set("".join(texts))):
        if ch.isspace():
            continue
        ids = tok.text_to_ids(ch)
        if (unk is not None and unk in ids) or tok.ids_to_text(ids).strip() != ch:
            bad.add(ch)
    return bad


def clean(text, bad):
    text = "".join(c for c in text if c not in bad)
    return re.sub(r"\s+", " ", text).strip()


def write_manifest(rows, path, bad):
    kept = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            text = clean(r["ref"], bad)
            if not text or r["seconds"] > MAX_SECONDS:
                continue
            f.write(json.dumps({"audio_filepath": r["wav"], "duration": r["seconds"], "text": text}, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def load_model():
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import HfApi, hf_hub_download

    nemo_file = next(f for f in HfApi().list_repo_files(HF_ID) if f.endswith(".nemo"))
    return nemo_asr.models.ASRModel.restore_from(hf_hub_download(HF_ID, nemo_file), map_location="cuda")


def transcribe(model, rows):
    model.eval()
    out = model.transcribe([r["wav"] for r in rows], batch_size=16, verbose=False)
    if isinstance(out, tuple):  # older NeMo RNNT returns (best, all_hypotheses)
        out = out[0]
    return [(o if isinstance(o, str) else o.text).strip() for o in out]


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


def set_decoding(model, beam):
    """Greedy (the model's default) or RNN-T beam search with BEAM_SIZE candidates."""
    import copy
    from omegaconf import open_dict

    cfg = copy.deepcopy(model.cfg.decoding)
    with open_dict(cfg):
        cfg.strategy = "beam" if beam else "greedy_batch"
        if beam:
            cfg.beam.beam_size = BEAM_SIZE
            cfg.beam.return_best_hypothesis = True
    model.change_decoding_strategy(cfg, verbose=False)


def average_checkpoints(paths):
    """Element-wise mean of several saved state dicts (non-float buffers taken from the last)."""
    import torch

    total, dtypes = None, {}
    for p in paths:
        sd = torch.load(p, map_location="cpu")
        if total is None:
            dtypes = {k: v.dtype for k, v in sd.items()}
            total = {k: v.double() if v.is_floating_point() else v for k, v in sd.items()}
        else:
            for k, v in sd.items():
                total[k] = total[k] + v.double() if v.is_floating_point() else v
    return {k: (v / len(paths)).to(dtypes[k]) if dtypes[k].is_floating_point else v for k, v in total.items()}


def evaluate(model, exp, epoch, parts, out_dir, summary, beam=False):
    """epoch is an epoch number, or a label such as "avg10-12-14" for averaged weights."""
    import pandas as pd

    tag = (f"ep{epoch}" if isinstance(epoch, int) else epoch) + ("_beam" if beam else "")
    set_decoding(model, beam)
    # val runs only need their fold-0 score (we submit the final twin), so skip test: about 40% of a val notebook's time
    if exp["eval"] is None or VAL_TEST_PREDS:
        test_rows = parts["test"]
        test_hyps = transcribe(model, test_rows)
        pd.DataFrame({"audio": [r["audio"] for r in test_rows], "hyp": test_hyps}).to_csv(
            os.path.join(out_dir, f"test_{exp['name']}_{tag}.csv"), index=False, encoding="utf-8")
    if exp["eval"] is None:  # trained on all labelled data: test predictions only
        print(f"[{exp['name']}] {tag}: test predictions written", flush=True)
    else:
        eval_rows = parts[exp["eval"]]
        hyps = transcribe(model, eval_rows)
        pd.DataFrame({"audio": [r["audio"] for r in eval_rows], "ref": [r["ref"] for r in eval_rows], "hyp": hyps}).to_csv(
            os.path.join(out_dir, f"pred_{exp['name']}_{tag}.csv"), index=False, encoding="utf-8")
        score = round(wer([r["ref"] for r in eval_rows], hyps), 4)
        summary["results"].setdefault(exp["name"], {})[tag] = score
        print(f"[{exp['name']}] {tag}: {exp['eval']} WER {score}", flush=True)
    if beam:
        set_decoding(model, False)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def add_lora(model, exp, summary):
    """Wrap the encoder's attention layers with LoRA and freeze every other weight.
    Hand-written instead of PEFT: PEFT's import fails on Kaggle's torchao version."""
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        """output = W x + (alpha / r) * B (A x), with W frozen and B starting at zero."""

        def __init__(self, base, r, alpha, dropout):
            super().__init__()
            self.base = base
            self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            self.scale = alpha / r
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale

    targets = [(name, module) for name, module in model.encoder.named_modules()
               if isinstance(module, nn.Linear) and name.split(".")[-1] in LORA["target_modules"]]
    for name, module in targets:
        parent = model.encoder.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], LoRALinear(module, LORA["r"], LORA["lora_alpha"], LORA["lora_dropout"]).to(module.weight.device))
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
    print(f"[{exp['name']}] LoRA wrapped {len(targets)} linear layers", flush=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    summary.setdefault("lora_params", {})[exp["name"]] = {"trainable": trainable, "total": total}
    print(f"[{exp['name']}] LoRA: training {trainable:,} of {total:,} weights ({100 * trainable / total:.2f}%)", flush=True)


def speed_copy(row, factor, work_dir):
    """The clip played at `factor` x speed (tempo and pitch change together, as in Kaldi speed
    perturbation); written once and cached."""
    if factor == 1.0:
        return row
    import librosa
    import soundfile as sf
    wav = os.path.join(work_dir, "wav_speed", f"{os.path.splitext(os.path.basename(row['wav']))[0]}_x{factor}.wav")
    if not os.path.exists(wav):
        os.makedirs(os.path.dirname(wav), exist_ok=True)
        y, _ = librosa.load(row["wav"], sr=SR)
        # read the samples as if recorded at SR * factor, then resample back to SR
        sf.write(wav, librosa.resample(y, orig_sr=SR * factor, target_sr=SR), SR)
    return {**row, "wav": wav, "seconds": row["seconds"] / factor}


def run_experiment(exp, parts, work_dir, out_dir, summary):
    import torch
    from omegaconf import OmegaConf, open_dict
    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl

    if exp.get("seed") is not None:
        pl.seed_everything(exp["seed"], workers=True)
    model = load_model()
    if exp.get("lora"):
        add_lora(model, exp, summary)
    bad = set(summary["unsupported_chars"])
    # "dev_rest*3" = that part repeated 3 times per epoch (oversampling); with exp["speed"],
    # copy i of an oversampled part is played at speed[i]
    train_rows = []
    for part in exp["train"]:
        name, _, times = part.partition("*")
        for i in range(int(times or 1)):
            factor = exp["speed"][i] if times and exp.get("speed") else 1.0
            train_rows += [speed_copy(r, factor, work_dir) for r in parts[name]]
    manifest = os.path.join(work_dir, f"train_{exp['name']}.json")
    kept = write_manifest(train_rows, manifest, bad)
    print(f"\n##### {exp['name']}: training on {kept}/{len(train_rows)} clips, eval on {exp['eval']}", flush=True)

    ckpt_dir = os.path.join(work_dir, exp["name"])
    os.makedirs(ckpt_dir, exist_ok=True)
    save_at = [e for e in exp["eval_epochs"] if e > 0]

    class SaveWeights(pl.Callback):
        def on_train_epoch_end(self, trainer, module):
            epoch = trainer.current_epoch + 1
            if epoch in save_at:
                torch.save(module.state_dict(), os.path.join(ckpt_dir, f"ep{epoch}.pt"))
                print(f"[{exp['name']}] saved epoch {epoch}, train loss {float(trainer.callback_metrics.get('train_loss', float('nan'))):.3f}", flush=True)

    trainer = pl.Trainer(
        devices=1, accelerator="gpu", max_epochs=exp["epochs"], precision="16-mixed",
        accumulate_grad_batches=ACCUMULATE, gradient_clip_val=1.0,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        limit_val_batches=0, num_sanity_val_steps=0, log_every_n_steps=25,
        callbacks=[SaveWeights()],
    )
    model.set_trainer(trainer)
    train_cfg = model.cfg.train_ds
    with open_dict(train_cfg):
        train_cfg.manifest_filepath = manifest
        train_cfg.batch_size = BATCH_SIZE
        train_cfg.shuffle = True
        train_cfg.num_workers = 4
        train_cfg.max_duration = MAX_SECONDS
        train_cfg.is_tarred = False
        train_cfg.tarred_audio_filepaths = None
    model.setup_training_data(train_cfg)
    # The RNN-T loss needs a (time x text x vocab) tensor per clip; a batch of several long clips with long
    # transcripts ran out of T4 memory (FL_final). Computing the joint + loss one clip at a time gives the
    # same loss with about a quarter of the peak memory.
    if hasattr(model.joint, "set_fused_batch_size") and getattr(model.joint, "fuse_loss_wer", False):
        model.joint.set_fused_batch_size(1)
        print(f"[{exp['name']}] RNN-T joint + loss computed one clip at a time (fused batch size 1)", flush=True)
    model.setup_optimization(optim_config=OmegaConf.create({
        "name": "adamw", "lr": exp["lr"], "betas": [0.9, 0.98], "weight_decay": 1e-3,
        "sched": {"name": "CosineAnnealing", "warmup_ratio": 0.1, "min_lr": 1e-6},
    }))

    if 0 in exp["eval_epochs"]:
        evaluate(model, exp, 0, parts, out_dir, summary)
    t0 = time.time()
    model.train()
    trainer.fit(model)
    summary["train_minutes"][exp["name"]] = round((time.time() - t0) / 60, 1)

    for epoch in save_at:
        path = os.path.join(ckpt_dir, f"ep{epoch}.pt")
        model.load_state_dict(torch.load(path, map_location="cuda"))
        evaluate(model, exp, epoch, parts, out_dir, summary)
        if epoch in exp.get("beam_epochs", []):
            evaluate(model, exp, epoch, parts, out_dir, summary, beam=True)
        if epoch == exp.get("keep_epoch"):  # keep these weights for later runs
            model.save_to(os.path.join(out_dir, f"{exp['name']}_ep{epoch}.nemo"))

    if exp.get("avg_epochs"):  # checkpoint averaging: one model whose weights are the mean of several epochs
        label = "avg" + "-".join(str(e) for e in exp["avg_epochs"])
        model.load_state_dict(average_checkpoints([os.path.join(ckpt_dir, f"ep{e}.pt") for e in exp["avg_epochs"]]))
        evaluate(model, exp, label, parts, out_dir, summary)
        if exp.get("avg_beam"):
            evaluate(model, exp, label, parts, out_dir, summary, beam=True)
        if exp.get("keep_avg"):
            model.save_to(os.path.join(out_dir, f"{exp['name']}_{label}.nemo"))
        if exp.get("soup_ingredient"):
            torch.save(model.state_dict(), os.path.join(work_dir, f"soup_{exp['name']}.pt"))
    for epoch in save_at:
        os.remove(os.path.join(ckpt_dir, f"ep{epoch}.pt"))

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


def make_soup(soup, parts, work_dir, out_dir, summary):
    """Average every soup ingredient (plus any saved .nemo models in soup["extra_nemo"]) into one
    model, then evaluate it with greedy and beam decoding."""
    import nemo.collections.asr as nemo_asr
    import torch

    paths = sorted(glob.glob(os.path.join(work_dir, "soup_*.pt")))
    for pattern in soup.get("extra_nemo", []):
        for p in glob.glob(pattern, recursive=True):
            extra = nemo_asr.models.ASRModel.restore_from(p, map_location="cpu")
            out = os.path.join(work_dir, f"soup_extra_{len(paths)}.pt")
            torch.save(extra.state_dict(), out)
            paths.append(out)
            print("soup: added saved model", p, flush=True)
            del extra
    print(f"soup of {len(paths)} models:", [os.path.basename(p) for p in paths], flush=True)
    model = load_model()
    model.load_state_dict(average_checkpoints(paths))
    exp = {"name": soup["name"], "eval": soup["eval"]}
    label = f"soup{len(paths)}"
    evaluate(model, exp, label, parts, out_dir, summary)
    evaluate(model, exp, label, parts, out_dir, summary, beam=True)
    if soup.get("keep"):
        model.save_to(os.path.join(out_dir, f"{soup['name']}_{label}.nemo"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--out-dir", default="/kaggle/working" if os.path.isdir("/kaggle") else "results/finetune_fastcon")
    ap.add_argument("--experiments", help="comma-separated experiment names (default: all)")
    ap.add_argument("--limit", type=int, help="only the first N clips per split (smoke test)")
    ap.add_argument("--epochs", type=int, help="override epochs for every experiment (smoke test)")
    ap.add_argument("--fleurs-dir", help="folder with FLEURS Thai train.tsv and its WAVs (enables the 'fleurs' part)")
    args = ap.parse_args()

    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError:
        sys.exit("No internet: enable it in the notebook settings (needs a phone-verified Kaggle account).")
    pip("nemo_toolkit[asr]", "pythainlp")
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        sys.exit("No GPU: switch on the accelerator.")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    work_dir = tempfile.mkdtemp()
    parts = load_splits(find_data_dir(args.data_dir), os.path.join(work_dir, "wav"), args.limit, args.data_dir)
    stats = {}
    for name in ("train", "dev", "test"):
        secs = np.array([r["seconds"] for r in parts[name]])
        stats[name] = {"clips": len(secs), "hours": round(secs.sum() / 3600, 2), "sec_mean": round(secs.mean(), 1),
                       "sec_max": round(secs.max(), 1), "over_40s": int((secs > MAX_SECONDS).sum())}
    stats["dev_fold0"], stats["dev_rest"] = len(parts["dev_fold0"]), len(parts["dev_rest"])
    if args.fleurs_dir:
        parts["fleurs"] = load_fleurs(args.fleurs_dir, parts["dev"])
        parts["fleurs_nolatin"] = [r for r in parts["fleurs"] if not re.search(r"[A-Za-z]", r["ref"])]
        for name in ("fleurs", "fleurs_nolatin"):
            stats[name] = {"clips": len(parts[name]), "hours": round(sum(r["seconds"] for r in parts[name]) / 3600, 2)}
    print(json.dumps(stats, indent=2), flush=True)

    probe = load_model()
    bad = unsupported_chars(probe, [r["ref"] for r in parts["train"] + parts["dev"] + parts.get("fleurs", [])])
    del probe
    torch.cuda.empty_cache()
    print("characters the tokenizer cannot encode (removed from training text):", "".join(sorted(bad)), flush=True)

    wanted = args.experiments.split(",") if args.experiments else DEFAULT_EXPERIMENTS
    experiments = [e for e in EXPERIMENTS if e["name"] in wanted]
    if args.epochs:
        experiments = [{**e, "epochs": args.epochs, "eval_epochs": [x for x in e["eval_epochs"] if x <= args.epochs]} for e in experiments]

    summary = {"model": HF_ID, "data": stats, "unsupported_chars": sorted(bad), "batch_size": BATCH_SIZE,
               "accumulate": ACCUMULATE, "experiments": experiments, "results": {}, "train_minutes": {}}
    for exp in experiments:
        run_experiment(exp, parts, work_dir, args.out_dir, summary)
    if SOUP:
        make_soup(SOUP, parts, work_dir, args.out_dir, summary)
    print(json.dumps(summary["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
