"""Zero-shot comparison of the leaderboard's pretrained models on the dev set.

Runs as a Kaggle GPU script (competition data attached, internet on), or anywhere
with --data-dir pointing at the unzipped competition data. Each model runs in its
own Python process, because Qwen3-ASR needs a newer transformers and the
FastConformer models need NeMo, and those installs can conflict.

Writes pred_<model>.csv (audio, ref, hyp) plus timings.json to --out-dir.
Score them locally with score_dev.py.
"""
import argparse
import glob
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

SR = 16_000

# Exactly the zero-shot models the host put on the leaderboard.
MODELS = {
    # name: (Hugging Face id, family)
    "typhoon-whisper-large-v3": ("typhoon-ai/typhoon-whisper-large-v3", "whisper"),
    "whisper-large-v3": ("openai/whisper-large-v3", "whisper"),
    "typhoon-whisper-turbo": ("typhoon-ai/typhoon-whisper-turbo", "whisper"),
    "pathumma-fastcon": ("nectec/Pathumma-stt-th-fastconformer-rnnt-large", "nemo"),
    "typhoon-asr-realtime": ("typhoon-ai/typhoon-asr-realtime", "nemo"),
    "qwen3-asr-0.6b": ("Qwen/Qwen3-ASR-0.6B-hf", "qwen"),
}
# Whisper large with beam 5 runs out of T4 memory at batch 16.
BATCH_OVERRIDE = {"typhoon-whisper-large-v3": 4, "whisper-large-v3": 4}
# Families are installed in this order: NeMo pins its own transformers, so the
# transformers upgrade for Qwen comes last.
FAMILY_ORDER = ["whisper", "nemo", "qwen"]
INSTALL = {
    "whisper": [],
    "nemo": ["nemo_toolkit[asr]"],
    "qwen": ["transformers>=5.13.0"],
}


def pip(*packages):
    if packages:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *packages], check=False)


def find_data_dir(given):
    if given:
        return given
    hits = glob.glob("/kaggle/input/**/dev.csv", recursive=True)
    if not hits:
        sys.exit("dev.csv not found; pass --data-dir")
    return os.path.dirname(hits[0])


def prepare_audio(data_dir, wav_dir, limit):
    """Decode dev and test MP3s once to 16 kHz mono WAV; every model reads the same files.
    Test rows have no reference; they only produce submission predictions."""
    import librosa
    import pandas as pd
    import soundfile as sf

    index = {os.path.basename(p): p for p in glob.glob(os.path.join(data_dir, "**", "*.mp3"), recursive=True)}
    os.makedirs(wav_dir, exist_ok=True)
    rows = []
    for split in ("dev", "test"):
        df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
        if limit:
            df = df.head(limit)
        refs = df["sentence"] if "sentence" in df else [""] * len(df)
        for audio, sentence in zip(df["audio"], refs):
            src = os.path.join(data_dir, audio)
            src = src if os.path.exists(src) else index[os.path.basename(audio)]
            wav = os.path.join(wav_dir, os.path.splitext(os.path.basename(audio))[0] + ".wav")
            y, _ = librosa.load(src, sr=SR, mono=True)
            sf.write(wav, y, SR)
            rows.append({"split": split, "audio": audio, "ref": sentence, "wav": wav, "seconds": len(y) / SR})
    return rows


# ---------- per-family transcription, run inside a worker process ----------

def transcribe_whisper(hf_id, wavs, batch_size):
    import torch
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition", model=hf_id, device=0,
        torch_dtype=torch.float16, chunk_length_s=30,
        generate_kwargs={"language": "th", "task": "transcribe", "num_beams": 5},
    )
    return [o["text"] for o in asr(wavs, batch_size=batch_size)]


def transcribe_nemo(hf_id, wavs, batch_size):
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import HfApi, hf_hub_download

    nemo_file = next(f for f in HfApi().list_repo_files(hf_id) if f.endswith(".nemo"))
    model = nemo_asr.models.ASRModel.restore_from(hf_hub_download(hf_id, nemo_file), map_location="cuda")
    model.eval()
    out = model.transcribe(wavs, batch_size=batch_size)
    if isinstance(out, tuple):  # older NeMo RNNT returns (best, all_hypotheses)
        out = out[0]
    return [o if isinstance(o, str) else o.text for o in out]


def transcribe_qwen(hf_id, wavs, batch_size):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(hf_id)
    model = AutoModelForMultimodalLM.from_pretrained(hf_id, device_map="cuda", torch_dtype=torch.float16)
    texts = []
    for wav in wavs:
        inputs = processor.apply_transcription_request(audio=wav, language="Thai").to(model.device, model.dtype)
        ids = model.generate(**inputs, max_new_tokens=448)
        texts.append(processor.decode(ids[:, inputs["input_ids"].shape[1]:], return_format="transcription_only")[0])
    return texts


def worker(name, rows_path, out_dir, batch_size):
    import pandas as pd

    hf_id, family = MODELS[name]
    rows = json.load(open(rows_path, encoding="utf-8"))
    transcribe = {"whisper": transcribe_whisper, "nemo": transcribe_nemo, "qwen": transcribe_qwen}[family]
    hyps = transcribe(hf_id, [r["wav"] for r in rows], BATCH_OVERRIDE.get(name, batch_size))
    df = pd.DataFrame({
        "split": [r["split"] for r in rows],
        "audio": [r["audio"] for r in rows],
        "ref": [r["ref"] for r in rows],
        "hyp": [str(h).strip() for h in hyps],
    })
    dev, test = df[df["split"] == "dev"], df[df["split"] == "test"]
    dev[["audio", "ref", "hyp"]].to_csv(os.path.join(out_dir, f"pred_{name}.csv"), index=False, encoding="utf-8")
    # Raw test transcripts; newmm segmentation for the submission happens locally.
    test[["audio", "hyp"]].to_csv(os.path.join(out_dir, f"test_{name}.csv"), index=False, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--out-dir", default="/kaggle/working" if os.path.isdir("/kaggle") else "results")
    ap.add_argument("--models", help="comma-separated names to run (default: all)")
    ap.add_argument("--limit", type=int, help="only the first N dev clips (smoke test)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--rows", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker(args.worker, args.rows, args.out_dir, args.batch_size)
        return

    import torch
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        online = True
    except OSError:
        online = False
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", "| internet:", online, flush=True)
    if not online or not torch.cuda.is_available():
        sys.exit("Needs a GPU and internet (Kaggle: phone-verified account, accelerator and internet switched on).")

    os.makedirs(args.out_dir, exist_ok=True)
    scratch = tempfile.mkdtemp()  # outside out-dir so the WAVs aren't saved as outputs
    rows = prepare_audio(find_data_dir(args.data_dir), os.path.join(scratch, "wav"), args.limit)
    rows_path = os.path.join(scratch, "rows.json")
    json.dump(rows, open(rows_path, "w", encoding="utf-8"), ensure_ascii=False)
    for split in ("dev", "test"):
        part = [r for r in rows if r["split"] == split]
        print(f"{split} clips: {len(part)}, audio {sum(r['seconds'] for r in part) / 60:.1f} min", flush=True)

    wanted = args.models.split(",") if args.models else list(MODELS)
    timings = {}
    for family in FAMILY_ORDER:
        names = [n for n in wanted if MODELS[n][1] == family]
        if not names:
            continue
        pip(*INSTALL[family])
        for name in names:
            print(f"\n=== {name} ({MODELS[name][0]})", flush=True)
            t0 = time.time()
            done = subprocess.run([sys.executable, __file__, "--worker", name, "--rows", rows_path,
                                   "--out-dir", args.out_dir, "--batch-size", str(args.batch_size)])
            timings[name] = {"seconds": round(time.time() - t0, 1), "ok": done.returncode == 0}
            print(name, timings[name], flush=True)
            json.dump(timings, open(os.path.join(args.out_dir, "timings.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
