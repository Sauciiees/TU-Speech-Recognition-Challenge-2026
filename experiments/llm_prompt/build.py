"""Collect the ASR hypotheses to correct and write the Kaggle kernel folder.

Usage (repo root): python experiments/llm_prompt/build.py
Writes results/llm_prompt/sets.json (local: which sentence id belongs to which clip of which set)
and experiments/llm_prompt/kernel/ (llm_prompt.py with the sentences embedded + kernel-metadata.json).
Only hypotheses go to Kaggle, never references.
"""
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import llm_prompt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = {  # name: (csv, is_dev)
    "hval": ("results/final_h/pred_H_val_avg6-8-10_beam.csv", True),   # closest to the best model, 50 held-out clips
    "gover": ("results/g_over/pred_G_over_ep10_beam.csv", True),       # same 50 clips, another fine-tuned model
    "blong": ("results/finetune_v3/pred_B_long_ep12.csv", True),
    "a10": ("results/finetune_v1/pred_A_train_ep10.csv", True),        # fine-tuned on train only, all 240 dev clips
    "zs": ("results/llm_only/llm_dev.csv", True),                     # zero-shot Pathumma, 240 clips (E10's set)
    "test": ("results/final_h/test_H_final_avg6-8-10_beam.csv", False),  # E15, our best submission (0.19)
}

sentences, sets = {}, {}
for name, (path, is_dev) in SETS.items():
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    sets[name] = []
    for r in rows:
        hyp = r["hyp"] or ""
        key = hashlib.md5(hyp.encode("utf-8")).hexdigest()[:12]
        sentences[key] = hyp
        sets[name].append({"audio": r["audio"], "id": key, "hyp": hyp, **({"ref": r["ref"]} if is_dev else {})})

# an example's "wrong" text must not occur in any real input (E10's model copied its example 29 times)
for _, fixes in llm_prompt.P1_EXAMPLES + llm_prompt.P0_EXAMPLES:
    for f in fixes:
        hits = [s for s in sentences.values() if f["wrong"] in s]
        assert not hits, f"example word {f['wrong']} occurs in {len(hits)} inputs"

os.makedirs("results/llm_prompt", exist_ok=True)
json.dump(sets, open("results/llm_prompt/sets.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

kdir = os.path.join(HERE, "kernel")
os.makedirs(kdir, exist_ok=True)
code = open(os.path.join(HERE, "llm_prompt.py"), encoding="utf-8").read()
code = code.replace("DATA = None  # filled in by build.py",
                    "DATA = " + json.dumps({"sentences": sentences}, ensure_ascii=False), 1)
open(os.path.join(kdir, "llm_prompt.py"), "w", encoding="utf-8").write(code)
json.dump({
    "id": "sauciiees/dsi443-llm-prompt", "title": "dsi443-llm-prompt", "code_file": "llm_prompt.py",
    "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
    "enable_internet": True, "competition_sources": [], "dataset_sources": [], "kernel_sources": [],
}, open(os.path.join(kdir, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
print({k: len(v) for k, v in sets.items()}, "unique sentences:", len(sentences))
