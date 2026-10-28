"""Re-score saved dev predictions (pred_<model>.csv with audio,ref,hyp) locally.

Primary metric: newmm on the text as written (whitespace only separates, it is
never a token), WER = edits / reference words over the whole dev set. This is
our best guess at the Kaggle scorer. Also reports a punctuation-stripped variant
and a character error rate, which ignores segmentation entirely.

Usage: python score_dev.py results/            # every pred_*.csv in the folder
"""
import csv
import glob
import os
import re
import sys

from pythainlp.tokenize import word_tokenize

PUNCT = r"[\"'“”‘’!?,.:;()\[\]{}…«»]"


def words(text, strip_punct=False):
    text = str(text or "")
    if strip_punct:
        text = re.sub(PUNCT, " ", text)
    return [t for t in word_tokenize(text, engine="newmm", keep_whitespace=False) if t.strip()]


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def rate(pairs):
    errors = total = 0
    for ref, hyp in pairs:
        errors += edit_distance(ref, hyp)
        total += len(ref)
    return errors / max(total, 1)


def score(rows):
    refs = [r["ref"] for r in rows]
    hyps = [r["hyp"] for r in rows]
    nospace = lambda s: re.sub(r"\s+", "", str(s or ""))
    return {
        "wer": rate([(words(r), words(h)) for r, h in zip(refs, hyps)]),
        "wer_no_punct": rate([(words(r, True), words(h, True)) for r, h in zip(refs, hyps)]),
        "cer": rate([(list(nospace(r)), list(nospace(h))) for r, h in zip(refs, hyps)]),
        "empty_hyps": sum(not str(h).strip() for h in hyps),
    }


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results"
    table = []
    for path in sorted(glob.glob(os.path.join(folder, "pred_*.csv"))):
        with open(path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        name = os.path.basename(path)[5:-4]
        table.append((name, len(rows), score(rows)))
    table.sort(key=lambda t: t[2]["wer"])
    print(f"{'model':22} {'clips':>5} {'WER':>7} {'WER-nopunct':>12} {'CER':>7} {'empty':>6}")
    for name, n, s in table:
        print(f"{name:22} {n:5d} {s['wer']:7.4f} {s['wer_no_punct']:12.4f} {s['cer']:7.4f} {s['empty_hyps']:6d}")


if __name__ == "__main__":
    main()
