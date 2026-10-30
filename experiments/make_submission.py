"""Turn raw test transcripts (test_<model>.csv: audio,hyp) into a Kaggle submission.

Segments each transcript with newmm (words joined by single spaces), keeps the
row order and file names of test.csv, and refuses to write if any clip is
missing or extra.

Usage: python make_submission.py results/zeroshot_v4/test_pathumma-fastcon.csv submissions/e04_pathumma_zeroshot.csv
"""
import csv
import os
import sys

from pythainlp.tokenize import word_tokenize

TEST_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "datasets", "test.csv")


def segment(text):
    return " ".join(t for t in word_tokenize(str(text or ""), engine="newmm", keep_whitespace=False) if t.strip())


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(TEST_CSV, encoding="utf-8-sig") as f:
        order = [r["audio"] for r in csv.DictReader(f)]
    with open(src, encoding="utf-8-sig") as f:
        hyps = {os.path.basename(r["audio"]): r["hyp"] for r in csv.DictReader(f)}

    missing = [a for a in order if os.path.basename(a) not in hyps]
    extra = set(hyps) - {os.path.basename(a) for a in order}
    if missing or extra:
        sys.exit(f"refusing to write: {len(missing)} missing, {len(extra)} extra clips")

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio", "sentence"])
        for a in order:
            w.writerow([a, segment(hyps[os.path.basename(a)])])
    empty = sum(not hyps[os.path.basename(a)].strip() for a in order)
    print(f"wrote {dst}: {len(order)} rows, {empty} empty transcripts")


if __name__ == "__main__":
    main()
