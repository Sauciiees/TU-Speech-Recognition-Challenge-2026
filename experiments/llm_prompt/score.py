"""Score the E17 LLM runs under several filters, and write corrected test files.

Usage (repo root): python -I experiments/llm_prompt/score.py results/llm_prompt
Needs sets.json (build.py) and replies_<run>.json (Kaggle output) in that folder.

Filters (each applies a proposal only if it passes):
  loose  "wrong" occurs in the sentence, no English letters or digits
  e10    loose + at most 2 characters and 34% of the word changed (the E10 filter)
  tok    e10 + the fix must help newmm: the corrected word splits into fewer newmm tokens than the
         wrong one, or the wrong one isn't a dictionary word and the corrected one is.
         (A real word swapped for another real word, e.g. รัน -> เรน, is blocked.)
"""
import collections
import csv
import glob
import json
import math
import os
import re
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO, "experiments"))
from score_dev import words  # noqa: E402
from pythainlp.corpus import thai_words  # noqa: E402
from pythainlp.tokenize import word_tokenize  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9]")
DICT = thai_words()


def lev(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def parse(reply):
    m = re.search(r"\[.*\]", reply or "", re.S)
    try:
        fixes = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    return [(str(f.get("wrong", "")).strip(), str(f.get("correct", "")).strip())
            for f in fixes if isinstance(f, dict)]


def ok(wrong, correct, sentence, level):
    if not wrong or not correct or wrong == correct or wrong not in sentence:
        return False
    if LATIN_OR_DIGIT.search(wrong + correct):
        return False
    if level == "loose":
        return True
    d = lev(wrong, correct)
    if d > 2 or d > math.ceil(0.34 * len(wrong)):
        return False
    if level == "e10":
        return True
    n_w = len(word_tokenize(wrong, engine="newmm"))
    n_c = len(word_tokenize(correct, engine="newmm"))
    return n_c < n_w or (wrong not in DICT and correct in DICT)


def apply(sentence, fixes, level):
    for w, c in fixes:
        if ok(w, c, sentence, level):
            sentence = sentence.replace(w, c, 1)
    return sentence


def errors(ref, hyp):
    return lev(words(ref), words(hyp))


def main(folder):
    sets = json.load(open(os.path.join(folder, "sets.json"), encoding="utf-8"))
    runs = {os.path.basename(p)[8:-5]: json.load(open(p, encoding="utf-8"))
            for p in sorted(glob.glob(os.path.join(folder, "replies_*.json")))}
    levels = ("loose", "e10", "tok")
    table, changes = [], collections.defaultdict(list)
    for name, rows in sets.items():
        if name == "test":
            continue
        n_ref = sum(len(words(r["ref"])) for r in rows)
        base = sum(errors(r["ref"], r["hyp"]) for r in rows)
        line = {"set": name, "clips": len(rows), "no LLM": round(base / n_ref, 4)}
        for run, replies in runs.items():
            for level in levels:
                e, changed, better, worse = 0, 0, 0, 0
                for r in rows:
                    fixed = apply(r["hyp"], parse(replies.get(r["id"], "")), level)
                    fe = errors(r["ref"], fixed)
                    e += fe
                    if fixed != r["hyp"]:
                        changed += 1
                        b = errors(r["ref"], r["hyp"])
                        better += fe < b
                        worse += fe > b
                        if level == "tok":
                            changes[run].append((name, fe - b, r["hyp"], fixed))
                line[f"{run}/{level}"] = f"{e / n_ref:.4f} ({changed}: +{better} -{worse})"
        table.append(line)
    for line in table:
        print(json.dumps(line, ensure_ascii=False, indent=1))
    for run, ch in changes.items():
        print(f"\n== {run} / tok: sentence changes (set, delta errors) ==")
        for s, d, h, f in sorted(ch, key=lambda x: x[1]):
            diff = [(a, b) for a, b in zip(words(h), words(f)) if a != b][:3]
            print(f"{s:6} {d:+d}  {diff}")
    # corrected test files for every run and filter
    for run, replies in runs.items():
        for level in levels:
            out = os.path.join(folder, f"test_E17_{run}_{level}.csv")
            with open(out, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["audio", "hyp"])
                n = 0
                for r in sets["test"]:
                    fixed = apply(r["hyp"], parse(replies.get(r["id"], "")), level)
                    n += fixed != r["hyp"]
                    w.writerow([r["audio"], fixed])
            print(f"{out}: {n} test sentences changed")


if __name__ == "__main__":
    main(sys.argv[1])
