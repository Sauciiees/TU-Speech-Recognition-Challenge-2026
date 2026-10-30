"""ROVER: combine several systems' transcripts by word-level voting.

Each transcript is split into newmm words. The systems are aligned one by one into a
word network: a list of slots, each mapping a word to its total vote weight
("" means "no word here"). Each slot then keeps its highest-voted word, and a slot
whose winner is "" is dropped.

Usage:
  python rover.py out.csv sysA.csv:1.0 sysB.csv:0.8 ...
Each input CSV has columns audio and hyp (ref is carried over from the first if present).
The first system is the backbone and wins ties, so list the best system first.
"""
import csv
import os
import sys

from pythainlp.tokenize import word_tokenize


def words(text):
    return [w for w in word_tokenize(str(text or ""), engine="newmm", keep_whitespace=False) if w.strip()]


def align(slots, tokens):
    """Edit-distance alignment of tokens to slots. A token already present in a slot matches
    for free. Returns a list of (slot index or None for a new slot, token or "")."""
    n, m = len(slots), len(tokens)
    hit = lambda i, j: 0 if tokens[j] in slots[i] else 1
    cost = [[i + j if i == 0 or j == 0 else 0 for j in range(m + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j - 1] + hit(i - 1, j - 1), cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    path, i, j = [], n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + hit(i - 1, j - 1):
            path.append((i - 1, tokens[j - 1]))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            path.append((i - 1, ""))      # this system has no word in an existing slot
            i -= 1
        else:
            path.append((None, tokens[j - 1]))  # a word no earlier system had
            j -= 1
    return path[::-1]


def combine(hyps, weights):
    """hyps: one transcript per system (best first); weights: vote weight per system."""
    slots = [{w: weights[0]} for w in words(hyps[0])]
    for k in range(1, len(hyps)):
        earlier = sum(weights[:k])
        new_slots = []
        for idx, token in align(slots, words(hyps[k])):
            if idx is None:
                new_slots.append({"": earlier, token: weights[k]})
            else:
                slot = dict(slots[idx])
                slot[token] = slot.get(token, 0.0) + weights[k]
                new_slots.append(slot)
        slots = new_slots
    # dicts keep insertion order and max() returns the first maximum, so ties go to the earliest system
    return " ".join(w for w in (max(s.items(), key=lambda kv: kv[1])[0] for s in slots) if w)


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        return {os.path.basename(r["audio"]): r for r in csv.DictReader(f)}


def main():
    out_path, specs = sys.argv[1], sys.argv[2:]
    systems = [(load(p), float(w)) for p, w in (s.rsplit(":", 1) for s in specs)]
    first = systems[0][0]
    has_ref = "ref" in next(iter(first.values()))
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        out = csv.writer(f)
        out.writerow(["audio", "ref", "hyp"] if has_ref else ["audio", "hyp"])
        for clip, row in first.items():
            hyp = combine([s[clip]["hyp"] for s, _ in systems], [w for _, w in systems])
            out.writerow([row["audio"], row["ref"], hyp] if has_ref else [row["audio"], hyp])
    print("wrote", out_path, len(first), "clips")


if __name__ == "__main__":
    main()
