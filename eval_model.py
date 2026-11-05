#!/usr/bin/env python3
"""Evaluate a pisikopa.modoki model: holdout accuracy, banana-20,
and a seeded sample of training prompts (all EN->PI, greedy).

Usage:
    python3 eval_model.py --model pisikopa.modoki [--train-sample 500]
Prompts come from holdout.tsv / corpus.tsv in this directory.
banana20 is rebuilt from the same 20 (verb, evidential) combos as the
v0.2 baseline (TRAINING.md), but in the normalised v2 format.
"""
import argparse
import os
import random
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/workspace/lmm-modoki"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modoki import parse_modoki  # noqa: E402
from generate_pi import generate  # noqa: E402
from gen_corpus import en_evidential  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
BANANA20 = [("eat", "ta"), ("drink", "te"), ("see", "to"), ("read", "tə"),
            ("buy", "tɔ"), ("eat", "ti"), ("drink", "tu"), ("see", "tæ"),
            ("read", "ta"), ("buy", "te"), ("eat", "tɔ"), ("drink", "tə"),
            ("see", "ta"), ("read", "te"), ("buy", "ti"), ("eat", "tu"),
            ("drink", "tɔ"), ("see", "tə"), ("read", "tu"), ("buy", "tæ")]
VMAP = {"eat": ("ra", "eating"), "drink": ("ri", "drinking"),
        "see": ("ro", "seeing"), "read": ("kæro", "reading"),
        "buy": ("kiro", "buying")}
SUBJ = [("i", "pa"), ("you", "pi"), ("he", "pu"), ("we", "pe")]


def banana20_rows():
    rows = []
    for i, (v, ev) in enumerate(BANANA20):
        s, sp = SUBJ[i % 4]
        p, g = VMAP[v]
        rows.append((en_evidential(s, v, g, "banana", ev), f"kasæ {sp} {p}-{ev}"))
    return rows


def load_tsv(name):
    rows = []
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                rows.append(tuple(line.split("\t")))
    return rows


def evaluate(m, rows, label, show=8):
    ok, oov, fails = 0, 0, []
    for en, exp in rows:
        try:
            got = generate(m, f"{en} =", tokens=12)
        except ValueError as e:
            oov += 1
            fails.append((en, exp, f"OOV: {e}"))
            continue
        if got.strip() == exp.strip():
            ok += 1
        else:
            fails.append((en, exp, got))
    n = len(rows)
    print(f"{label}: {ok}/{n} = {ok / n * 100:.1f}%  (oov {oov}, wrong {n - ok - oov})")
    for en, exp, got in fails[:show]:
        print(f"  FAIL {en!r}\n       exp {exp!r}\n       got {got!r}")
    return ok, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "pisikopa.modoki"))
    ap.add_argument("--train-sample", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = parse_modoki(args.model)
    print(f"model: {args.model}")
    evaluate(m, load_tsv("holdout.tsv"), "holdout (kasæ+sell, unseen combo)")
    evaluate(m, banana20_rows(), "banana20 (same combos as v0.2 baseline)")
    train = load_tsv("corpus.tsv")
    rng = random.Random(args.seed)
    sample = rng.sample(train, min(args.train_sample, len(train)))
    evaluate(m, sample, f"train sample (seed {args.seed})", show=5)


if __name__ == "__main__":
    main()
