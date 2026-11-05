#!/usr/bin/env python3
"""pisikopa.modoki を読み、英語プロンプトから Pisikopa を貪欲生成する。

単語レベル: プロンプトは空白区切りで語彙と照合する。プロンプトは
translator.normalize_en で正規化される（小文字化、句読点除去、
三人称単数 -> 原形）。正規化後も語彙にない単語があるとエラーで
一覧を出す (学習データにない単語は生成できない、という実験上の
制限。未知語の Pisikopa 語は translator.py vec で最近傍を調べる)。
modoki.py のパーサ / forward を流用し、トークナイザだけ単語版に差し替えた。

使い方:
    python3 generate_pi.py --prompt "I eat an apple. =" [--tokens 10]
    python3 generate_pi.py --prompt "I eat an apple. =" --temp 0.9 --seed 1
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/workspace/lmm-modoki"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from modoki import forward, parse_modoki, softmax  # noqa: E402
from translator import normalize_en  # noqa: E402


def generate(m, prompt, tokens=10, temp=0.0, seed=0):
    vocab = m["vocab"]
    idx = {t: i for i, t in enumerate(vocab)}
    # Normalise the prompt the same way the corpus is stored:
    # lowercase, punctuation stripped ("You" -> "you", "sea." -> "sea",
    # "reads" -> "read"). What is still unknown after this is a
    # genuinely unknown word, not a surface variant.
    prompt = normalize_en(prompt)
    words = prompt.split(" ")
    unknown = [w for w in words if w not in idx]
    if unknown:
        raise ValueError(
            f"語彙にない単語: {unknown}（正規化済み。未知語は生成できません。"
            "translator.py vec <単語> で一番近い Pisikopa 語を確認してください）")
    ids = [idx["<bos>"]] + [idx[w] for w in words]
    rng = np.random.default_rng(seed)
    out = list(ids)
    for _ in range(tokens):
        ctx = out[-m["model"]["max_len"]:]
        logits = forward(m, ctx)
        if temp == 0:
            nxt = int(np.argmax(logits))
        else:
            z = logits / temp
            p = softmax(z)
            nxt = int(rng.choice(len(p), p=p))
        out.append(nxt)
        if vocab[nxt] == "<eos>":
            break
    return " ".join(vocab[i] for i in out[len(ids):] if vocab[i] != "<eos>")


def main():
    ap = argparse.ArgumentParser(description="pisikopa.modoki から貪欲生成")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pisikopa.modoki"))
    ap.add_argument("--prompt", required=True, help='例: "I eat an apple. ="')
    ap.add_argument("--tokens", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.0, help="0 なら貪欲法")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = parse_modoki(args.model)
    try:
        text = generate(m, args.prompt, args.tokens, args.temp, args.seed)
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    print(f"{args.prompt} {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
