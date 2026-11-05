#!/usr/bin/env python3
"""Pisikopa翻訳機の lmm風学習: 単語レベル tiny transformer を勾配学習で鍛える。

lmm-modoki/train.py の Net (手書き逆伝播 + Adam) と fit.py の fmt を流用。
文字レベルではなく空白区切りの単語レベル。学習データは corpus.tsv の各行を
「英語 = Pisikopa」の1シーケンスにしたもの。できた .modoki は modoki.py の
パーサが読めるテキスト形式 (単語が語彙に入っている)。

使い方:
    python3 train_pi.py [--steps 800] [--lr 0.005] [--d 48] [--layers 2]
    python3 train_pi.py --help

出力: pisikopa.modoki (同ディレクトリ)
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/workspace/lmm-modoki"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from fit import fmt  # noqa: E402
from train import Net  # noqa: E402


def load_seqs(corpus_path):
    seqs = []
    vocab_set = set()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            en, pi = line.split("\t")
            seq = f"{en} = {pi}"
            toks = seq.split(" ")
            vocab_set.update(toks)
            seqs.append(toks)
    vocab = ["<bos>", "<eos>", "<pad>"] + sorted(vocab_set)
    idx = {t: i for i, t in enumerate(vocab)}
    idseqs = [[idx["<bos>"]] + [idx[t] for t in s] + [idx["<eos>"]] for s in seqs]
    return vocab, idx, idseqs


def write_modoki(net, vocab, out_path):
    P = net.P
    out = ["# lmmもどき 1 (Pisikopa translator, word-level)", "[model]",
           f"d_model = {net.cfg['d']}", f"n_layer = {net.cfg['n_layer']}",
           f"n_head = {net.cfg['n_head']}", f"d_ff = {net.cfg['d_ff']}",
           f"max_len = {net.cfg['max_len']}", "",
           "[vocab]"] + [json.dumps(t, ensure_ascii=False) for t in vocab] + ["",
           "[embed]", fmt(P["embed"]), "",
           "[pos]", fmt(P["pos"])]
    for i in range(net.cfg["n_layer"]):
        p = f"L{i}_"
        out.append("")
        out.append(f"[layer {i}]")
        for name in ("ln1_w", "ln1_b", "attn_q_w", "attn_q_b", "attn_k_w",
                     "attn_k_b", "attn_v_w", "attn_v_b", "attn_o_w", "attn_o_b",
                     "ln2_w", "ln2_b", "mlp1_w", "mlp1_b", "mlp2_w", "mlp2_b"):
            out.append(f"{name}:")
            out.append(fmt(P[p + name]))
    out += ["", "[head]", "ln_f_w:", fmt(P["ln_f_w"]), "ln_f_b:", fmt(P["ln_f_b"]),
            "lm_head_w:", fmt(P["lm_head_w"]), "lm_head_b:", fmt(P["lm_head_b"]), ""]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description="Pisikopa 翻訳機 lmm風学習 (単語レベル)")
    ap.add_argument("--corpus", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.tsv"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pisikopa.modoki"))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--d", type=int, default=48)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    vocab, idx, seqs = load_seqs(args.corpus)
    V, pad = len(vocab), idx["<pad>"]
    max_len = max(len(s) for s in seqs)
    print(f"ペア数: {len(seqs)}, 語彙数: {V}, 最大長: {max_len} トークン")
    print(f"d_model={args.d}, n_layer={args.layers}, steps={args.steps}, lr={args.lr}")

    net = Net(V, d=args.d, n_layer=args.layers, n_head=3, d_ff=96,
              max_len=64, seed=args.seed)
    P = net.P
    mt = {k: np.zeros_like(v) for k, v in P.items()}
    vt = {k: np.zeros_like(v) for k, v in P.items()}
    rng = np.random.default_rng(1)
    step = 0
    first_loss = None
    loss = float("nan")
    for epoch in range(10000):
        order = rng.permutation(len(seqs))
        for bi in range(0, len(seqs), 16):
            batch = [seqs[j] for j in order[bi:bi + 16]]
            T = max(len(s) for s in batch)
            B = len(batch)
            ids = np.full((B, T), pad, dtype=int)
            lengths = np.array([len(s) for s in batch])
            for b, s in enumerate(batch):
                ids[b, :len(s)] = s
            x_ids, y_ids = ids[:, :-1], ids[:, 1:]
            y_lengths = lengths - 1
            logits, cache = net.forward(x_ids, y_lengths)
            valid = (np.arange(x_ids.shape[1])[None, :] < y_lengths[:, None])
            lp = logits - logits.max(axis=-1, keepdims=True)
            lp = lp - np.log(np.exp(lp).sum(axis=-1, keepdims=True))
            nll = -lp[np.arange(B)[:, None], np.arange(x_ids.shape[1])[None, :], y_ids]
            loss = (nll * valid).sum() / valid.sum()
            if first_loss is None:
                first_loss = float(loss)
            dlogits = np.exp(lp)
            dlogits[np.arange(B)[:, None], np.arange(x_ids.shape[1])[None, :], y_ids] -= 1.0
            dlogits *= (valid / valid.sum())[:, :, None]
            G = net.backward(dlogits, cache)
            step += 1
            b1, b2, eps = 0.9, 0.999, 1e-8
            for k in P:
                mt[k] = b1 * mt[k] + (1 - b1) * G[k]
                vt[k] = b2 * vt[k] + (1 - b2) * G[k] ** 2
                mh = mt[k] / (1 - b1**step)
                vh = vt[k] / (1 - b2**step)
                P[k] -= args.lr * mh / (np.sqrt(vh) + eps)
            if step % 100 == 0:
                print(f"step {step} loss {loss:.3f}", flush=True)
            if step >= args.steps:
                break
        if step >= args.steps:
            break
    print(f"初期 loss {first_loss:.3f} -> 最終 loss {loss:.3f} (step {step})")
    write_modoki(net, vocab, args.out)
    print(f"書き出し: {args.out}")


if __name__ == "__main__":
    main()
