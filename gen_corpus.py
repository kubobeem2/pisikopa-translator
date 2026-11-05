#!/usr/bin/env python3
"""Generate corpus.tsv (~1000 pairs) from templates.
Evidential markers on the English side are systematic, so each
English sentence maps to exactly one Pisikopa sentence:
  ta plain:            "I eat an apple."
  ti heard-sound:      "I heard the sound of you eating an apple."
  tu felt:             "I felt you eating an apple."
  te maybe:            "Maybe I eat an apple."
  to hearsay:          "I heard that you eat an apple."
  tə dream:            "In a dream, I eat an apple."
  tæ quoted:           '"I eat an apple," someone said.'
  tɔ common:           "Everyone knows I eat an apple."
Run: python3 gen_corpus.py  -> writes corpus.tsv + prints a
duplicate / collision report. Exit 1 if any collision remains.
"""
import itertools
import sys
from collections import Counter

SUBJ = [("I", "pa"), ("you", "pi"), ("he", "pu"), ("we", "pe")]
SUBJ_OBJ = {"I": "me", "you": "you", "he": "him", "we": "us"}
TRANS = [("eat", "ra", "eating"), ("drink", "ri", "drinking"),
         ("see", "ro", "seeing"), ("read", "kæro", "reading"),
         ("buy", "kiro", "buying")]
OBJS = [("apple", "ka"), ("water", "ke"), ("book", "kæ"), ("rice", "ki")]
INTRANS = [("go", "ru", "going"), ("sleep", "lulo", "sleeping"),
           ("run", "ruru", "running"), ("swim", "siru", "swimming")]
EVID = ["ta", "ti", "tu", "te", "to", "tə", "tæ", "tɔ"]


def en_evidential(subj, verb, ger, obj, ev):
    s = subj
    if ev == "ta":
        return f"{s} {verb} an {obj}." if obj else f"{s} {verb}."
    if ev == "ti":
        return f"I heard the sound of {SUBJ_OBJ[s]} {ger} an {obj}." if obj else f"I heard the sound of {SUBJ_OBJ[s]} {ger}."
    if ev == "tu":
        return f"I felt {SUBJ_OBJ[s]} {ger} an {obj}." if obj else f"I felt {SUBJ_OBJ[s]} {ger}."
    if ev == "te":
        return f"Maybe {s} {verb} an {obj}." if obj else f"Maybe {s} {verb}."
    if ev == "to":
        return f"I heard that {s} {verb} an {obj}." if obj else f"I heard that {s} {verb}."
    if ev == "tə":
        return f"In a dream, {s} {verb} an {obj}." if obj else f"In a dream, {s} {verb}."
    if ev == "tæ":
        core = f"{s} {verb} an {obj}" if obj else f"{s} {verb}"
        return f'"{core}," someone said.'
    if ev == "tɔ":
        return f"Everyone knows {s} {verb} an {obj}." if obj else f"Everyone knows {s} {verb}."
    raise ValueError(ev)


def main():
    pairs = []  # (en, pi, category)

    def add(en, pi, cat):
        pairs.append((en, pi, cat))

    # 1. transitive, all evidentials: 4 x 5 x 4 x 8 = 640
    for (se, sp), (ve, vp, ger), (oe, op), ev in itertools.product(SUBJ, TRANS, OBJS, EVID):
        add(en_evidential(se, ve, ger, oe, ev), f"{op} {sp} {vp}-{ev}", "transitive")
    # 2. intransitive, all evidentials: 4 x 4 x 8 = 128
    for (se, sp), (ve, vp, ger), ev in itertools.product(SUBJ, INTRANS, EVID):
        add(en_evidential(se, ve, ger, None, ev), f"{sp} {vp}-{ev}", "intransitive")
    # 3. negation (ta only): 4 x 3 verbs x 4 objs = 48
    for (se, sp), (ve, vp, _g), (oe, op) in itertools.product(SUBJ, TRANS[:3], OBJS):
        add(f"{se} did not {ve} an {oe}.", f"{op} {sp} mæ {vp}-ta", "negation")
    # 4. yes/no question (ta): 4 x 3 x 4 = 48
    for (se, sp), (ve, vp, _g), (oe, op) in itertools.product(SUBJ, TRANS[:3], OBJS):
        add(f"Did {se} {ve} an {oe}?", f"{op} {sp} {vp}-ta lo", "question-yesno")
    # 5. wh-question (ta): 4 subjects x 3 verbs = 12
    for (se, sp), (ve, vp, _g) in itertools.product(SUBJ, TRANS[:3]):
        add(f"What did {se} {ve}?", f"lɔ {sp} {vp}-ta lo", "question-wh")
    # 6. imperative: 5 verbs x 4 objs = 20
    for (ve, vp, _g), (oe, op) in itertools.product(TRANS, OBJS):
        add(f"{ve.capitalize()} an {oe}!", f"{op} {vp} rɔ", "imperative")
    # 7. want: subjects I/you/he x verbs eat/drink/go/see x objs, ev ta/te = 96
    for (se, sp), (ve, vp, _g), (oe, op), ev in itertools.product(
            SUBJ[:3], TRANS[:4], OBJS, ["ta", "te"]):
        maybe = "might " if ev == "te" else ""
        add(f"{se} {maybe}want to {ve} an {oe}.", f"{op} {sp} mə{vp}-{ev}", "want")
    # 8. time adverbs (ta): subjects x verbs eat/drink x objs x 3 times = 96
    #    (kept to 2 verbs to stay near 1000 total; see counts below)
    for (se, sp), (ve, vp, _g), (oe, op), (te_en, te_pi) in itertools.product(
            SUBJ, TRANS[:2], OBJS, [("today", "la"), ("tomorrow", "li"), ("yesterday", "lu")]):
        add(f"{se} {ve} an {oe} {te_en}.", f"{op} {te_pi} {sp} {vp}-ta", "time")
    # 9. embedding (læ): saw-that, 3 subj2 x 3 verbs x 2 objs = 18
    for (se2, sp2), (ve, vp, _g), (oe, op) in itertools.product(SUBJ[1:], TRANS[:3], OBJS[:2]):
        add(f"I saw that {se2} {ve} an {oe}.", f"{op} {sp2} {vp}-ta læ pa ro-ta", "embedding-saw-that")
    # 10. embedding gerund (banana/cry example + variants): 12
    for (oe, op) in [("banana", "kasæ"), ("apple", "ka")]:
        for (se2, sp2) in SUBJ[1:3]:
            add(f"I saw {SUBJ_OBJ[se2]} stepping on an {oe} and crying.",
                f"{op} {sp2} moru-ta {sp2} məso-ta læ pa ro-ta", "embedding-gerund")
            add(f"I saw {SUBJ_OBJ[se2]} stepping on an {oe}.",
                f"{op} {sp2} moru-ta læ pa ro-ta", "embedding-gerund")

    # deduplicate exact pairs, then check collisions
    seen, uniq = set(), []
    for en, pi, cat in pairs:
        if (en, pi) not in seen:
            seen.add((en, pi))
            uniq.append((en, pi, cat))
    en_to_pi = {}
    pi_to_en = {}
    collisions = []
    for en, pi, _cat in uniq:
        if en in en_to_pi and en_to_pi[en] != pi:
            collisions.append(f"EN collision: {en!r} -> {en_to_pi[en]!r} and {pi!r}")
        if pi in pi_to_en and pi_to_en[pi] != en:
            collisions.append(f"PI collision: {pi!r} -> {pi_to_en[pi]!r} and {en!r}")
        en_to_pi[en] = pi
        pi_to_en[pi] = en
    counts = Counter(c for _, _, c in uniq)
    print(f"total pairs: {len(uniq)}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"collisions: {len(collisions)}")
    for c in collisions[:20]:
        print(" ", c)
    with open("corpus.tsv", "w", encoding="utf-8") as f:
        for en, pi, _cat in uniq:
            f.write(f"{en}\t{pi}\n")
    print("wrote corpus.tsv")
    if collisions:
        sys.exit(1)


if __name__ == "__main__":
    main()
