#!/usr/bin/env python3
"""Generate corpus.tsv + holdout.tsv (v2, 2026-11-05).

v2 changes vs v1 (1114 pairs):
- English side is NORMALISED: lowercase, no periods/commas/quotes,
  "?" / "!" kept as separate tokens ("did i eat an apple ?").
  This is the same normalisation generate_pi.py / translator.py apply
  to user input, so "You see the sea." and "He reads a book." no longer
  die on OOV ("You", "sea.", "He", "reads").
- Objects: all 69 lexicon nouns (was: 4). Verbs: 8 transitive + 6
  intransitive (was: 5 + 4). Subjects: 6 incl. she/they (was: 4).
  Colours/numbers appear as objects of "see".
- HOLDOUT: kasæ (banana) + sell, all 6 subjects x 8 evidentials
  (48 pairs) is EXCLUDED from corpus.tsv and written to holdout.tsv.
  The model sees kasæ with 7 other verbs and sell with all other
  objects, but never the kasæ+sell combination. That is the
  compositional-generalisation test (baseline: old model 0/32 on the
  4-subject subset, all OOV on "sell"/"selling").
- Collision check: same EN -> different PI is an error; same PI from
  different EN is an error EXCEPT he/she (Pisikopa has one pronoun pu
  for both, so "he" and "she" rows legitimately share a PI sentence).
  The "who ... ?" form is deliberately NOT generated: it collides
  with yes/no "did he ... ?" (both -> "ka pu ra-ta lo"), a real
  ambiguity in Pisikopa, not a corpus bug.

Run: python3 gen_corpus.py  -> writes corpus.tsv + holdout.tsv,
prints per-category counts and a duplicate / collision report.
Exit 1 if any (non-allowed) collision remains.
"""
import itertools
import re
import sys
from collections import Counter

SUBJ = [("i", "pa"), ("you", "pi"), ("he", "pu"), ("she", "pu"),
        ("we", "pe"), ("they", "po")]
SUBJ_OBJ = {"i": "me", "you": "you", "he": "him", "she": "her",
            "we": "us", "they": "them"}
TRANS = [("eat", "ra", "eating"), ("drink", "ri", "drinking"),
         ("see", "ro", "seeing"), ("read", "kæro", "reading"),
         ("buy", "kiro", "buying"), ("sell", "kuro", "selling"),
         ("write", "kærɔ", "writing"), ("make", "rɔ", "making")]
INTRANS = [("go", "ru", "going"), ("come", "re", "coming"),
           ("sleep", "lulo", "sleeping"), ("wake", "laro", "waking"),
           ("run", "ruru", "running"), ("swim", "siru", "swimming")]
CORE_OBJS = [("apple", "ka"), ("water", "ke"), ("book", "kæ"),
             ("rice", "ki"), ("banana", "kasæ"), ("sea", "si"),
             ("bread", "kiki"), ("fruit", "kaka")]
# Every other lexicon noun (69 nouns total - 8 core = 61).
EXT_OBJS = [
    ("meat", "ku"), ("fire", "ko"), ("house", "kə"), ("road", "kɔ"),
    ("sky", "sa"), ("mountain", "su"), ("forest", "se"), ("rain", "so"),
    ("moon", "sə"), ("sun", "sæ"), ("star", "sɔ"),
    ("hand", "ma"), ("eye", "mi"), ("ear", "mu"), ("mouth", "me"),
    ("foot", "mo"), ("heart", "mə"), ("pain", "mæ"), ("head", "mɔ"),
    ("cat", "sumi"), ("dog", "sumu"), ("bird", "susa"), ("fish", "susi"),
    ("horse", "suru"), ("cow", "sumo"), ("insect", "suso"), ("snake", "sulo"),
    ("school", "kælə"), ("shop", "kəkɔ"), ("hospital", "mækə"),
    ("park", "sekə"), ("beach", "sisu"), ("airport", "sako"),
    ("library", "kækə"),
    ("mother", "pama"), ("father", "papa"), ("child", "paki"),
    ("friend", "papi"), ("teacher", "pakæ"), ("doctor", "pamæ"),
    ("king", "paro"), ("god", "pasa"),
    ("vegetable", "kuku"), ("salt", "keke"), ("alcohol", "koko"),
    ("snack", "kækæ"), ("noodle", "kiku"),
    ("snow", "susu"), ("wind", "saro"), ("thunder", "saka"),
    ("cloud", "saku"), ("river", "sisi"), ("lake", "siso"),
    ("phone", "ræro"), ("car", "ruko"), ("train", "rusi"),
    ("computer", "lærɔ"), ("smartphone", "miko"), ("camera", "miro"),
    ("clock", "lolo"), ("notebook", "kæki"),
]
COLOUR_NUM = [("red", "sæko"), ("blue", "sæsi"), ("green", "sæse"),
              ("yellow", "sæsæ"), ("white", "sæsa"), ("black", "sæso"),
              ("purple", "sæsɔ"), ("orange", "sæka"),
              ("one", "na"), ("two", "ni"), ("three", "nu"), ("four", "ne"),
              ("five", "no"), ("six", "nə"), ("seven", "næ"), ("eight", "nɔ")]
EVID = ["ta", "ti", "tu", "te", "to", "tə", "tæ", "tɔ"]
TIMES = [("today", "la"), ("tomorrow", "li"), ("yesterday", "lu"),
         ("now", "le")]


def art(word):
    if word == "one":
        return "a"
    return "an" if word[0] in "aeiou" else "a"


def en_evidential(subj, verb, ger, obj, ev):
    """Normalised English for one (subj, verb, obj, evidential) combo.
    obj=None for intransitives."""
    s = subj
    o = f" {art(obj)} {obj}" if obj else ""
    if ev == "ta":
        return f"{s} {verb}{o}"
    if ev == "ti":
        return f"i heard the sound of {SUBJ_OBJ[s]} {ger}{o}"
    if ev == "tu":
        return f"i felt {SUBJ_OBJ[s]} {ger}{o}"
    if ev == "te":
        return f"maybe {s} {verb}{o}"
    if ev == "to":
        return f"i heard that {s} {verb}{o}"
    if ev == "tə":
        return f"in a dream {s} {verb}{o}"
    if ev == "tæ":
        return f"{s} {verb}{o} someone said"
    if ev == "tɔ":
        return f"everyone knows {s} {verb}{o}"
    raise ValueError(ev)


def en_key(en):
    """Collision key: he/she (and him/her) are the same Pisikopa
    pronoun, so they must not count as a PI collision."""
    en = re.sub(r"\b(she|he)\b", "PRON3", en)
    en = re.sub(r"\b(her|him)\b", "OBJ3", en)
    return en


def main():
    pairs = []    # (en, pi, category) -> corpus.tsv
    holdout = []  # (en, pi, category) -> holdout.tsv

    def add(en, pi, cat, held_out=False):
        (holdout if held_out else pairs).append((en, pi, cat))

    # 1. transitive core, all evidentials: 6 x 8 x 8 x 8 = 3072
    #    minus kasæ+sell holdout (6 x 8 = 48).
    for (se, sp), (ve, vp, ger), (oe, op), ev in itertools.product(
            SUBJ, TRANS, CORE_OBJS, EVID):
        held = (oe == "banana" and ve == "sell")
        add(en_evidential(se, ve, ger, oe, ev), f"{op} {sp} {vp}-{ev}",
            "transitive-core", held_out=held)
    # 2. transitive extended objects: each object gets 2 rotating verbs
    #    x 2 rotating evidentials x all 6 subjects = 24 per object.
    for j, (oe, op) in enumerate(EXT_OBJS):
        verbs = [TRANS[j % len(TRANS)], TRANS[(j + 3) % len(TRANS)]]
        evs = [EVID[j % len(EVID)], EVID[(j + 5) % len(EVID)]]
        for (se, sp), (ve, vp, ger), ev in itertools.product(SUBJ, verbs, evs):
            add(en_evidential(se, ve, ger, oe, ev), f"{op} {sp} {vp}-{ev}",
                "transitive-extended")
    # 3. intransitive, all evidentials: 6 x 6 x 8 = 288
    for (se, sp), (ve, vp, ger), ev in itertools.product(SUBJ, INTRANS, EVID):
        add(en_evidential(se, ve, ger, None, ev), f"{sp} {vp}-{ev}",
            "intransitive")
    # 4. negation (ta): 6 x 4 verbs x 8 core objs = 192
    for (se, sp), (ve, vp, _g), (oe, op) in itertools.product(
            SUBJ, TRANS[:4], CORE_OBJS):
        add(f"{se} did not {ve} {art(oe)} {oe}", f"{op} {sp} mæ {vp}-ta",
            "negation")
    # 5. yes/no question (ta): 6 x 4 x 8 = 192
    for (se, sp), (ve, vp, _g), (oe, op) in itertools.product(
            SUBJ, TRANS[:4], CORE_OBJS):
        add(f"did {se} {ve} {art(oe)} {oe} ?", f"{op} {sp} {vp}-ta lo",
            "question-yesno")
    # 6. wh-question "what" (ta): 6 subjects x 4 verbs = 24
    #    ("who ... ?" is NOT generated: it collides with yes/no
    #    "did he ... ?" -> same PI "ka pu ra-ta lo".)
    for (se, sp), (ve, vp, _g) in itertools.product(SUBJ, TRANS[:4]):
        add(f"what did {se} {ve} ?", f"lɔ {sp} {vp}-ta lo", "question-wh")
    # 7. imperative: 8 verbs x 8 core objs = 64, + 6 bare intransitives
    for (ve, vp, _g), (oe, op) in itertools.product(TRANS, CORE_OBJS):
        add(f"{ve} {art(oe)} {oe} !", f"{op} {vp} rɔ", "imperative")
    for (ve, vp, _g) in INTRANS:
        add(f"{ve} !", f"{vp} rɔ", "imperative-intrans")
    # 8. want: 6 subj x 4 verbs x 8 objs x ev ta/te = 384
    for (se, sp), (ve, vp, _g), (oe, op), ev in itertools.product(
            SUBJ, TRANS[:4], CORE_OBJS, ["ta", "te"]):
        maybe = "might " if ev == "te" else ""
        add(f"{se} {maybe}want to {ve} {art(oe)} {oe}",
            f"{op} {sp} mə{vp}-{ev}", "want")
    # 9. time adverbs (ta): 6 x 2 verbs x 8 objs x 4 times = 384
    for (se, sp), (ve, vp, _g), (oe, op), (te_en, te_pi) in itertools.product(
            SUBJ, TRANS[:2], CORE_OBJS, TIMES):
        add(f"{se} {ve} {art(oe)} {oe} {te_en}",
            f"{op} {te_pi} {sp} {vp}-ta", "time")
    # 10. embedding saw-that: 3 subj2 x 3 verbs x 4 objs = 36
    for (se2, sp2), (ve, vp, _g), (oe, op) in itertools.product(
            [SUBJ[1], SUBJ[2], SUBJ[5]], TRANS[:3], CORE_OBJS[:4]):
        # subj2 = you, he, they (skip she: same PI as he)
        add(f"i saw that {se2} {ve} {art(oe)} {oe}",
            f"{op} {sp2} {vp}-ta læ pa ro-ta", "embedding-saw-that")
    # 11. embedding gerund (step on + cry): 3 objs x 3 subj2 x 2 = 18
    for (oe, op) in [("banana", "kasæ"), ("apple", "ka"), ("sea", "si")]:
        for (se2, sp2) in [SUBJ[1], SUBJ[2], SUBJ[5]]:
            add(f"i saw {SUBJ_OBJ[se2]} stepping on {art(oe)} {oe} and crying",
                f"{op} {sp2} moru-ta {sp2} məso-ta læ pa ro-ta",
                "embedding-gerund")
            add(f"i saw {SUBJ_OBJ[se2]} stepping on {art(oe)} {oe}",
                f"{op} {sp2} moru-ta læ pa ro-ta", "embedding-gerund")
    # 12. see + colour/number (ta): 16 x 6 = 96
    for (oe, op), (se, sp) in itertools.product(COLOUR_NUM, SUBJ):
        add(f"{se} see {oe}", f"{op} {sp} ro-ta", "see-colour-number")

    # deduplicate exact pairs, then check collisions
    def dedup(rows):
        seen, uniq = set(), []
        for en, pi, cat in rows:
            if (en, pi) not in seen:
                seen.add((en, pi))
                uniq.append((en, pi, cat))
        return uniq
    uniq = dedup(pairs)
    uniq_hold = dedup(holdout)
    all_rows = uniq + uniq_hold
    en_to_pi, pi_to_en, collisions = {}, {}, []
    for en, pi, _cat in all_rows:
        if en in en_to_pi and en_to_pi[en] != pi:
            collisions.append(f"EN collision: {en!r} -> {en_to_pi[en]!r} and {pi!r}")
        key = en_key(en)
        if pi in pi_to_en and en_key(pi_to_en[pi]) != key:
            collisions.append(f"PI collision: {pi!r} -> {pi_to_en[pi]!r} and {en!r}")
        en_to_pi[en] = pi
        pi_to_en[pi] = en
    counts = Counter(c for _, _, c in uniq)
    print(f"total pairs: {len(uniq)} (train) + {len(uniq_hold)} (holdout)")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"holdout: {Counter(c for _, _, c in uniq_hold)}")
    print(f"collisions: {len(collisions)}")
    for c in collisions[:20]:
        print(" ", c)
    with open("corpus.tsv", "w", encoding="utf-8") as f:
        for en, pi, _cat in uniq:
            f.write(f"{en}\t{pi}\n")
    with open("holdout.tsv", "w", encoding="utf-8") as f:
        for en, pi, _cat in uniq_hold:
            f.write(f"{en}\t{pi}\n")
    print("wrote corpus.tsv + holdout.tsv")
    if collisions:
        sys.exit(1)


if __name__ == "__main__":
    main()
