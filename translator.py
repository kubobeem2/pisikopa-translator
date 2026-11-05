#!/usr/bin/env python3
"""Pisikopa <-> English translator, v0.2
Rule-based + vector (shared embedding space) hybrid.

Grammar (see ~/workspace/conlang/SPEC.md):
- OSV, no tense, evidential suffix mandatory on statements:
  ta saw / ti heard-sound / tu felt / te inferred-maybe / to hearsay /
  tә dream / tæ quoted / tɔ common-knowledge
- Negation: function word mæ immediately before the verb.
- Question: function word lo after the evidential. Wh-word = the
  corresponding root in function position at the front
  (lɔ what, pu who, lə where/when).
- Imperative: no subject, no evidential. Object - verb - rɔ.
- Want: mə + verb (məra want-to-eat, məri want-to-drink, ...).
- Embedding: clause(s) + læ ("that/thing") as the object, then
  subject + main verb-evidential. læ is optional when the object
  position alone makes the clause the object.

Vector part: each Pisikopa root has a 16-dim vector
  [consonant-field one-hot (8) | vowel one-hot (8)].
  English words share the same space via lexicon_en.json.
  Unknown words fall back to nearest neighbour by cosine on the
  mean vector of their known parts, else string overlap (reported
  as a fallback, not as a cosine match).

Usage:
  python3 translator.py en "I eat an apple"
  python3 translator.py en "I might want to eat an apple"
  python3 translator.py en "Did I eat an apple?"
  python3 translator.py en "Eat an apple!"
  python3 translator.py pi "ka pa ra-ta"
  python3 translator.py pi "ka pa ra-ta lo"
  python3 translator.py pi "ka ra rɔ"
  python3 translator.py vec banana
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
EN_PI = json.loads((ROOT / "lexicon_en.json").read_text(encoding="utf-8"))
PI_EN = {}
for _k, _v in EN_PI.items():
    PI_EN.setdefault(_v, _k)
# Verb-context overrides: a root used as the verb means the action,
# even if its default gloss is a noun/emotion (e.g. məso = sad as
# adjective, cry as verb; ra = meal as noun, eat as verb).
VERB_EN = {
    "ra": "eat", "ri": "drink", "ru": "go", "re": "come", "ro": "see",
    "rə": "hear", "ræ": "speak", "rɔ": "make", "məso": "cry",
    "moru": "step", "kiro": "buy", "kuro": "sell", "lulo": "sleep",
    "laro": "wake", "ruru": "run", "siru": "swim", "kæro": "read",
    "kærɔ": "write",
}
CONS = ["p", "t", "k", "s", "m", "n", "l", "r"]
VOWELS = ["a", "i", "u", "e", "o", "ə", "æ", "ɔ"]
EVID = {
    "ta": "saw", "ti": "heard", "tu": "felt", "te": "inferred/maybe",
    "to": "hearsay", "tə": "dream", "tæ": "quoted", "tɔ": "common knowledge",
}
TIME_EN = {"la": "today", "li": "tomorrow", "lu": "yesterday", "le": "now"}
WH_EN = {"lɔ": "what", "pu": "who", "lə": "where/when"}
# English evidential markers -> Pisikopa suffix. Order matters:
# longer / more specific markers first.
EVID_MARKERS = [
    ("in a dream", "tə"), ("i dreamed", "tə"), ("dreamed", "tə"),
    ("everyone knows", "tɔ"), ("everybody knows", "tɔ"),
    ("someone said", "tæ"), ("quoted", "tæ"),
    ("they say", "to"),
    ("i heard the sound", "ti"), ("heard the sound", "ti"),
    ("i felt", "tu"), ("felt", "tu"),
    ("might", "te"), ("maybe", "te"), ("perhaps", "te"),
]
# Gerund -> base verb (English -ing forms used in embedded clauses)
GERUND = {
    "eating": "eat", "drinking": "drink", "going": "go", "coming": "come",
    "seeing": "see", "hearing": "hear", "speaking": "speak",
    "making": "make", "stepping": "step", "crying": "cry",
    "reading": "read", "writing": "write", "running": "run",
    "swimming": "swim", "sleeping": "sleep", "buying": "buy",
    "selling": "sell",
}
OBJ_PRON = {"him": "pu", "her": "pu", "me": "pa", "us": "pe", "them": "po", "you": "pi", "i": "pa", "he": "pu", "she": "pu", "we": "pe", "they": "po"}


def _syllables(word):
    """Split a Pisikopa word into CV syllables (2 chars each)."""
    return [word[i:i + 2] for i in range(0, len(word), 2)]


def vec_of_pi(syl):
    vecs = []
    for p in _syllables(syl):
        if len(p) < 2 or p[0] not in CONS or p[1] not in VOWELS:
            continue
        v = np.zeros(16)
        v[CONS.index(p[0])] = 1.0
        v[8 + VOWELS.index(p[1])] = 1.0
        vecs.append(v)
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def nearest(word):
    """Return (pisikopa, score, method)."""
    if word in EN_PI:
        return EN_PI[word], 1.0, "lexicon"
    # Compound guess: banana = fruit + yellow etc. is already in the
    # lexicon; for genuinely unknown words use cosine on a mean vector
    # built from any known English morphemes found inside the word.
    parts = [k for k in EN_PI if len(k) > 2 and k in word]
    vecs = [vec_of_pi(EN_PI[k]) for k in parts]
    vecs = [v for v in vecs if v is not None]
    if vecs:
        target = np.mean(vecs, axis=0)
        best, best_score = None, -1.0
        for pi in set(EN_PI.values()):
            v = vec_of_pi(pi)
            if v is None:
                continue
            s = float(target @ v / (np.linalg.norm(target) * np.linalg.norm(v)))
            if s > best_score:
                best, best_score = pi, s
        return best, round(best_score, 3), "cosine-compound"
    best, best_score = None, -1.0
    for k, v in EN_PI.items():
        s = len(set(word) & set(k)) / max(len(set(word) | set(k)), 1)
        if s > best_score:
            best, best_score = v, s
    return best, round(best_score, 3), "string-overlap-fallback"


def _stem(word):
    if word in GERUND:
        return GERUND[word]
    if word.endswith("ing") and word[:-3] in EN_PI:
        return word[:-3]
    if word.endswith("s") and word[:-1] in EN_PI:
        return word[:-1]
    return word


def _pi_of_en(word):
    word = _stem(word)
    if word in EN_PI:
        return EN_PI[word]
    got, _score, _m = nearest(word)
    return got


def en_to_pi(text):
    t = text.strip()
    is_q = t.endswith("?")
    t = t.rstrip("?.!").strip()
    low = t.lower()
    # --- quoted: '"I eat an apple," someone said.' -> inner clause, ev=tæ ---
    mq = re.match(r'"(.+?)",?\s+someone said\.?$', text.strip())
    if mq:
        inner = en_to_pi(mq.group(1))
        return re.sub(r"-(ta|ti|tu|te|to|tə|tæ|tɔ)(\s+lo)?$", r"-tæ\2", inner)
    # --- heard-sound / felt templates (corpus forms) ---
    # "I heard the sound of you eating an apple." / "I felt you eating an apple."
    mt = re.match(r"i heard the sound of (\w+) (\w+ing)(?: an? (.+))?\.?$", low)
    if mt:
        subj_pi = OBJ_PRON.get(mt.group(1), mt.group(1))
        verb_pi = _pi_of_en(_stem(mt.group(2)))
        obj_pi = _pi_of_en(mt.group(3).split()[0]) if mt.group(3) else None
        parts = ([obj_pi] if obj_pi else []) + [subj_pi, f"{verb_pi}-ti"]
        return " ".join(parts)
    mt = re.match(r"i felt (\w+) (\w+ing)(?: an? (.+))?\.?$", low)
    if mt:
        subj_pi = OBJ_PRON.get(mt.group(1), mt.group(1))
        verb_pi = _pi_of_en(_stem(mt.group(2)))
        obj_pi = _pi_of_en(mt.group(3).split()[0]) if mt.group(3) else None
        parts = ([obj_pi] if obj_pi else []) + [subj_pi, f"{verb_pi}-tu"]
        return " ".join(parts)
    # --- evidential markers (strip them, keep the suffix) ---
    ev = "ta"
    for marker, suffix in EVID_MARKERS:
        if marker in low:
            ev = suffix
            low = low.replace(marker, " ")
            break
    # "I heard that X" -> ev=to, clause is X (drop the reporting frame)
    if low.startswith("i heard that "):
        ev = "to"; low = low[len("i heard that "):]
    elif low.startswith("they say "):
        ev = "to"; low = low[len("they say "):]
    # --- imperative: "Eat an apple!" / "Please eat an apple" ---
    imperative = False
    if low.startswith("please "):
        low = low[len("please "):]
        imperative = True
    first = low.split()[0] if low.split() else ""
    if _stem(first) in ("eat", "drink", "see", "read", "go", "sleep",
                        "buy", "sell", "run", "swim", "write", "speak",
                        "make", "come", "hear", "step", "cry"):
        imperative = True
    # --- negation ---
    neg = bool(re.search(r"\b(not|never|n't|don't|doesn't|didn't)\b", low))
    low = re.sub(r"\b(don't|doesn't|didn't|do not|does not|did not|not|never)\b", " ", low)
    # --- auxiliaries (question / past) ---
    low = re.sub(r"^(did|do|does|will|would|can|could)\s+", "", low)
    # --- want to X ---
    want = None
    m = re.search(r"\bwant to (\w+)", low)
    if m:
        verb_en = _stem(m.group(1))
        want = "mə" + EN_PI.get(verb_en, "ra")
        low = low[:m.start()] + " WANT " + low[m.end():]
    # --- embedding: "I saw that ..." / "I saw him ...-ing ..." ---
    # Returns a læ-nominalised object clause, handled below.
    embed = None
    def _en_clause_to_pi(clause):
        cw = [w.strip(",") for w in clause.split() if w.strip(",") not in ("a", "an", "the")]
        if not cw:
            return ""
        s_en = OBJ_PRON.get(cw[0], cw[0]); v_en = _stem(cw[1]) if len(cw) > 1 else "eat"
        o_en = cw[2] if len(cw) > 2 else None
        s_pi = EN_PI.get(s_en, OBJ_PRON.get(s_en, s_en))
        v_pi = _pi_of_en(v_en)
        parts = ([_pi_of_en(o_en)] if o_en else []) + [s_pi, f"{v_pi}-ta"]
        return " ".join(parts)
    m = re.search(r"\bi saw that (.+)$", low)
    if m:
        embed = _en_clause_to_pi(m.group(1))
        low = "i see"
    m = re.search(r"\bi saw (\w+) (\w+ing) (?:on |in |at )?(.+)$", low)
    if m:
        subj2, ger, obj2 = m.group(1), m.group(2), m.group(3).strip()
        verb2 = _stem(ger)
        obj_words = [w for w in obj2.split() if w not in ("a", "an", "the", "and")]
        obj2_pi = _pi_of_en(obj_words[0]) if obj_words else None
        subj2_pi = OBJ_PRON.get(subj2, EN_PI.get(subj2, subj2))
        embed = f"{obj2_pi} {subj2_pi} {_pi_of_en(verb2)}-ta" if obj2_pi else f"{subj2_pi} {_pi_of_en(verb2)}-ta"
        low = "i see"
        m2 = re.search(r"\band (\w+ing|\w+)\.?$", text.lower())
        if m2:
            v2 = _stem(m2.group(1))
            embed += f" {subj2_pi} {_pi_of_en(v2)}-ta"
    words = [w.strip(",") for w in low.split() if w.strip(",")]
    words = [w for w in words if w not in ("a", "an", "the", "to", "on", "in", "at")]
    if embed is not None:
        # main clause: [embedded] læ pa ro-<ev>
        return f"{embed} læ pa ro-{ev}" + (" lo" if is_q else "")
    if imperative:
        # Object - verb - rɔ ; no subject, no evidential
        verb_en = _stem(words[0]) if words else "eat"
        verb_pi = _pi_of_en(verb_en)
        obj_pi = _pi_of_en(words[1]) if len(words) > 1 else None
        parts = ([obj_pi] if obj_pi else []) + [verb_pi, "rɔ"]
        return " ".join(parts)
    # --- wh-question ---
    wh = None
    wh_subj = False
    if words and words[0] in ("what", "who", "where", "when"):
        if words[0] == "who":
            # "Who ate an apple?" -> wh is the subject: ka pu ra-ta lo
            wh_subj = True
            words = words[1:]
            is_q = True
        else:
            wh_map = {"what": "lɔ", "where": "lə", "when": "lə"}
            wh = wh_map[words[0]]
            words = words[1:]
            # drop auxiliary after the wh-word: "What did I eat?"
            words = [w for w in words if w not in ("did", "do", "does")]
            is_q = True
    # --- simple SVO / intransitive, with optional time adverb ---
    time_pi = None
    rest = []
    for w in words:
        if w in ("today", "tomorrow", "yesterday", "now"):
            time_pi = EN_PI[w]
        else:
            rest.append(w)
    subj_en = rest[0] if rest else "i"
    verb_en = rest[1] if len(rest) > 1 else "eat"
    obj_en = rest[2] if len(rest) > 2 else None
    if wh_subj:
        # rest is [verb, obj] after the wh-word was removed
        subj_pi = "pu"
        verb_en2 = _stem(rest[0]) if rest else "eat"
        obj_en2 = rest[1] if len(rest) > 1 else None
        obj_pi = _pi_of_en(obj_en2) if obj_en2 else None
        parts = ([obj_pi] if obj_pi else []) + [subj_pi]
        if neg:
            parts.append("mæ")
        parts.append(f"{_pi_of_en(verb_en2)}-{ev}")
        out = " ".join(parts)
        if is_q:
            out += " lo"
        return out
    subj_pi = EN_PI.get(subj_en, subj_en)
    verb_pi = want if want else _pi_of_en(verb_en)
    if verb_en == "WANT":
        verb_pi = want
        obj_pi = _pi_of_en(obj_en) if obj_en else None
    else:
        obj_pi = _pi_of_en(obj_en) if obj_en else None
    parts = []
    if wh:
        parts.append(wh)
    if obj_pi:
        parts.append(obj_pi)
    if time_pi:
        parts.append(time_pi)
    parts.append(subj_pi)
    if neg:
        parts.append("mæ")
    parts.append(f"{verb_pi}-{ev}")
    out = " ".join(parts)
    if is_q:
        out += " lo"
    return out


def _translate_clause(toks):
    """Translate one OSV clause (tokens, no læ). Returns English words."""
    # toks: [obj?, time?, subj, (mæ)?, verb-ev] or [subj, verb-ev]
    neg = "mæ" in toks
    toks = [x for x in toks if x != "mæ"]
    if not toks:
        return "", "ta"
    verb_ev = toks[-1]
    if "-" in verb_ev:
        verb, ev = verb_ev.split("-", 1)
    else:
        verb, ev = verb_ev, "ta"
    rest = toks[:-1]
    time_en = None
    rest2 = []
    for x in rest:
        if x in TIME_EN:
            time_en = TIME_EN[x]
        else:
            rest2.append(x)
    subj_pi = rest2[-1] if rest2 else "pa"
    obj_pi = rest2[0] if len(rest2) >= 2 else None
    subj = PI_EN.get(subj_pi, subj_pi)
    obj = PI_EN.get(obj_pi, obj_pi) if obj_pi else None
    verb_en = VERB_EN.get(verb, PI_EN.get(verb, verb))
    if verb.startswith("mə") and len(verb) > 2 and verb[2:] in VERB_EN:
        verb_en = "want to " + VERB_EN[verb[2:]]
    return (subj, verb_en, obj, neg, time_en), ev


def pi_to_en(text):
    t = text.strip()
    is_q = t.endswith(" lo")
    if is_q:
        t = t[:-3].strip()
    toks = t.split()
    # --- imperative: ... verb rɔ, no evidential ---
    if toks and toks[-1] == "rɔ":
        verb = toks[-2] if len(toks) >= 2 else "ra"
        obj = toks[0] if len(toks) >= 3 else None
        verb_en = VERB_EN.get(verb, PI_EN.get(verb, verb))
        obj_en = PI_EN.get(obj, obj) if obj else ""
        return f"{verb_en.capitalize()} {obj_en}!".strip()
    # --- embedded: ... læ subj mainverb-ev ---
    if "læ" in toks:
        i = toks.index("læ")
        emb_toks, main_toks = toks[:i], toks[i + 1:]
        # embedded part may be 2 clauses: split at each verb-ev token
        clauses, cur = [], []
        for x in emb_toks:
            cur.append(x)
            if "-" in x:
                clauses.append(cur)
                cur = []
        if cur:
            clauses.append(cur)
        emb_en = []
        for c in clauses:
            (subj, verb_en, obj, neg, _tm), _ev = _translate_clause(c)
            s = f"{subj} {verb_en} {obj}" if obj else f"{subj} {verb_en}"
            if neg:
                s = f"{subj} did not {verb_en} {obj or ''}".strip()
            emb_en.append(s.strip())
        (subj, verb_en, obj, _neg, _tm), ev = _translate_clause(main_toks)
        ev_en = EVID.get(ev, ev)
        return f"{subj.capitalize()} {verb_en} that {' and '.join(emb_en)} ({ev_en})"
    # --- wh-question: first token is a wh root in function position ---
    wh = None
    if toks and toks[0] in WH_EN:
        wh = WH_EN[toks[0]]
        toks = toks[1:]
        is_q = True
    (subj, verb_en, obj, neg, time_en), ev = _translate_clause(toks)
    ev_en = EVID.get(ev, ev)
    subj_cap = subj.capitalize() if subj not in ("i",) else "I"
    if subj == "i":
        subj_cap = "I"
    base = f"{subj_cap} {verb_en} {obj or ''}".strip()
    if neg:
        base = f"{subj_cap} did not {verb_en} {obj or ''}".strip()
    if time_en:
        base += f" {time_en}"
    base += f" ({ev_en})"
    if is_q:
        if wh:
            return f"{wh.capitalize()} did {subj_cap} {verb_en} {obj or ''}? ({ev_en})".strip()
        return f"Did {subj_cap} {verb_en} {obj or ''}? ({ev_en})".strip()
    return base


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    mode, text = sys.argv[1], " ".join(sys.argv[2:])
    if mode == "en":
        print(en_to_pi(text))
    elif mode == "pi":
        print(pi_to_en(text))
    elif mode == "vec":
        print(nearest(text))
    else:
        print("mode: en | pi | vec")
