#!/usr/bin/env python3
"""Test cases for translator v0.2. Run: python3 test_translator.py"""
import sys
sys.path.insert(0, ".")
from translator import en_to_pi, pi_to_en

EN_PI = [
    ("I eat an apple", "ka pa ra-ta"),
    ("I eat an apple yesterday", "ka lu pa ra-ta"),
    ("I eat an apple today", "ka la pa ra-ta"),
    ("I eat an apple tomorrow", "ka li pa ra-ta"),
    ("I might eat an apple", "ka pa ra-te"),
    ("I did not eat an apple", "ka pa mæ ra-ta"),
    ("I do not drink water", "ke pa mæ ri-ta"),
    ("Did I eat an apple?", "ka pa ra-ta lo"),
    ("What did I eat?", "lɔ pa ra-ta lo"),
    ("Who ate an apple?", "ka pu ra-ta lo"),
    ("Eat an apple!", "ka ra rɔ"),
    ("I want to eat an apple", "ka pa məra-ta"),
    ("I might want to eat an apple", "ka pa məra-te"),
    ("I want to drink water", "ke pa məri-ta"),
    ("I see the sky", "sa pa ro-ta"),
    ("You see the sea", "si pi ro-ta"),
    ("He reads a book", "kæ pu kæro-ta"),
    ("I heard that I eat an apple", "ka pa ra-to"),
    ("In a dream, I eat an apple", "ka pa ra-tə"),
    ("Everyone knows I eat an apple", "ka pa ra-tɔ"),
    ("I saw that he eats an apple", "ka pu ra-ta læ pa ro-ta"),
    ("I saw him stepping on a banana and crying",
     "kasæ pu moru-ta pu məso-ta læ pa ro-ta"),
    # normalisation (v0.3): capitals / trailing punctuation / 3rd-person -s
    ("You see the sea.", "si pi ro-ta"),
    ("He reads a book.", "kæ pu kæro-ta"),
    ("She sells a banana.", "kasæ pu kuro-ta"),
    ("They sell a banana.", "kasæ po kuro-ta"),
    ("I see red.", "sæko pa ro-ta"),
    ("She doesn't eat a banana.", "kasæ pu mæ ra-ta"),
    ('"I eat an apple," someone said.', "ka pa ra-tæ"),
]
PI_EN = [
    ("ka pa ra-ta", "I eat apple (saw)"),
    ("ka pa ra-ta lo", "Did I eat apple? (saw)"),
    ("ka pa mæ ra-ta", "I did not eat apple (saw)"),
    ("ka ra rɔ", "Eat apple!"),
    ("ka pa məra-te", "I want to eat apple (inferred/maybe)"),
    ("lɔ pa ra-ta lo", "What did I eat ? (saw)"),
    ("kasæ pu moru-ta pu məso-ta læ pa ro-ta",
     "I see that he step banana and he cry (saw)"),
    ("si pi ru-te", "You go sea (inferred/maybe)"),
]

def main():
    passed = failed = 0
    for en, exp in EN_PI:
        got = en_to_pi(en)
        ok = got == exp
        passed += ok; failed += not ok
        print(("PASS" if ok else "FAIL"), f"EN>PI {en!r} -> {got!r} (expected {exp!r})")
    for pi, exp in PI_EN:
        got = pi_to_en(pi)
        ok = got == exp
        passed += ok; failed += not ok
        print(("PASS" if ok else "FAIL"), f"PI>EN {pi!r} -> {got!r} (expected {exp!r})")
    print(f"\n{passed} passed, {failed} failed, total {passed+failed}")

if __name__ == "__main__":
    main()
