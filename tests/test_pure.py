import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kmt.prep_data import clean_pairs, move_near_dups, split_of  # noqa: E402
from kmt.relex import apply_lexicon, load_lexicon, relexify  # noqa: E402
from kmt.text import (build_hindi_only, is_dev, leak_rate, normalize,  # noqa: E402
                      tm_compatible, words)


def test_normalize_nukta_and_digits():
    assert normalize("\u0958 ३४\u200d  x") == "\u0915\u093c 34 x"           # NFC + digits + ZW + spaces
    assert "\u093c" not in normalize("\u0958", strip_nukta=True)


def test_words_and_danda():
    assert words("मैं 25 साल।") == ["मैं", "25", "साल", "।"]
    assert is_dev("साल") and not is_dev("।") and not is_dev("25")


def test_leak_metric_and_list():
    hi = ["वह घर जा रहा है"] * 40 + ["यह अच्छा है"] * 40
    km = ["वो घर जां रौछ"] * 40
    lo = build_hindi_only(hi, km, min_hi=10, max_ratio=0.1)
    assert "है" in lo and "घर" not in lo
    assert leak_rate("घर है है", lo) > 0.5 and leak_rate("घर", lo) == 0


def test_tm_guard():
    assert tm_compatible("I have 3 books", "I have 3 books.")
    assert not tm_compatible("I have 3 books", "I have 2 books")          # numbers
    assert not tm_compatible("I do not like tea", "I do like tea")        # negation
    assert not tm_compatible("Then Ravi left", "Then Rahul left")         # names
    assert tm_compatible("He can't go", "He cannot go")


def test_relexify_and_reviewed_flag():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "lex.tsv")
        open(p, "w", encoding="utf-8").write(
            "# hdr\nहै\tछ\t9\t0.9\ty\nका\tको\t9\t0.9\t\nबुरा\tखराब\t9\t0.9\tn\n")
        assert load_lexicon(p) == {"है": "छ", "का": "को"}
        assert load_lexicon(p, require_reviewed=True) == {"है": "छ"}
        out, n, tot = apply_lexicon("घर का है।", load_lexicon(p))
        assert out == "घर को छ।" and n == 2 and tot == 3
        _, frac = relexify("घर", {}, [])
        assert frac == 0


def test_split_deterministic_and_grouped():
    assert split_of("hello world") == split_of("hello world")
    rows = [{"english": "I like tea", "kumaoni": "म चा पसंद करूं"},
            {"english": "I like tea", "kumaoni": "म चा पसंद करूं"},          # exact dup
            {"english": "Where is it", "kumaoni": "यो कां छ"}]
    kept, stats = clean_pairs(rows)
    assert len(kept) == 2 and stats["duplicate"] == 1


def test_near_dup_moved_to_train():
    sp = {"train": [{"src": "I am going to the market today", "tgt": "x"}],
          "dev": [{"src": "I am going to the market today.", "tgt": "y"},
                  {"src": "Completely different sentence here", "tgt": "z"}],
          "test": []}
    assert move_near_dups(sp) == 1 and len(sp["dev"]) == 1 and len(sp["train"]) == 2
