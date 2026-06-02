"""
Unit tests for the data pipeline's core domain logic.

Covers three functions that are the most complex and bug-prone:
  - _to_formal       : Korean irregular verb conjugation (no Mecab dependency)
  - label_sentence   : morpheme-based formality labeling via Mecab
  - is_valid_pair    : length and ratio filtering via Mecab

Run: pytest tests/test_data_pipeline.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.data_pipeline.augment import _to_formal
from backend.data_pipeline.label import label_sentence
from backend.data_pipeline.filter import is_valid_pair


# ---------------------------------------------------------------------------
# _to_formal — pure Python, no Mecab required
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stem,expected", [
    # Regular consonant-final stem
    ("먹",       "먹습니다"),
    # ㄹ-irregular: ㄹ종성 swapped for ㅂ종성, then 니다 appended
    ("알",       "압니다"),
    ("살",       "삽니다"),
    ("만들",     "만듭니다"),
    # ㄷ-irregular: standalone last word matched against lookup table
    ("들",       "듣습니다"),
    ("걸",       "걷습니다"),
    # ㄷ-irregular in a multi-word stem (last word matches the lookup key)
    ("음악을 들", "음악을 듣습니다"),
    # ㄹ-regular in a multi-word stem (last word "만들" is NOT in ㄷ table)
    ("음악을 만들", "음악을 만듭니다"),
    # ㅅ-irregular
    ("지",       "짓습니다"),
    ("나",       "낫습니다"),
])
def test_to_formal(stem, expected):
    assert _to_formal(stem) == expected, f"_to_formal({stem!r}) → {_to_formal(stem)!r}, expected {expected!r}"


# ---------------------------------------------------------------------------
# label_sentence — requires Mecab
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence,expected", [
    # Formal: 습니다 tagged as EF when preceded by an adjective stem
    ("오늘 날씨가 좋습니다", "formal"),
    ("네 알겠습니다",       "formal"),
    # Polite: 어요 tagged as EF; 해요 caught by -요 fallback regex
    ("잘 지내고 있어요",    "polite"),
    ("감사해요",           "polite"),
    # Casual: -야 copula ending reliably tagged as VCP+EF by Mecab
    ("지금 어디야",         "casual"),
    ("그거 뭐야",           "casual"),
])
def test_label_sentence(sentence, expected):
    result = label_sentence(sentence)
    assert result == expected, f"label_sentence({sentence!r}) → {result!r}, expected {expected!r}"


def test_label_sentence_returns_none_for_unlabeled():
    # Non-Korean input has no EF morpheme and no -요 fallback
    assert label_sentence("hello world") is None


# ---------------------------------------------------------------------------
# is_valid_pair — requires Mecab
# ---------------------------------------------------------------------------

def test_valid_pair_accepted():
    assert is_valid_pair("Can you help me please", "도와주실 수 있어요", "polite") is True


def test_en_too_short_rejected():
    # 1 EN word — below the minimum of 3
    assert is_valid_pair("hi", "안녕", "casual") is False


def test_none_formality_rejected():
    assert is_valid_pair("hello world this", "안녕하세요", None) is False


def test_unknown_formality_rejected():
    assert is_valid_pair("hello world this", "안녕하세요", "unknown") is False


def test_en_too_long_rejected():
    # 30 EN words — above the formal max of 25
    long_en = " ".join(["word"] * 30)
    assert is_valid_pair(long_en, "감사합니다", "formal") is False
