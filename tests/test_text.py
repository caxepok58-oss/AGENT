from __future__ import annotations

import pytest

from shorts_agent.text import normalize_key, word_set

# Any of these reduced to an empty string under the previous ASCII-only
# normalization, which silently dropped the topic from trend aggregation.
NON_LATIN = [
    ("как экономить деньги", "как экономить деньги"),
    ("如何省钱", "如何省钱"),
    ("كيف توفر المال", "كيف توفر المال"),
    ("बजट कैसे बनाएं", "बजट कैसे बनाएं"),
]


@pytest.mark.parametrize(("raw", "expected"), NON_LATIN)
def test_non_latin_text_survives_normalization(raw, expected):
    assert normalize_key(raw) == expected


def test_normalization_is_case_and_punctuation_insensitive():
    assert normalize_key("Как Экономить, Деньги!") == normalize_key("как экономить деньги")


def test_latin_normalization_is_unchanged():
    assert normalize_key("Save MONEY, fast!") == "save money fast"


def test_casefold_handles_cases_lower_does_not():
    # "ß" lowercases to itself but casefolds to "ss", so these must compare equal.
    assert normalize_key("Straße") == normalize_key("STRASSE")


def test_compatibility_normalization_unifies_equivalent_codepoints():
    """Full-width Latin must not be treated as a different topic."""
    assert normalize_key("ＭＯＮＥＹ") == normalize_key("money")


def test_decomposed_and_composed_accents_compare_equal():
    assert normalize_key("café") == normalize_key("café")


def test_empty_and_punctuation_only_input():
    assert normalize_key("") == ""
    assert normalize_key("!!! ??? ...") == ""


def test_word_set_extracts_non_latin_words():
    assert word_set("Как экономить деньги студенту") == {
        "как",
        "экономить",
        "деньги",
        "студенту",
    }


def test_word_set_applies_length_floor():
    assert word_set("a to the budgeting", min_length=3) == {"the", "budgeting"}


def test_word_set_removes_stopwords():
    assert word_set("the best budgeting tips", stopwords={"the", "best"}) == {
        "budgeting",
        "tips",
    }


def test_word_set_keeps_internal_apostrophes():
    assert word_set("don't overspend") == {"don't", "overspend"}


def test_word_set_ignores_underscores_and_digits_separately():
    assert word_set("save_money 2026") == {"save", "money", "2026"}
