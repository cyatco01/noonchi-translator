"""
Unit tests for FormalityResolver.

Tests the rule-based sociolinguistic inference engine in isolation — no API calls.
Run: pytest tests/test_formality_resolver.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'formality'))
from resolver import FormalityResolver, SocialContext, RelationshipType, SettingType, FormalityToken

resolver = FormalityResolver()


@pytest.mark.parametrize("relationship,age_diff,setting,expected", [
    # Superior relationships → always formal regardless of setting
    ("boss",      -10, "workplace", "formal"),
    ("boss",        5, "social",    "formal"),
    ("professor",   0, "academic",  "formal"),
    ("professor",  10, "intimate",  "formal"),
    ("elder",     -15, "intimate",  "formal"),
    ("elder",       0, "social",    "formal"),

    # Stranger → always formal
    ("stranger",    0, "social",    "formal"),
    ("stranger",    5, "intimate",  "formal"),

    # Public setting → always formal regardless of relationship
    ("friend",      0, "public",    "formal"),
    ("colleague",  -3, "public",    "formal"),
    ("subordinate", 10, "public",   "formal"),

    # Notably younger in professional setting → formal
    ("colleague",  -6, "workplace", "formal"),
    ("peer",      -10, "academic",  "formal"),

    # Boundary: -5 is NOT < -5, no formal trigger
    ("colleague",  -5, "workplace", "polite"),
    ("colleague",  -4, "academic",  "polite"),

    # Friend + similar age + intimate/social → casual
    ("friend",      0, "intimate",  "casual"),
    ("friend",     -3, "social",    "casual"),   # -3 is the boundary (>= -3)
    ("friend",      2, "intimate",  "casual"),

    # Friend but wrong setting → polite
    ("friend",      0, "workplace", "polite"),
    ("friend",     -3, "academic",  "polite"),

    # Friend but too much younger → polite
    ("friend",     -4, "intimate",  "polite"),   # -4 < -3, no casual trigger

    # Subordinate + notably older + intimate/social → casual
    ("subordinate",  6, "intimate", "casual"),
    ("subordinate", 10, "social",   "casual"),

    # Subordinate at boundary: 5 is NOT > 5
    ("subordinate",  5, "intimate", "polite"),

    # Default: everything else → polite
    ("colleague",    0, "workplace", "polite"),
    ("peer",         0, "academic",  "polite"),
    ("acquaintance", 0, "social",    "polite"),
    ("subordinate",  2, "workplace", "polite"),
    ("acquaintance", 5, "intimate",  "polite"),
])
def test_resolve(relationship, age_diff, setting, expected):
    ctx = SocialContext(
        relationship=RelationshipType(relationship),
        age_differential=age_diff,
        setting=SettingType(setting),
    )
    result = resolver.resolve(ctx)
    assert result == FormalityToken(expected), (
        f"({relationship}, age_diff={age_diff}, {setting}) → got {result.value!r}, expected {expected!r}"
    )


def test_override_bypasses_all_rules():
    ctx = SocialContext(
        relationship=RelationshipType.BOSS,
        age_differential=-10,
        setting=SettingType.WORKPLACE,
        formality_override=FormalityToken.CASUAL,
    )
    assert resolver.resolve(ctx) == FormalityToken.CASUAL


def test_override_all_tokens():
    ctx = SocialContext(
        relationship=RelationshipType.FRIEND,
        age_differential=0,
        setting=SettingType.INTIMATE,
    )
    for token in FormalityToken:
        ctx.formality_override = token
        assert resolver.resolve(ctx) == token


def test_override_none_falls_through_to_rules():
    ctx = SocialContext(
        relationship=RelationshipType.BOSS,
        age_differential=-10,
        setting=SettingType.WORKPLACE,
        formality_override=None,
    )
    assert resolver.resolve(ctx) == FormalityToken.FORMAL
