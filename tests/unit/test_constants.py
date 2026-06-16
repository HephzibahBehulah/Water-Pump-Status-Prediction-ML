"""Tests for configuration constants"""

import pytest
from src.config.constants import (
    FEATURE_SELECTION_K,
    RANDOM_STATE,
    CLASS_NAMES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def test_feature_selection_k():
    """Test feature selection constant"""
    assert FEATURE_SELECTION_K == 15
    assert isinstance(FEATURE_SELECTION_K, int)
    assert FEATURE_SELECTION_K > 0


def test_random_state():
    """Test random state constant"""
    assert RANDOM_STATE == 42
    assert isinstance(RANDOM_STATE, int)


def test_class_names():
    """Test class names are defined"""
    assert len(CLASS_NAMES) == 3
    assert 'Functional' in CLASS_NAMES
    assert 'Functional needs repair' in CLASS_NAMES
    assert 'Non-functional' in CLASS_NAMES


def test_feature_lists():
    """Test feature lists are properly defined"""
    assert len(NUMERIC_FEATURES) == 10
    assert len(CATEGORICAL_FEATURES) == 25
    assert 'pump_age' in NUMERIC_FEATURES
    assert 'funder' in CATEGORICAL_FEATURES


def test_no_feature_overlap():
    """Test numeric and categorical features don't overlap"""
    overlap = set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
    assert len(overlap) == 0, f"Features overlap: {overlap}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
