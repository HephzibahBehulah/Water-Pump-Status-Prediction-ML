# Contributing Guidelines

This document outlines the standards and processes for contributing to the Water Pump Status Prediction project.

## Code Standards

### 1. **Type Hints (Required)**
All functions must have type hints for parameters and return values.

```python
# ❌ WRONG
def predict_batch(df):
    return predictions, probabilities

# ✅ RIGHT
from typing import Tuple
import pandas as pd
import numpy as np

def predict_batch(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions for multiple pumps"""
    return predictions, probabilities
```

### 2. **Docstrings (Required)**
All functions must have docstrings explaining purpose, args, and return values.

```python
# ✅ REQUIRED FORMAT
def predict_batch(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions for multiple pumps.

    Args:
        df: Input dataframe with pump features

    Returns:
        Tuple of (predictions, repair_probabilities)
    """
    pass
```

### 3. **Constants (Required)**
- No magic numbers in code
- All constants must be in `src/config/constants.py`
- Use `from src.config.constants import CONSTANT_NAME`

```python
# ❌ WRONG
for i in range(15):  # What is 15?
    pass

# ✅ RIGHT
from src.config.constants import FEATURE_SELECTION_K
for i in range(FEATURE_SELECTION_K):
    pass
```

### 4. **Code Organization**
- Keep files under 300 lines
- One class/major function per file
- Related functions in modules (e.g., all data functions in `src/data/`)

### 5. **Naming Conventions**
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPERCASE_WITH_UNDERSCORES`
- Variables: `snake_case`

```python
# ✅ CORRECT
def load_training_data() -> pd.DataFrame:
    pass

class DataPreprocessor:
    pass

FEATURE_SELECTION_K = 15
input_dataframe = pd.read_csv('data.csv')
```

### 6. **Testing**
- Every new function must have unit tests
- Place tests in `tests/unit/` or `tests/integration/`
- Run tests before committing: `pytest tests/`

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/unit/test_constants.py -v

# Run with coverage
pytest tests/ --cov=src/
```

### 7. **Logging (Not Printing)**
- Use `logger` instead of `print()`
- Logger is already set up in `src/utils/logger.py`

```python
# ❌ WRONG
print("Model trained successfully")

# ✅ RIGHT
from src.utils import setup_logger
logger = setup_logger(__name__)
logger.info("Model trained successfully")
```

## File Organization

### Where to Put Code

| Code Type | Location | Example |
|-----------|----------|---------|
| Data loading | `src/data/` | `loader.py` |
| Feature engineering | `src/features/` | `engineering.py` |
| Model training | `src/models/` | `trainer.py` |
| Model evaluation | `src/evaluation/` | `metrics.py` |
| Utilities | `src/utils/` | `helpers.py` |
| Configuration | `src/config/` | `constants.py` |
| Scripts (entry points) | `scripts/` | `train.py` |
| Web app | `water_pump_app/` | `app.py` |
| Tests | `tests/` | `test_*.py` |

### Never Leave Duplicates

- Delete old files, don't leave them in the directory
- Archive old versions in `archive/legacy_versions/`
- Keep only ONE version of each file

## Git Workflow

### Before Committing

1. **Add type hints** to all new functions
2. **Add docstrings** to all functions
3. **Run tests** locally: `pytest tests/`
4. **Format code** with Black: `black src/ scripts/`
5. **Check imports** with isort: `isort src/ scripts/`
6. **Lint code** with Flake8: `flake8 src/ scripts/`

### Commit Messages

```
[TYPE] Brief description (under 70 chars)

Longer explanation (if needed)

Types: feature, fix, refactor, docs, test, style, chore
```

Examples:
```
[feature] Add SMOTE balancing for minority class
[fix] Fix missing column error in batch prediction
[test] Add unit tests for feature selection
[docs] Update README with new setup instructions
```

### Avoid in Commits

- ❌ Multiple unrelated changes
- ❌ Large files (>10 MB) - use DVC
- ❌ Sensitive data (.env, credentials)
- ❌ Generated files (__pycache__, .pyc)

## New Feature Checklist

When adding a new feature:

- [ ] Code has type hints
- [ ] Code has docstrings
- [ ] No magic numbers (all in constants.py)
- [ ] Unit tests added
- [ ] Integration tests added (if applicable)
- [ ] Code formatted with Black
- [ ] Code linted with Flake8
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No duplicate files
- [ ] Commit message is clear

## Code Review Checklist

When reviewing code:

- [ ] Type hints present?
- [ ] Docstrings complete?
- [ ] Magic numbers in constants?
- [ ] Tests included?
- [ ] Duplicates removed?
- [ ] No print() statements (using logger)?
- [ ] Clear variable names?
- [ ] Follows project structure?

## Testing Standards

### Unit Tests
- Test individual functions
- Location: `tests/unit/`
- File naming: `test_<module>.py`

### Integration Tests
- Test multiple components together
- Location: `tests/integration/`
- File naming: `test_<feature>.py`

### Running Tests
```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=src/ --cov-report=html

# Specific file
pytest tests/unit/test_constants.py -v

# Watch mode (auto-rerun on changes)
pytest-watch tests/
```

## Questions?

1. Check the documentation in `docs/`
2. Look at existing code examples
3. Check commit history for similar changes
4. Ask in the project discussion

## Code Examples

### ✅ Good Example
```python
"""Data preprocessing utilities"""

from typing import Tuple
import pandas as pd
import numpy as np
from src.utils import setup_logger
from src.config.constants import NUMERIC_FEATURES, CATEGORICAL_FEATURES

logger = setup_logger(__name__)

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Preprocess input data for model.

    Args:
        df: Raw input dataframe

    Returns:
        Tuple of (processed_dataframe, preprocessing_info)
    """
    logger.info(f"Preprocessing {len(df)} records")

    # Fill missing numeric values
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Fill missing categorical values
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col].fillna('unknown', inplace=True)

    logger.info("Preprocessing complete")
    return df, {'preprocessed': True}
```

### ❌ Bad Example
```python
# No docstring, no type hints, magic numbers, no logging
def preprocess_data(df):
    logger.info(f"Processing")
    for i in range(10):  # What is 10?
        df[i].fillna(0, inplace=True)
    print("Done")  # Don't print!
    return df
```

---

Thank you for contributing!
