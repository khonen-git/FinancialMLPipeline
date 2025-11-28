# Coding Standards & Best Practices

This document defines the coding rules for this project, intended for both developers and AI code assistants.

---

## 1. General Principles

- Code must be **modular**, each file with a single responsibility
- Avoid monolithic scripts
- Functions must be pure when possible
- No global variables
- Use type hints everywhere
- Logging instead of print
- Prefer small, composable utilities

---

## 2. Folder Responsibilities

| Folder | Responsibility |
|--------|----------------|
| data/ | ingestion, format detection, bar builders |
| features/ | all feature engineering |
| labeling/ | triple barrier, meta labels |
| models/ | RF, HMM, meta-model, MLflow registry |
| validation/ | CV, purging, embargo, walk-forward |
| backtest/ | Backtrader integration |
| risk/ | Monte Carlo, DD metrics |
| interpretability/ | feature importance, SHAP, regime analysis |
| reporting/ | Jinja reports |
| deployment/ | inference logic |
| benchmarks/ | Pandas vs Polars vs cuDF, profiling |

You must respect these boundaries.

---

## 3. Logging

Use the standard Python `logging` module.

Guidelines:

- No `print()` calls in production code
- Use a module-level logger:

```python
import logging

logger = logging.getLogger(__name__)
```

- Log at appropriate levels: `debug`, `info`, `warning`, `error`
- Include timestamp, level, module, and message in the log format
- For experiment runs, logs should be saved as artifacts in MLflow (e.g. `run.log`)
- Configure logging centrally in `src/utils/logging.py`

---

## 4. Style Rules

- Use `snake_case` for functions and variables
- Class names in `PascalCase`
- One class per file if possible
- Prefer dataclasses for containers
- Use enums for static categories (bar type, model type)
- No emojis in code, comments, or commit messages

---

## 5. Function Size and Structure

Prefer small, focused functions.

**Guidelines**:

- Aim for functions under ~60 lines
- Longer functions are acceptable if:
  - They implement a single, well-defined responsibility
  - Splitting them would make the code harder to understand
  - The control flow remains clear
- If a function grows beyond ~80 lines, consider splitting it into helpers

**AI guidance**:

- Do not aggressively split functions into tiny pieces just to obey a hard line count
- Prioritize readability and clear responsibilities over arbitrary line limits

---

## 6. Docstrings

Use Google-style docstrings with `Args:` and `Returns:` sections.

Example:

```python
def compute_triple_barrier(events, prices, tp, sl, max_horizon):
    """Compute triple-barrier labels for long-only events.

    Args:
        events: DataFrame with event start times and metadata.
        prices: Series or DataFrame with bid/ask prices.
        tp: Profit target distance (in price units or pips).
        sl: Stop loss distance.
        max_horizon: Maximum holding period (in bars).

    Returns:
        DataFrame with:
            - t1: event end time
            - label: {-1, 0, 1}
    """
```

**Rules**:

- Public functions and class methods must have a docstring
- Private helpers (e.g. `_helper`) may have a shorter docstring or be documented via comments if trivial
- Use a `Raises:` section when the function deliberately raises specific exceptions

---

## 7. Comments

- Comments should explain **why** the code is written this way, not restate what the code already says
- Avoid noisy or trivial comments, such as:
  - `# increment counter`
  - `# call function`
- Do not duplicate docstring content in inline comments
- Avoid auto-generated banners or decorative comments

---

## 8. Testing

- Each module must have unit tests
- Use small static datasets in `tests/data/`
- Add regression tests for critical logic (bar construction, triple barrier)

---

## 9. AI-Specific Guidelines

This project allows AI-assistance. The following rules prevent common AI mistakes:

### 9.1 Configuration Integrity

**Rule**: The AI must not invent new configuration keys or change existing ones silently.

- Any new config field must be documented in `docs/`
- Any new config field must be added to the appropriate config reference in `configs/`
- Never change the meaning of existing config keys without explicit instruction

### 9.2 API Stability

**Rule**: Do not modify public API signatures without explicit instruction.

If you change the signature of a public function or class, you must:

- Update all call sites
- Update the corresponding documentation
- Update or add tests

Examples of public APIs:

- `compute_triple_barrier(...)`
- `BarBuilder` classes
- MLflow registry functions
- Feature engineering pipelines

### 9.3 Protected Zones

**Rule**: Respect strict boundaries for certain directories.

- **docs/**: Do not modify unless explicitly requested
- **configs/**: Do not restructure without explicit instruction (adding new files is OK)
- **tests/**: Never delete existing tests (adapt them if needed, but do not remove)

### 9.4 Expensive Operations

**Rule**: The AI must not create scripts that automatically launch heavy computations on import.

All expensive operations must be behind an explicit CLI or function call:

- ✅ `python scripts/run_backtest.py --config ...`
- ✅ `if __name__ == "__main__": run_monte_carlo()`
- ❌ `import my_module` → triggers a full walk-forward run

Examples of expensive operations:

- Backtests with Backtrader
- Monte Carlo simulations with 1000+ iterations
- Walk-forward validation
- Model training

---

## 10. Git and Commits

Make small, coherent commits:

- One feature
- One bug fix
- One refactor per commit when possible

Use clear commit messages, for example:

```
feat: add triple barrier labeling
fix: handle missing volumes in schema detection
refactor: split backtest adapter into smaller functions
docs: update data pipeline architecture
```

**Rules**:

- No emojis or "chatty" messages in commit logs
- Commit message must describe what changed in concrete terms

**When using AI assistants**:

- Ask the AI to propose a single commit per logical change
- Do not rewrite entire files unless explicitly requested; prefer minimal, localized changes
