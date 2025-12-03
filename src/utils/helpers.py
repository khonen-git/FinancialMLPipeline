"""General helper functions."""

import subprocess
from pathlib import Path
from typing import Optional


def get_git_commit_hash() -> str:
    """Get current git commit hash for reproducibility.

    Returns:
        Git commit hash or 'unknown' if not in git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def check_git_clean() -> bool:
    """Check if git repository is clean (no uncommitted changes).

    Returns:
        True if clean, False otherwise.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return len(result.stdout.strip()) == 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if not.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def convert_ticks_to_price(ticks: int, tick_size: float) -> float:
    """Convert a distance expressed in ticks to a price distance.

    Args:
        ticks: Number of ticks.
        tick_size: Size of one tick (e.g. 0.00001 on EURUSD, 0.001 on USDJPY).

    Returns:
        Price distance (ticks * tick_size).
    """
    return ticks * tick_size

