"""
config.py — Secure Configuration Loader
-----------------------------------------
Loads all settings from the .env file (never from code).
Import this module everywhere instead of reading os.getenv() directly.

Usage:
    from config import settings
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.api_base_url)

Security guarantees:
  - API key is never logged, printed, or serialised
  - .env is loaded once at startup and never re-read
  - Fails fast with a clear error if required keys are missing
  - Key is masked in __repr__ so it can't leak into logs
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Load .env file before anything else touches os.environ
# find_dotenv() searches upward from cwd so works from any subdirectory
try:
    from dotenv import load_dotenv, find_dotenv
    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        load_dotenv(_dotenv_path, override=False)   # don't override already-set shell vars
    else:
        # Fallback: look next to this file
        _local_env = Path(__file__).parent / ".env"
        if _local_env.exists():
            load_dotenv(_local_env, override=False)
except ImportError:
    # python-dotenv not installed — fall back to raw os.environ (e.g. Docker with -e flags)
    pass


class _MaskedStr(str):
    """String subclass that masks its value in repr/str to prevent log leakage."""
    def __repr__(self) -> str:
        if len(self) <= 8:
            return "****"
        return self[:4] + "****" + self[-2:]

    def __str__(self) -> str:
        return repr(self)


class Settings:
    """
    Single source of truth for all runtime configuration.
    Reads from environment variables (populated from .env by load_dotenv above).
    """

    # ── LLM API ─────────────────────────────────────────────────────────────
    @property
    def openai_api_key(self) -> _MaskedStr:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key or key == "sk-your-api-key-here":
            _warn("OPENAI_API_KEY is not set. Copy .env.example → .env and add your key.")
        return _MaskedStr(key)

    @property
    def api_base_url(self) -> str:
        return os.environ.get("API_BASE_URL", "https://api.openai.com/v1").strip()

    @property
    def model_name(self) -> str:
        return os.environ.get("MODEL_NAME", "gpt-4o-mini").strip()

    # ── Server ───────────────────────────────────────────────────────────────
    @property
    def port(self) -> int:
        try:
            return int(os.environ.get("PORT", "8000"))
        except ValueError:
            return 8000

    # ── Environment ──────────────────────────────────────────────────────────
    @property
    def default_seed(self) -> int:
        try:
            return int(os.environ.get("DEFAULT_SEED", "42"))
        except ValueError:
            return 42

    @property
    def log_level(self) -> str:
        return os.environ.get("LOG_LEVEL", "INFO").upper()

    # ── Validation ───────────────────────────────────────────────────────────
    def validate(self) -> None:
        """
        Call at application startup to fail fast on missing required config.
        Raises SystemExit with a helpful message — never prints the key value.
        """
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key or key == "sk-your-api-key-here":
            print(
                "\n[CONFIG ERROR] OPENAI_API_KEY is missing or still set to placeholder.\n"
                "  1. cp .env.example .env\n"
                "  2. Edit .env and set OPENAI_API_KEY=sk-...\n"
                "  3. Re-run the script.\n"
                "  (Never hardcode keys in source files.)\n",
                file=sys.stderr,
            )
            sys.exit(1)

    def safe_summary(self) -> dict:
        """Return a loggable summary — API key is always masked."""
        return {
            "api_base_url":  self.api_base_url,
            "model_name":    self.model_name,
            "openai_api_key": repr(self.openai_api_key),   # masked via __repr__
            "port":          self.port,
            "default_seed":  self.default_seed,
            "log_level":     self.log_level,
        }


def _warn(msg: str) -> None:
    print(f"[CONFIG WARNING] {msg}", file=sys.stderr)


# Singleton — import and use `settings` everywhere
settings = Settings()
