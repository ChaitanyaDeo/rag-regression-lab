"""
ragregress/api/settings.py

API read and writes. Every path is relative.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ragregress.report import DEFAULT_PASSAGES, DEFAULT_SPLITS_DIR, DEFAULT_TOP
from ragregress.storage.database import DB_PATH_DEFAULT

DEFAULT_CONFIGS_DIR = Path("config")

DEFAULT_TARGET = DEFAULT_CONFIGS_DIR / "targets" / "squad_val.yaml"

@dataclass
class ApiSettings:
    db_path: Path = DB_PATH_DEFAULT
    configs_dir: Path = DEFAULT_CONFIGS_DIR  # POST /runs loads <configs_dir>/<name>.yaml
    passages_path: Path = DEFAULT_PASSAGES
    splits_dir: Path = DEFAULT_SPLITS_DIR    # report gold answers
    targets_path: Optional[Path] = DEFAULT_TARGET     # report target table; skipped when the file is absent
    top: int = DEFAULT_TOP

    def targets_file(self)-> Optional[Path]:
        """Target file if there is one: the default is optional, so a missing file mean no table"""

        if self.targets_path is not None and Path(self.targets_path).is_file():
            return Path(self.targets_path)

        return None