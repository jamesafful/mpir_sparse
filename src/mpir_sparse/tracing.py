from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class JSONLTraceWriter:
    """Write one JSON object per line for iterative solver traces.

    This is intentionally lightweight and dependency-free.
    """
    path: str
    flush: bool = True

    def __post_init__(self) -> None:
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        if self.flush:
            self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "JSONLTraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
