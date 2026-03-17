"""Hierarchical prompt-response logging utilities.

Each interaction record has the form:
{
  session_id,
  turn_id,
  parent_turn_id,
  prompt,
  response,
  model,
  context_refs,
  metadata,
  hash,
  created_at,
  schema_version
}
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ConversationTurn:
    session_id: str
    turn_id: str
    parent_turn_id: Optional[str]
    prompt: str
    response: str
    model: str
    context_refs: list[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash: str = ""
    created_at: str = ""
    schema_version: str = "v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationLogger:
    """Append-only JSONL conversation logger supporting conversation trees."""

    def __init__(self, log_dir: Path | None = None) -> None:
        root = Path(log_dir or os.getenv("RAG_LOG_DIR") or Path(__file__).resolve().parents[1] / "RAG_LOGS")
        root.mkdir(parents=True, exist_ok=True)
        self.log_path = root / "prompts.jsonl"

    def _compute_hash(self, turn: ConversationTurn) -> str:
        h = hashlib.sha256()
        payload = json.dumps(
            {
                "session_id": turn.session_id,
                "turn_id": turn.turn_id,
                "parent_turn_id": turn.parent_turn_id,
                "prompt": turn.prompt,
                "response": turn.response,
                "model": turn.model,
                "metadata": turn.metadata,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        h.update(payload.encode("utf-8"))
        return h.hexdigest()

    def log_turn(self, turn: ConversationTurn) -> None:
        if not turn.created_at:
            turn.created_at = datetime.now(timezone.utc).isoformat()
        if not turn.hash:
            turn.hash = self._compute_hash(turn)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(turn.to_dict(), ensure_ascii=False) + "\n")
