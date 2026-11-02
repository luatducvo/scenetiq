"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MATCH_THRESHOLD = 0.5
"""Default relevance threshold for a Match.

With the real Pipeline this is a SigLIP2 match probability. Replace it with the
``recommended_threshold`` of an accepted evaluation-harness run (see AGENTS.md).
"""


@dataclass(frozen=True)
class Settings:
    """Backend settings.

    Attributes:
        storage_root: Directory holding the SQLite database and all Scene data.
        token: Shared secret every request must present as a bearer token.
        match_threshold: Minimum relevance score for an Instance to be a Match.
        pipeline: Which Pipeline to run: ``"fake"`` or ``"real"``.
        cors_origins: Browser origins allowed to call the API.
    """

    storage_root: Path
    token: str
    match_threshold: float = DEFAULT_MATCH_THRESHOLD
    pipeline: str = "fake"
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from ``SCENETIQ_*`` environment variables.

        Raises:
            RuntimeError: If ``SCENETIQ_TOKEN`` is not set.
        """
        token = os.environ.get("SCENETIQ_TOKEN")
        if not token:
            raise RuntimeError("SCENETIQ_TOKEN must be set")
        origins = os.environ.get("SCENETIQ_CORS_ORIGINS")
        kwargs: dict = {}
        if origins:
            kwargs["cors_origins"] = [o.strip() for o in origins.split(",") if o.strip()]
        return cls(
            storage_root=Path(os.environ.get("SCENETIQ_STORAGE_ROOT", "./data")),
            token=token,
            match_threshold=float(
                os.environ.get("SCENETIQ_MATCH_THRESHOLD", DEFAULT_MATCH_THRESHOLD)
            ),
            pipeline=os.environ.get("SCENETIQ_PIPELINE", "fake"),
            **kwargs,
        )
