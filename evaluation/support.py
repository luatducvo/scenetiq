import os

from scenetiq.config import Settings, overrides_from_json
from scenetiq.domain.scene import Stage


def configured_settings() -> Settings:
    return Settings.from_overrides("", overrides_from_json(os.environ.get("SCENETIQ_CONFIG")))


class StageRecorder:
    def __init__(self, stages: list[Stage]) -> None:
        self.stages = stages
        self.notes: list[str] = []

    def stage(self, stage: Stage) -> None:
        self.stages.append(stage)

    def checkpoint(self) -> None:
        pass

    def note(self, message: str) -> None:
        self.notes.append(message)
