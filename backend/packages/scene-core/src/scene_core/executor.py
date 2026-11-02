import gc
import sys
import threading
from pathlib import Path

from .contracts import SceneWorkspace
from .registry import Registry


class Executor:
    # The optional capabilities an embed Algorithm offers beyond run(), and the key of each result.
    CAPABILITIES = {"embed_text": "vectors", "rerank": "scores"}

    def __init__(self, data_dir: Path, registry: Registry | None = None):
        self.data_dir = Path(data_dir)
        self.registry = registry or Registry()
        self._instances = {}
        self._lock = threading.RLock()

    def algorithm(self, name: str):
        if name not in self._instances:
            instance = self.registry.resolve(name)()
            instance.setup()
            self._instances[name] = instance
        return self._instances[name]

    def _release_others(self, keep: str):
        """Drop every Algorithm instance but keep's, so only one Stage's models occupy the GPU.

        Stages run one after another and each loads its own models; kept alive together they
        exhaust GPU and host memory, and the compute process is killed mid-Job.
        """
        released = [name for name in self._instances if name != keep]
        for name in released:
            del self._instances[name]
        if released:
            gc.collect()
            # Only an Algorithm imports torch; the Executor clears its cache when one has.
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(self, algorithm: str, scene_id: str, params: dict, on_progress=None) -> dict:
        with self._lock:
            self._release_others(algorithm)
            instance = self.algorithm(algorithm)
            workspace = SceneWorkspace(self.data_dir, scene_id)
            instance._progress = on_progress
            try:
                result = instance.manifest_type.model_validate(instance.run(workspace, params))
            finally:
                instance._progress = None
            for output in instance.writes:
                workspace.dir(output).joinpath("manifest.json").write_text(
                    result.model_dump_json(indent=2), encoding="utf-8"
                )
            return result.model_dump(mode="json")

    def invoke(self, algorithm: str, method: str, payload: dict) -> dict:
        if method not in self.CAPABILITIES:
            raise ValueError("Unsupported invoke method")
        with self._lock:
            # SOLID review (ISP/DIP): embed_text and rerank are optional capabilities absent from
            # StageAlgorithm. Give embedding Algorithms a narrow capability interface instead of
            # adding them to every Stage.
            value = getattr(self.algorithm(algorithm), method)(**payload)
            return {
                self.CAPABILITIES[method]: value.tolist() if hasattr(value, "tolist") else value
            }
