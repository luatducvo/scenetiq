"""Target Workspace: the only way the api and the worker reach a Scene's Target Workspace.

`open_target_workspace` picks the pipeline's one Target and returns a `TargetWorkspace`. Behind it,
one of two adapters moves files and runs Algorithms: in-process on this machine, or HTTP to a
scene-compute host. Either way the Target Workspace is separate from the Scene's published results
in `DATA_DIR/scenes/{id}`, which change only through `replace_results`.
"""

import logging
import os
import shutil
from pathlib import Path

from scene_core.config import ComputeConfig, PipelineConfig
from scene_core.contracts import OUTPUTS, SceneWorkspace
from scene_core.resources import backend_folder
from scene_core.results import SceneResults

from ._http import HttpAdapter
from ._inprocess import InProcessAdapter


def configuration():
    """The pipeline and compute configuration named by PIPELINE_CONFIG and COMPUTE_CONFIG; unset, the files in CONFIG_DIR."""

    def folder():
        return backend_folder("config", "CONFIG_DIR")

    pipeline = PipelineConfig.load(os.getenv("PIPELINE_CONFIG") or folder() / "pipeline.fake.yaml")
    compute_path = Path(os.getenv("COMPUTE_CONFIG") or folder() / "compute.yaml")
    return pipeline, ComputeConfig.load(compute_path) if compute_path.exists() else ComputeConfig()


class TargetWorkspace:
    """Scene inputs, Stage execution, results, query embedding and reranking on the pipeline's Target, by scene_id."""

    def __init__(self, pipeline: PipelineConfig, data_dir, adapter):
        # SOLID review (DIP): both adapters satisfy an implicit interface. A typed Protocol would
        # make their shared contract explicit without coupling TargetWorkspace to either implementation.
        self.pipeline = pipeline
        self.data_dir = Path(data_dir)
        self._adapter = adapter

    def put_inputs(self, scene_id: str, files: dict):
        """Store uploaded files {name: seekable stream} under raw/ on the Target, several at once.

        On failure some files may already be stored; the caller removes them by name.
        """
        self._adapter.put(scene_id, {f"raw/{name}": stream for name, stream in files.items()})

    def put_input(self, scene_id: str, name: str, stream):
        """Store one uploaded file under raw/ on the Target; stream must be seekable."""
        self.put_inputs(scene_id, {name: stream})

    def list_inputs(self, scene_id: str) -> list[str]:
        return [path.removeprefix("raw/") for path in self._adapter.list(scene_id, "raw")]

    def remove_inputs(self, scene_id: str, names: list[str]):
        if names:
            self._adapter.clear(scene_id, [f"raw/{name}" for name in names])

    def clear_outputs(self, scene_id: str):
        """Delete every Stage output on the Target and keep raw/: every Job runs from the start."""
        self._adapter.clear(scene_id, [d for dirs in OUTPUTS.values() for d in dirs])

    def run_stage(self, scene_id: str, stage: str, on_progress=None) -> dict:
        """Run the Stage's configured Algorithm on the Target and return its Manifest; outputs stay there."""
        config = self.pipeline.stages[stage]
        return self._adapter.run(config.algorithm, scene_id, config.params, on_progress)

    def replace_results(self, scene_id: str, version: str, publish):
        """Make this version the Scene's publication: download, finish its folder, then call publish.

        The results land in staging and are renamed to DATA_DIR/scenes/{scene}/{version}, complete and
        never modified again, before publish(that Workspace) commits the database. If anything
        fails first, the folder is removed and the earlier publication stays served. After the commit,
        every other publication folder of the Scene is removed.
        """
        staging = SceneWorkspace(self.data_dir / "staging", scene_id, version)
        published = SceneWorkspace(self.data_dir, scene_id, version)
        # One Job per Scene at a time, so the Scene's staging folder holds only this download.
        shutil.rmtree(staging.root.parent, ignore_errors=True)
        try:
            self._adapter.download(scene_id, list(SceneResults.DIRECTORIES), staging)
            published.root.parent.mkdir(parents=True, exist_ok=True)
            staging.root.rename(published.root)
            publish(published)
        except BaseException:
            shutil.rmtree(published.root, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(staging.root.parent, ignore_errors=True)
        for other in published.root.parent.iterdir():
            if other != published.root:
                if other.is_dir():
                    shutil.rmtree(other, ignore_errors=True)
                else:
                    other.unlink(missing_ok=True)

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Embed query text with the embed Stage's Algorithm and params, in the same space as Object embeddings."""
        config = self.pipeline.stages["embed"]
        return self._adapter.embed(config.algorithm, texts, config.params)

    def rerank(self, query: str, descriptions: list[str]) -> list[float]:
        """Score how well each Object Description answers the query, in 0..1, with the embed Stage's Algorithm and params."""
        if not descriptions:
            return []
        config = self.pipeline.stages["embed"]
        return self._adapter.rerank(config.algorithm, query, descriptions, config.params)

    def delete(self, scene_id: str):
        """Remove the Scene's Target Workspace; its published results are the api's to remove."""
        self._adapter.delete(scene_id)

    def close(self):
        self._adapter.close()


def open_target_workspace(
    data_dir,
    pipeline: PipelineConfig | None = None,
    compute: ComputeConfig | None = None,
    ready_timeout: float | None = None,
    client=None,
) -> TargetWorkspace:
    """Build the TargetWorkspace for the pipeline's Target; ValueError names what to set.

    ready_timeout caps how long a call waits for a cold Target. client replaces the HTTP client, so
    tests can talk to an in-memory scene-compute app.
    """
    if pipeline is None or compute is None:
        defaults = configuration()
        pipeline = pipeline or defaults[0]
        compute = compute or defaults[1]
    name = pipeline.target
    if name.startswith("${"):
        raise ValueError(f"Set {name[2:-1]} to a compute Target name")
    if name not in compute.targets:
        raise ValueError(f"Pipeline uses unknown compute Target {name}")
    target = compute.targets[name]
    if target.backend == "inprocess":
        return TargetWorkspace(pipeline, data_dir, InProcessAdapter(Path(data_dir) / "target"))
    if target.backend != "http":
        raise ValueError(f"Unsupported compute backend {target.backend}")
    if target.missing_url:
        raise ValueError(f"compute Target {name} has no URL; set its URL variable")
    if ready_timeout is not None:
        target = target.model_copy(
            update={"startup_timeout": min(target.startup_timeout, ready_timeout)}
        )
    return TargetWorkspace(pipeline, data_dir, HttpAdapter(target, client))


def delete_quietly(target: TargetWorkspace, scene_id: str):
    """Delete a Scene's copies, logging rather than raising when the Target is unreachable."""
    try:
        target.delete(scene_id)
    except Exception:
        logging.exception("Could not delete the Target Workspace of Scene %s", scene_id)


__all__ = ["TargetWorkspace", "configuration", "delete_quietly", "open_target_workspace"]
