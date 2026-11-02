from pydantic import BaseModel
from scene_core.executor import Executor
from scene_core.ports import StageAlgorithm


class Done(BaseModel):
    ok: bool = True


def algorithm(algorithm_name):
    class Algorithm(StageAlgorithm):
        name = algorithm_name
        stage = algorithm_name
        manifest_type = Done
        reads = []
        writes = []

        def run(self, workspace, params):
            return Done()

    return Algorithm


class Registry:
    def __init__(self, *names):
        self.classes = {name: algorithm(name) for name in names}

    def resolve(self, name):
        return self.classes[name]


def test_running_a_stage_releases_the_algorithms_of_earlier_stages(tmp_path):
    executor = Executor(tmp_path, Registry("recon", "segment"))
    executor.run("recon", "scene", {})
    recon = executor.algorithm("recon")
    executor.run("recon", "scene", {})
    assert executor.algorithm("recon") is recon
    executor.run("segment", "scene", {})
    assert list(executor._instances) == ["segment"]
