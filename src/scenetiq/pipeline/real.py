"""The real Pipeline: chains the GPU stages behind the ``Pipeline`` interface."""

import numpy as np

from scenetiq.domain import Capture, Reconstruction, Stage, StageCallback
from scenetiq.pipeline.fusion import FusionOptions, fuse
from scenetiq.pipeline.reconstruct import DA3Reconstructor, Reconstructor


class RealPipeline:
    """Turns a Capture into a Reconstruction with the real model stack.

    Args:
        reconstructor: The Reconstructing stage (DA3 by default, ADR 0001).
        fusion: Options of the TSDF Fusing stage (ADR 0002).
    """

    def __init__(
        self,
        reconstructor: Reconstructor | None = None,
        fusion: FusionOptions | None = None,
    ) -> None:
        """Assemble the stages; models load lazily on the first run."""
        self.reconstructor = reconstructor or DA3Reconstructor()
        self.fusion = fusion or FusionOptions()

    def run(self, capture: Capture, on_stage: StageCallback) -> Reconstruction:
        """Run every stage on ``capture``, reporting each one through ``on_stage``."""
        on_stage(Stage.RECONSTRUCTING)
        frames = self.reconstructor.reconstruct(capture.views)

        on_stage(Stage.FUSING)
        geometry = fuse(frames, self.fusion)

        return Reconstruction(
            points=geometry.vertices,
            colors=geometry.colors,
            instance_ids=np.full(len(geometry.vertices), -1, dtype=np.int32),
            cameras=frames.cameras,
        )
