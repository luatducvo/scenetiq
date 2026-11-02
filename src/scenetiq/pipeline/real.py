"""The real Pipeline: chains the GPU stages behind the ``Pipeline`` interface."""

from dataclasses import dataclass, field

import numpy as np

from scenetiq.domain import Capture, Reconstruction, Stage, StageCallback
from scenetiq.geometry import build_instances
from scenetiq.pipeline.fusion import FusionOptions, fuse
from scenetiq.pipeline.masking import SAM2Masker
from scenetiq.pipeline.merging import MergingOptions, merge_instances
from scenetiq.pipeline.primitives import voxel_primitives
from scenetiq.pipeline.reconstruct import DA3Reconstructor, Reconstructor


@dataclass
class RealPipelineOptions:
    """Tuning of the stages that are plain functions (models carry their own options).

    Attributes:
        fusion: TSDF Fusing stage (ADR 0002).
        merging: MV3DIS Merging stage.
        primitive_voxel_size: Edge, in metres, of the voxel-grid primitives fed to MV3DIS.
    """

    fusion: FusionOptions = field(default_factory=FusionOptions)
    merging: MergingOptions = field(default_factory=MergingOptions)
    primitive_voxel_size: float = 0.15


class RealPipeline:
    """Turns a Capture into a Reconstruction with the real model stack.

    Args:
        reconstructor: The Reconstructing stage (DA3 by default, ADR 0001).
        masker: The Masking stage (SAM2 automatic masks by default, ADR 0003).
        options: Settings of the remaining stages.
    """

    def __init__(
        self,
        reconstructor: Reconstructor | None = None,
        masker: SAM2Masker | None = None,
        options: RealPipelineOptions | None = None,
    ) -> None:
        """Assemble the stages; models load lazily on the first run."""
        self.reconstructor = reconstructor or DA3Reconstructor()
        self.masker = masker or SAM2Masker()
        self.options = options or RealPipelineOptions()

    def run(self, capture: Capture, on_stage: StageCallback) -> Reconstruction:
        """Run every stage on ``capture``, reporting each one through ``on_stage``."""
        opts = self.options
        on_stage(Stage.RECONSTRUCTING)
        frames = self.reconstructor.reconstruct(capture.views)

        on_stage(Stage.FUSING)
        geometry = fuse(frames, opts.fusion)
        points = geometry.vertices

        on_stage(Stage.SUPERPOINTS)
        primitive_ids = voxel_primitives(points, opts.primitive_voxel_size)

        on_stage(Stage.MASKING)
        view_masks = self.masker.masks_for(frames)

        on_stage(Stage.MERGING)
        instance_ids = merge_instances(points, primitive_ids, frames, view_masks, opts.merging)

        return Reconstruction(
            points=points,
            colors=geometry.colors,
            instance_ids=instance_ids.astype(np.int32),
            cameras=frames.cameras,
            instances=build_instances(points, instance_ids),
        )
