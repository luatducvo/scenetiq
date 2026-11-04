"""The real Pipeline: chains the GPU stages behind the ``Pipeline`` interface."""

from dataclasses import dataclass, field

import numpy as np

from scenetiq.config import EmbeddingOptions, FusionOptions, MergingOptions, SuperpointOptions
from scenetiq.domain import Capture, Reconstruction, Stage, StageCallback
from scenetiq.geometry import build_instances
from scenetiq.pipeline.embedding import ImageEncoder, embed_instances
from scenetiq.pipeline.fusion import fuse
from scenetiq.pipeline.masking import SAM2Masker
from scenetiq.pipeline.merging import merge_instances
from scenetiq.pipeline.primitives import voxel_primitives
from scenetiq.pipeline.reconstruct import DA3Reconstructor, Reconstructor
from scenetiq.pipeline.siglip import SigLIP2Encoder
from scenetiq.pipeline.superpoints import mesh_superpoints


@dataclass
class RealPipelineOptions:
    """Tuning of the stages that are plain functions (models carry their own options).

    Attributes:
        fusion: TSDF Fusing stage (ADR 0002).
        superpoints: Superpoints stage.
        merging: MV3DIS Merging stage.
        embedding: SigLIP2 Embedding stage.
    """

    fusion: FusionOptions = field(default_factory=FusionOptions)
    superpoints: SuperpointOptions = field(default_factory=SuperpointOptions)
    merging: MergingOptions = field(default_factory=MergingOptions)
    embedding: EmbeddingOptions = field(default_factory=EmbeddingOptions)


class RealPipeline:
    """Turns a Capture into a Reconstruction with the real model stack.

    Args:
        reconstructor: The Reconstructing stage (DA3 by default, ADR 0001).
        masker: The Masking stage (SAM2 automatic masks by default, ADR 0003).
        image_encoder: Image tower of the Embedding stage (SigLIP2 by default). Pass
            the same encoder the API uses for Query text.
        options: Settings of the remaining stages.
    """

    def __init__(
        self,
        reconstructor: Reconstructor | None = None,
        masker: SAM2Masker | None = None,
        image_encoder: ImageEncoder | None = None,
        options: RealPipelineOptions | None = None,
    ) -> None:
        """Assemble the stages; models load lazily on the first run."""
        self.reconstructor = reconstructor or DA3Reconstructor()
        self.masker = masker or SAM2Masker()
        self.image_encoder = image_encoder or SigLIP2Encoder()
        self.options = options or RealPipelineOptions()

    def run(self, capture: Capture, on_stage: StageCallback) -> Reconstruction:
        """Run every stage on ``capture``, reporting each one through ``on_stage``."""
        opts = self.options
        on_stage(Stage.RECONSTRUCTING)
        view_data = self.reconstructor.reconstruct(capture.views)

        on_stage(Stage.FUSING)
        geometry = fuse(view_data, opts.fusion)
        points = geometry.vertices

        on_stage(Stage.SUPERPOINTS)
        superpoints = opts.superpoints
        if superpoints.primitives == "superpoints":
            primitive_ids = mesh_superpoints(
                points, geometry.triangles, superpoints.k_thresh, superpoints.min_verts
            )
        else:
            primitive_ids = voxel_primitives(points, superpoints.primitive_voxel_size)

        on_stage(Stage.MASKING)
        view_masks = self.masker.masks_for(view_data)

        on_stage(Stage.MERGING)
        instance_ids = merge_instances(points, primitive_ids, view_data, view_masks, opts.merging)

        on_stage(Stage.EMBEDDING)
        embeddings = embed_instances(
            points,
            instance_ids,
            view_data,
            [v.path for v in capture.views],
            self.image_encoder,
            opts.embedding,
        )
        # Instances no View sees well enough cannot be queried: drop them.
        unseen = np.isin(instance_ids, list(embeddings), invert=True)
        instance_ids = np.where(unseen, -1, instance_ids).astype(np.int32)

        return Reconstruction(
            points=points,
            colors=geometry.colors,
            instance_ids=instance_ids,
            cameras=view_data.cameras,
            instances=build_instances(points, instance_ids, embeddings),
        )
