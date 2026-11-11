"""The real Pipeline: chains the GPU stages behind the ``Pipeline`` interface."""

from dataclasses import dataclass, field

import numpy as np

from scenetiq.config import (
    AlignmentOptions,
    CaptioningOptions,
    EmbeddingOptions,
    FusionOptions,
    MergingOptions,
    SuperpointOptions,
)
from scenetiq.domain import (
    CaptionEncoder,
    Capture,
    JobReporter,
    Reconstruction,
    Stage,
)
from scenetiq.geometry import build_instances
from scenetiq.pipeline.alignment import GeoCalibUpEstimator, UpEstimator, gravity_align
from scenetiq.pipeline.captioning import Captioner, caption_instances, embed_descriptions
from scenetiq.pipeline.embedding import ImageEncoder, embed_instances, plan_views
from scenetiq.pipeline.fusion import fuse
from scenetiq.pipeline.gpu_memory import free_gpu_memory, report_merge_input, track_stage_memory
from scenetiq.pipeline.masking import SAM2Masker
from scenetiq.pipeline.merging import merge_instances
from scenetiq.pipeline.primitives import voxel_primitives
from scenetiq.pipeline.qwen_embedding import Qwen3TextEmbedder
from scenetiq.pipeline.qwen_vl import QwenVLCaptioner
from scenetiq.pipeline.reconstruct import DA3Reconstructor, Reconstructor
from scenetiq.pipeline.siglip import SigLIP2Encoder
from scenetiq.pipeline.superpoints import mesh_superpoints


@dataclass
class RealPipelineOptions:
    """Tuning of the stages that are plain functions (models carry their own options).

    Attributes:
        fusion: TSDF Fusing stage (ADR 0002).
        alignment: Aligning stage (ADR 0007).
        superpoints: Superpoints stage.
        merging: MV3DIS Merging stage.
        embedding: SigLIP2 Embedding stage.
        captioning: Captioning stage (ADR 0008).
    """

    fusion: FusionOptions = field(default_factory=FusionOptions)
    alignment: AlignmentOptions = field(default_factory=AlignmentOptions)
    superpoints: SuperpointOptions = field(default_factory=SuperpointOptions)
    merging: MergingOptions = field(default_factory=MergingOptions)
    embedding: EmbeddingOptions = field(default_factory=EmbeddingOptions)
    captioning: CaptioningOptions = field(default_factory=CaptioningOptions)


class RealPipeline:
    """Turns a Capture into a Reconstruction with the real model stack.

    Args:
        reconstructor: The Reconstructing stage (DA3 by default, ADR 0001).
        masker: The Masking stage (SAM2 automatic masks by default, ADR 0003).
        image_encoder: Image tower of the Embedding stage (SigLIP2 by default). Pass
            the same encoder the API uses for Query text.
        options: Settings of the remaining stages.
        up_estimator: The Aligning stage's source of up vectors (GeoCalib by default,
            ADR 0007).
        captioner: The Captioning stage's vision-language model (Qwen3-VL by default,
            ADR 0008). It stays on the GPU after the stage: pass the captioner built in
            ``server.build_components``.
        caption_encoder: Embeds Captions (Qwen3-Embedding by default). Pass the same
            encoder the API uses for Query text.
    """

    def __init__(
        self,
        reconstructor: Reconstructor | None = None,
        masker: SAM2Masker | None = None,
        image_encoder: ImageEncoder | None = None,
        options: RealPipelineOptions | None = None,
        up_estimator: UpEstimator | None = None,
        captioner: Captioner | None = None,
        caption_encoder: CaptionEncoder | None = None,
    ) -> None:
        """Assemble the stages; models load lazily on the first run."""
        self.reconstructor = reconstructor or DA3Reconstructor()
        self.masker = masker or SAM2Masker()
        self.image_encoder = image_encoder or SigLIP2Encoder()
        self.options = options or RealPipelineOptions()
        self.up_estimator = up_estimator or GeoCalibUpEstimator()
        self.captioner = captioner or QwenVLCaptioner(self.options.captioning)
        self.caption_encoder = caption_encoder or Qwen3TextEmbedder()

    def run(self, capture: Capture, reporter: JobReporter) -> Reconstruction:
        """Run every stage on ``capture``, reporting each one through ``reporter``."""
        opts = self.options
        # A job that ran out of memory could only be collected after its exception was handled.
        free_gpu_memory()

        reporter.stage(Stage.RECONSTRUCTING)
        with track_stage_memory(Stage.RECONSTRUCTING):
            view_data = self.reconstructor.reconstruct(capture.views)
        reporter.note(f"Reconstructed {len(capture.views)} Views.")

        reporter.stage(Stage.FUSING)
        geometry = fuse(view_data, opts.fusion)

        reporter.stage(Stage.ALIGNING)
        with track_stage_memory(Stage.ALIGNING):
            aligned, view_data = gravity_align(
                geometry, view_data, self.up_estimator, opts.alignment
            )
        # gravity_align returns its inputs unchanged when the Views do not agree on up.
        if aligned is geometry:
            reporter.note(
                "The Scene could not be aligned to gravity: its Views did not agree on up, "
                "so above, below, on, under, tallest and lowest may be wrong."
            )
        geometry = aligned
        points = geometry.vertices

        reporter.stage(Stage.SUPERPOINTS)
        superpoints = opts.superpoints
        if superpoints.primitives == "superpoints":
            primitive_ids = mesh_superpoints(
                points, geometry.triangles, superpoints.k_thresh, superpoints.min_verts
            )
        else:
            primitive_ids = voxel_primitives(points, superpoints.primitive_voxel_size)

        reporter.stage(Stage.MASKING)
        with track_stage_memory(Stage.MASKING):
            view_masks = self.masker.masks_for(view_data, reporter.checkpoint)

        reporter.stage(Stage.MERGING)
        report_merge_input(len(points), len(np.unique(primitive_ids)))
        with track_stage_memory(Stage.MERGING):
            instance_ids = merge_instances(
                points, primitive_ids, view_data, view_masks, opts.merging
            )
        free_gpu_memory()

        reporter.stage(Stage.EMBEDDING)
        with track_stage_memory(Stage.EMBEDDING):
            plans = plan_views(points, instance_ids, view_data, opts.embedding)
            embeddings = embed_instances(
                plans,
                view_data,
                [v.path for v in capture.views],
                self.image_encoder,
                opts.embedding,
            )
        # Instances no View sees well enough cannot be queried: drop them.
        merged = len(np.unique(instance_ids[instance_ids >= 0]))
        unseen = np.isin(instance_ids, list(embeddings), invert=True)
        instance_ids = np.where(unseen, -1, instance_ids).astype(np.int32)
        reporter.note(
            f"Found {len(embeddings)} Instances; dropped {merged - len(embeddings)} "
            "that no View sees well enough."
        )

        reporter.stage(Stage.CAPTIONING)
        with track_stage_memory(Stage.CAPTIONING):
            # The vision-language model stays on the GPU after the stage, also on failure.
            captions = caption_instances(
                {iid: plan for iid, plan in plans.items() if iid in embeddings},
                [v.path for v in capture.views],
                view_data.cameras,
                self.captioner,
                opts.captioning,
                reporter.checkpoint,
            )
            caption_data = embed_descriptions(captions, self.caption_encoder)
        reporter.note(f"{len(captions)} of {len(embeddings)} Instances got a Caption.")

        return Reconstruction(
            points=points,
            colors=geometry.colors,
            instance_ids=instance_ids,
            cameras=view_data.cameras,
            instances=build_instances(points, instance_ids, embeddings, caption_data),
        )
