"""SigLIP2: one model encodes Instance crops (Embedding stage) and Query text.

The same instance serves as the Pipeline's image encoder and the API's
``TextEncoder``, so the model is loaded once per process.
"""

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from scenetiq.config import SigLIP2Options
from scenetiq.domain import FloatArray


def sigmoid_score(cosine: npt.ArrayLike, logit_scale: float, logit_bias: float) -> FloatArray:
    """Turn cosine similarity into SigLIP's match probability ``sigmoid(scale * cos + bias)``."""
    z = logit_scale * np.asarray(cosine, dtype=np.float64) + logit_bias
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


class SigLIP2Encoder:
    """SigLIP2 image and text towers; the model loads on first use."""

    def __init__(self, options: SigLIP2Options | None = None, device: str = "cuda") -> None:
        """Configure the encoder; nothing is loaded until first used."""
        self.options = options or SigLIP2Options()
        self.device = device
        self._model: Any = None
        self._processor: Any = None
        self._scale = 1.0
        self._bias = 0.0

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoProcessor

            model = AutoModel.from_pretrained(self.options.model_id, torch_dtype=torch.bfloat16).to(
                self.device
            )
            model.eval()
            self._processor = AutoProcessor.from_pretrained(self.options.model_id)
            self._scale = math.exp(float(model.logit_scale.float().item()))
            self._bias = float(model.logit_bias.float().item())
            self._model = model
        return self._model, self._processor

    def encode_images(self, images: list[Any]) -> FloatArray:
        """Return ``(len(images), D)`` L2-normalised embeddings of PIL images."""
        import torch

        model, processor = self._load()
        out: list[np.ndarray] = []
        for start in range(0, len(images), self.options.batch_size):
            batch = images[start : start + self.options.batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                feats = model.get_image_features(**inputs).float()
            feats = torch.nn.functional.normalize(feats, dim=-1)
            out.append(feats.cpu().numpy())
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    def encode(self, text: str) -> FloatArray:
        """Return the L2-normalised embedding of Query ``text``."""
        import torch

        model, processor = self._load()
        prompt = self.options.prompt.format(text.strip().lower())
        inputs = processor(
            text=[prompt], padding="max_length", max_length=64, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            feats = model.get_text_features(**inputs).float()
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats[0].cpu().numpy().astype(np.float32)

    def to_score(self, cosine: FloatArray) -> FloatArray:
        """Map cosine similarities to SigLIP match probabilities in [0, 1]."""
        self._load()
        return sigmoid_score(cosine, self._scale, self._bias)
