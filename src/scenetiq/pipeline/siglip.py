"""SigLIP2: one model encodes Instance crops (Embedding stage) and Query text.

The same instance serves as the Pipeline's image encoder and the API's
``TextEncoder``, so the model is loaded once per process and stays on the GPU
for Queries. Queries and the job run on different threads, so loading takes a lock.

The checkpoint loads ``SiglipProcessor``, which does not truncate text, and the text tower
has 64 positions: Query text must be truncated explicitly, or a long Query raises.
"""

import logging
import threading
from typing import Any

import numpy as np

from scenetiq.config import SigLIP2Options
from scenetiq.domain import FloatArray
from scenetiq.query import QUERY_LOGGER

MAX_TEXT_TOKENS = 64
"""Positions of the SigLIP2 text tower; it was trained on text padded to this length."""

_logger = logging.getLogger(QUERY_LOGGER)


def siglip_query_text(text: str, prompt: str, prompt_max_words: int) -> str:
    """The text SigLIP2 encodes for a Query.

    A short Query, like a label, is wrapped in ``prompt``; a longer one is kept as
    written. The SigLIP2 paper lowercases all training text, so the result is lowercased,
    template included.

    Args:
        text: The Query text.
        prompt: Template with ``{}`` for the text.
        prompt_max_words: Longest Query, in words, that is wrapped; 0 never wraps.
    """
    text = text.strip()
    if 0 < len(text.split()) <= prompt_max_words:
        text = prompt.format(text)
    return text.lower()


class SigLIP2Encoder:
    """SigLIP2 image and text towers; the model loads on first use."""

    def __init__(self, options: SigLIP2Options | None = None, device: str = "cuda") -> None:
        """Configure the encoder; nothing is loaded until first used."""
        self.options = options or SigLIP2Options()
        self.device = device
        self._model: Any = None
        self._processor: Any = None
        self._load_lock = threading.Lock()

    def _load(self) -> tuple[Any, Any]:
        with self._load_lock:
            if self._model is None:
                import torch
                from transformers import AutoModel, AutoProcessor

                model = AutoModel.from_pretrained(self.options.model_id, dtype=torch.bfloat16).to(
                    self.device
                )
                model.eval()
                self._processor = AutoProcessor.from_pretrained(self.options.model_id)
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
                feats = model.get_image_features(**inputs).pooler_output.float()
            feats = torch.nn.functional.normalize(feats, dim=-1)
            out.append(feats.cpu().numpy())
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    def encode(self, text: str) -> FloatArray:
        """Return the L2-normalised embedding of Query ``text``."""
        import torch

        model, processor = self._load()
        query = siglip_query_text(text, self.options.prompt, self.options.prompt_max_words)
        length = len(processor.tokenizer(query)["input_ids"])
        if length > MAX_TEXT_TOKENS:
            # The Caption channel still sees the whole Query; never log its text.
            _logger.info("siglip query truncated to %d tokens (had %d)", MAX_TEXT_TOKENS, length)
        inputs = processor(
            text=[query],
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_TOKENS,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            feats = model.get_text_features(**inputs).pooler_output.float()
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats[0].cpu().numpy().astype(np.float32)
