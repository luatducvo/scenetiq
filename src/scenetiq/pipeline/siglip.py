import logging
import threading
from typing import Any

import numpy as np

from scenetiq.config import SigLIP2Options
from scenetiq.domain import FloatArray
from scenetiq.query import QUERY_LOGGER

MAX_TEXT_TOKENS = 64

_logger = logging.getLogger(QUERY_LOGGER)


def siglip_query_text(text: str, prompt: str, prompt_max_words: int) -> str:
    text = text.strip()
    if 0 < len(text.split()) <= prompt_max_words:
        text = prompt.format(text)
    return text.lower()


class SigLIP2Encoder:
    def __init__(self, options: SigLIP2Options | None = None, device: str = "cuda") -> None:
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
