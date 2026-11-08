"""Qwen3-Embedding: embeds Caption documents (Captioning stage) and Query text (ADR 0008).

The same instance serves the Pipeline and the API, so the model is loaded once per process
and stays on the GPU for Queries. Queries and the job run on different threads, so loading
takes a lock.
"""

import threading
from typing import Any

import numpy as np

from scenetiq.config import CaptionEmbeddingOptions
from scenetiq.domain import FloatArray


class Qwen3TextEmbedder:
    """Qwen3-Embedding behind the ``CaptionEncoder`` interface; the model loads on first use."""

    def __init__(
        self, options: CaptionEmbeddingOptions | None = None, device: str = "cuda"
    ) -> None:
        """Configure the encoder; nothing is loaded until first used."""
        self.options = options or CaptionEmbeddingOptions()
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_lock = threading.Lock()

    def _load(self) -> tuple[Any, Any]:
        with self._load_lock:
            if self._model is None:
                import torch
                from transformers import AutoModel, AutoTokenizer

                # Left padding puts every sequence's last token in the last column.
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.options.model_id, padding_side="left"
                )
                model = AutoModel.from_pretrained(self.options.model_id, dtype=torch.bfloat16)
                self._model = model.to(self.device).eval()
            return self._model, self._tokenizer

    def _encode(self, texts: list[str]) -> FloatArray:
        import torch

        model, tokenizer = self._load()
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.options.batch_size):
            batch = tokenizer(
                texts[start : start + self.options.batch_size],
                padding=True,
                truncation=True,
                max_length=self.options.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = model(**batch).last_hidden_state[:, -1].float()
            out.append(torch.nn.functional.normalize(hidden, dim=-1).cpu().numpy())
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    def encode_documents(self, texts: list[str]) -> FloatArray:
        """Return ``(len(texts), D)`` L2-normalised embeddings of Caption documents."""
        return self._encode(texts)

    def encode_query(self, text: str) -> FloatArray:
        """Return the L2-normalised embedding of Query ``text``, with the query instruction."""
        prompt = f"Instruct: {self.options.query_instruction}\nQuery:{text.strip()}"
        return self._encode([prompt])[0]
