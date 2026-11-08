"""Qwen3-VL: the vision-language model of the Captioning stage (ADR 0008).

The model is on the GPU only while the stage runs: the Pipeline calls :meth:`release`
when the stage ends, also on failure. Replies are forced to follow
:data:`~scenetiq.pipeline.captioning.CAPTION_SCHEMA` by xgrammar's transformers logits
processor; the schema is compiled once per load, and each processor serves one
``generate`` call only.
"""

import json
import math
from typing import Any

from scenetiq.config import CaptioningOptions
from scenetiq.pipeline.captioning import CAPTION_PROMPT, CAPTION_SCHEMA
from scenetiq.pipeline.gpu_memory import free_gpu_memory


def fit_pixels(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """The largest size with the aspect ratio of ``width x height`` and at most ``max_pixels``.

    Sizes already within the budget are returned unchanged.
    """
    if width * height <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / (width * height))
    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


class QwenVLCaptioner:
    """Qwen3-VL behind the ``Captioner`` interface; the model loads on first use."""

    def __init__(self, options: CaptioningOptions | None = None, device: str = "cuda") -> None:
        """Configure the model; nothing is loaded until :meth:`describe`."""
        self.options = options or CaptioningOptions()
        self.device = device
        self._model: Any = None
        self._processor: Any = None
        self._grammar: Any = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is None:
            import torch
            import xgrammar
            from transformers import AutoModelForImageTextToText, AutoProcessor

            model = AutoModelForImageTextToText.from_pretrained(
                self.options.model_id, dtype=torch.bfloat16
            )
            self._model = model.to(self.device).eval()
            self._processor = AutoProcessor.from_pretrained(self.options.model_id)
            # The mask must cover every logit, so size it by the output layer, which is
            # padded beyond the tokenizer's vocabulary.
            vocab_size = self._model.get_output_embeddings().weight.shape[0]
            info = xgrammar.TokenizerInfo.from_huggingface(
                self._processor.tokenizer, vocab_size=vocab_size
            )
            compiler = xgrammar.GrammarCompiler(info)
            self._grammar = compiler.compile_json_schema(json.dumps(CAPTION_SCHEMA))
        return self._model, self._processor, self._grammar

    def describe(self, images: list[Any]) -> str:
        """Return the model's schema-forced reply to :data:`CAPTION_PROMPT` for one Instance."""
        import torch
        from xgrammar.contrib.hf import LogitsProcessor

        model, processor, grammar = self._load()
        budget = self.options.max_pixels
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image.resize(fit_pixels(*image.size, budget))}
            for image in images
        ]
        content.append({"type": "text", "text": CAPTION_PROMPT})
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.options.max_new_tokens,
                do_sample=False,
                # A processor keeps its matcher state, so it serves one generate call only.
                logits_processor=[LogitsProcessor(grammar)],
            )
        reply = generated[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(reply, skip_special_tokens=True)[0]

    def release(self) -> None:
        """Drop the model and free its GPU memory; the next call loads it again."""
        self._model = None
        self._processor = None
        self._grammar = None
        free_gpu_memory()
