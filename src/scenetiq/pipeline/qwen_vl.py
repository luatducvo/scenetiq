"""Qwen3-VL: the vision-language model of the Captioning stage (ADR 0008).

One instance is shared by the Pipeline and the API. The model loads on first use and stays
on the GPU after the stage, so a later Query needs no load. Each generation holds the
model's lock, so callers on other threads take turns between two generations, never in
the middle of one.

Replies are forced to follow a JSON schema by xgrammar's transformers logits processor.
Each schema is compiled once per load; each processor serves one ``generate`` call only.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Any

from scenetiq.config import CaptioningOptions
from scenetiq.pipeline.captioning import CAPTION_PROMPT, CAPTION_SCHEMA
from scenetiq.pipeline.resident import ResidentModel


def fit_pixels(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """The largest size with the aspect ratio of ``width x height`` and at most ``max_pixels``.

    Sizes already within the budget are returned unchanged.
    """
    if width * height <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / (width * height))
    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


@dataclass
class _LoadedModel:
    """The loaded model and what it needs to generate; used only under the model's lock."""

    model: Any
    processor: Any
    compiler: Any
    grammars: dict[str, Any] = field(default_factory=dict)

    def grammar(self, schema: dict[str, Any]) -> Any:
        """The compiled grammar of ``schema``, compiled on its first use."""
        key = json.dumps(schema, sort_keys=True)
        if key not in self.grammars:
            self.grammars[key] = self.compiler.compile_json_schema(json.dumps(schema))
        return self.grammars[key]


class QwenVLCaptioner:
    """Qwen3-VL behind the ``Captioner`` interface; the model loads on first use and stays."""

    def __init__(self, options: CaptioningOptions | None = None, device: str = "cuda") -> None:
        """Configure the model; nothing is loaded until the first generation."""
        self.options = options or CaptioningOptions()
        self.device = device
        self._resident = ResidentModel(self._load)

    def _load(self) -> _LoadedModel:
        import torch
        import xgrammar
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(
            self.options.model_id, dtype=torch.bfloat16
        )
        model = model.to(self.device).eval()
        processor = AutoProcessor.from_pretrained(self.options.model_id)
        # The mask must cover every logit, so size it by the output layer, which is
        # padded beyond the tokenizer's vocabulary.
        vocab_size = model.get_output_embeddings().weight.shape[0]
        info = xgrammar.TokenizerInfo.from_huggingface(processor.tokenizer, vocab_size=vocab_size)
        return _LoadedModel(model, processor, xgrammar.GrammarCompiler(info))

    def generate(
        self, content: list[dict[str, Any]], schema: dict[str, Any], max_new_tokens: int
    ) -> str:
        """Return the greedy reply to one user message, forced to follow ``schema``.

        Holds the model's lock for this one generation, loading the model first if needed.

        Args:
            content: The user message's parts in the processor's chat format, such as
                ``{"type": "image", "image": <PIL image>}`` and ``{"type": "text", "text": ...}``.
            schema: JSON schema the reply must follow; compiled once per schema.
            max_new_tokens: Longest reply. A reply cut off here does not close its JSON.

        Returns:
            The decoded reply; it holds no GPU tensors.
        """
        import torch
        from xgrammar.contrib.hf import LogitsProcessor

        with self._resident.use() as loaded:
            inputs = loaded.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                generated = loaded.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    # A processor keeps its matcher state, so it serves one generate call only.
                    logits_processor=[LogitsProcessor(loaded.grammar(schema))],
                )
            reply = generated[:, inputs["input_ids"].shape[1] :]
            return loaded.processor.batch_decode(reply, skip_special_tokens=True)[0]

    def describe(self, images: list[Any]) -> str:
        """Return the model's schema-forced reply to :data:`CAPTION_PROMPT` for one Instance."""
        budget = self.options.max_pixels
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image.resize(fit_pixels(*image.size, budget))}
            for image in images
        ]
        content.append({"type": "text", "text": CAPTION_PROMPT})
        return self.generate(content, CAPTION_SCHEMA, self.options.max_new_tokens)
