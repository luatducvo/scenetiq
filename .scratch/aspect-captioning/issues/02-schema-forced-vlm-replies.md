# 02: Schema-forced VLM replies

**What to build:** The vision-language model's reply is forced, token by token, to follow one fixed JSON schema. Every key is required, Synonyms are capped at three and every string has a bounded length. A reply can then be lost only by being cut off at the token limit, and the token limit is raised so that four sentences fit. The GPU stack gains the constrained-decoding library and moves to the latest transformers patch release, without touching Colab's CUDA torch.

**Blocked by:** 01 (the schema's keys come from the Aspects)

**Status:** done

- [x] The schema below is a constant in the Captioning stage, built from the Aspects, and is not configurable. An empty Description is allowed.

  ```json
  {
    "type": "object",
    "properties": {
      "name":       {"type": "string", "minLength": 1, "maxLength": 40},
      "synonyms":   {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 3},
      "identity":   {"type": "string", "maxLength": 200},
      "appearance": {"type": "string", "maxLength": 200},
      "material":   {"type": "string", "maxLength": 200},
      "use":        {"type": "string", "maxLength": 200}
    },
    "required": ["name", "synonyms", "identity", "appearance", "material", "use"],
    "additionalProperties": false
  }
  ```

- [x] The Qwen3-VL wrapper compiles the schema once when the model loads, using its tokenizer and the model's padded vocabulary size. It passes a new xgrammar transformers logits processor to every `generate` call, and keeps greedy decoding and its own chat template. Releasing the model drops the compiled grammar too.
- [x] The reply parser stays as the second check. A reply cut off at the token limit gives no Caption, with the reason logged.
- [x] The default `max_new_tokens` for Captioning is 320, and the config tests check it.
- [x] The GPU requirements pin `xgrammar==0.2.8` and `transformers==5.22.2`, with a comment to confirm torch is unchanged after install.
- [x] A unit test checks that the schema's keys and required list match the Aspects plus Name and Synonyms.
- [x] The Seam B smoke test asserts at least one Instance has a Caption with a non-empty Name and at least one non-empty Description, and a Query returns `query_top_k` Matches with `caption_aspect` set.
- [x] AGENTS.md records two constraints: build one logits processor per `generate` call, and do not unpin xgrammar or transformers without checking Colab's torch. The configuration doc shows the new default. ADR 0008 mentions schema-forced replies.
- [x] Ruff, ty and pytest pass.
