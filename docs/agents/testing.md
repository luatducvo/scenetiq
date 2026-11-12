# Testing

## Seams

- **Seam A**, `tests/api/`: the HTTP API through TestClient, with `FakePipeline`, `FakeTextEncoder` and `FakeCaptionEncoder` (from `scenetiq.fakes`) on a temp storage root. Most behaviour is tested here.
- **Unit**, `tests/unit/`: pure, model-free functions, wherever they live (stage modules included).
- **Seam B**, `tests/eval/`: the real Pipeline on GPU, marked `@pytest.mark.gpu`, deselected by default.

## Model-loading code

Anything that loads a model (DA3, SAM2, SigLIP2, Qwen3-VL, Qwen3-Embedding, a full MV3DIS run, `RealPipeline.run`) is covered by Seam B alone. Elsewhere, stand it in only with the fakes in `scenetiq.fakes`, and only at Seam A; write no other mocks of it and no unit tests of it. Constructing those classes is fine in any test, because models load on first use.

## Expected values

Derive every expectation from hand-worked values, so a test checks the formula instead of restating it.

## Suites the CPU run skips

`tests/unit/test_fusion.py` and `tests/unit/test_plane_search.py` need open3d. After changing fusion or the plane search, run:

```bash
uv run --with open3d==0.20.0 pytest tests/unit/test_fusion.py tests/unit/test_plane_search.py
```
