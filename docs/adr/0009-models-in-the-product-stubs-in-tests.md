# The product always runs the models; Stubs live in tests

`scenetiq.server` used to pick its Ports by a `pipeline` setting, `"fake"` by default, so the product carried test doubles and a GPU runtime that missed the setting served a made-up Scene without an error. Now `scenetiq.server.build_components` always builds the Model Pipeline and the model-backed encoders and parser; the setting is gone, and `src/scenetiq/` never imports `tests`. The Stubs (`StubPipeline`, `StubTextEncoder`, `StubCaptionEncoder`, `StubQueryParser` and the stub Scene) live in `tests/stubs/components.py`; `tests/stubs/server.py` serves them as the local backend for frontend work (`uvicorn --factory tests.stubs.server:build_app`).

## Considered Options

- Keep the setting and make it required: the product would still ship test doubles, and one wrong value still serves a made-up Scene.
- A stub package at the repo root, shared by tests and the local backend: one more top-level import name, for code only tests and local work use.

## Consequences

- The GPU runtime can no longer start a backend without the models; checking the tunnel alone costs the full install. Its notebook drops the `pipeline` key, and an old CONFIG that still has it is refused as an unknown setting before anything is installed.
- A new Port needs a Stub next to the others, or the integration tests cannot build the app.
