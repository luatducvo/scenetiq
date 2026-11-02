# Configuration

Choosing an Algorithm, its params or a Target is a configuration change, not a code change.

## Files

| File | Holds |
| --- | --- |
| `backend/config/pipeline.yaml` | the real Algorithm for each Stage and its params |
| `backend/config/pipeline.fake.yaml` | Fake Algorithms on the in-process `cpu` Target; the default when `PIPELINE_CONFIG` is unset, and what the tests use |
| `backend/config/compute.yaml` | every Target: `inprocess` or `http` with URL, token, chunk size, timeouts |
| `backend/config/vocabularies/*.txt` | Vocabularies for the segment Stage, one concept per line |
| `backend/.env.example` | every backend environment variable, with a comment and the value the Colab notebook sets |
| `frontend/.env.example` | `VITE_API_URL` (the Colab tunnel URL), `VITE_API_MODE` (`http` or `mock`) |

## Rules

- A pipeline file declares all eight Stages in order and one `target` that every Stage runs on.
- `${VAR}` in a YAML file expands from the environment. In a pipeline file an unset variable stays literal, so the worker's start-up error names it.
- `PIPELINE_CONFIG`, `COMPUTE_CONFIG` and `GPU_TARGET` select the files and the Target.
- A new backend environment variable goes into `backend/.env.example` with a comment, and into the environment the notebook's backend cell (`backend/deploy/colab/GPU.ipynb`) passes to each process. A new frontend variable goes into `frontend/.env.example`.
- Secrets (`HF_TOKEN`) come from Colab secrets; never commit, print or copy them.
