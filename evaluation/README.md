# Evaluation

Checks the Model Pipeline on a GPU runtime. `uv run pytest` does not collect this directory, and without CUDA every test here is skipped (ADR 0008).

## The suite

```bash
SCENETIQ_SMOKE_CAPTURE=/path/to/photos uv run pytest evaluation -s
```

| Variable | Needed by |
|---|---|
| `SCENETIQ_SMOKE_CAPTURE` | everything that runs the Pipeline: a directory of photos of one static Scene |
| `SCENETIQ_SMOKE_QUERY`, `SCENETIQ_SMOKE_QUERY_BOX` | `test_queries.py`: an object visible in the Capture and its box `view,left,top,right,bottom` in that photo (`view` indexes the sorted photos) |
| `SCENETIQ_SMOKE_SPATIAL_QUERY` | optional; a Query with a Relation, default "the object nearest to the chair" |
| `SCENETIQ_CONFIG` | optional; the same settings JSON the backend reads |

- `test_pipeline.py`: the stages, the point cloud, Instances, Captions, Gravity alignment and GPU memory after a run.
- `test_models.py`: each model on its own (SigLIP2, Qwen3-Embedding, GeoCalib).
- `test_queries.py`: Queries with and without a Relation against the reconstructed Scene.

## Scoring Queries against hand-labelled Instances

Against a running backend, with `SCENETIQ_TOKEN` set:

```bash
python -m scenetiq.evaluation --url https://<backend> --instances <scene_id>   # find the Instance ids
python -m scenetiq.evaluation --url https://<backend> --labels labels.json
```

`labels.example.json` shows the format: one object per Query, with the Scene, the Query text and every Instance it should find. The command prints precision, recall and whether the best Match is right, per Query and averaged.
