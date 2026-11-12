# Deployment and configuration

## Colab notebook

Deployment is `notebooks/colab_server.ipynb` only. The notebook holds parameters; its logic belongs in `scenetiq.colab`.

- `scenetiq.colab`, `scenetiq.config` and `scenetiq.runtime_log` use only the standard library, because the notebook imports them before anything is installed.
- Colab logs are never written to files or Drive. `colab_logging.json` decides what reaches the server's stdout, which the notebook's last cell displays. Stderr is kept only to explain a crash, so anything the owner must see goes through a logger that file lets through.

## Configuration

Every default lives in `scenetiq.config`, stage Options included. Add a new setting there and pass it to its stage in `server.build_components`. The notebook's `CONFIG` cell overrides it through `SCENETIQ_CONFIG`; only the token travels separately.

## Uploads

Cloudflare returns 413 for large bodies (105 MB was refused), so a Capture is uploaded one View per request (ADR 0006).
