# 07: Run the backend on Colab behind a Cloudflare named tunnel

**What to build:** The owner runs one Colab notebook. It starts the backend on the Colab GPU, stores all data on Google Drive, and exposes the backend at a stable Cloudflare named-tunnel URL. The localhost frontend works end to end against it, still using the fake Pipeline. See ADR 0004.

Spec: `.scratch/scene-reconstruction-query/spec.md` (Deployment)

**Blocked by:** 02 (walking skeleton)

**Status:** ready-for-agent

- [ ] The bootstrap notebook:
  - [ ] mounts Drive
  - [ ] installs the package
  - [ ] reads the token and tunnel credentials from Colab secrets
  - [ ] sets the storage root to a Drive folder
  - [ ] starts uvicorn in the background
  - [ ] starts the Cloudflare named tunnel
- [ ] The notebook contains no business logic. Everything it calls lives in the package.
- [ ] Running the notebook again after a runtime restart brings back the same Scenes at the same URL. Restart recovery (ticket 04) applies if that ticket has landed.
- [ ] Configuration selects the Pipeline used on Colab. The notebook defaults to `PIPELINE = "real"` (switched by ticket 08); `"fake"` remains for testing the deployment without models.
- [ ] AGENTS.md documents how to run the notebook, the Colab secrets it needs, and the one-time Cloudflare named-tunnel setup.
- [ ] The Definition of Done in AGENTS.md is met. The notebook is exempt from docstring rules; the package code it calls is not.
