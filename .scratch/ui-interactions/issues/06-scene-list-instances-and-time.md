# 06: Scene list: Instance count and processing time

**What to build:** Each Scene in the list shows its number of Instances and, once finished, how long it took to process. Both are blank while the Scene is processing, and for a failed or interrupted Scene.

Spec: *Scene list* user stories 37–39; *HTTP API* (`GET /scenes`); *Storage* (`instance_count`); *Frontend* (Scene list).

**Blocked by:** 05 (the timestamps and the schema migration)

**Status:** ready-for-agent

- [ ] The store sets `instance_count` from the Reconstruction's Instances when it is saved. The schema migration of ticket 05 also covers this column.
- [ ] `GET /scenes` adds `started_at`, `finished_at` and `instance_count`. `instance_count` is null unless the job succeeded. All existing fields are unchanged.
- [ ] Seam A tests: a succeeded fake Scene lists `instance_count` 3 and both timestamps, and a failed one lists null `instance_count` and `finished_at` set.
- [ ] The Scene list shows an Instances column and a processing time column, formatted as minutes and seconds from the two timestamps, blank when missing.
- [ ] The README mentions the new columns.
- [ ] Ruff, ty and pytest are green, and the frontend passes lint, typecheck and build.
- [ ] Manual check against the local fake backend: a finished fake Scene shows 3 Instances and a processing time of a few seconds.
