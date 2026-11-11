# 04: Heatmap Matches with an adjustable K

**What to build:**

- A Query returns 5 Matches by default. A slider in the viewer sets K from 1 to a configured maximum and re-runs the current Query when changed.
- Each Match's points are coloured by a heatmap of its score, stretched over the scores of the Matches shown: the best shown Match gets the hottest colour, the lowest shown Match the coolest, and the others in proportion to their score. No colour legend is shown. The list swatches use the same colours.
- The ranking is unchanged: `rrf_k` stays 60 and the API's scores are untouched; the stretch is frontend-only.

Spec: *Queries and heatmap* user stories 14–19 and 28; *Configuration*; *HTTP API* (`top_k`, `/options`); *Frontend* (viewer heatmap, K slider); *Further Notes* on why the scale is stretched.

**Blocked by:** 02 (`GET /options`), 03 (the viewer's colouring and Match list)

**Status:** ready-for-agent

- [ ] Settings defaults: `query_top_k` 5, new `query_max_top_k` 20 (refused below `query_top_k`). `rrf_k` stays 60. Config unit tests cover the new validation.
- [ ] `POST /scenes/{id}/query` accepts an optional `top_k`:
  - a missing value falls back to `query_top_k`;
  - 0 or a value above `query_max_top_k` answers 422;
  - it applies to plain and spatial Queries, including the wanted object's shortlist.
- [ ] `GET /options` also returns `query_top_k` and `query_max_top_k`.
- [ ] Seam A tests cover `top_k` returning that many Matches best first, the fallback, both 422 cases, and a spatial Query honouring `top_k`.
- [ ] The viewer colours each Match's points by a heatmap of its score stretched over the Matches shown (position = (score − lowest) / (highest − lowest); all hottest when the scores are equal or K = 1), shows no colour legend, and uses the same colour for each list swatch. The rest of the Scene keeps its original colour.
- [ ] The K slider runs from 1 to `query_max_top_k`, starts at `query_top_k`, is sent as `top_k`, and re-runs the shown Query when changed.
- [ ] The configuration docs and the pipeline docs show the new `query_top_k` default and `query_max_top_k`. The README describes the heatmap and the K slider.
- [ ] Ruff, ty and pytest are green, and the frontend passes lint, typecheck and build.
- [ ] Manual check against the local fake backend: `furniture light` with K = 3 shows three Instances in visibly different heatmap colours (hottest to coolest), and moving the slider to 1 leaves one.
