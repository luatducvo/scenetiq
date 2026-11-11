# Matches by standardized similarity with a relative cut, and Instance masks in the crops

A Query for "cái ghế" (a chair) coloured the floor as a Match. Reading the code gave five causes, none of them a bug in the usual sense:

- ADR 0008 returned the top K of a Reciprocal Rank Fusion with no cut, so a Scene with two chairs and K = 5 always returned three Instances that are not chairs.
- Reciprocal Rank Fusion keeps only ranks. With `rrf_k = 60` the Instance ranked 30th on both channels still scored 0.68, and a real chair and the floor were one rank apart. The viewer then stretched the colours over the Matches shown, so every Query looked equally confident.
- An Instance was cropped by the bounding box of its pixels. The box of a floor or a wall is most of the photo, with every chair in it, so its image embedding was the embedding of the room, and the vision-language model was asked to describe "the object inside the red box" with the same box.
- The Caption similarity was the maximum over all four Descriptions, so the floor's Use ("for walking on and placing furniture such as chairs") could answer a Query that only names a kind of thing.
- A Query without a Relation was ranked with its original text, so a Vietnamese Query reached SigLIP2 inside an English template and was compared with English Descriptions, although the parser had already written an English phrase for it.

This ADR supersedes the ranking, the cut and the "original text" rule of ADR 0008 and ADR 0009. Captions, the two channels, the parser and the solver stay as those ADRs describe them.

**Fusion.** On each channel the similarities are standardized over the Reconstruction's Instances (subtract the mean, divide by the population standard deviation; all zero when they do not differ). An Instance's fused value is the mean of its two standardized values. An Instance without a Caption takes 0 on the Caption channel, the average of the captioned ones: no evidence either way. A Reconstruction where no Instance has a Caption uses the image channel alone. A Match's score is its fused value divided by the best one, so the best Match always scores 1.

**Cut.** A Query without a Relation keeps an Instance as a Match when its score is at least `query_score_ratio` (0.5) of the best one's, and returns at most K of them. K is now a maximum. The best Instance always passes, so a Query on a Reconstruction with Instances still has at least one Match. When no Instance stands out (every fused value is 0), every score is 1 and the first K are returned. The phrase rankings of a spatial Query are not cut: its candidates are still chosen by the candidate margin of ADR 0009.

**Aspects of a phrase.** The parser now also says, for each object, which Aspects its phrase speaks about. The Caption similarity is the best Description among those Aspects; an Instance with no Description of any of them is compared on all it has, and so is an object for which the parser named none. "chair" reads only Identity, so a floor's Use no longer answers it; "something to sit on" reads Use.

**Text of a plain Query.** A Query without a Relation is ranked by the parsed object's English phrase and Aspects when it names one object, and so is a Query whose Relations found no valid assignment (by its wanted object). A Query the parser could not read, or one that lists several objects without relating them, is still ranked by its original text on every Aspect.

**Instance masks.** The plan of an Instance's Views now carries, per visible point, a splat radius: half the pixel distance between neighbouring points at that depth (the TSDF voxel size times the focal length over the depth), rounded up and capped. The squares of an Instance's points are its mask in that View. The Embedding stage adds, per View, one crop at the tightest scale with everything outside the mask filled with grey; the embedding is `masked_weight` (0.5) of the normalised mean of those crops plus the rest of the normalised mean of the context crops. The Captioning stage darkens everything outside the mask to `background_dim` (0.35) of its brightness before drawing the red box, and the prompt says that the object is only the part at full brightness and that a bright surface is the surface, not what stands on it.

**Heatmap.** The viewer colours a Match by its score on the fixed scale from 0 to 1 and no longer stretches the colours over the Matches shown.

**Measuring.** `python -m scenetiq.evaluation` asks hand-labelled Queries against a running backend and reports precision, recall and how often the best Match is a wanted Instance. The repo still holds no labelled Queries.

## Considered Options

- Keeping Reciprocal Rank Fusion and only adding a cut: ranks carry no gap to cut at.
- An absolute similarity threshold: ADR 0008 removed it because similarities are uncalibrated per channel and per Scene. Standardizing within the Scene and cutting relative to the best Match needs no absolute value and can never return nothing.
- A cut at the largest gap between consecutive scores: unstable with few Instances, and it returns one Match when two chairs differ slightly and everything else is far below.
- A margin in standard deviations below the best instead of a ratio: with one clear Match at five deviations, a second real one at three would be dropped.
- The lowest value, instead of the average, on the Caption channel for an Instance without a Caption: an uncaptioned chair could then hardly ever be a Match.
- Weighting the Identity Description above the others for every Query: breaks the Queries about material or use that ADR 0008 added Captions for.
- A keyword test for "is this Query just a noun": the same objection as in ADR 0009; the parser already reads every Query.
- Dense masks from the SAM2 Masks of each View: an Instance is merged from several Masks per View and Masks overlap, so choosing them needs MV3DIS's own matching.
- Dense masks by looking up every depth pixel's nearest Reconstruction point: exact, but tens of millions of nearest-neighbour lookups per job.
- Blurring the background of the masked crop instead of filling it: a blurred chair on the floor is still chair-coloured; grey removes it, and the context crops still carry the surroundings.
- A second generation per Query in which the vision-language model judges the shortlisted Names: the most direct fix for "is this a chair", but it doubles the wait of every Query.

## Consequences

A Query returns between one and K Matches, and the count now depends on the Query. A Query for something the Scene does not have still returns the best of a poor ranking, usually with several Instances near it, because a relative cut cannot tell that the best is poor.

Scores remain relative to one Query and one Scene. A standardized value depends on what else is in the Scene: one chair among thirty Instances stands out more than one among three, and with two or three Instances the values are coarse (two Instances always standardize to +1 and −1).

`query_score_ratio`, `embedding.masked_weight`, `captioning.background_dim` and the splat radius are starting guesses. None has run on a real Scene, and neither has the parser's new `aspects` field. A parser that names the wrong Aspect hides the right Description from the Caption channel, with no sign in the viewer.

Scenes reconstructed before this change keep their old embeddings and Captions: the Query-time changes apply to them at once, the masks only after the Capture is reconstructed again. The Embedding stage encodes one more crop per View and Instance, and both stages draw a mask per crop.

The mask is as coarse as the Reconstruction: it follows the Instance's fused points, reaches a few pixels past its outline, and has holes where the TSDF has none of its points.
