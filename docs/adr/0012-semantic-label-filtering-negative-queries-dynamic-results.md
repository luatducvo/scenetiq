# Semantic Label filtering, Negative Queries returning 0 Matches, and dynamic results without fixed K

ADR 0011 replaced rank-only Reciprocal Rank Fusion with standardized fusion and a relative cut (`query_score_ratio`). However, it kept two limitations:

- It mandated that a Query on a Scene with Instances must always return at least one Match. For a negative Query—asking for an object that does not exist in the Scene, such as "con mèo" (a cat) or a refrigerator in an office—standardization forced the least poor Instance (such as a wall or a cabinet) to score 1.0, and every Instance near it was coloured as a Match on the 3D viewer.
- The viewer provided a "Results to show" slider that artificially capped returned Matches at $K$ (default 5). A room with eight chairs hid three of them unless the user dragged the slider, while a room with one trash can still invited the user to guess how many results to ask for.

This ADR supersedes the "must always return at least one Match" rule of ADR 0011 and removes the fixed $K$ constraint from the viewer.

## Decision

**Semantic Label filtering.** Each captioned Instance carries Semantic Labels: its Name, Synonyms, and Aspect Descriptions (Identity, Appearance, Material, Use). When a Query is parsed into a wanted object phrase:

- An Instance is a semantic match when its Name, Synonyms, or relevant Aspect Descriptions share significant word stems with the phrase, or when its Caption similarity on the queried Aspect exceeds `negative_query_threshold` (0.35).
- Unrelated Instances (such as a floor or a wall when querying "chair") are excluded from the candidate shortlist before score normalization.

**Negative Queries return 0 Matches.** When no Instance in the Scene matches the wanted object semantically, the Query is recognized as a negative Query and returns an empty list (`matches: []`). The 3D viewer colours no points and displays an informative notification ("No matching objects found in scene"), preventing arbitrary false-positive highlights.

**Dynamic results without fixed $K$.** The "Results to show" slider is removed from the viewer. A Query returns all Instances that semantically match and pass `query_score_ratio`, up to the backend's safety ceiling (`query_max_top_k`). When a caller specifies `top_k`, at most that many are returned; when omitted, all qualifying Matches are returned.

**Uncaptioned Scenes.** Scenes without Captions lack Semantic Labels and fall back to image-channel similarity ranking without lexical label filtering.

## Consequences

- A Query for an object that does not exist in the Scene cleanly yields 0 Matches instead of highlighting random surfaces.
- Queries for common objects (e.g. "cái ghế") return exactly the matching objects (all chairs in the room) and never include unrelated furniture or floors.
- The user interface is simplified: users do not need to adjust a "Results to show" slider.
