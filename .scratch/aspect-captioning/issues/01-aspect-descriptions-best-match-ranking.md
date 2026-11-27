# 01: Captions as per-Aspect Descriptions, ranked by the best Description

**What to build:** A Caption is now a Name, up to three Synonyms, and one short English Description for each Aspect: Identity, Appearance, Material and Use. Each non-empty Description is embedded on its own. When a Query runs, an Instance's Caption similarity is that of its best-matching Description, so a Query about an object's look, material or use is matched by the sentence about that Aspect. Each Match reports which Aspect won. The one-document Caption (Attributes, Use and the joined document) disappears entirely. The vision-language model is asked for the new fields by prompt only; ticket 02 adds the forced schema.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] One ordered set of Aspects (identity, appearance, material, use) exists in the domain, and prompt, parser, storage and scoring all read it.
- [x] A Caption holds a Name, at most three Synonyms and one Description per Aspect; Attributes and Use are gone. An Instance holds one Caption embedding per non-empty Description, and still cannot have Caption embeddings without a Caption.
- [x] The Captioning prompt asks, with no example, for one plain-English sentence of at most about 25 words per Aspect, about the boxed object only.
- [x] The reply parser reads the new fields tolerantly. A missing or empty Name, or unreadable JSON, gives no Caption, and the reason is logged at the Captioning logger. Synonyms beyond three are dropped. An empty Identity is built from Name and Synonyms, for example "A sofa, also called a couch or settee.".
- [x] The Captioning stage embeds every non-empty Description as one Caption-encoder document.
- [x] The Caption ranking orders captioned Instances by the maximum cosine over their Descriptions. RRF, normalisation and tie-breaks are unchanged. A Match carries `caption_score` (that maximum) and `caption_aspect` (the winning Aspect, or null).
- [x] `instances.json` stores the new Caption fields or null. `caption_embeddings.npy` is (Instances, 4, D) in Aspect order, with zero rows for empty Descriptions and for Instances without a Caption, and is written only when some Instance has a Caption.
- [x] The Query API returns `caption_aspect`. The Instances API still returns only `name`. The frontend types include the new field and the display is unchanged.
- [x] The fake Pipeline and fake Caption encoder give Captions with known Descriptions and vectors, including one Instance with some empty Descriptions and one without a Caption.
- [x] Seam A covers, with hand-worked values:
  - an Appearance-matching Query wins, with `caption_aspect` "appearance";
  - a different Aspect flips the result;
  - partial Descriptions;
  - an Instance without a Caption;
  - a Scene with no Captions;
  - `name` only on Instances;
  - job progress including Captioning.
- [x] Unit tests cover the parser cases, the generated Identity, the best-Description choice with ties, and the storage round-trip.
- [x] The joined document builder, Attributes, Use and every reference to them are removed from code, tests and docs.
- [x] GLOSSARY adds Aspect and Description, redefines Caption, and lists "sentence" and "facet" under Avoid. ADR 0008 is amended in place: per-Aspect Descriptions, embedded separately, best Description per Instance, and the rejected options. The pipeline doc and AGENTS.md describe the new storage shape and Match field.
- [x] Ruff, ty and pytest pass, and the frontend checks pass.
