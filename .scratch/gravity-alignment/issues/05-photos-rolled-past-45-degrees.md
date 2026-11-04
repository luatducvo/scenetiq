# 05: Photos rolled past 45° can turn a Scene the wrong way

**What to build:** A Capture of portrait phone photos must not come out turned about 90°. The way to do it is undecided, so this ticket needs a decision first.

**Blocked by:** None, but ticket 04's result with portrait photos should inform the decision.

**Status:** needs-triage

## What was found

Measured with the real GeoCalib estimator on CPU, on one photo (OpenCV's `building.jpg`), rolling the pixels and comparing against the unrolled result:

| Roll applied | GeoCalib measured |
|---|---|
| 15° | 14.8° |
| 30° | 29.4° |
| 45° | 44.3° |
| 60° | −33.9° |
| 90° | −15.5° |
| 120° | 29.8° |
| 180° | −0.1° |
| −30° | −30.0° |
| −60° | 30.4° |
| −90° | 5.4° |

Past about ±45° it still answers, with no sign of doubt. It is a single photo, so the exact limit is not established.

Nothing in the upload path, DA3 or the Embedding stage applies EXIF orientation. A portrait photo stored as landscape pixels plus an EXIF flag therefore reaches every stage sideways.

## Expected failure

Not yet reproduced end to end. Every sideways View reports up as roughly the image's top, which in the world is a horizontal direction. The Views agree, so consensus passes and the Scene turns about 90°. RANSAC may then accept a wall as the "level" plane. The log shows an ordinary `method=geocalib` or `method=plane` with an angle near 90°, which is the only clue. Captures from video frames, or from cameras that store upright pixels, are not affected.

## Options

1. **Apply EXIF orientation when photos are stored at upload.** Every stage then sees upright photos, which should also help SAM2 and SigLIP2, since they may be seeing these photos sideways today. It touches upload and re-encodes the photos unless done losslessly. Rolls that are in the pixels themselves are not helped.
2. **Apply EXIF orientation only for GeoCalib,** rotating each photo before the estimate and the vector back after it. The smallest change, limited to this stage, but the other stages keep seeing sideways photos.
3. **Refuse large corrections,** treating a rotation above some angle as no consensus. It needs no knowledge of the cause and also catches rolls in the pixels, but it would also refuse a Scene whose DA3 frame really is that far off.

Options 1 and 3 can be combined. Option 1 looks like the most useful first step, but it should be checked against ticket 04's result.
