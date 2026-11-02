# Masks come from SAM2 automatic generation, not Grounded-SAM2

By default MV3DIS gets its 2D Masks from a detector (Grounded-SAM2 / YOLO-World) prompted with the 200 ScanNet200 class names. An object outside that list never becomes an Instance, so no Query can find it. That would defeat open-vocabulary querying. We generate Masks with SAM2's prompt-free automatic mask generator, so segmentation never depends on any vocabulary. Semantics come only later, from Instance embeddings.

## Consequences

Automatic generation gives many more Masks, including over-segmented fragments. MV3DIS's 3D-guided merging has to filter them, and runtime per Capture goes up.
