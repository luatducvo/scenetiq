"""The real (GPU) Pipeline: DA3 reconstruction, TSDF fusion, Masks, Instances, embeddings.

Model libraries are imported lazily inside the stage implementations, so this
package imports cleanly on a CPU-only machine without the GPU stack.
"""
