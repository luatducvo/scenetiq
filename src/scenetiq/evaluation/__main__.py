"""Command line for the evaluation harness (GPU, Colab).

Example::

    python -m scenetiq.evaluation --scans-dir /data/scannet/scans
        --frames-dir /data/scannet/frames --scenes scene0011_00,scene0015_00
        --out /content/drive/MyDrive/scenetiq/eval/2026-11-03
        --baseline /content/drive/MyDrive/scenetiq/eval/baseline.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from scenetiq.evaluation.harness import HarnessConfig, regressions, run


def main(argv: list[str] | None = None) -> int:
    """Run the harness; return 1 if AP regressed against ``--baseline``."""
    parser = argparse.ArgumentParser(prog="python -m scenetiq.evaluation")
    parser.add_argument("--scans-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--scenes", required=True, help="comma-separated ScanNet scene ids")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-views", type=int, default=300)
    parser.add_argument("--baseline", type=Path, help="earlier report.json to compare against")
    parser.add_argument("--tolerance", type=float, default=0.02, help="allowed AP drop")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from scenetiq.pipeline.real import RealPipeline
    from scenetiq.pipeline.siglip import SigLIP2Encoder

    encoder = SigLIP2Encoder()
    cfg = HarnessConfig(
        scans_dir=args.scans_dir,
        frames_dir=args.frames_dir,
        scene_ids=[s.strip() for s in args.scenes.split(",") if s.strip()],
        out_dir=args.out,
        max_views=args.max_views,
    )
    report = run(cfg, RealPipeline(image_encoder=encoder), encoder)
    print((args.out / "report.md").read_text())

    if args.baseline:
        baseline = json.loads(args.baseline.read_text())["aggregate"]
        failures = regressions(report["aggregate"], baseline, args.tolerance)
        for failure in failures:
            print(f"REGRESSION: {failure}", file=sys.stderr)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
