"""Compare duplicate-message weights 0, 0.1, 0.25 and 0.5 on matched inputs.

From the extracted aragcl_dp folder on Paperspace:
    python run_lambda_comparison.py

The default dataset is the existing /storage/weibo1 cohort. All four weights
use the unfiltered duplicate graph, copies-to-reference messages and five
training seeds. Exact augmented-view sequences are checked within each seed.
Results remain exploratory validation scores, not independent test results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from run_matched_comparisons import LAMBDA_VARIANTS, run_comparisons


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=Path("/storage/weibo1"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/lambda-comparison-2026-09-29"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.set_defaults(models=list(LAMBDA_VARIANTS))
    return parser.parse_args(argv)


def main(argv=None):
    return run_comparisons(**vars(parse_args(argv)))


if __name__ == "__main__":
    main()
