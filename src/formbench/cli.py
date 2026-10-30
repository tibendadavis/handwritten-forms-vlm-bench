"""The `formbench` command. Every subcommand and option is defined here.

formbench eval      one photo, one model: reading, wrong cells, scores
formbench bench     a scenario (photos + truth), one or more models, end to end
formbench compare   combine saved benchmark runs per model
formbench synth     make a synthetic handwritten scenario with known answers
formbench template  build the blank-page template the synthetic pages are drawn on
formbench metadata  download DHIS2 data set metadata (read-only)
formbench run       weekly: read this week's photos into a draft for review
formbench approve   weekly: approve a reviewed draft into a dataValueSet (never sent)
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

DESCRIPTION = (
    "Find the vision model that reads your handwritten form best: read photos with a local "
    "model, score the readings against the truth, and track every run in Langfuse."
)
METADATA = Path("schemas/ri_v2/metadata.json")
ENGINE_HELP = "ollama:<model-tag>, e.g. ollama:qwen3.5:4b"
MAX_SIDE_HELP = "longest image side sent to the model in pixels (default 3072)"


def model_tag(engine: str) -> str:
    kind, _, model = engine.partition(":")
    if kind != "ollama" or not model:
        raise SystemExit(f"--engine must be 'ollama:<model-tag>', got {engine!r}")
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="formbench", description=DESCRIPTION)
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    p = sub.add_parser("eval", help="test one photo with one model")
    p.add_argument("--engine", required=True, help=ENGINE_HELP)
    p.add_argument("--image", type=Path, help="a register page photo")
    p.add_argument("--truth", type=Path, help="what is really written on --image (JSON)")
    p.add_argument("--scenario", type=Path, help="synthetic scenario folder (truth comes from it)")
    p.add_argument("--photo", help="photo inside the scenario, e.g. week-01/page-2026-07-s1.jpg")
    p.add_argument("--max-side", type=int, help=MAX_SIDE_HELP)
    p.add_argument("--think", action="store_true", help="let thinking models think")
    p.add_argument("--metadata", type=Path, default=METADATA)
    p.add_argument("--out", type=Path, default=Path("results/register/eval"))

    p = sub.add_parser("bench", help="benchmark models on a scenario, end to end")
    p.add_argument("--scenario", type=Path, required=True, help="data/private/synthetic/<name>")
    p.add_argument("--engine", action="append", help=f"{ENGINE_HELP}; repeat for several")
    p.add_argument("--rescore", type=Path, nargs="+", help="re-score saved run files, no model")
    p.add_argument("--max-side", type=int, help=MAX_SIDE_HELP)
    p.add_argument("--think", action="store_true", help="let thinking models think")
    p.add_argument("--metadata", type=Path, default=METADATA)
    p.add_argument("--out", type=Path, default=Path("results/register/bench"))

    p = sub.add_parser("compare", help="combine saved benchmark runs per model")
    p.add_argument("runs", type=Path, nargs="+", help="results/register/bench/*.json")
    p.add_argument("--metadata", type=Path, default=METADATA)

    p = sub.add_parser("synth", help="make a synthetic handwritten scenario")
    p.add_argument("--name", help="scenario folder name (default seed-N)")
    p.add_argument("--seed", type=int, default=1, help="same seed, same scenario")
    p.add_argument("--weeks", type=int, default=3)
    p.add_argument("--pages", type=int, default=3, help="cohort pages in the register")
    p.add_argument("--messiness", choices=["mixed", "neat", "average", "bad", "terrible"], default="mixed")  # fmt: skip
    p.add_argument("--quality", choices=["mixed", "scan", "good", "poor", "terrible"], default="mixed",
                   help="photo quality")  # fmt: skip
    p.add_argument("--paper", choices=["a3", "a4"], default="a3", help="physical register size")
    p.add_argument("--go-live", type=dt.date.fromisoformat, default=dt.date(2026, 9, 7))
    p.add_argument("--metadata", type=Path, default=METADATA)
    p.add_argument("--out", type=Path, default=Path("data/private/synthetic"))

    p = sub.add_parser("template", help="build the blank-page template from the R-06 PDF")
    p.add_argument("--pdf", type=Path, default=Path("data/private/samples/R-06_EPI Register.pdf"))
    p.add_argument("--out", type=Path, default=Path("data/private/register"))

    p = sub.add_parser("metadata", help="download DHIS2 data set metadata (read-only)")
    p.add_argument("--dataset", required=True, help="DHIS2 data set UID")
    p.add_argument("--out", type=Path, help="folder (default schemas/<data set name>)")

    p = sub.add_parser("run", help="weekly: read photos into a draft for review")
    p.add_argument("--week", required=True, help="ISO week, e.g. 2026W44")
    p.add_argument("--cover", type=Path, required=True, help="photo of the register cover")
    p.add_argument("--pages", type=Path, nargs="+", required=True, help="cohort page photos")
    p.add_argument("--engine", required=True, help=ENGINE_HELP)
    p.add_argument("--think", action="store_true", help="let thinking models think")
    p.add_argument("--metadata", type=Path, default=METADATA)
    p.add_argument("--out", type=Path, default=Path("results/register"))

    p = sub.add_parser("approve", help="weekly: approve a reviewed draft (never sent to DHIS2)")
    p.add_argument("--run", type=Path, required=True, help="the run folder from `formbench run`")
    p.add_argument("--reviewer", required=True, help="your name")
    p.add_argument("--metadata", type=Path, default=METADATA)
    return parser


def main(argv: list[str] | None = None) -> None:
    from dotenv import find_dotenv, load_dotenv

    args = build_parser().parse_args(argv)
    load_dotenv(find_dotenv(usecwd=True))
    if args.command == "eval":
        from formbench.evaluation.photo import command
    elif args.command == "bench":
        from formbench.evaluation.bench import command
    elif args.command == "compare":
        from formbench.evaluation.bench import compare as command
    elif args.command == "synth":
        from formbench.synthetic.simulate import command
    elif args.command == "template":
        from formbench.synthetic.template import command
    elif args.command == "metadata":
        from formbench.dhis2 import command
    elif args.command == "run":
        from formbench.weekly.draft import command
    else:
        from formbench.weekly.approve import command
    command(args)


if __name__ == "__main__":
    main()
