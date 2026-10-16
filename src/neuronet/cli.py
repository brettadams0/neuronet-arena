"""Command line: ``neuronet train | view | experiment | plot``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from neuronet import __version__, snapshot
from neuronet import experiment as exp
from neuronet.config import Config
from neuronet.simulation import GenerationStats, Simulation
from neuronet.svgplot import Series, line_chart

STATS_FIELDS = ("generation", "best", "mean", "survivors", "hazard_steps", "champion")


def _seed(text: str) -> int:
    value = int(text)
    if not 0 <= value < 2**32:
        raise argparse.ArgumentTypeError("seed must be in [0, 2^32)")
    return value


def _override(text: str) -> tuple[str, Any]:
    key, sep, raw = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.strip(), value


def _config(args: argparse.Namespace) -> Config:
    data: dict[str, Any] = {}
    if getattr(args, "config", None):
        data.update(json.loads(Path(args.config).read_text()))
    for key, value in getattr(args, "set", None) or []:
        data[key] = value
    try:
        return Config.from_dict(data)
    except (TypeError, ValueError) as err:
        raise SystemExit(f"neuronet: bad config: {err}") from err


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", metavar="FILE", help="JSON file of config overrides")
    p.add_argument(
        "--set",
        metavar="KEY=VALUE",
        type=_override,
        action="append",
        help="override one config field, e.g. --set population=60 (repeatable)",
    )


def _print_stats(s: GenerationStats, elapsed: float) -> None:
    print(
        f"gen {s.generation:4d}  best {s.best:3d}  mean {s.mean:6.2f}  "
        f"survivors {s.survivors:3d}  hazard-steps {s.hazard_steps:5d}  [{elapsed:6.1f}s]",
        flush=True,
    )


def cmd_train(args: argparse.Namespace) -> int:
    sim = snapshot.load(args.resume) if args.resume else Simulation(_config(args), args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    for _ in range(args.generations):
        stats = sim.run_generation()
        if not args.quiet:
            _print_stats(stats, time.perf_counter() - start)

    with (out / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATS_FIELDS)
        writer.writeheader()
        writer.writerows(s.to_dict() for s in sim.history)

    # Record what the next generation scores from this exact state, so whoever
    # resumes the snapshot — the browser included — can check it gets the same.
    probe = snapshot.from_dict(snapshot.to_dict(sim))
    expect = probe.run_generation()
    snapshot.save(sim, out / "snapshot.json", expect=expect)
    print(f"wrote {out / 'history.csv'} and {out / 'snapshot.json'}")
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    try:
        from neuronet.viewer import run_viewer
    except ImportError as err:  # pygame is an optional extra
        print(f"neuronet view needs pygame: pip install 'neuronet-arena[viewer]' ({err})")
        return 1
    sim = snapshot.load(args.snapshot) if args.snapshot else Simulation(_config(args), args.seed)
    run_viewer(sim)
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    seeds = list(range(1, args.seeds + 1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    def progress(condition: str, seed: int) -> None:
        print(f"  {condition:<10} seed {seed:2d} done  [{time.perf_counter() - start:6.1f}s]")

    print(f"{len(seeds)} seeds × 2 conditions × {args.generations} generations")
    result = exp.run_experiment(
        _config(args), seeds, args.generations, workers=args.workers, progress=progress
    )
    exp.save(result, out / "experiment.json")
    _write_plots(result, out)
    _print_summary(result)
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    result = exp.load(args.experiment)
    _write_plots(result, Path(args.out) if args.out else Path(args.experiment).parent)
    _print_summary(result)
    return 0


def _write_plots(result: dict[str, Any], out: Path) -> None:
    def band(condition: str) -> tuple[list[float], list[float], list[float]]:
        per_seed = exp.series(result, condition)
        by_gen = list(zip(*per_seed, strict=True))
        return (
            [exp.median(g) for g in by_gen],
            [min(g) for g in by_gen],
            [max(g) for g in by_gen],
        )

    on, off = band("tournament"), band("random")
    series = [
        Series("Selection on", *on),
        Series("Selection off (control)", *off),
    ]
    n = len(result["seeds"])
    for theme in ("light", "dark"):
        svg = line_chart(
            series,
            title="Food eaten per agent, by generation",
            subtitle=f"Median of {n} seeds; the band is the range across seeds. "
            "Same seeds and config; only selection differs.",
            x_label="Generation",
            y_label="Food eaten per agent",
            theme=theme,
        )
        path = out / f"fitness-{theme}.svg"
        path.write_text(svg)
        print(f"wrote {path}")


def _print_summary(result: dict[str, Any]) -> None:
    summary = exp.summarise(result)
    on, off = summary["tournament"], summary["random"]
    print()
    print(f"{'':28}{'selection on':>14}{'selection off':>15}")
    print(f"{'food per agent, generation 0':28}{on['start']:14.2f}{off['start']:15.2f}")
    print(f"{'food per agent, last 10':28}{on['final']:14.2f}{off['final']:15.2f}")
    print(
        f"{'  range across seeds':28}{on['final_min']:>8.2f}–{on['final_max']:<5.2f}"
        f"{off['final_min']:>9.2f}–{off['final_max']:<5.2f}"
    )
    print(f"{'survivors, last 10':28}{on['survivors']:14.1f}{off['survivors']:15.1f}")
    hazard = on["hazard_per_food"], off["hazard_per_food"]
    print(f"{'hazard steps per food':28}{hazard[0]:14.2f}{hazard[1]:15.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuronet", description="Evolve small neural networks in a 2D arena."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="evolve headless and write history + snapshot")
    p.add_argument("--seed", type=_seed, default=1)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--out", default="runs/latest", help="output directory (default runs/latest)")
    p.add_argument("--resume", metavar="SNAPSHOT", help="continue from a snapshot.json")
    p.add_argument("--quiet", action="store_true")
    _add_config_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("view", help="watch a run live (needs pygame)")
    p.add_argument("--seed", type=_seed, default=1)
    p.add_argument("--snapshot", help="start from a snapshot.json instead of from scratch")
    _add_config_args(p)
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("experiment", help="selection on vs off, across seeds")
    p.add_argument("--seeds", type=int, default=8, help="number of seeds, 1..N (default 8)")
    p.add_argument("--generations", type=int, default=150)
    p.add_argument("--workers", type=int, help="processes (default: one per CPU)")
    p.add_argument("--out", default="results")
    _add_config_args(p)
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("plot", help="redraw the charts from an experiment.json")
    p.add_argument("experiment")
    p.add_argument("--out", help="directory for the SVGs (default: next to the JSON)")
    p.set_defaults(func=cmd_plot)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
