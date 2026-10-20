"""Command line interface.

    framebreaker fuzz --seconds 600 --workers 8      run a campaign
    framebreaker list                                 table of findings
    framebreaker show <id>                            report + reproducer for one finding
    framebreaker replay <id>                          re-run a finding on the installed versions
    framebreaker known                                print the known-differences atlas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _find(root: Path, fid: str) -> Path:
    for sub in ("new", "known"):
        matches = sorted((root / sub).glob(f"{fid}*"))
        if matches:
            return matches[0]
    sys.exit(f"no finding matching {fid!r} under {root}")


def cmd_fuzz(args) -> None:
    from .campaign import run_campaign
    names = args.backends.split(",") if args.backends else None
    run_campaign(args.out, args.seconds, workers=args.workers, seed=args.seed,
                 backend_names=names, tlp_rate=args.tlp_rate, shrink_budget=args.shrink_budget,
                 log=lambda msg: print(msg, flush=True))


def cmd_list(args) -> None:
    index = Path(args.out) / "index.jsonl"
    if not index.exists():
        sys.exit(f"no findings yet in {args.out}")
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    if not args.all:
        rows = [r for r in rows if not r.get("known")]
    for r in rows:
        who = r["who"]
        if who and isinstance(who[0], list):
            who = " vs ".join("{" + ",".join(g) + "}" for g in who)
        label = r.get("known") or "NEW"
        print(f"{r['id']}  {r['kind']:<8} {label:<30} {', '.join(r['features'])[:60]:<60}  {who}")
    print(f"\n{len(rows)} findings" + ("" if args.all else " (use --all to include known)"))


def cmd_show(args) -> None:
    d = _find(Path(args.out), args.id)
    print((d / "README.md").read_text())
    for repro in sorted(d.glob("repro_*")):
        print(f"--- {repro.name}\n{repro.read_text()}")


def cmd_replay(args) -> None:
    from .compare import show_rows
    from .engines import make_backends, run_case
    from .oracle import check
    d = _find(Path(args.out), args.id)
    case = json.loads((d / "case.json").read_text())
    for b in make_backends():
        print(f"--- {b.name}\n{show_rows(run_case(case, b))}")
    verdict = check(case, make_backends())
    print("\nSTILL FAILS" if verdict else "\nFIXED (all oracles pass)")


def cmd_known(args) -> None:
    from .known import KNOWN, POLARS_KNOWN
    for k in POLARS_KNOWN + KNOWN:
        print(f"[{k.kind}] {k.id}\n    {k.title}\n    {k.link or '(not yet reported upstream)'}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="framebreaker", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="findings", help="findings directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fuzz", help="run a fuzzing campaign")
    f.add_argument("--seconds", type=float, default=60)
    f.add_argument("--workers", type=int, default=4)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--backends", help="comma-separated subset of backends")
    f.add_argument("--tlp-rate", type=float, default=0.3)
    f.add_argument("--shrink-budget", type=int, default=1500)
    f.set_defaults(fn=cmd_fuzz)

    ls = sub.add_parser("list", help="list findings")
    ls.add_argument("--all", action="store_true")
    ls.set_defaults(fn=cmd_list)

    for name, fn in (("show", cmd_show), ("replay", cmd_replay)):
        s = sub.add_parser(name)
        s.add_argument("id")
        s.set_defaults(fn=fn)

    k = sub.add_parser("known", help="print the known-differences atlas")
    k.set_defaults(fn=cmd_known)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
