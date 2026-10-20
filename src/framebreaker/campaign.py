"""Fuzzing campaigns: parallel workers, crash/hang detection, de-duplicated storage.

Workers generate, check and shrink cases independently. Before every oracle
evaluation a worker writes the case it is about to run to an "in-flight" file,
so if the process segfaults or hangs the parent can still save the exact input
that killed it, then restart the worker and carry on.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import queue
import random
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

from .ir import OP_KINDS, data_tags, features

LIBRARIES = ("polars", "pandas", "pyarrow", "duckdb", "narwhals")


def versions() -> dict[str, str]:
    return {lib: metadata.version(lib) for lib in LIBRARIES}


# ------------------------------------------------------------------ bandit
class Bandit:
    """Thompson sampling over operation kinds.

    Each case draws op weights from Beta posteriors; cases that produce a new
    finding reward every op kind they used. Over a campaign this steers effort
    towards the parts of the API that are actually yielding bugs.
    """

    def __init__(self, rng: random.Random, arms=OP_KINDS):
        self.rng = rng
        self.alpha = {a: 1.0 for a in arms}
        self.beta = {a: 1.0 for a in arms}

    def sample(self) -> dict[str, float]:
        return {a: self.rng.betavariate(self.alpha[a], self.beta[a]) for a in self.alpha}

    def update(self, used: set[str], reward: float) -> None:
        for a in used:
            if a in self.alpha:
                self.alpha[a] += reward
                self.beta[a] += 1.0 - reward


# ------------------------------------------------------------------ storage
def finding_id(sig: str) -> str:
    return hashlib.sha1(sig.encode()).hexdigest()[:10]


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = root / "index.jsonl"
        self.seen: set[str] = set()
        if self.index_path.exists():
            for line in self.index_path.read_text().splitlines():
                if line.strip():
                    self.seen.add(json.loads(line)["signature"])

    def save(self, rec: dict) -> bool:
        """Persist a finding; returns False if it is a duplicate."""
        from .compare import show_rows
        from .render import duckdb_sql, narwhals_script, polars_script

        if rec["signature"] in self.seen:
            return False
        self.seen.add(rec["signature"])
        fid = finding_id(rec["signature"])
        sub = "known" if rec.get("known") else "new"
        d = self.root / sub / fid
        d.mkdir(parents=True, exist_ok=True)
        case = rec["case"]
        (d / "case.json").write_text(json.dumps(case, indent=1))
        if rec.get("original"):
            (d / "original.json").write_text(json.dumps(rec["original"], indent=1))
        (d / "repro_all_backends.py").write_text(narwhals_script(case))
        involved = set(sum(rec.get("partition") or [], [])) | set(rec.get("backends") or [])
        polars_modes = {"polars": "eager", "polars-chunked": "chunked", "polars-lazy": "lazy",
                        "polars-noopt": "noopt", "polars-streaming": "streaming"}
        for backend, mode in polars_modes.items():
            if backend in involved:
                (d / f"repro_polars_{mode}.py").write_text(polars_script(case, mode))
        if "duckdb" in involved:
            sql = duckdb_sql(case)
            if sql:
                (d / "repro_duckdb.sql").write_text(sql)
        outcomes = rec.get("outcomes") or {}
        lines = [f"# {rec['kind']} — {fid}", "",
                 f"- signature: `{rec['signature']}`",
                 f"- known: {rec.get('known') or 'no'}",
                 f"- features: {', '.join(rec['features'])}",
                 f"- data tags: {', '.join(rec['tags']) or 'none'}",
                 f"- versions: {', '.join(f'{k} {v}' for k, v in rec['versions'].items())}",
                 f"- found: {rec['time']} (worker seed {rec.get('seed')})", ""]
        if rec.get("error"):
            lines += ["## Error", "", "```", rec["error"], "```", ""]
        if rec.get("partition"):
            lines += ["## Groups of agreeing backends", ""]
            lines += [f"- {', '.join(g)}" for g in rec["partition"]]
            lines.append("")
        if outcomes:
            lines += ["## Outcomes", ""]
            shown: dict[str, list[str]] = {}
            for name, o in outcomes.items():
                shown.setdefault(show_rows(o), []).append(name)
            for text, names in shown.items():
                lines += [f"**{', '.join(names)}**", "", "```", text, "```", ""]
        (d / "README.md").write_text("\n".join(lines))
        index_rec = {k: rec[k] for k in ("signature", "kind", "features", "tags", "time", "known")}
        index_rec.update(id=fid, dir=str(d.relative_to(self.root)),
                         who=rec.get("partition") or rec.get("backends"))
        with self.index_path.open("a") as f:
            f.write(json.dumps(index_rec) + "\n")
        return True


# ------------------------------------------------------------------- worker
def _worker(wid: int, seed: int, deadline: float, out: mp.Queue, inflight: str,
            backend_names: list[str] | None, tlp_rate: float, shrink_budget: int) -> None:
    from .engines import make_backends
    from .gen import Generator
    from .known import classify, signature
    from .oracle import check, fingerprint, tlp_valid
    from .shrink import shrink

    rng = random.Random(seed)
    bandit = Bandit(rng)
    backends = make_backends(backend_names)
    seen: set[str] = set()
    n_checks = 0

    def hook(case: dict) -> None:
        nonlocal n_checks
        n_checks += 1
        with open(inflight, "w") as f:
            json.dump(case, f)
        if n_checks % 20 == 0:
            out.put(("tick", wid, n_checks))

    cases = 0
    while time.time() < deadline:
        gen = Generator(rng, weights=bandit.sample())
        case = gen.gen_case()
        if rng.random() < tlp_rate:
            _add_tlp(gen, case)
        cases += 1
        verdict = check(case, backends, hook)
        reward = 0.0
        if verdict is not None:
            fp = fingerprint(verdict)

            def still_fails(c: dict, fp=fp) -> bool:
                w = check(c, backends, hook)
                return w is not None and fingerprint(w) == fp

            small, _ = shrink(case, still_fails, tlp_valid, budget=shrink_budget)
            final = check(small, backends, hook)
            if final is None or fingerprint(final) != fp:
                # The failure did not reproduce: keep the original input and flag it.
                small, final = case, {**verdict, "kind": "flaky"}
            sig = signature(final, small)
            if sig not in seen:
                seen.add(sig)
                known = classify(final, small, rerun=lambda c: check(c, backends, hook))
                reward = 0.0 if known else 1.0
                out.put(("finding", wid, {
                    "signature": sig, "kind": final["kind"], "case": small, "original": case,
                    "partition": final.get("partition"), "backends": final.get("backends"),
                    "error": final.get("error"), "outcomes": _jsonable(final.get("outcomes")),
                    "features": features(small), "tags": data_tags(small),
                    "known": known, "seed": seed,
                }))
        bandit.update({op["op"] for op in case["pipeline"]}, reward)
        if cases % 25 == 0:
            out.put(("cases", wid, 25))
    out.put(("cases", wid, cases % 25))
    out.put(("done", wid, n_checks))


def _add_tlp(gen, case: dict) -> None:
    from .ir import BOOL, InvalidCase, validate
    try:
        schema = validate(case)
    except InvalidCase:
        return
    case["tlp"] = gen.expr(schema, BOOL, depth=2)


def _jsonable(outcomes: dict | None) -> dict | None:
    if outcomes is None:
        return None
    return json.loads(json.dumps(outcomes, default=str))


# ------------------------------------------------------------------- parent
@dataclass
class Stats:
    cases: int = 0
    findings_new: int = 0
    findings_known: int = 0
    duplicates: int = 0
    crashes: int = 0
    hangs: int = 0
    restarts: int = 0
    per_kind: dict = field(default_factory=dict)


def run_campaign(out_dir: str | Path, seconds: float, workers: int = 4, seed: int = 0,
                 backend_names: list[str] | None = None, tlp_rate: float = 0.3,
                 shrink_budget: int = 1500, hang_timeout: float = 60.0,
                 log=print) -> Stats:
    out_dir = Path(out_dir)
    store = Store(out_dir)
    inflight_dir = out_dir / ".inflight"
    inflight_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    deadline = time.time() + seconds
    vers = versions()
    stats = Stats()
    procs: dict[int, mp.Process] = {}
    last_seen: dict[int, float] = {}
    seeds: dict[int, int] = {}
    next_seed = [seed * 1000]

    def start(wid: int) -> None:
        s = next_seed[0]
        next_seed[0] += 1
        seeds[wid] = s
        p = ctx.Process(target=_worker, daemon=True, args=(
            wid, s, deadline, q, str(inflight_dir / f"w{wid}.json"),
            backend_names, tlp_rate, shrink_budget))
        p.start()
        procs[wid] = p
        last_seen[wid] = time.time()

    def record_death(wid: int, kind: str, detail: str) -> None:
        path = inflight_dir / f"w{wid}.json"
        try:
            case = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            case = None
        if case is None:
            return
        sig = f"{kind}|{detail}|{','.join(features(case))}"
        rec = {"signature": sig, "kind": kind, "case": case, "error": detail,
               "features": features(case), "tags": data_tags(case), "known": None,
               "time": time.strftime("%Y-%m-%d %H:%M:%S"), "versions": vers, "seed": seeds[wid]}
        if store.save(rec):
            stats.findings_new += 1
            log(f"  !! {kind}: {detail} -> {out_dir / 'new' / finding_id(sig)}")

    for wid in range(workers):
        start(wid)
    log(f"fuzzing for {seconds:.0f}s with {workers} workers -> {out_dir}")
    log("versions: " + ", ".join(f"{k} {v}" for k, v in vers.items()))
    last_report = time.time()

    def handle(msg) -> None:
        tag, wid, payload = msg
        last_seen[wid] = time.time()
        if tag == "cases":
            stats.cases += payload
        elif tag == "finding":
            payload.update(time=time.strftime("%Y-%m-%d %H:%M:%S"), versions=vers)
            if not store.save(payload):
                stats.duplicates += 1
            elif payload["known"]:
                stats.findings_known += 1
            else:
                stats.findings_new += 1
                stats.per_kind[payload["kind"]] = stats.per_kind.get(payload["kind"], 0) + 1
                log(f"  ++ {payload['kind']}: {payload['signature']}")
        elif tag == "done":
            finished.add(wid)

    finished: set[int] = set()
    while procs:
        try:
            handle(q.get(timeout=0.5))
        except queue.Empty:
            pass
        now = time.time()
        for wid, p in list(procs.items()):
            if wid in finished or (not p.is_alive() and p.exitcode == 0):
                p.join(timeout=5)
                procs.pop(wid)
            elif not p.is_alive():
                procs.pop(wid)
                stats.crashes += 1
                record_death(wid, "crash", f"exit code {p.exitcode}")
                if now < deadline:
                    stats.restarts += 1
                    start(wid)
            elif now - last_seen[wid] > hang_timeout:
                p.kill()
                p.join(timeout=5)
                procs.pop(wid)
                stats.hangs += 1
                record_death(wid, "hang", f"no progress for {hang_timeout:.0f}s")
                if now < deadline:
                    stats.restarts += 1
                    start(wid)
        if now - last_report > 15:
            last_report = now
            remaining = max(0, deadline - now)
            log(f"  [{remaining:5.0f}s left] cases={stats.cases} new={stats.findings_new} "
                f"known={stats.findings_known} dup={stats.duplicates} crashes={stats.crashes}")

    # workers may exit before the parent has read everything they sent
    while True:
        try:
            handle(q.get(timeout=1.0))
        except queue.Empty:
            break

    for path in inflight_dir.glob("*.json"):
        path.unlink(missing_ok=True)
    log(f"done: {stats.cases} cases, {stats.findings_new} new findings, "
        f"{stats.findings_known} known, {stats.duplicates} duplicates, "
        f"{stats.crashes} crashes, {stats.hangs} hangs")
    return stats


if __name__ == "__main__":  # pragma: no cover
    run_campaign(os.environ.get("FB_OUT", "findings"), float(os.environ.get("FB_SECONDS", 30)))
