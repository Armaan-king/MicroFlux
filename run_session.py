"""One session, end to end, under PROTOCOL.md: general eligibility, optional
batch rules, classical replication, neural comparison, final test -- each
stage once. Already-evaluated data is recognised by its fingerprint, not by
the output folder name; an existing stage is skipped only if its artifacts
record the same data and the frozen configuration, else the run refuses.

    python run_session.py --date 2026-09-14 --batch docs/collection-batch-1.json
    python run_session.py --root C:/tickforge-runs/archive-2h --date 2026-09-09 --session BTCUSDT-2026-09-09-early
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from microflux.batch import batch_check, load_json, record_failure, register
from microflux.experiment import session_eligibility

FROZEN = {"N": 64, "width": 64, "layers": 2, "heads": 4, "max_epochs": 40, "min_epochs": 10, "patience": 8,
          "targets_per_epoch": 40_000}
SEEDS = [0, 1, 2, 3, 4]


def verify_stage(out: Path, marker: str, facts: dict) -> str | None:
    """Why an existing stage's artifacts do NOT match this data and the frozen configuration; None if they do."""
    art = load_json(out / marker, None)
    if art is None:
        return "unreadable"
    if marker == "replication.json":
        if art["data"]["fingerprint"] != facts["fingerprint"]:
            return f"replication.json is for fingerprint {art['data']['fingerprint']}, data is {facts['fingerprint']}"
    if marker == "summary.json":
        if art["seeds"] != SEEDS:
            return f"summary.json seeds {art['seeds']} != {SEEDS}"
        for kind in ("MLP", "attention"):
            for seed in SEEDS:
                cfg = load_json(out / f"{kind}_seed{seed}" / "config.json", None)
                if cfg is None or cfg["data"]["fingerprint"] != facts["fingerprint"]:
                    return f"{kind}_seed{seed}: config missing or for other data"
                bad = {k: cfg.get(k) for k, v in FROZEN.items() if k != "N" and cfg.get(k) != v}
                if kind == "attention" and cfg.get("N") != FROZEN["N"]:
                    bad["N"] = cfg.get("N")
                if bad:
                    return f"{kind}_seed{seed}: configuration differs from frozen defaults: {bad}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/tickforge-runs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", required=True)
    ap.add_argument("--session", default=None)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--batch", default=None, help="batch plan JSON; enforces its rules and records the outcome")
    ap.add_argument("--role", default=None, help="registry role; default 'batch:<name>' or 'fresh'")
    args = ap.parse_args()
    session = args.session or f"{args.symbol}-{args.date}"
    out = Path(args.runs) / session
    out.mkdir(parents=True, exist_ok=True)
    registry_path = Path(args.runs) / "registry.json"
    batch_path = Path(args.batch) if args.batch else None
    batch = load_json(batch_path, None) if batch_path else None
    if batch_path and batch is None:
        raise SystemExit(f"batch file {batch_path} not found; refusing to run without its rules")
    registry = load_json(registry_path, [])

    facts = session_eligibility(args.root, args.symbol, args.date)
    (out / "eligibility.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(f"{session}: " + ", ".join(f"{k}={v}" for k, v in facts.items() if k not in ("first_ns", "last_ns")))
    if not facts["eligible"]:
        if batch_path:
            record_failure(batch_path, session, args.date, facts, ["fails general eligibility"])
        raise SystemExit("not eligible under PROTOCOL.md s.1b; nothing run")

    already = next((r for r in registry if r["fingerprint"] == facts["fingerprint"]), None)
    if batch is not None:
        ok, why = batch_check(facts, args.date, registry, batch)
        if not ok:
            record_failure(batch_path, session, args.date, facts, why)
            raise SystemExit("refused for batch " + batch["name"] + ":\n  - " + "\n  - ".join(why))
        print(f"batch {batch['name']}: accepted as session {len(batch['sessions']) + 1} of {batch['sessions_required']}")
    elif already is not None and already["session"] != session:
        raise SystemExit(f"this data was already evaluated as {already['session']} (fingerprint {facts['fingerprint']})")

    common = ["--root", args.root, "--symbol", args.symbol, "--date", args.date, "--session", session, "--runs", args.runs]
    for marker, script in (("replication.json", "replicate.py"), ("summary.json", "fit_neural.py"), ("final_test.json", "final_test.py")):
        if (out / marker).exists():
            problem = verify_stage(out, marker, facts)
            if problem:
                raise SystemExit(f"{script}: existing {marker} does not match -- {problem}. Not skipping, not overwriting.")
            print(f"-- {script}: {marker} exists and matches this data and the frozen configuration; skipping")
            continue
        print(f"-- {script}")
        subprocess.run([sys.executable, "-u", script, *common], check=True)

    role = args.role or (f"batch:{batch['name']}" if batch else "fresh")
    if already is None:
        register(registry_path, batch_path, session, facts, args.root, args.symbol, args.date, role)
        print(f"registered {session} ({role}) in {registry_path}" + (f" and {batch_path}" if batch_path else ""))
    else:
        print(f"already registered as {already['session']} ({already['role']}); registry unchanged")


if __name__ == "__main__":
    main()
