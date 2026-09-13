"""One fresh session, end to end, under PROTOCOL.md: eligibility gate,
classical replication, neural comparison, final test. Refuses to run any
stage whose artifacts already exist, so a session is evaluated once.

    python run_session.py --date 2026-09-14 [--root C:/tickforge-runs] [--symbol BTCUSDT]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from microflux.experiment import session_eligibility


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/tickforge-runs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", required=True)
    ap.add_argument("--session", default=None)
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()
    session = args.session or f"{args.symbol}-{args.date}"
    out = Path(args.runs) / session
    out.mkdir(parents=True, exist_ok=True)

    facts = session_eligibility(args.root, args.symbol, args.date)
    (out / "eligibility.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(f"{session}: " + ", ".join(f"{k}={v}" for k, v in facts.items()))
    if not facts["eligible"]:
        raise SystemExit("not eligible under PROTOCOL.md s.1b; nothing run")

    common = ["--root", args.root, "--symbol", args.symbol, "--date", args.date, "--session", session, "--runs", args.runs]
    stages = [("replication.json", ["replicate.py"]), ("summary.json", ["fit_neural.py"]), ("final_test.json", ["final_test.py"])]
    for marker, script in stages:
        if (out / marker).exists():
            print(f"-- {script[0]}: {marker} exists, skipping (a session is evaluated once)")
            continue
        print(f"-- {script[0]}")
        subprocess.run([sys.executable, "-u", *script, *common], check=True)


if __name__ == "__main__":
    main()
