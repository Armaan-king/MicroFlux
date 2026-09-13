"""Collection-batch safeguards (docs/collection-batch-1.md, PROTOCOL.md s.1b).

A *registry* lists every session that has ever been evaluated, keyed on a
fingerprint of its order timestamps -- so the same data under another
folder name, root or date label is still recognised. A *batch* file records
the prospective plan (how many sessions, how long) and fills in as sessions
are accepted, in capture order.

`batch_check` sits on top of the unchanged general eligibility
(`experiment.session_eligibility`): a session can be eligible in general
and still be refused for a batch.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np


def fingerprint(t_ns: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(t_ns).tobytes()).hexdigest()[:16]


def load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def batch_check(facts: dict, date: str, registry: list[dict], batch: dict) -> tuple[bool, list[str]]:
    """Reasons the candidate fails the batch rules; empty means accepted.

    facts: from `session_eligibility` (span_hours, first_ns, last_ns, fingerprint, eligible).
    registry: every evaluated session, any role.
    batch: the plan, with its accepted `sessions` so far.
    """
    reasons = []
    if not facts.get("eligible"):
        reasons.append("fails general eligibility (PROTOCOL.md s.1b)")
    if facts["span_hours"] < batch["min_hours"]:
        reasons.append(f"span {facts['span_hours']:.2f} h < batch minimum {batch['min_hours']} h")
    if len(batch["sessions"]) >= batch["sessions_required"]:
        reasons.append(f"batch already has {batch['sessions_required']} sessions")
    for r in registry:
        if r["fingerprint"] == facts["fingerprint"]:
            reasons.append(f"same data as evaluated session {r['session']} (fingerprint {r['fingerprint']})")
        if facts["first_ns"] < r["last_ns"] and r["first_ns"] < facts["last_ns"]:
            reasons.append(f"overlaps evaluated session {r['session']}")
        if r["date"] == date and r["symbol"] == batch["symbol"]:
            reasons.append(f"UTC date {date} already holds evaluated session {r['session']}")
    accepted = batch["sessions"]
    if accepted and facts["first_ns"] <= max(s["first_ns"] for s in accepted):
        reasons.append("captured before a session already accepted into the batch; sessions are evaluated in capture order")
    return not reasons, reasons


def register(registry_path: Path, batch_path: Path | None, session: str, facts: dict, root: str, symbol: str, date: str, role: str) -> None:
    """Record an evaluated session in the registry and, if part of a batch, in the batch."""
    entry = {"session": session, "root": root, "symbol": symbol, "date": date, "role": role,
             "fingerprint": facts["fingerprint"], "first_ns": facts["first_ns"], "last_ns": facts["last_ns"],
             "span_hours": facts["span_hours"], "orders": facts["orders"], "evaluated": time.strftime("%Y-%m-%d")}
    registry = load_json(registry_path, [])
    if not any(r["fingerprint"] == entry["fingerprint"] for r in registry):
        registry.append(entry)
        registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    if batch_path is not None:
        batch = load_json(batch_path, None)
        if batch is not None and not any(s["fingerprint"] == entry["fingerprint"] for s in batch["sessions"]):
            batch["sessions"].append(entry)
            batch_path.write_text(json.dumps(batch, indent=2), encoding="utf-8")


def record_failure(batch_path: Path, session: str, date: str, facts: dict, reasons: list[str]) -> None:
    batch = load_json(batch_path, None)
    if batch is None:
        return
    batch.setdefault("excluded", []).append({"session": session, "date": date, "reasons": reasons,
                                             "span_hours": facts.get("span_hours"), "orders": facts.get("orders"),
                                             "fingerprint": facts.get("fingerprint"), "when": time.strftime("%Y-%m-%d")})
    batch_path.write_text(json.dumps(batch, indent=2), encoding="utf-8")
