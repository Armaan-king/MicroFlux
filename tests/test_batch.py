"""The batch safeguards refuse what the plan forbids and nothing else."""

import numpy as np
import pytest

from microflux.batch import batch_check, fingerprint

H = 3_600_000_000_000  # ns per hour


def facts(first_h, span_h, fp="new", eligible=True, orders=100_000):
    return {"eligible": eligible, "span_hours": span_h, "first_ns": first_h * H, "last_ns": (first_h + span_h) * H,
            "fingerprint": fp, "orders": orders}


REGISTRY = [
    {"session": "BTCUSDT-2026-09-09", "symbol": "BTCUSDT", "date": "2026-09-09", "fingerprint": "expl",
     "first_ns": 11.5 * H, "last_ns": 19.5 * H, "role": "exploratory"},
    {"session": "BTCUSDT-2026-09-09-early", "symbol": "BTCUSDT", "date": "2026-09-09", "fingerprint": "fresh1",
     "first_ns": 5.0 * H, "last_ns": 7.0 * H, "role": "fresh-1"},
]


def batch(sessions=()):
    return {"symbol": "BTCUSDT", "min_hours": 8.0, "sessions_required": 2, "sessions": list(sessions)}


def test_an_eight_hour_session_on_a_new_date_is_accepted():
    ok, why = batch_check(facts(24 + 1, 8.2), "2026-09-14", REGISTRY, batch())
    assert ok, why


def test_general_eligibility_is_not_batch_eligibility():
    """Two hours passes the protocol's one-hour rule and fails the batch's eight."""
    f = facts(24 + 1, 2.0)
    assert f["eligible"]
    ok, why = batch_check(f, "2026-09-14", REGISTRY, batch())
    assert not ok and any("8.0 h" in w for w in why)


def test_same_data_under_another_name_is_refused():
    f = facts(24 + 1, 8.2, fp="expl")
    ok, why = batch_check(f, "2026-09-14", REGISTRY, batch())
    assert not ok and any("same data" in w for w in why)


def test_overlap_with_an_evaluated_session_is_refused():
    ok, why = batch_check(facts(15.0, 8.2), "2026-09-14", REGISTRY, batch())  # 15:00-23:12 on the 9th
    assert not ok and any("overlaps" in w for w in why)


def test_a_date_that_already_holds_evaluated_data_is_refused():
    ok, why = batch_check(facts(22 + 30, 8.2), "2026-09-09", REGISTRY, batch())
    assert not ok and any("2026-09-09 already holds" in w for w in why)


def test_capture_order_is_enforced():
    accepted = [{"session": "BTCUSDT-2026-09-15", "fingerprint": "b1", "first_ns": (48 + 1) * H, "last_ns": (48 + 9) * H}]
    ok, why = batch_check(facts(24 + 1, 8.2, fp="b0"), "2026-09-14", REGISTRY, batch(accepted))
    assert not ok and any("capture order" in w for w in why)


def test_a_full_batch_takes_no_more():
    accepted = [{"session": "a", "fingerprint": "b1", "first_ns": 24 * H, "last_ns": 33 * H},
                {"session": "b", "fingerprint": "b2", "first_ns": 48 * H, "last_ns": 57 * H}]
    ok, why = batch_check(facts(72 + 1, 8.2, fp="b3"), "2026-09-16", REGISTRY, batch(accepted))
    assert not ok and any("already has 2" in w for w in why)


def test_fingerprint_depends_on_the_data_not_the_name():
    a = np.arange(10, dtype=np.int64) * 1_000_000
    assert fingerprint(a) == fingerprint(a.copy())
    assert fingerprint(a) != fingerprint(a + 1)
