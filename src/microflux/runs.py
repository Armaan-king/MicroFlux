"""Reproducible run artifacts for neural fits.

One directory per (session, model, seed):

    config.json           every argument, the seed, the code revision, torch and
                          numpy versions, and the data identity (root, symbol,
                          date, order count, split boundaries, a fingerprint of
                          the order timestamps)
    learning_curve.csv    epoch, train NLL/event, val NLL/event, cumulative seconds
    checkpoint.pt         model and optimiser state, torch and numpy RNG state,
                          epoch, best validation NLL -- enough to resume
    best.pt               the model state at the best validation epoch
    result.json           final metrics, epochs run, runtime
    blocks_<size>s.csv    per-block log-likelihood contributions on the
                          evaluation window, so paired intervals can be
                          recomputed without the model
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from microflux.batch import fingerprint


def code_revision(root: Path) -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def data_identity(root: str, symbol: str, date: str, t_ns: np.ndarray, split: dict) -> dict:
    return {
        "root": root, "symbol": symbol, "date": date, "orders": int(len(t_ns)),
        "first_ns": int(t_ns[0]), "last_ns": int(t_ns[-1]),
        "fingerprint": fingerprint(t_ns),
        "split": {k: [float(a), float(b)] for k, (a, b) in split.items()},
    }


class Run:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def config(self, cfg: dict) -> None:
        cfg = dict(cfg, torch=torch.__version__, numpy=np.__version__, python=sys.version.split()[0],
                   started=time.strftime("%Y-%m-%dT%H:%M:%S"))
        (self.path / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    def curve(self, epoch: int, train_nll: float, val_nll: float, seconds: float) -> None:
        f = self.path / "learning_curve.csv"
        if not f.exists():
            f.write_text("epoch,train_nll,val_nll,seconds\n", encoding="utf-8")
        with f.open("a", encoding="utf-8") as h:
            h.write(f"{epoch},{train_nll:.6f},{val_nll:.6f},{seconds:.1f}\n")

    def checkpoint(self, model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int, best: float, bad: int) -> None:
        torch.save({
            "model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best": best, "bad": bad,
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
        }, self.path / "checkpoint.pt")

    def best(self, state: dict) -> None:
        torch.save(state, self.path / "best.pt")

    def resume(self, model: torch.nn.Module, opt: torch.optim.Optimizer) -> dict | None:
        f = self.path / "checkpoint.pt"
        if not f.exists():
            return None
        ck = torch.load(f, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        return ck

    def result(self, res: dict) -> None:
        (self.path / "result.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")

    def blocks(self, block_s: float, ll: np.ndarray, n: np.ndarray, edges: np.ndarray) -> None:
        rows = "\n".join(f"{lo:.3f},{hi:.3f},{l:.6f},{int(c)}" for lo, hi, l, c in zip(edges[:-1], edges[1:], ll, n))
        (self.path / f"blocks_{int(block_s)}s.csv").write_text("start,end,loglik,events\n" + rows + "\n", encoding="utf-8")
