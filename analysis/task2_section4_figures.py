"""Section-4 figures, drawn straight from the logged per-update metrics (no hand-entered numbers).
Run from the repo root:  python analysis/task2_section4_figures.py
Reads  results/task2_ppo/forks/fork_eps*_beta*/metrics.jsonl
Writes results/task2_ppo/analysis/section4/fig_beta.pdf|png, fig_eps.pdf|png"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FORKS = ROOT / "results/task2_ppo/forks"
OUT = ROOT / "results/task2_ppo/analysis/section4"
OUT.mkdir(parents=True, exist_ok=True)

def load(name):
    return [json.loads(l) for l in open(FORKS / name / "metrics.jsonl")]

plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
C = ["#d95f02", "#1b9e77", "#7570b3"]

# Fig: beta sweep (eps=0.2)
betas = ["0", "0.1", "0.2"]
runs = {b: load(f"fork_eps0.2_beta{b}") for b in betas}
u = np.arange(1, len(runs["0"]) + 1)
fig, ax = plt.subplots(1, 3, figsize=(6.8, 1.9))
for b, c in zip(betas, C):
    r = runs[b]
    ax[0].plot(u, [x["entropy"] for x in r], "-o", ms=2.5, c=c, label=rf"$\beta$={b}")
    ax[1].plot(u, [x["kl"] * 1e3 for x in r], "-o", ms=2.5, c=c)
    ax[2].plot(u, [x["reward"] for x in r], "-o", ms=2.5, c=c)
ax[0].set_ylabel("entropy (nats/tok)")
ax[1].set_ylabel(r"sampled KL ($10^{-3}$/tok)")
ax[2].set_ylabel("RM reward")
for a in ax: a.set_xlabel("fork update")
ax[0].legend(frameon=False)
plt.tight_layout()
for ext in ("pdf", "png"): plt.savefig(OUT / f"fig_beta.{ext}", dpi=200)
plt.close()

# Fig: epsilon sweep (beta=0.1)
eps = ["0.05", "0.2", "0.5"]
runs = {e: load(f"fork_eps{e}_beta0.1") for e in eps}
fig, ax = plt.subplots(1, 3, figsize=(6.8, 1.9))
for e, c in zip(eps, C):
    r = runs[e]
    ax[0].plot(u, [x["max_abs_ratio_dev"] for x in r], "-o", ms=2.5, c=c, label=rf"$\epsilon$={e}")
    ax[0].axhline(float(e), ls="--", c=c, lw=.8)
    ax[1].plot(u, [x["clip_fraction"] for x in r], "-o", ms=2.5, c=c)
    ax[2].plot(u, [x["reward"] for x in r], "-o", ms=2.5 + (e == "0.05"), c=c, alpha=.8)
ax[0].set_yscale("log"); ax[0].set_ylabel(r"$\max|\rho-1|$ (dashed: $\epsilon$)")
ax[1].set_ylabel("clip fraction"); ax[1].set_ylim(-0.01, 0.2)
ax[2].set_ylabel("RM reward")
for a in ax: a.set_xlabel("fork update")
ax[0].legend(frameon=False, fontsize=6)
plt.tight_layout()
for ext in ("pdf", "png"): plt.savefig(OUT / f"fig_eps.{ext}", dpi=200)
print("wrote", OUT)
