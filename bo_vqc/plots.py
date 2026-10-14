"""Figures for the write-up."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"-1": "#1f77b4", "+1": "#d62728"}


def plot_dataset(X_tr, y_tr, X_te, y_te, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for y, c, lab in [(-1, COLORS["-1"], "class -1"), (1, COLORS["+1"], "class +1")]:
        ax.scatter(*X_tr[y_tr == y].T, c=c, s=18, label=f"{lab} (train)")
        ax.scatter(*X_te[y_te == y].T, c=c, s=30, marker="x", label=f"{lab} (test)")
    ax.set_xlabel("x₁ (scaled)"); ax.set_ylabel("x₂ (scaled)")
    ax.set_title("Two-moons toy task")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_decision_boundaries(panels, X_te, y_te, path, grid_n=80):
    """panels: list of (title, predict_fn) where predict_fn(X) -> continuous score."""
    xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, grid_n), np.linspace(-1.2, 1.2, grid_n))
    G = np.c_[xx.ravel(), yy.ravel()]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8), squeeze=False)
    for ax, (title, fn) in zip(axes[0], panels):
        Z = fn(G).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=20, cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.8)
        ax.contour(xx, yy, Z, levels=[0], colors="k", linewidths=1.2)
        for y, c in [(-1, COLORS["-1"]), (1, COLORS["+1"])]:
            ax.scatter(*X_te[y_te == y].T, c=c, s=16, edgecolors="k", linewidths=0.4)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_convergence(runs, path):
    """runs: dict name -> list of history arrays (one per seed)."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, hists in runs.items():
        n = min(len(h) for h in hists)
        best = np.array([np.minimum.accumulate(h[:n]) for h in hists])
        m, s = best.mean(0), best.std(0)
        x = np.arange(1, n + 1)
        ax.plot(x, m, label=name, ls="--" if "polish" in name else "-")
        ax.fill_between(x, m - s, m + s, alpha=0.15)
    ax.set_xlabel("circuit loss evaluations"); ax.set_ylabel("best training loss so far")
    ax.set_title("Optimizer convergence (mean ± std over seeds)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_accuracy_bars(rows, path):
    """rows: list of (name, mean_test_acc, std_test_acc, kind) kind in {'quantum','classical'}."""
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r[0] for r in rows]
    means = [r[1] for r in rows]; stds = [r[2] for r in rows]
    cols = ["#6a3d9a" if r[3] == "quantum" else "#33a02c" for r in rows]
    ax.barh(names, means, xerr=stds, color=cols, alpha=0.85, capsize=3)
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("test accuracy")
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    for i, m in enumerate(means):
        ax.text(m + 0.005, i, f"{m:.3f}", va="center", fontsize=8)
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    ax.set_title("Quantum (purple) vs classical (green) test accuracy")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
