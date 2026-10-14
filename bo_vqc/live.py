"""Live terminal dashboard for watching the circuit train.

TrainingMonitor is a callback: every optimizer in bo_vqc.optimize calls it
with (loss, theta) after each circuit-loss evaluation. It keeps the history
and redraws a rich Live layout: loss curve (plotext), progress/stats panel,
and an ASCII heatmap of the current decision boundary over the data.
"""
from __future__ import annotations

import shutil
import time
import numpy as np
import plotext as plt
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text


def _ramp(v: float) -> str:
    """Map a value in [-1, 1] to a blue-white-red background colour."""
    v = float(np.clip(v, -1, 1))
    if v < 0:
        t = -v
        r, g, b = int(255 - 200 * t), int(255 - 150 * t), 255
    else:
        t = v
        r, g, b = 255, int(255 - 180 * t), int(255 - 215 * t)
    return f"rgb({r},{g},{b})"


class TrainingMonitor:
    def __init__(self, vqc, X_tr, y_tr, X_te, y_te, total_calls: int,
                 title: str = "VQC training", eval_every: int = 5, boundary_every: int = 20,
                 refresh_per_second: float = 6.0):
        self.vqc, self.X_tr, self.y_tr, self.X_te, self.y_te = vqc, X_tr, y_tr, X_te, y_te
        self.total = total_calls
        self.title = title
        self.eval_every, self.boundary_every = eval_every, boundary_every
        self.stage = "starting"
        self.losses: list[float] = []
        self.best_so_far: list[float] = []
        self.best_loss, self.best_theta = np.inf, None
        self.train_acc = self.test_acc = float("nan")
        self.t0 = time.time()
        self.stage_marks: list[tuple[int, str]] = []
        # boundary grid
        cols, rows = 46, 20
        self.gx, self.gy = np.meshgrid(np.linspace(-1.2, 1.2, cols), np.linspace(1.2, -1.2, rows))
        self.grid = np.c_[self.gx.ravel(), self.gy.ravel()]
        self.field = np.zeros(rows * cols)
        self._cells = {}
        for X, y, mark in [(X_tr, y_tr, "•"), (X_te, y_te, "x")]:
            ci = np.clip(np.round((X[:, 0] + 1.2) / 2.4 * (cols - 1)).astype(int), 0, cols - 1)
            ri = np.clip(np.round((1.2 - X[:, 1]) / 2.4 * (rows - 1)).astype(int), 0, rows - 1)
            for r, c, lab in zip(ri, ci, y):
                self._cells[(r, c)] = (mark, lab)
        self.progress = Progress(
            TextColumn("[bold]{task.description}"), BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total} evals"), TimeElapsedColumn(),
        )
        self.task = self.progress.add_task("", total=total_calls)
        self.live = Live(self._render(), refresh_per_second=refresh_per_second, screen=False)

    # ---- callback -------------------------------------------------------
    def set_stage(self, name: str):
        self.stage = name
        if name != "done":
            self.stage_marks.append((len(self.losses), name))
        self.live.update(self._render())

    def __call__(self, loss: float, theta: np.ndarray):
        self.losses.append(loss)
        improved = loss < self.best_loss
        if improved:
            self.best_loss, self.best_theta = loss, theta.copy()
        self.best_so_far.append(self.best_loss)
        n = len(self.losses)
        self.progress.update(self.task, completed=min(n, self.total), description=self.stage)
        if improved and (n % self.eval_every == 0 or n < 5):
            self.train_acc = self.vqc.accuracy(self.best_theta, self.X_tr, self.y_tr)
            self.test_acc = self.vqc.accuracy(self.best_theta, self.X_te, self.y_te)
        if self.best_theta is not None and (n % self.boundary_every == 0 or n == 1):
            self.field = self.vqc.expectation(self.best_theta, self.grid)
        self.live.update(self._render())

    def __enter__(self):
        self.live.__enter__(); return self

    def __exit__(self, *a):
        if self.best_theta is not None:
            self.train_acc = self.vqc.accuracy(self.best_theta, self.X_tr, self.y_tr)
            self.test_acc = self.vqc.accuracy(self.best_theta, self.X_te, self.y_te)
            self.field = self.vqc.expectation(self.best_theta, self.grid)
        self.live.update(self._render())
        self.live.__exit__(*a)

    # ---- rendering ------------------------------------------------------
    def _loss_plot(self, width: int, height: int) -> Text:
        plt.clf(); plt.theme("clear"); plt.plotsize(max(width, 20), max(height, 6))
        if self.losses:
            x = list(range(1, len(self.losses) + 1))
            plt.scatter(x, self.losses, marker="dot", color="gray", label="loss")
            plt.plot(x, self.best_so_far, color="green", label="best so far")
            for i, name in self.stage_marks[1:]:
                plt.vline(i, "blue")
        plt.xlabel("evaluations"); plt.ylim(0, max(1.5, max(self.losses[:50], default=1.5)))
        plt.title("training loss  (vertical lines = stage change)")
        return Text.from_ansi(plt.build())

    def _boundary(self) -> Text:
        rows, cols = self.gx.shape
        f = self.field.reshape(rows, cols)
        t = Text()
        for r in range(rows):
            for c in range(cols):
                bg = _ramp(f[r, c])
                cell = self._cells.get((r, c))
                if cell:
                    mark, lab = cell
                    fg = "bold red" if lab > 0 else "bold blue"
                    t.append(mark + " ", style=f"{fg} on {bg}")
                else:
                    t.append("  ", style=f"on {bg}")
            if r < rows - 1:
                t.append("\n")
        return t

    def _stats(self) -> Table:
        tb = Table.grid(padding=(0, 2))
        tb.add_column(style="dim"); tb.add_column(style="bold")
        tb.add_row("stage", self.stage)
        tb.add_row("evaluations", f"{len(self.losses)} / {self.total}")
        tb.add_row("best train loss", f"{self.best_loss:.4f}" if self.losses else "–")
        tb.add_row("train accuracy", f"{self.train_acc:.3f}")
        tb.add_row("test accuracy", f"[green]{self.test_acc:.3f}[/green]")
        tb.add_row("elapsed", f"{time.time() - self.t0:6.1f} s")
        tb.add_row("circuit", f"{self.vqc.n_qubits} qubits · {self.vqc.n_layers} layers · {self.vqc.n_params} params")
        return tb

    def _render(self):
        w, h = shutil.get_terminal_size((120, 40))
        layout = Layout()
        layout.split_column(
            Layout(Panel(self.progress, title=self.title, border_style="cyan"), size=3),
            Layout(name="body"),
        )
        layout["body"].split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=2))
        plot_h = max(h - 3 - 4, 8)
        layout["left"].update(Panel(self._loss_plot(int(w * 0.6) - 6, plot_h), border_style="green"))
        layout["right"].split_column(
            Layout(Panel(self._stats(), title="status", border_style="magenta"), size=11),
            Layout(Panel(self._boundary(), title="decision boundary  (• train  x test)",
                         border_style="blue")),
        )
        return layout
