"""Train every optimizer in parallel and compare them live in the terminal.

    uv run python compare_live.py                          # all five methods
    uv run python compare_live.py --methods pipeline cobyla bo
    uv run python compare_live.py --quick --seed 1

One worker process per method (each ~200 MB RAM, one CPU core). Workers stream
(loss, best-so-far accuracies, decision-boundary field) back over a queue; the
main process draws: a status table with a progress bar per method, overlaid
best-so-far convergence curves, and a small decision-boundary heatmap per
method. Full-space BO uses 150 evaluations (GP cost grows as calls^3); the
others get the pipeline's budget, as in run_experiment.py.
"""
import argparse, json, multiprocessing as mp, shutil, time
from pathlib import Path
import numpy as np

METHODS = {
    "pipeline": "Block BO + polish",
    "block": "Block BO",
    "bo": "Full-space BO",
    "cobyla": "COBYLA",
    "random": "Random search",
}
COLORS = {"pipeline": "green", "block": "cyan", "bo": "magenta", "cobyla": "yellow", "random": "red"}


# ---------------------------------------------------------------- worker --
def worker(method, seed, layers, quick, q):
    import warnings; warnings.filterwarnings("ignore")
    from bo_vqc import (VQC, make_dataset, bayesian_optimize, block_bayesian_optimize,
                        random_search, cobyla_from_random, local_polish)
    X_tr, y_tr, X_te, y_te, _ = make_dataset(seed=seed)
    vqc = VQC(n_layers=layers, seed=seed)
    sweeps, cpb, cinit = (1, 12, 5) if quick else (3, 40, 10)
    polish = 30 if quick else 150
    bo_calls, bo_init = (40, 10) if quick else (150, 25)
    n_blocks = vqc.n_layers
    block_calls = 1 + sweeps * n_blocks * cpb
    budget = {"pipeline": block_calls + polish, "block": block_calls, "bo": bo_calls,
              "cobyla": block_calls + polish, "random": block_calls + polish}[method]

    gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 22), np.linspace(1.2, -1.2, 11))
    grid = np.c_[gx.ravel(), gy.ravel()]
    state = dict(n=0, best=np.inf, theta=None)
    q.put(dict(method=method, type="start", budget=budget))

    def cb(loss, theta):
        state["n"] += 1
        msg = dict(method=method, type="eval", n=state["n"], loss=float(loss))
        if loss < state["best"]:
            state["best"], state["theta"] = float(loss), theta.copy()
            if state["n"] % 5 == 0 or state["n"] < 5:
                msg["train_acc"] = vqc.accuracy(theta, X_tr, y_tr)
                msg["test_acc"] = vqc.accuracy(theta, X_te, y_te)
        if state["theta"] is not None and (state["n"] % 20 == 0 or state["n"] == 1):
            msg["field"] = vqc.expectation(state["theta"], grid).tolist()
        q.put(msg)

    def stage(name):
        q.put(dict(method=method, type="stage", stage=name))

    t0 = time.time()
    if method in ("pipeline", "block"):
        stage("block BO")
        r = block_bayesian_optimize(vqc, X_tr, y_tr, n_sweeps=sweeps, calls_per_block=cpb,
                                    n_initial_points=cinit, seed=seed, callback=cb)
        if method == "pipeline":
            stage("COBYLA polish")
            local_polish(vqc, X_tr, y_tr, r["theta"], n_calls=polish, callback=cb)
    elif method == "bo":
        stage("GP-EI")
        bayesian_optimize(vqc, X_tr, y_tr, n_calls=bo_calls, n_initial_points=bo_init, seed=seed, callback=cb)
    elif method == "cobyla":
        stage("COBYLA")
        cobyla_from_random(vqc, X_tr, y_tr, n_calls=budget, seed=seed, callback=cb)
    else:
        stage("random")
        random_search(vqc, X_tr, y_tr, n_calls=budget, seed=seed, callback=cb)
    th = state["theta"]
    q.put(dict(method=method, type="done", seconds=time.time() - t0, best=state["best"],
               train_acc=vqc.accuracy(th, X_tr, y_tr), test_acc=vqc.accuracy(th, X_te, y_te),
               field=vqc.expectation(th, grid).tolist(), theta=th.tolist()))


# ------------------------------------------------------------- dashboard --
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fps", type=float, default=4.0)
    args = ap.parse_args()

    import plotext as plt
    from rich.console import Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.table import Table
    from rich.text import Text
    from bo_vqc.live import _ramp
    from bo_vqc import make_dataset
    X_tr, y_tr, X_te, y_te, _ = make_dataset(seed=args.seed)
    cells = {}  # only test points, to keep the small heatmaps readable
    ci = np.clip(np.round((X_te[:, 0] + 1.2) / 2.4 * 21).astype(int), 0, 21)
    ri = np.clip(np.round((1.2 - X_te[:, 1]) / 2.4 * 10).astype(int), 0, 10)
    for r, c, lab in zip(ri, ci, y_te):
        cells[(r, c)] = ("•", lab)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(m, args.seed, args.layers, args.quick, q), daemon=True)
             for m in args.methods]
    for p in procs:
        p.start()

    S = {m: dict(budget=1, n=0, stage="starting", losses=[], best=[], train=float("nan"),
                 test=float("nan"), field=None, done=False, seconds=None) for m in args.methods}
    progress = Progress(TextColumn("[bold]{task.fields[label]:<18}"), BarColumn(bar_width=None),
                        TextColumn("{task.completed:>4}/{task.total:<4}"), TextColumn("{task.fields[stage]}"))
    tasks = {m: progress.add_task("", total=1, label=METHODS[m], stage="starting") for m in args.methods}
    t0 = time.time()

    def table():
        tb = Table(box=None, pad_edge=False)
        for col in ["method", "best loss", "train acc", "test acc", "time"]:
            tb.add_column(col, style="bold" if col == "test acc" else None)
        for m in args.methods:
            s = S[m]
            best = f"{s['best'][-1]:.4f}" if s["best"] else "–"
            sec = f"{s['seconds']:.0f} s" if s["seconds"] else f"{time.time() - t0:.0f} s"
            tb.add_row(Text(METHODS[m], style=COLORS[m]), best, f"{s['train']:.3f}",
                       Text(f"{s['test']:.3f}", style="bold green" if s["done"] else ""), sec)
        return tb

    def chart(width, height):
        plt.clf(); plt.theme("clear"); plt.plotsize(max(width, 30), max(height, 8))
        ymax = 1.0
        for m in args.methods:
            b = S[m]["best"]
            if b:
                plt.plot(list(range(1, len(b) + 1)), b, color=COLORS[m], label=METHODS[m])
                ymax = max(ymax, b[0])
        plt.ylim(0, min(ymax, 2.0)); plt.xlabel("evaluations"); plt.title("best training loss so far")
        return Text.from_ansi(plt.build())

    def heat(m):
        f = S[m]["field"]
        t = Text()
        if f is None:
            return Text("waiting…", style="dim")
        f = np.asarray(f).reshape(11, 22)
        for r in range(11):
            for c in range(22):
                bg = _ramp(f[r, c]); cell = cells.get((r, c))
                if cell:
                    t.append(cell[0], style=f"{'bold red' if cell[1] > 0 else 'bold blue'} on {bg}")
                else:
                    t.append(" ", style=f"on {bg}")
            if r < 10:
                t.append("\n")
        return t

    def render():
        w, h = shutil.get_terminal_size((120, 40))
        n = len(args.methods)
        layout = Layout()
        layout.split_column(
            Layout(Panel(progress, title=f"parallel training · seed {args.seed} · {args.layers} layers",
                         border_style="cyan"), size=n + 2),
            Layout(name="mid"),
            Layout(name="heat", size=13),  # decision boundary per method, test points overlaid
        )
        layout["mid"].split_row(Layout(name="chart", ratio=3), Layout(name="tbl", ratio=2))
        chart_h = max(h - (n + 2) - 13 - 2, 8)
        layout["chart"].update(Panel(chart(int(w * 0.6) - 4, chart_h), border_style="green"))
        layout["tbl"].update(Panel(table(), title="status", border_style="magenta"))
        layout["heat"].split_row(*[Layout(Panel(heat(m), title=Text(METHODS[m], style=COLORS[m]),
                                                border_style=COLORS[m])) for m in args.methods])
        return layout

    def handle(msg):
        s = S[msg["method"]]; m = msg["method"]
        if msg["type"] == "start":
            s["budget"] = msg["budget"]; progress.update(tasks[m], total=msg["budget"])
        elif msg["type"] == "stage":
            s["stage"] = msg["stage"]; progress.update(tasks[m], stage=msg["stage"])
        elif msg["type"] == "eval":
            s["n"] = msg["n"]; s["losses"].append(msg["loss"])
            s["best"].append(min(msg["loss"], s["best"][-1]) if s["best"] else msg["loss"])
            if "train_acc" in msg:
                s["train"], s["test"] = msg["train_acc"], msg["test_acc"]
            if "field" in msg:
                s["field"] = msg["field"]
            progress.update(tasks[m], completed=msg["n"])
        elif msg["type"] == "done":
            s.update(done=True, seconds=msg["seconds"], train=msg["train_acc"], test=msg["test_acc"],
                     field=msg["field"], theta=msg["theta"], stage="done")
            progress.update(tasks[m], stage="done", completed=s["budget"])

    with Live(render(), refresh_per_second=args.fps, screen=False) as live:
        last = 0
        while not all(S[m]["done"] for m in args.methods):
            try:
                handle(q.get(timeout=0.25))
            except Exception:
                pass
            if time.time() - last > 1 / args.fps:
                live.update(render()); last = time.time()
        live.update(render())
    for p in procs:
        p.join(timeout=5)

    Path("results").mkdir(exist_ok=True)
    out = {m: {k: v for k, v in S[m].items() if k not in ("field",)} for m in args.methods}
    path = f"results/compare_seed{args.seed}{'_quick' if args.quick else ''}.json"
    json.dump(out, open(path, "w"))
    print(f"\n{'method':<20}{'best loss':>10}{'train':>8}{'test':>8}{'time':>8}")
    for m in args.methods:
        s = S[m]
        print(f"{METHODS[m]:<20}{s['best'][-1]:>10.4f}{s['train']:>8.3f}{s['test']:>8.3f}{s['seconds']:>7.0f}s")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
