"""Watch the circuit train in the terminal.

    uv run python train_live.py                       # main pipeline: block BO -> COBYLA polish
    uv run python train_live.py --method bo           # full-space Bayesian optimization
    uv run python train_live.py --method cobyla       # COBYLA from a random start
    uv run python train_live.py --method random
    uv run python train_live.py --seed 1 --layers 3 --quick

Left: loss per evaluation (dots) and best-so-far (line). Right: current best
loss / accuracies and the decision boundary of the best circuit so far drawn
over the data. Writes results/live_<method>_seed<k>.json at the end.
"""
import argparse, json
from pathlib import Path
import numpy as np

from bo_vqc import (VQC, make_dataset, bayesian_optimize, block_bayesian_optimize,
                    random_search, cobyla_from_random, local_polish)
from bo_vqc.live import TrainingMonitor

METHODS = {
    "pipeline": "Block-coordinate BO + COBYLA polish",
    "block": "Block-coordinate BO",
    "bo": "Full-space Bayesian optimization (GP, EI)",
    "cobyla": "COBYLA from random init",
    "random": "Random search",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, default="pipeline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="smaller budgets")
    ap.add_argument("--fps", type=float, default=6.0, help="dashboard refresh rate")
    args = ap.parse_args()
    q = args.quick

    sweeps, cpb, cinit = (1, 12, 5) if q else (3, 40, 10)
    polish = 30 if q else 150
    bo_calls, bo_init = (40, 10) if q else (150, 25)

    X_tr, y_tr, X_te, y_te, _ = make_dataset(seed=args.seed)
    vqc = VQC(n_layers=args.layers, seed=args.seed)
    n_blocks = vqc.n_params // (vqc.n_params // vqc.n_layers)
    block_calls = 1 + sweeps * n_blocks * cpb
    budget = {"pipeline": block_calls + polish, "block": block_calls,
              "bo": bo_calls, "cobyla": block_calls + polish, "random": block_calls + polish}[args.method]

    with TrainingMonitor(vqc, X_tr, y_tr, X_te, y_te, total_calls=budget,
                         title=f"{METHODS[args.method]}  ·  seed {args.seed}",
                         refresh_per_second=args.fps) as mon:
        if args.method in ("pipeline", "block"):
            mon.set_stage(f"block BO  ({sweeps} sweeps × {n_blocks} blocks × {cpb} calls)")
            r = block_bayesian_optimize(vqc, X_tr, y_tr, n_sweeps=sweeps, calls_per_block=cpb,
                                        n_initial_points=cinit, seed=args.seed, callback=mon)
            if args.method == "pipeline":
                mon.set_stage(f"COBYLA polish  ({polish} iters)")
                r = local_polish(vqc, X_tr, y_tr, r["theta"], n_calls=polish, callback=mon)
        elif args.method == "bo":
            mon.set_stage(f"GP-EI BO  ({bo_init} LHS points, then {bo_calls - bo_init} model-chosen)")
            r = bayesian_optimize(vqc, X_tr, y_tr, n_calls=bo_calls, n_initial_points=bo_init,
                                  seed=args.seed, callback=mon)
        elif args.method == "cobyla":
            mon.set_stage("COBYLA")
            r = cobyla_from_random(vqc, X_tr, y_tr, n_calls=budget, seed=args.seed, callback=mon)
        else:
            mon.set_stage("random search")
            r = random_search(vqc, X_tr, y_tr, n_calls=budget, seed=args.seed, callback=mon)
        mon.set_stage("done")

    theta = mon.best_theta
    out = dict(method=args.method, seed=args.seed, layers=args.layers, quick=q,
               best_loss=mon.best_loss, train_acc=mon.train_acc, test_acc=mon.test_acc,
               n_evals=len(mon.losses), history=mon.losses, theta=theta.tolist())
    Path("results").mkdir(exist_ok=True)
    path = f"results/live_{args.method}_seed{args.seed}{'_quick' if q else ''}.json"
    json.dump(out, open(path, "w"))
    print(f"\n{METHODS[args.method]}: best loss {mon.best_loss:.4f}, "
          f"train acc {mon.train_acc:.3f}, test acc {mon.test_acc:.3f}  ->  {path}")


if __name__ == "__main__":
    main()
