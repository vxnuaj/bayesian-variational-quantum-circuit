"""Train the VQC with several optimizers and the classical baselines for one seed.

Usage: uv run python run_experiment.py --seed 0 [--layers 4] [--quick]
Writes results/seed_<k>.json and results/seed_<k>_theta.npz.
"""
import argparse, json, time
from pathlib import Path
import numpy as np

from bo_vqc import (VQC, make_dataset, bayesian_optimize, block_bayesian_optimize,
                    random_search, cobyla_from_random, local_polish, classical_baselines)
from bo_vqc.shots import shot_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="small budgets for a smoke test")
    args = ap.parse_args()
    q = args.quick
    cfg = dict(
        layers=args.layers,
        bo_calls=40 if q else 150, bo_init=10 if q else 25,
        block_sweeps=1 if q else 3, block_calls=12 if q else 40, block_init=5 if q else 10,
        polish_calls=20 if q else 150,
    )
    Path("results").mkdir(exist_ok=True)
    X_tr, y_tr, X_te, y_te, _ = make_dataset(seed=args.seed)
    vqc = VQC(n_layers=cfg["layers"], seed=args.seed)
    vqc.param_order_check()
    print(f"seed {args.seed}: {vqc.n_qubits} qubits, {vqc.n_layers} layers, {vqc.n_params} params, "
          f"{len(X_tr)} train / {len(X_te)} test")

    runs = {}

    def record(key, r, note=""):
        runs[key] = dict(
            name=r["name"], loss=r["loss"], n_calls=int(r["n_calls"]), seconds=r["seconds"],
            train_acc=vqc.accuracy(r["theta"], X_tr, y_tr),
            test_acc=vqc.accuracy(r["theta"], X_te, y_te),
            history=r["history"].tolist(), theta=np.asarray(r["theta"]).tolist(),
        )
        print(f"  {r['name']:<34} loss {r['loss']:.3f}  train {runs[key]['train_acc']:.3f}  "
              f"test {runs[key]['test_acc']:.3f}  ({r['n_calls']} evals, {r['seconds']:.0f}s) {note}")

    # 1. Full-space GP Bayesian optimization
    record("bo_full", bayesian_optimize(vqc, X_tr, y_tr, n_calls=cfg["bo_calls"],
                                        n_initial_points=cfg["bo_init"], seed=args.seed))
    # 2. Block-coordinate BO (one layer at a time)
    blk = block_bayesian_optimize(vqc, X_tr, y_tr, n_sweeps=cfg["block_sweeps"],
                                  calls_per_block=cfg["block_calls"],
                                  n_initial_points=cfg["block_init"], seed=args.seed)
    record("bo_block", blk)
    # 3. Block BO followed by a short local polish (the main pipeline)
    pol = local_polish(vqc, X_tr, y_tr, blk["theta"], n_calls=cfg["polish_calls"])
    pol["history"] = np.concatenate([blk["history"], pol["history"]])
    pol["n_calls"] = len(pol["history"]); pol["seconds"] += blk["seconds"]
    pol["name"] = "Block BO + COBYLA polish"
    record("bo_block_polish", pol)
    budget = pol["n_calls"]
    # 4/5. Controls at the same evaluation budget
    record("random", random_search(vqc, X_tr, y_tr, n_calls=budget, seed=args.seed))
    record("cobyla", cobyla_from_random(vqc, X_tr, y_tr, n_calls=budget, seed=args.seed))

    # Finite-shot check of the main pipeline's circuit
    shots = 32 if q else 256
    shot_acc = shot_accuracy(vqc, np.array(runs["bo_block_polish"]["theta"]), X_te, y_te,
                             shots=shots, seed=args.seed)
    print(f"  main pipeline, {shots}-shot Aer readout: test {shot_acc:.3f}")

    classical = []
    for r in classical_baselines(X_tr, y_tr, X_te, y_te, seed=args.seed):
        classical.append({k: v for k, v in r.items() if k != "model"})
        print(f"  {r['name']:<34} params {r['n_params']:<5} train {r['train_acc']:.3f}  test {r['test_acc']:.3f}")

    out = dict(seed=args.seed, config=cfg, n_qubits=vqc.n_qubits, n_params=vqc.n_params,
               n_train=len(X_tr), n_test=len(X_te), quantum=runs, classical=classical,
               shot_eval=dict(shots=shots, test_acc=shot_acc))
    tag = f"seed_{args.seed}" + ("_quick" if q else "")
    with open(f"results/{tag}.json", "w") as f:
        json.dump(out, f)
    print(f"wrote results/{tag}.json")


if __name__ == "__main__":
    main()
