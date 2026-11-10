"""Aggregate results/seed_*.json into a table, figures/, and results/summary.md."""
import glob, json
import numpy as np
from bo_vqc import VQC, make_dataset
from bo_vqc.classical import classical_baselines
from bo_vqc.plots import plot_dataset, plot_decision_boundaries, plot_convergence, plot_accuracy_bars

files = sorted(f for f in glob.glob("results/seed_*.json") if "quick" not in f)
R = [json.load(open(f)) for f in files]
seeds = [r["seed"] for r in R]
print(f"aggregating seeds {seeds}")

qkeys = ["bo_full", "bo_block", "bo_block_polish", "random", "cobyla"]
rows, md = [], []
md.append(f"| Method | Params | Train acc | Test acc | Train loss | Evals |")
md.append("|---|---|---|---|---|---|")
for k in qkeys:
    name = R[0]["quantum"][k]["name"]
    tr = np.array([r["quantum"][k]["train_acc"] for r in R])
    te = np.array([r["quantum"][k]["test_acc"] for r in R])
    lo = np.array([r["quantum"][k]["loss"] for r in R])
    ev = np.mean([r["quantum"][k]["n_calls"] for r in R])
    rows.append((f"VQC: {name}", te.mean(), te.std(), "quantum"))
    md.append(f"| VQC: {name} | {R[0]['n_params']} | {tr.mean():.3f} ± {tr.std():.3f} | "
              f"**{te.mean():.3f} ± {te.std():.3f}** | {lo.mean():.3f} | {ev:.0f} |")
for i, c in enumerate(R[0]["classical"]):
    tr = np.array([r["classical"][i]["train_acc"] for r in R])
    te = np.array([r["classical"][i]["test_acc"] for r in R])
    rows.append((c["name"], te.mean(), te.std(), "classical"))
    md.append(f"| {c['name']} | {c['n_params']} | {tr.mean():.3f} ± {tr.std():.3f} | "
              f"**{te.mean():.3f} ± {te.std():.3f}** | – | – |")
sh = np.array([r["shot_eval"]["test_acc"] for r in R])
md.append("")
md.append(f"Main pipeline re-evaluated with {R[0]['shot_eval']['shots']} measurement shots per point "
          f"(Aer): test acc {sh.mean():.3f} ± {sh.std():.3f}")
table = "\n".join(md)
print(table)
open("results/summary.md", "w").write(f"Seeds: {seeds}\n\n" + table + "\n")

# figures
plot_accuracy_bars(rows, "figures/accuracy.png")
plot_convergence({R[0]["quantum"][k]["name"]: [np.array(r["quantum"][k]["history"]) for r in R]
                  for k in qkeys}, "figures/convergence.png")

s0 = R[0]["seed"]
X_tr, y_tr, X_te, y_te, _ = make_dataset(seed=s0)
plot_dataset(X_tr, y_tr, X_te, y_te, "figures/dataset.png")
vqc = VQC(n_layers=R[0]["config"]["layers"], seed=s0)
cls = classical_baselines(X_tr, y_tr, X_te, y_te, seed=s0)
panels = []
for k in ["bo_full", "bo_block_polish"]:
    th = np.array(R[0]["quantum"][k]["theta"]); acc = R[0]["quantum"][k]["test_acc"]
    panels.append((f"VQC – {R[0]['quantum'][k]['name']}\ntest acc {acc:.2f}",
                   lambda G, th=th: vqc.expectation(th, G)))
for c in cls[1:]:
    panels.append((f"{c['name']}\ntest acc {c['test_acc']:.2f}",
                   lambda G, m=c["model"]: 2 * m.predict_proba(G)[:, 1] - 1))
plot_decision_boundaries(panels, X_te, y_te, "figures/decision_boundaries.png")
print("wrote figures/*.png and results/summary.md")
