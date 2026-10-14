# Bayesian-Optimized Variational Quantum Circuits vs. Classical Neural Networks

Code and instructions: [`README.md`](../README.md).
The mathematics of the circuit and the optimizers: [`MECHANISM.md`](MECHANISM.md).

## 1. Question

Can a tiny variational quantum circuit (VQC), trained with Bayesian
optimization (BO) instead of gradients, classify a nonlinear toy dataset as
well as a classical neural network with a similar number of parameters? And
does BO actually help compared with simpler gradient-free optimizers?

## 2. Setup

**Task.** Two interleaving moons (`sklearn.datasets.make_moons`, noise 0.2),
300 points, 200 train / 100 test, features scaled to $[-1, 1]$, labels $\pm 1$.
A linear classifier tops out near 84 %, so beating that requires a curved
decision boundary.

![dataset](../figures/dataset.png)

**Quantum model.** 2 qubits, 4 layers, 24 trainable parameters, built in
Qiskit 2.5 and simulated exactly with `StatevectorEstimator`. Each layer
re-encodes the inputs with trainable scales, applies trainable RZ/RY rotations,
and entangles with CZ. Readout is $\langle Z_0\rangle$; prediction is its sign.
Loss is mean squared error between $\langle Z_0\rangle$ and the label.

**Optimizers for the circuit** (all gradient-free; the circuit is a black box):

| Method | Description | Loss evals |
|---|---|---|
| Full-space BO | scikit-optimize `gp_minimize`, Matérn GP, Expected Improvement, over all 24 parameters | 150 |
| Block-coordinate BO | BO on one layer's 6 parameters at a time, 40 calls per block, 3 sweeps | 481 |
| **Block BO + COBYLA polish** | the above, then 150 COBYLA iterations from its best point (main pipeline) | 631 |
| Random search | uniform in $[-\pi,\pi]^{24}$, control | 631 |
| COBYLA from random init | standard derivative-free local optimizer, control | 631 |

**Classical baselines** (scikit-learn, L-BFGS, tanh): logistic regression (3
parameters), a 2-4-1 MLP (17 parameters, the fair comparison to the 24-parameter
circuit), and a 2-32-32-1 MLP (1185 parameters) as an unconstrained ceiling.

Everything is repeated for three random seeds (which change the data split,
the BO initial design, and the random starts). Numbers are mean ± std.

## 3. Results

| Method | Params | Train acc | Test acc | Final train loss |
|---|---|---|---|---|
| VQC: full-space BO | 24 | 0.818 ± 0.019 | 0.790 ± 0.014 | 0.569 |
| VQC: block-coordinate BO | 24 | 0.938 ± 0.027 | 0.933 ± 0.039 | 0.283 |
| **VQC: block BO + COBYLA polish** | 24 | 0.960 ± 0.007 | **0.960 ± 0.000** | 0.187 |
| VQC: random search | 24 | 0.812 ± 0.017 | 0.790 ± 0.029 | 0.570 |
| VQC: COBYLA from random init | 24 | 0.972 ± 0.008 | 0.963 ± 0.005 | 0.164 |
| Logistic regression | 3 | 0.855 ± 0.004 | 0.843 ± 0.017 | – |
| MLP 2-4-1 (parameter-matched) | 17 | 0.985 ± 0.004 | 0.953 ± 0.017 | – |
| MLP 2-32-32-1 (large) | 1185 | 1.000 ± 0.000 | 0.960 ± 0.016 | – |

Main pipeline re-evaluated with 256 measurement shots per point on
`AerSimulator` instead of exact expectation values: **0.953 ± 0.005** test
accuracy.

![accuracy](../figures/accuracy.png)

![convergence](../figures/convergence.png)

![decision boundaries](../figures/decision_boundaries.png)

## 4. Discussion

**The BO-trained circuit matches the classical networks.** The main pipeline
reaches 96.0 % test accuracy on every seed, equal to the 1185-parameter MLP
and slightly above the parameter-matched 17-parameter MLP (95.3 %). The test
set has 100 points, so differences of one or two points are within noise; the
honest reading is "parity", not "quantum advantage". Note also that the
classical models fit the training set more tightly (98.5 to 100 %) while the
VQC sits at 96 %, so the circuit is the less expressive model here and
generalizes about as well.

**Shot noise costs little.** Reading out with 256 shots instead of exact
expectation values drops accuracy by under one point. The trained boundary is
not relying on tiny margins.

**Plain Bayesian optimization does not work in 24 dimensions.** Full-space BO
with 150 evaluations is indistinguishable from random search (79 % vs 79 %).
The convergence plot makes this concrete: the two curves overlap. A Gaussian
process conditioned on 150 points in a 24-dimensional periodic landscape has
almost no information, so the Expected Improvement surface is flat and each
query is effectively random. We tried more calls (200, 300), other acquisition
functions (LCB, gp_hedge) and more initial points during development; none
changed this. This is the well-known dimensionality ceiling of GP-based BO,
and VQC parameter counts cross it quickly.

**Block-coordinate BO recovers most of the gap.** Running BO on 6 parameters
at a time, where the GP is informative, brings the loss from 0.57 to 0.28 and
test accuracy to 93 %, with the same optimizer and no gradients. It is seed
sensitive (88 to 97 %) because coordinate descent can stop at a point that is
optimal per block but not jointly.

**A short local polish closes the rest.** 150 COBYLA iterations from the block
BO solution remove the seed sensitivity entirely (96.0 ± 0.0 %). This is the
usual global-then-local pattern: BO finds the basin, a local method descends it.

**The uncomfortable control.** COBYLA alone, from a random start and with the
same budget, does just as well (96.3 %) and converges faster. On a noiseless
simulator with cheap evaluations, a good local optimizer simply has no
disadvantage, and the multi-start robustness BO is supposed to provide was not
needed on this landscape. BO's expected advantages, robustness to evaluation
noise and sample efficiency when each evaluation is very costly, are exactly
the conditions this experiment does not create. The natural follow-up is to
train with shot-noisy losses (where COBYLA's linear interpolation becomes
unreliable and the GP's noise model should pay off) and to compare at smaller
budgets.

**Model shapes differ.** The decision-boundary plot shows the structural
difference between the two model families. The MLP boundary is a few straight
segments joined by bends (sum of tanh ridges). The VQC boundary is smooth and
repeats outside the data region because the circuit output is a band-limited
Fourier series in the inputs. Both fit the moons; they extrapolate very
differently.

## 5. Conclusions

1. A 2-qubit, 24-parameter re-uploading circuit trained without gradients
   reaches 96 % test accuracy on two-moons, on par with classical MLPs with
   17 and 1185 parameters.
2. Gaussian-process Bayesian optimization over all circuit parameters at once
   fails at this dimensionality. Restricting BO to one layer at a time makes it
   work, and a short local polish makes it reliable.
3. On an exact simulator the simpler COBYLA optimizer is equally good, so the
   case for BO must be made in the noisy or very-expensive-evaluation regime,
   which is the planned extension.

## 6. Reproduce

```bash
uv sync
for s in 0 1 2; do uv run python run_experiment.py --seed $s; done
uv run python aggregate.py
```

Each seed takes about 5 minutes on a laptop. Raw per-seed results, including
every loss evaluation and the trained parameter vectors, are in
`results/seed_<k>.json` (relative to the repo root); the aggregated table is `results/summary.md`.

## References

- Pérez-Salinas, Cervera-Lierta, Gil-Fuster, Latorre. *Data re-uploading for a universal quantum classifier.* Quantum 4, 226 (2020).
- Schuld, Sweke, Meyer. *Effect of data encoding on the expressive power of variational quantum machine learning models.* Phys. Rev. A 103, 032430 (2021).
- Frazier. *A tutorial on Bayesian optimization.* arXiv:1807.02811 (2018).
- Powell. *A direct search optimization method that models the objective and constraint functions by linear interpolation.* (1994).
- scikit-optimize `gp_minimize`; Qiskit 2.5; Qiskit Aer 0.17.
