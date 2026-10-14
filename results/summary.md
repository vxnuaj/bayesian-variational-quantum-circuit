Seeds: [0, 1, 2]

| Method | Params | Train acc | Test acc | Train loss | Evals |
|---|---|---|---|---|---|
| VQC: Bayesian optimization (GP, EI) | 24 | 0.818 ± 0.019 | **0.790 ± 0.014** | 0.569 | 150 |
| VQC: Block-coordinate BO | 24 | 0.938 ± 0.027 | **0.933 ± 0.039** | 0.283 | 481 |
| VQC: Block BO + COBYLA polish | 24 | 0.960 ± 0.007 | **0.960 ± 0.000** | 0.187 | 631 |
| VQC: Random search | 24 | 0.812 ± 0.017 | **0.790 ± 0.029** | 0.570 | 631 |
| VQC: COBYLA (random init) | 24 | 0.972 ± 0.008 | **0.963 ± 0.005** | 0.164 | 631 |
| Logistic regression (linear) | 3 | 0.855 ± 0.004 | **0.843 ± 0.017** | – | – |
| MLP 2-4-1 (parameter-matched) | 17 | 0.985 ± 0.004 | **0.953 ± 0.017** | – | – |
| MLP 2-32-32-1 (large) | 1185 | 1.000 ± 0.000 | **0.960 ± 0.016** | – | – |

Main pipeline re-evaluated with 256 measurement shots per point (Aer): test acc 0.953 ± 0.005
