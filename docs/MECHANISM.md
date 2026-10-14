# Mechanism: the mathematics behind the pipeline

This document describes, in math, what the code in `bo_vqc/` computes. It
covers (1) the quantum model and why it can represent a curved decision
boundary, (2) the loss, (3) how the simulator evaluates it, (4) the Bayesian
optimization loop, (5) why plain BO stalls and how block-coordinate BO and the
local polish fix that, and (6) the classical baseline we compare against.

Notation: input $\mathbf{x} = (x_1, x_2) \in [-1, 1]^2$, label $y \in \{-1, +1\}$,
trainable parameters $\boldsymbol{\theta} \in \mathbb{R}^P$ with $P = 24$, $n = 2$ qubits, $L = 4$ layers.

---

## 1. The quantum model

### 1.1 Gates

Single-qubit rotations are exponentials of Pauli matrices:

$$
R_Y(\phi) = e^{-i\phi Y/2} = \begin{pmatrix}\cos\frac{\phi}{2} & -\sin\frac{\phi}{2}\\ \sin\frac{\phi}{2} & \cos\frac{\phi}{2}\end{pmatrix},
\qquad
R_Z(\phi) = e^{-i\phi Z/2} = \begin{pmatrix} e^{-i\phi/2} & 0\\ 0 & e^{i\phi/2}\end{pmatrix}.
$$

The two-qubit entangler is the controlled-Z gate $\mathrm{CZ} = \mathrm{diag}(1, 1, 1, -1)$.

### 1.2 One layer

Layer $\ell$ owns six parameters: encoding scales $w_{\ell,1}, w_{\ell,2}$ and
rotation angles $\alpha_{\ell,1}, \beta_{\ell,1}, \alpha_{\ell,2}, \beta_{\ell,2}$.
It applies, in order,

$$
U_\ell(\mathbf{x};\boldsymbol{\theta}) \;=\; \mathrm{CZ}
\;\cdot\; \bigotimes_{q=1}^{2} R_Y(\beta_{\ell,q})\, R_Z(\alpha_{\ell,q})
\;\cdot\; \bigotimes_{q=1}^{2} R_Y(w_{\ell,q}\, x_q).
$$

The first factor (rightmost, applied first) is the **data encoding**: qubit $q$
is rotated by an angle proportional to feature $x_q$. The second is a general
trainable rotation. The third entangles the qubits.

### 1.3 Full circuit and readout

Starting from $|00\rangle$, the state after $L$ layers is

$$
|\psi(\mathbf{x};\boldsymbol{\theta})\rangle = U_L U_{L-1}\cdots U_1 \,|00\rangle .
$$

The model output is the expectation value of $Z$ on qubit 0:

$$
f(\mathbf{x};\boldsymbol{\theta}) = \langle \psi(\mathbf{x};\boldsymbol{\theta})|\, Z\otimes I \,|\psi(\mathbf{x};\boldsymbol{\theta})\rangle \;\in\; [-1, 1],
$$

and the predicted class is $\hat y = \operatorname{sign} f(\mathbf{x};\boldsymbol{\theta})$.
The decision boundary is the level set $f = 0$.

### 1.4 Why re-uploading gives a curved boundary

Because $R_Y(w x) = \cos\frac{wx}{2}\, I - i \sin\frac{wx}{2}\, Y$, every matrix
element of an encoding gate is a linear combination of $e^{\pm i w x/2}$. A
product of $L$ such gates (interleaved with $x$-independent unitaries) has
matrix elements that are linear combinations of $e^{i \omega x}$ with
$\omega \in \{\tfrac{1}{2}\sum_{\ell} s_\ell w_\ell : s_\ell \in \{-1,+1\}\}$.
Since $f$ is quadratic in the amplitudes, it is a finite Fourier series

$$
f(\mathbf{x};\boldsymbol{\theta}) = \sum_{\boldsymbol{\omega} \in \Omega(\mathbf{w})} c_{\boldsymbol{\omega}}(\boldsymbol{\theta})\, e^{i \boldsymbol{\omega}\cdot\mathbf{x}},
\qquad
\Omega(\mathbf{w}) = \Big\{ \big(\textstyle\sum_\ell s_{\ell,1} w_{\ell,1},\ \sum_\ell s_{\ell,2} w_{\ell,2}\big) : s_{\ell,q} \in \{-1, 0, +1\} \Big\}.
$$

Two consequences drive the design:

- The **rotation angles** $\alpha, \beta$ control the Fourier coefficients $c_{\boldsymbol{\omega}}$.
- The **encoding scales** $w$ control which frequencies appear. Making them
  trainable lets the optimizer pick the spectrum instead of fixing it.
- With a single encoding layer ($L = 1$, fixed $w = 1$) the model is just
  $a + b\cos x_1 + c \sin x_1 + \dots$ with one frequency per feature, far too
  smooth for two interleaved moons. Each extra re-upload adds frequencies, so
  depth buys expressivity. This is the data re-uploading result of
  Pérez-Salinas et al. (2020) and the Fourier analysis of Schuld, Sweke and
  Meyer (2021).

Qubit 1 never gets measured directly; its job is to carry correlations
through the CZ gates so that $f$ depends on products of $x_1$- and $x_2$-dependent
terms, i.e. on *interactions* between the features.

---

## 2. Loss

For training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ with $N = 200$,

$$
\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N}\sum_{i=1}^{N} \big(f(\mathbf{x}_i;\boldsymbol{\theta}) - y_i\big)^2 .
$$

Since $f \in [-1,1]$ and $y \in \{-1,+1\}$, $\mathcal{L} \in [0, 4]$ and
$\mathcal{L} = 0$ iff every point is classified with maximal margin. MSE is used
rather than hinge or cross-entropy because it is smooth in $\boldsymbol{\theta}$,
which suits a Gaussian-process surrogate. The optimizer treats $\mathcal{L}$ as a
**black box**: it only ever observes the scalar $\mathcal{L}(\boldsymbol{\theta})$.

### Why not gradients?

Gradients of $f$ exist in closed form via the parameter-shift rule,

$$
\frac{\partial f}{\partial \theta_k} = \tfrac{1}{2}\Big[ f(\boldsymbol{\theta} + \tfrac{\pi}{2}\mathbf{e}_k) - f(\boldsymbol{\theta} - \tfrac{\pi}{2}\mathbf{e}_k) \Big],
$$

but each gradient costs $2P = 48$ extra circuit evaluations per data point, each
of which on real hardware is a noisy estimate. That cost is why gradient-free
optimizers such as BO are studied for VQCs, and it is the setting this project
simulates.

---

## 3. Evaluating the loss on the simulator

### 3.1 Exact (statevector) evaluation, used during training

The 2-qubit state is a vector $\psi \in \mathbb{C}^4$ indexed by bitstrings
$b = b_1 b_0$. Qiskit's `StatevectorEstimator` applies the gate matrices and
returns

$$
f = \sum_{b \in \{00,01,10,11\}} (-1)^{b_0}\, |\psi_b|^2
  = \big(|\psi_{00}|^2 + |\psi_{10}|^2\big) - \big(|\psi_{01}|^2 + |\psi_{11}|^2\big).
$$

(Qiskit is little-endian: the rightmost bit is qubit 0, which is why the
observable is written `"IZ"`.) One loss evaluation is $N = 200$ such
computations, batched into a single estimator call.

### 3.2 Finite-shot evaluation, used as a hardware-realism check

On a device one cannot read $|\psi_b|^2$ directly. Running the circuit $S$ times
and counting outcomes $n_b$ gives the estimator

$$
\hat f = \frac{1}{S}\sum_b (-1)^{b_0} n_b, \qquad \mathrm{Var}[\hat f] = \frac{1 - f^2}{S}.
$$

With $S = 256$ the standard error is at most $1/16 \approx 0.06$. Points with
$|f| \lesssim 0.06$ can therefore flip class. `bo_vqc/shots.py` re-evaluates the
trained circuit this way on `AerSimulator` to measure how much accuracy that
noise costs (in our runs: about half a percentage point).

---

## 4. Bayesian optimization

BO minimizes an expensive black-box $\mathcal{L}$ by maintaining a probabilistic
model of it and choosing each query to maximize expected progress.

### 4.1 Gaussian-process surrogate

Prior: $\mathcal{L} \sim \mathcal{GP}(0, k)$ with a Matérn-5/2 kernel with
automatic relevance determination (one length-scale $\ell_j$ per dimension):

$$
k(\boldsymbol{\theta}, \boldsymbol{\theta}') = \sigma_f^2 \Big(1 + \sqrt{5}\, r + \tfrac{5}{3} r^2\Big) e^{-\sqrt{5}\, r},
\qquad
r^2 = \sum_{j=1}^{P} \frac{(\theta_j - \theta'_j)^2}{\ell_j^2}.
$$

Given observations $D_t = \{(\boldsymbol{\theta}_s, \mathcal{L}_s)\}_{s=1}^{t}$,
let $K$ be the $t\times t$ Gram matrix, $\mathbf{k}(\boldsymbol{\theta})$ the
vector of kernel values to the new point and $\boldsymbol{\ell}$ the observed
losses. The posterior at any $\boldsymbol{\theta}$ is Gaussian with

$$
\mu_t(\boldsymbol{\theta}) = \mathbf{k}^{\top}(K + \sigma_n^2 I)^{-1}\boldsymbol{\ell},
\qquad
\sigma_t^2(\boldsymbol{\theta}) = k(\boldsymbol{\theta},\boldsymbol{\theta}) - \mathbf{k}^{\top}(K + \sigma_n^2 I)^{-1}\mathbf{k}.
$$

The hyper-parameters $(\sigma_f, \ell_1, \dots, \ell_P)$ are refit at every
iteration by maximizing the marginal likelihood. Because the loss is computed
exactly, the noise term is set to $\sigma_n^2 = 10^{-10}$.

### 4.2 Acquisition: Expected Improvement

Let $\mathcal{L}^{*}_t = \min_s \mathcal{L}_s$ be the incumbent. The expected
improvement of querying $\boldsymbol{\theta}$ is

$$
\mathrm{EI}_t(\boldsymbol{\theta}) = \mathbb{E}\big[\max(\mathcal{L}^{*}_t - \mathcal{L}(\boldsymbol{\theta}), 0)\big]
= (\mathcal{L}^{*}_t - \mu_t)\,\Phi(z) + \sigma_t\,\varphi(z),
\qquad z = \frac{\mathcal{L}^{*}_t - \mu_t(\boldsymbol{\theta})}{\sigma_t(\boldsymbol{\theta})},
$$

with $\Phi, \varphi$ the standard normal CDF and PDF. The first term rewards
points the model predicts to be good (exploitation); the second rewards points
where the model is uncertain (exploration). EI is cheap to evaluate, so it is
maximized by sampling many candidates and polishing the best with L-BFGS.

### 4.3 The loop (`bayesian_optimize`)

1. Draw $t_0 = 25$ parameter vectors from a Latin hypercube over
   $[-\pi, \pi]^{P}$ and evaluate $\mathcal{L}$ at each.
2. For $t = t_0, \dots, 150$:
   fit the GP to $D_t$ (cost $O(t^3)$);
   choose $\boldsymbol{\theta}_{t+1} = \arg\max \mathrm{EI}_t$;
   run the circuit to get $\mathcal{L}(\boldsymbol{\theta}_{t+1})$;
   add it to $D_{t+1}$.
3. Return the best observed $\boldsymbol{\theta}$.

Every circuit evaluation is thus chosen by the model rather than by a fixed
rule, which is the whole appeal of BO when evaluations are expensive.

### 4.4 Why it stalls at $P = 24$

The convergence plot shows full-space BO tracking random search. The reason is
the curse of dimensionality in the surrogate: the loss is a sum of products of
sines and cosines in 24 periodic variables, so it has many local minima, and a
GP conditioned on 150 points in a 24-dimensional box has posterior variance
close to the prior almost everywhere. With $\sigma_t \approx \sigma_f$ and
$\mu_t \approx 0$ nearly everywhere, EI is nearly flat and its maximizer is
effectively a random draw. GP-based BO is known to be reliable up to roughly
10 to 15 dimensions and to degrade past that.

### 4.5 Block-coordinate BO (`block_bayesian_optimize`)

Partition the parameters by layer, $\boldsymbol{\theta} = (\boldsymbol{\theta}_{B_1}, \dots, \boldsymbol{\theta}_{B_L})$
with $|B_\ell| = 6$. Starting from a random $\boldsymbol{\theta}^{(0)}$, repeat
for sweeps $s = 1, \dots, 3$ and blocks $\ell = 1, \dots, L$:

$$
\boldsymbol{\theta}_{B_\ell} \leftarrow \arg\min_{\mathbf{u} \in [-\pi,\pi]^{6}}
\mathcal{L}\big(\boldsymbol{\theta}_{B_1}, \dots, \mathbf{u}, \dots, \boldsymbol{\theta}_{B_L}\big),
$$

where the inner minimization is a 40-call GP-EI run in six dimensions, seeded
with the current block values and their known loss as the incumbent. The update
is accepted only if it improves the incumbent, so the loss is monotone
non-increasing across blocks. Each inner problem lives in a space small enough
for the GP to be informative, at the cost of being a coordinate-descent method
that can stall at a point that is block-wise optimal but not globally optimal
(the sweep logs show improvements becoming rare after the second sweep).

### 4.6 Local polish (`local_polish`)

From the block-BO result, run 150 iterations of COBYLA (Powell's Constrained
Optimization BY Linear Approximation). At each step COBYLA keeps a simplex of
$P + 1$ points, fits the unique linear interpolant of $\mathcal{L}$ through them,
and minimizes it inside a trust region of radius $\rho$ around the best vertex;
$\rho$ starts at $0.2$ and shrinks when a step fails to improve. It is
derivative-free, so the total cost of the pipeline stays measured in circuit
evaluations. The combination is the standard global-then-local pattern: BO
finds a good basin, COBYLA descends to its bottom.

Total budget for the main pipeline: $1 + 3 \times 4 \times 40 + 150 = 631$
loss evaluations, each of which runs $N = 200$ circuits. Random search and
COBYLA-from-random are given the same 631 evaluations as controls.

---

## 5. Classical baseline

The parameter-matched comparison is a one-hidden-layer network

$$
g(\mathbf{x}) = \mathbf{w}_2^{\top} \tanh(W_1 \mathbf{x} + \mathbf{b}_1) + b_2,
\qquad W_1 \in \mathbb{R}^{4\times 2},\ \mathbf{b}_1 \in \mathbb{R}^{4},\ \mathbf{w}_2 \in \mathbb{R}^{4},\ b_2 \in \mathbb{R},
$$

with $4\cdot 2 + 4 + 4 + 1 = 17$ parameters, trained by minimizing the logistic
loss $\frac{1}{N}\sum_i \log\big(1 + e^{-y_i g(\mathbf{x}_i)}\big)$ with L-BFGS
(exact gradients by backpropagation). Prediction is $\operatorname{sign} g$.
A 2-32-32-1 network with 1185 parameters gives the unconstrained ceiling, and
logistic regression ($g = \mathbf{w}^{\top}\mathbf{x} + b$, 3 parameters) shows
how much a linear boundary can do.

The structural contrast: the MLP is a sum of sigmoidal ridges, so its boundary
is piecewise-smooth with a few bends (visible as the V shape in the decision
plot). The VQC is a band-limited Fourier series, so its boundary is smooth and
periodic in the input, which is why its contour plot shows repeating lobes
outside the data region.

---

## 6. Summary of the algorithm as run

```
data  ← two moons, N=300, scaled to [-1,1]^2, labels ±1, 200/100 split
f     ← 2-qubit, 4-layer re-uploading circuit, readout <Z_0>, P=24
L(θ)  ← mean (f(x_i;θ) − y_i)^2 over the training set, exact statevector

θ_BO  ← block-coordinate GP-EI BO: 3 sweeps × 4 layers × 40 calls
θ*    ← COBYLA polish from θ_BO, 150 iterations
report test accuracy of sign f(·;θ*), exact and with 256 shots
compare to: full-space BO, random search, COBYLA-from-random (same budget)
            logistic regression, MLP 2-4-1, MLP 2-32-32-1
repeat for seeds 0, 1, 2; report mean ± std
```
