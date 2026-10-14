"""Optimizers for the circuit parameters.

- bayesian_optimize: Gaussian-process BO (scikit-optimize gp_minimize)
- random_search:     same evaluation budget, no model -> control for BO
- cobyla_from_random: classical gradient-free local optimizer from a random start
- local_polish:      short COBYLA refinement from a given start (used after BO)
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real


def _trace(vqc, X, y, history):
    """Wrap loss so every evaluation is recorded as (loss, running-best)."""
    def f(theta):
        l = vqc.loss(theta, X, y)
        history.append(l)
        return l
    return f


def bayesian_optimize(vqc, X, y, n_calls=150, n_initial_points=25, seed=0,
                      acq_func="EI", bound=np.pi, verbose=False):
    space = [Real(-bound, bound, name=f"t{i}") for i in range(vqc.n_params)]
    history = []
    t0 = time.time()
    res = gp_minimize(
        _trace(vqc, X, y, history), space,
        n_calls=n_calls, n_initial_points=n_initial_points,
        initial_point_generator="lhs", acq_func=acq_func,
        random_state=seed, noise=1e-10, verbose=verbose,
    )
    return dict(
        name="Bayesian optimization (GP, EI)", theta=np.array(res.x), loss=float(res.fun),
        history=np.array(history), n_calls=n_calls, seconds=time.time() - t0, skopt_result=res,
    )


def random_search(vqc, X, y, n_calls=150, seed=0, bound=np.pi):
    rng = np.random.default_rng(seed)
    history, best, best_theta = [], np.inf, None
    t0 = time.time()
    for _ in range(n_calls):
        theta = rng.uniform(-bound, bound, vqc.n_params)
        l = vqc.loss(theta, X, y)
        history.append(l)
        if l < best:
            best, best_theta = l, theta
    return dict(name="Random search", theta=best_theta, loss=best, history=np.array(history),
                n_calls=n_calls, seconds=time.time() - t0)


def cobyla_from_random(vqc, X, y, n_calls=150, seed=0, bound=np.pi):
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-bound, bound, vqc.n_params)
    history = []
    t0 = time.time()
    res = minimize(_trace(vqc, X, y, history), theta0, method="COBYLA",
                   options=dict(maxiter=n_calls, rhobeg=0.5))
    return dict(name="COBYLA (random init)", theta=res.x, loss=float(res.fun),
                history=np.array(history), n_calls=len(history), seconds=time.time() - t0)


def local_polish(vqc, X, y, theta0, n_calls=100):
    history = []
    t0 = time.time()
    res = minimize(_trace(vqc, X, y, history), np.asarray(theta0), method="COBYLA",
                   options=dict(maxiter=n_calls, rhobeg=0.2))
    return dict(name="BO + COBYLA polish", theta=res.x, loss=float(res.fun),
                history=np.array(history), n_calls=len(history), seconds=time.time() - t0)


def block_bayesian_optimize(vqc, X, y, n_sweeps=3, calls_per_block=40, n_initial_points=10,
                            seed=0, bound=np.pi, block_size=None, verbose=False):
    """Block-coordinate Bayesian optimization.

    GP-based BO scales poorly past ~10-15 dimensions, so instead of modelling
    all parameters at once we cycle over blocks (one circuit layer each by
    default), running a short BO on each block while the others are frozen.
    The incumbent is passed to each block's BO as a known point so it never
    gets worse than the current best.
    """
    rng = np.random.default_rng(seed)
    d = vqc.n_params
    block_size = block_size or d // vqc.n_layers
    blocks = [list(range(i, min(i + block_size, d))) for i in range(0, d, block_size)]
    theta = rng.uniform(-bound, bound, d)
    history = []
    best = vqc.loss(theta, X, y)
    history.append(best)
    sweep_log = []
    t0 = time.time()
    for sweep in range(n_sweeps):
        for b, idx in enumerate(blocks):
            def f(sub, idx=idx):
                t = theta.copy(); t[idx] = sub
                l = vqc.loss(t, X, y)
                history.append(l)
                return l
            space = [Real(-bound, bound, name=f"t{i}") for i in idx]
            res = gp_minimize(
                f, space, n_calls=calls_per_block, n_initial_points=n_initial_points,
                initial_point_generator="lhs", acq_func="EI",
                x0=[list(theta[idx])], y0=[best],
                random_state=int(rng.integers(1 << 31)), noise=1e-10,
            )
            if res.fun < best:
                best = float(res.fun)
                theta[idx] = res.x
            sweep_log.append((sweep, b, best))
            if verbose:
                print(f"sweep {sweep} block {b}: best loss {best:.4f}")
    return dict(name="Block-coordinate BO", theta=theta, loss=best, history=np.array(history),
                n_calls=len(history), seconds=time.time() - t0, sweep_log=sweep_log)
