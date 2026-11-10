"""Data re-uploading variational quantum circuit (Pérez-Salinas et al., 2020).

Each layer re-encodes the two input features with trainable scales, applies
trainable single-qubit rotations, then entangles the qubits. Re-uploading makes
the model a truncated Fourier series in the inputs, which is what lets a tiny
2-qubit circuit draw a curved decision boundary.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp


class VQC:
    def __init__(self, n_qubits: int = 2, n_layers: int = 3, seed: int = 0):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = 2
        # per layer: n_qubits encoding scales + 2 rotations per qubit
        self.n_params = n_layers * (n_qubits + 2 * n_qubits)
        self.x = ParameterVector("x", self.n_features)
        self.theta = ParameterVector("θ", self.n_params)
        self.circuit = self._build()
        # Readout: Z on qubit 0. Qiskit is little-endian so "I...IZ" is qubit 0.
        self.observable = SparsePauliOp("I" * (n_qubits - 1) + "Z")
        self.estimator = StatevectorEstimator(seed=seed)
        self.n_evals = 0

    def _build(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        k = 0
        for layer in range(self.n_layers):
            # encoding: qubit q gets feature q mod 2, scaled by a trainable weight
            for q in range(self.n_qubits):
                qc.ry(self.theta[k] * self.x[q % self.n_features], q)
                k += 1
            # trainable rotations
            for q in range(self.n_qubits):
                qc.rz(self.theta[k], q); k += 1
                qc.ry(self.theta[k], q); k += 1
            # entangle (ring for >2 qubits, single CZ for 2)
            if self.n_qubits == 2:
                qc.cz(0, 1)
            else:
                for q in range(self.n_qubits):
                    qc.cz(q, (q + 1) % self.n_qubits)
            if layer < self.n_layers - 1:
                qc.barrier()
        assert k == self.n_params
        return qc

    def expectation(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        """<Z_0> for every row of X, batched through the estimator primitive."""
        theta = np.asarray(theta, dtype=float)
        X = np.atleast_2d(X)
        # parameter order follows circuit.parameters (sorted by name): x[0], x[1], θ[0..]
        params = np.hstack([X, np.tile(theta, (len(X), 1))])
        job = self.estimator.run([(self.circuit, self.observable, params)])
        self.n_evals += len(X)
        return np.asarray(job.result()[0].data.evs)

    def predict(self, theta, X):
        return np.where(self.expectation(theta, X) >= 0, 1, -1)

    def loss(self, theta, X, y) -> float:
        """Mean squared error between <Z> and the ±1 label (smooth, bounded in [0, 4])."""
        return float(np.mean((self.expectation(theta, X) - y) ** 2))

    def accuracy(self, theta, X, y) -> float:
        return float(np.mean(self.predict(theta, X) == y))

    def param_order_check(self):
        names = [p.name for p in self.circuit.parameters]
        expected = [f"x[{i}]" for i in range(self.n_features)] + [f"θ[{i}]" for i in range(self.n_params)]
        assert names == expected, names
