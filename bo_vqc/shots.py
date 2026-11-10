"""Evaluate a trained circuit with finite measurement shots on the Aer simulator.

The optimization uses exact statevector expectation values. This checks that
the trained classifier still works when <Z> is estimated from sampled bitstrings,
as it would be on real hardware.
"""
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator


def shot_accuracy(vqc, theta, X, y, shots=256, seed=0):
    sim = AerSimulator(seed_simulator=seed)
    qc = vqc.circuit.assign_parameters({vqc.theta: theta})
    qc.measure_all()
    circuits = [qc.assign_parameters({vqc.x: list(x)}) for x in X]
    tqc = transpile(circuits, sim)
    counts = sim.run(tqc, shots=shots).result().get_counts()
    preds = []
    for c in counts:
        # qubit 0 is the rightmost bit in Qiskit's bitstring convention
        z = sum((1 if k[-1] == "0" else -1) * v for k, v in c.items()) / shots
        preds.append(1 if z >= 0 else -1)
    return float(np.mean(np.array(preds) == y))
