# %%

import numpy as np
from copy import deepcopy
from typing import Any
from qiskit import QuantumCircuit, Aer  # type: ignore
from qiskit.circuit import Parameter  # type: ignore
from qiskit.compiler import transpile  # type: ignore
from qiskit.quantum_info import XXDecomposer  # type: ignore
from qiskit.transpiler import PassManager  # type: ignore
from qiskit.transpiler.passes import FullAncillaAllocation, Optimize1qGatesDecomposition, BasisTranslator, RZXCalibrationBuilder  # type: ignore
from qiskit.transpiler.coupling import CouplingMap  # type: ignore
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary  # type: ignore
from qiskit.providers.fake_provider import FakeNairobi  # type: ignore

MAIN = __name__ == "__main__"

# %%


def get_unitary(quantum_circuit: QuantumCircuit) -> list[list[float]]:
    """
    Args:
        qc: Input quantum circuit
        num_dec_pt: Number of decimal points of the unitary matrix
    Returns:
        unitary: Unitary matrix corresponding to the quantum circuit
    """
    # Determine the unitary matrix corresponding to the quantum circuit
    backend = Aer.get_backend("aer_simulator")
    new_circuit = deepcopy(quantum_circuit)
    new_circuit.save_unitary()  # type: ignore
    result = backend.run(new_circuit).result()
    unitary = result.get_unitary(new_circuit, 16).data
    return unitary


# %%


def w_original(theta: float) -> QuantumCircuit:
    """
    Args:
        theta: A rotation angle
    Returns:
        w_gate: A Qiskit circuit representing the w(theta) gate
    """
    w_gate = QuantumCircuit(2, name="w")
    w_gate.h([0, 1])
    w_gate.cz(0, 1)
    w_gate.sdg([0, 1])
    w_gate.h([0, 1])
    w_gate.rz(theta, 0)
    w_gate.rz(theta - np.pi / 2, 1)
    w_gate.h([0, 1])
    w_gate.s([0, 1])
    w_gate.cz(0, 1)
    w_gate.h([0, 1])
    return w_gate


def w(theta: float) -> QuantumCircuit:
    """
    Args:
        theta: A rotation angle
    Returns:
        w_gate: A Qiskit circuit representing the w(theta) gate
    """
    # Set up the decomposer and pass
    xx_decomposer = XXDecomposer()
    pm = PassManager(
        [
            BasisTranslator(
                SessionEquivalenceLibrary,
                target_basis=["id", "rz", "sx", "x", "rzx", "cx", "reset"],
            ),
            Optimize1qGatesDecomposition(
                basis=["id", "rz", "sx", "x", "rzx", "cx", "reset"]
            ),
        ]
    )
    # Decompose the unitary
    w_unitary = np.array(
        [
            [np.cos(theta - np.pi / 4), 0.0, 0.0, np.sin(theta - np.pi / 4)],
            [0.0, np.cos(np.pi / 4), -np.sin(np.pi / 4), 0.0],
            [0.0, np.sin(np.pi / 4), np.cos(np.pi / 4), 0.0],
            [-np.sin(theta - np.pi / 4), 0.0, 0.0, np.cos(theta - np.pi / 4)],
        ]
    )
    w_gate: QuantumCircuit = pm.run(xx_decomposer(w_unitary))  # type: ignore
    return w_gate


if MAIN:
    theta_range = np.linspace(-np.pi, np.pi, 101)
    theta_param = Parameter("theta")
    r_zx_angle_w = []
    for theta in theta_range:
        w_orig = w_original(theta)
        w_unitary_orig = get_unitary(w_orig)
        w_decomp = w(theta)
        w_unitary_decomp = get_unitary(w_decomp)
        assert np.allclose(w_unitary_orig, w_unitary_decomp, atol=1e-7)  # type: ignore
        r_zx_angle_w.append(sum(abs(gate.operation.params[0]) for gate in w_decomp.get_instructions("rzx")) / np.pi)  # type: ignore
    print(f"For w(theta), the total accumulated rotation angle for the R_ZX gates in multiples of pi has maximum {max(r_zx_angle_w):.2f}, minimum {min(r_zx_angle_w):.2f}, mean {np.mean(r_zx_angle_w):.2f}, and median {np.median(r_zx_angle_w):.2f}. Note that a CX gate corresponds to an angle 0.50.")  # type: ignore


# %%


def u_original(theta: float) -> QuantumCircuit:
    """
    Args:
        theta: A rotation angle
    Returns:
        u_gate: A Qiskit circuit representing the u(theta) gate
    """
    u_gate = QuantumCircuit(2, name="u")
    u_gate.h([0, 1])
    u_gate.cz(0, 1)
    u_gate.sdg([0, 1])
    u_gate.h([0, 1])
    u_gate.rz(theta, 0)
    u_gate.rz(theta, 1)
    u_gate.h([0, 1])
    u_gate.s([0, 1])
    u_gate.cz(0, 1)
    u_gate.h([0, 1])
    return u_gate


def u(theta: float, backend: Any = None) -> QuantumCircuit:
    """
    Args:
        theta: A rotation angle
    Returns:
        u_gate: A Qiskit circuit representing the u(theta) gate
    """
    if backend is None:
        backend = FakeNairobi()
    # Set up the decomposer and pass
    xx_decomposer = XXDecomposer()
    # pm = PassManager(
    #     [
    #         SetLayout(self.layout),
    #         FullAncillaAllocation(CouplingMap(backend.configuration().coupling_map)),
    #         BasisTranslator(
    #             SessionEquivalenceLibrary,
    #             ["id", "rz", "sx", "x", "rzx", "cx", "reset"],
    #         ),
    #         Optimize1qGatesDecomposition(["id", "rz", "sx", "x", "rzx", "cx", "reset"]),
    #         RZXCalibrationBuilder(
    #             backend.defaults().instruction_schedule_map,
    #             backend.configuration().qubit_channel_mapping,
    #         ),
    #     ]
    # )
    pm = PassManager(
        [
            BasisTranslator(
                SessionEquivalenceLibrary,
                target_basis=["id", "rz", "sx", "x", "rzx", "cx", "reset"],
            ),
            Optimize1qGatesDecomposition(
                basis=["id", "rz", "sx", "x", "rzx", "cx", "reset"]
            ),
        ]
    )
    # Decompose the unitary
    u_unitary = np.array(
        [
            [np.cos(theta), 0.0, 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, 0.0, np.cos(theta)],
        ]
    )
    u_gate: QuantumCircuit = pm.run(xx_decomposer(u_unitary))  # type: ignore
    return u_gate


if MAIN:
    theta_range = np.linspace(-np.pi, np.pi, 101)
    theta_param = Parameter("theta")
    r_zx_angle_u = []
    for theta in theta_range:
        u_orig = u_original(theta)
        u_unitary_orig = get_unitary(u_orig)
        u_decomp = u(theta)
        u_unitary_decomp = get_unitary(u_decomp)
        assert np.allclose(u_unitary_orig, u_unitary_decomp, atol=1e-7)  # type: ignore
        r_zx_angle_u.append(sum(abs(gate.operation.params[0]) for gate in u_decomp.get_instructions("rzx")) / np.pi)  # type: ignore
    print(f"For u(theta), the total accumulated rotation angle for the R_ZX gates in multiples of pi has maximum {max(r_zx_angle_u):.2f}, minimum {min(r_zx_angle_u):.2f}, mean {np.mean(r_zx_angle_u):.2f}, and median {np.median(r_zx_angle_u):.2f}. Note that a CX gate corresponds to an angle 0.50.")  # type: ignore


# %%


def q_original() -> QuantumCircuit:
    """
    Returns:
        q_gate: A Qiskit circuit representing the q initialisation gate
    """
    q_gate = QuantumCircuit(3, name="q")
    init_state: list[float] = [
        1.0 / np.sqrt(109),
        0.0,
        0.0,
        6.0 / np.sqrt(109),
        0.0,
        6.0 / np.sqrt(109),
        6.0 / np.sqrt(109),
        0.0,
    ]
    q_gate.initialize(init_state, [0, 1, 2])  # type: ignore
    q_gate = transpile(
        q_gate.decompose(),
        backend=Aer.get_backend("aer_simulator"),
        optimization_level=3,
    )
    q_gate.remove_final_measurements()  # type: ignore
    return q_gate  # type: ignore


def q() -> QuantumCircuit:
    q_gate = QuantumCircuit(3, name="q")
    q_gate.ry(2.191045812777718, 1)
    q_gate.ry(1.897688061737172, 2)
    q_gate.cx(2, 1)
    q_gate.ry(0.620249485982821, 1)
    q_gate.cx(1, 0)
    q_gate.cx(2, 0)
    return q_gate


if MAIN:
    backend = Aer.get_backend("aer_simulator")
    q_orig = q_original()
    q_orig_cx = len(q_orig.get_instructions("cx"))
    q_orig.save_statevector()  # type: ignore
    result = backend.run(q_orig).result()
    statevector_orig = result.get_statevector()
    q_decomp = q()
    q_decomp_cx = len(q_decomp.get_instructions("cx"))
    q_decomp.save_statevector()  # type: ignore
    result_decomp = backend.run(q_decomp).result()
    statevector_decomp = result_decomp.get_statevector()
    assert np.allclose(statevector_orig, statevector_decomp, atol=1e-7)  # type: ignore
    print(f"The original q gate uses {q_orig_cx} CX gates, whereas the decomposed q gate uses {q_decomp_cx} CX gates.")  # type: ignore


# %%


if MAIN:
    backend = FakeNairobi()
    rzx_calibrate = PassManager(RZXCalibrationBuilder(backend.defaults().instruction_schedule_map, backend.configuration().qubit_channel_mapping))  # type: ignore
    w_test = rzx_calibrate.run(w(np.pi / 12))  # type: ignore
    u_test = rzx_calibrate.run(u(-np.pi / 6, backend))  # type: ignore
    w_test.measure_all()  # type: ignore
    u_test.measure_all()  # type: ignore
    rzx_w: list[float] = []
    rzx_u: list[float] = []
    optimization_levels = [0, 1, 2, 3]
    for optimization_level in optimization_levels:
        w_transpiled = transpile(
            w_test, backend=backend, optimization_level=optimization_level
        )
        u_transpiled = transpile(
            u_test, backend=backend, optimization_level=optimization_level
        )
        # display(w_transpiled.draw("mpl"))
        w_job = backend.run(w_transpiled, shots=1e4)  # type: ignore
        w_results = w_job.result().get_counts()  # type: ignore
        u_job = backend.run(u_transpiled, shots=1e4)  # type: ignore
        u_results = u_job.result().get_counts()  # type: ignore
        rzx_w.append((sum(abs(gate.operation.params[0]) for gate in w_test.get_instructions("rzx")) - sum(abs(gate.operation.params[0]) for gate in w_transpiled.get_instructions("rzx"))) / np.pi)  # type: ignore
        rzx_u.append((sum(abs(gate.operation.params[0]) for gate in u_test.get_instructions("rzx")) - sum(abs(gate.operation.params[0]) for gate in u_transpiled.get_instructions("rzx"))) / np.pi)  # type: ignore
    print(
        f"For optimisation levels {optimization_levels}, the differences in the total accumulated rotation angle for the R_ZX gates in multiples of pi for w(theta) are {[float(f'{theta:.4f}') for theta in rzx_w]}, and for u(theta) are {[float(f'{theta:.4f}') for theta in rzx_u]}. That is, the RZX gates are turned into CX gates when the optimisation level is 3."
    )


# %%


if MAIN:
    backend = FakeNairobi()
    rzx_calibrate = PassManager(RZXCalibrationBuilder(backend.defaults().instruction_schedule_map, backend.configuration().qubit_channel_mapping))  # type: ignore
    w_test = w_original(np.pi / 12)  # type: ignore
    u_test = u_original(-np.pi / 6)  # type: ignore
    w_test.measure_all()
    u_test.measure_all()
    rzx_w: list[float] = []
    rzx_u: list[float] = []
    optimization_levels = [0, 1, 2, 3]
    for optimization_level in optimization_levels:
        w_transpiled = transpile(
            w_test, backend=backend, optimization_level=optimization_level
        )
        u_transpiled = transpile(
            u_test, backend=backend, optimization_level=optimization_level
        )
        # display(w_transpiled.draw("mpl"))
        w_job = backend.run(w_transpiled, shots=1e4)  # type: ignore
        w_results = w_job.result().get_counts()  # type: ignore
        u_job = backend.run(u_transpiled, shots=1e4)  # type: ignore
        u_results = u_job.result().get_counts()  # type: ignore
        rzx_w.append(len(w_transpiled.get_instructions("cx")))  # type: ignore
        rzx_u.append(len(u_transpiled.get_instructions("cx")))  # type: ignore
    print(
        f"For optimisation levels {optimization_levels}, the number of CX gates for w(theta) are {[float(f'{theta:.4f}') for theta in rzx_w]}, and for u(theta) are {[float(f'{theta:.4f}') for theta in rzx_u]}."
    )


# %%

"""
Note that we can get the PassManager for certain optimisation levels using this:
https://qiskit.org/documentation/stubs/qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager.html
"""
