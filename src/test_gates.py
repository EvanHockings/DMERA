# %%

from protocol import w, u, q  # type: ignore
from qiskit import QuantumCircuit, Aer  # type: ignore
from qiskit.quantum_info import XXDecomposer  # type: ignore
from qiskit.circuit import Parameter  # type: ignore
from qiskit.transpiler import PassManager  # type: ignore
from qiskit.transpiler.passes import Optimize1qGatesDecomposition, BasisTranslator  # type: ignore
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary  # type: ignore

MAIN = __name__ == "__main__"


# %%


def get_unitary(qc: QuantumCircuit, num_dec_pt: int = 5) -> list[list[float]]:
    """
    Args:
        qc: Input quantum circuit
        num_dec_pt: Number of decimal points of the unitary matrix
    Returns:
        unitary: Unitary matrix corresponding to the quantum circuit
    """
    # Determine the unitary matrix corresponding to the quantum circuit
    backend = Aer.get_backend("aer_simulator")
    qc.save_unitary()  # type: ignore
    result = backend.run(qc).result()
    unitary = result.get_unitary(qc, num_dec_pt).data
    return unitary


# %%

if MAIN:
    # Initialise parameters
    theta = Parameter("theta")
    theta_value = {theta: 0.5}
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
    draw_options = {"output": "mpl", "fold": -1}
    # w gate
    try:
        w_gate = w(theta)
        w_bound = w_gate.bind_parameters(theta_value)
        w_unitary = get_unitary(w_bound)
        w_decomposed = xx_decomposer(w_unitary)  # type: ignore
        w_transpiled = pm.run(w_decomposed)  # type: ignore
        display(w_transpiled.draw(**draw_options))  # type: ignore
    except:
        print("w gate decomposition failed.")
    # u gate
    try:
        u_gate = u(theta)
        u_bound = u_gate.bind_parameters(theta_value)
        u_unitary = get_unitary(u_bound)
        u_decomposed = xx_decomposer(u_unitary)  # type: ignore
        u_transpiled = pm.run(u_decomposed)  # type: ignore
        display(u_transpiled.draw(**draw_options))  # type: ignore
    except:
        print("u gate decomposition failed.")
    # q gate
    try:
        q_gate = q()
        q_unitary = get_unitary(q_gate)
        q_decomposed = xx_decomposer(q_gate)  # type: ignore
        q_transpiled = pm.run(q_decomposed)  # type: ignore
        display(q_transpiled.draw(**draw_options))  # type: ignore
    except:
        print("q gate decomposition failed.")


# %%
