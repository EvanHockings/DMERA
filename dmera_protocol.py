# %%

import math
from copy import deepcopy
from typing import Optional, Union
from qiskit import IBMQ, ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.compiler import transpile

MAIN = __name__ == "__main__"


# %%


def dmera(n: int, d: int) -> list[list[tuple[int, int]]]:
    """
    Args:
        n: Number of scales
        d: Depth at each scale
    Returns:
        circuit: DMERA circuit
    """
    circuit: list[list[tuple[int, int]]] = []
    for s in range(n):
        sites = [i * (1 << (n - (s + 1))) for i in range(2 ** (s + 1))]
        for j in range(d):
            if j % 2 == 0:
                even_sites = [
                    (
                        sites[(2 * i + 0) % (2 ** (s + 1))],
                        sites[(2 * i + 1) % (2 ** (s + 1))],
                    )
                    for i in range(2**s)
                ]
                circuit.append(even_sites)
            else:
                odd_sites = [
                    (
                        sites[(2 * i + 1) % (2 ** (s + 1))],
                        sites[(2 * i + 2) % (2 ** (s + 1))],
                    )
                    for i in range(2**s)
                ]
                circuit.append(odd_sites)
    return circuit


# %%


def pcc(
    circuit: list[list[tuple[int, int]]], support: list[int]
) -> list[list[tuple[int, int]]]:
    """
    Args:
        circuit: Circuit
        support: Support of the observable
    Returns:
        pcc_circuit: Past causal cone circuit for the observable
    """
    circuit_reduced: list[list[tuple[int, int]]] = []
    # Store the qubits on which the circuit is supported
    supported_qubits = deepcopy(support)
    # Construct the pcc circuit in reverse order
    for layer in reversed(circuit):
        layer_temp: list[tuple[int, int]] = []
        for gate in layer:
            supported_0 = gate[0] in supported_qubits
            supported_1 = gate[1] in supported_qubits
            # If the gate is in the support of the observable, add it to the circuit, and grow the support appropriately
            if supported_0 or supported_1:
                layer_temp.append(gate)
                if not supported_0:
                    supported_qubits.append(gate[0])
                if not supported_1:
                    supported_qubits.append(gate[1])
        circuit_reduced.append(layer_temp)
    pcc_circuit = list(reversed(circuit_reduced))
    return pcc_circuit


# %%


def used(circuit: list[list[tuple[int, int]]], n: int) -> tuple[list[int], list[int]]:
    """
    Args:
        circuit: Circuit
        n: Number of scales
    Returns:
        start_use: The circuit layer at which each qubit starts being used
        stop_use: The circuit layer at which each qubit stops being used
    """
    circuit_sorted: list[list[int]] = []
    L = len(circuit)
    for layer in circuit:
        layer_sorted = sorted([qubit for gate in layer for qubit in gate])
        circuit_sorted.append(layer_sorted)
    # Mark unused qubits with the default value L
    start_use: list[int] = [L] * (2**n)
    stop_use: list[int] = [L] * (2**n)
    for qubit in range(2**n):
        # Determine the layers in which the qubit appears
        qubit_used: list[int] = [
            layer_index
            for (layer_index, layer_sorted) in enumerate(circuit_sorted)
            if qubit in layer_sorted
        ]
        # Store the timesteps at which the qubit starts and stops being used.
        if qubit_used:
            start_use[qubit] = qubit_used[0]
            stop_use[qubit] = qubit_used[-1]
    return (start_use, stop_use)


# %%


def dmera_reset(
    n: int,
    d: int,
    circuit: list[list[tuple[int, int]]],
    support: list[int],
    reset_time: Optional[int] = None,
    reset_count: int = 1,
) -> tuple[
    list[list[tuple[int, int]]],
    list[list[int]],
    dict[tuple[int, int], int],
    dict[int, int],
]:
    """
    Args:
        n: Number of scales
        d: Depth at each scale
        circuit: DMERA circuit
        support: Support of the observable
        reset_time: Time taken to perform a reset operation as a multiple of the time taken for a circuit layer
        reset_count: Number of reset operations to perform a reset
    Returns:
        pcc_circuit: DMERA past causal cone circuit of the observable
        resets: Qubits which are reset in each layer
        reset_map: Reset mapping for the qubits upon which the gates in each layer act
        inverse_map: Inverse mapping for the qubit reset mapping
    """
    # Initialise parameters
    pcc_circuit = pcc(circuit, support)
    L = n * d
    assert L == len(pcc_circuit)
    if reset_time is None:
        reset_length = L
    else:
        reset_length = reset_time * reset_count
    (start_use, stop_use) = used(pcc_circuit, n)
    qubit_map = list(range(2**n))
    reset_in_progress: list[int] = []
    # Store the resets and the mapping on the qubits
    resets: list[list[int]] = []
    reset_map: dict[tuple[int, int], int] = {}
    # Determine the resets
    for (layer_index, layer) in enumerate(pcc_circuit):
        # Update the reset tracker and store the qubit mapping for the gates
        for gate in layer:
            if gate[0] in reset_in_progress:
                reset_in_progress.remove(gate[0])
            if gate[1] in reset_in_progress:
                reset_in_progress.remove(gate[1])
            reset_map[(layer_index, gate[0])] = qubit_map[gate[0]]
            reset_map[(layer_index, gate[1])] = qubit_map[gate[1]]
        # Determine the qubits that can be reset
        reset_qubits = [
            qubit_index
            for (qubit_index, stop_index) in enumerate(stop_use)
            if stop_index == layer_index
        ]
        # Determine the qubits that reset qubits could be used to replace
        reset_allocation_layers = [
            [
                qubit_index
                for (qubit_index, start_index) in enumerate(start_use)
                if start_index == future_layer_index
            ]
            for future_layer_index in range(layer_index + 1 + reset_length, L)
        ]
        # Filter out the qubits currently being reset
        reset_allocation = [
            qubit_index
            for layer in reset_allocation_layers
            for qubit_index in layer
            if qubit_index not in reset_in_progress
        ]
        # Reset qubits and relabel them appropriately with the qubit map
        reset: list[int] = []
        for j in range(min(len(reset_qubits), len(reset_allocation))):
            reset.append(qubit_map[reset_qubits[j]])
            reset_in_progress.append(reset_allocation[j])
            qubit_map[reset_allocation[j]] = qubit_map[reset_qubits[j]]
            qubit_map[reset_qubits[j]] = 2**n
        resets.append(reset)
    # Generate the inverse map
    qubits = list(set(reset_map.values()))
    inverse_map: dict[int, int] = {}
    for (qubit_index, qubit) in enumerate(qubits):
        inverse_map[qubit] = qubit_index
    return (pcc_circuit, resets, reset_map, inverse_map)


# %%


def w(param: Parameter) -> QuantumCircuit:
    """
    Args:
        param: A Qiskit parameter
    Returns:
        w_gate: A Qiskit circuit representing the w(param) gate
    """
    w_gate = QuantumCircuit(2, name="w")
    w_gate.h([0, 1])
    w_gate.cz(0, 1)
    w_gate.sdg([0, 1])
    w_gate.h([0, 1])
    w_gate.rz(-param + math.pi / 2, 0)
    w_gate.rz(-param, 1)
    w_gate.h([0, 1])
    w_gate.s([0, 1])
    w_gate.cz(0, 1)
    w_gate.h([0, 1])
    return w_gate


def u(param: Parameter) -> QuantumCircuit:
    """
    Args:
        param: A Qiskit parameter
    Returns:
        u_gate: A Qiskit circuit representing the u(param) gate
    """
    u_gate = QuantumCircuit(2, name="u")
    u_gate.h([0, 1])
    u_gate.cz(0, 1)
    u_gate.sdg([0, 1])
    u_gate.h([0, 1])
    u_gate.rz(-param, 0)
    u_gate.rz(-param, 1)
    u_gate.h([0, 1])
    u_gate.s([0, 1])
    u_gate.cz(0, 1)
    u_gate.h([0, 1])
    return u_gate


# %%


def generate_params(
    circuit: list[list[tuple[int, int]]]
) -> tuple[dict[tuple[int, tuple[int, int]], Parameter], dict[Parameter, float]]:
    """
    Args:
        circuit: Circuit
    Returns:
        theta_dict: Dictonary of Qiskit parameters for each gate in each layer of the circuit
        theta_values: Dictionary of Qiskit parameter values for each parameter
    """
    theta_dict: dict[tuple[int, tuple[int, int]], Parameter] = {}
    theta_values: dict[Parameter, float] = {}
    for (layer_index, layer) in enumerate(circuit):
        for gate in layer:
            # Each gate is given an individual parameter, even though the parameter for each gate in a layer is the same, because it allows for easier calculation of derivatives without repeated transpilation
            theta_dict[(layer_index, gate)] = Parameter(f"theta({layer_index},{gate})")
            theta_values[theta_dict[(layer_index, gate)]] = 0.0
    return (theta_dict, theta_values)


# %%


def generate_dmera_reset(
    n: int,
    d: int,
    pcc_circuit: list[list[tuple[int, int]]],
    support: list[int],
    theta_dict: dict[tuple[int, tuple[int, int]], Parameter],
    resets: list[list[int]],
    reset_map: dict[tuple[int, int], int],
    inverse_map: dict[int, int],
    reset_count: int = 1,
    barriers: bool = False,
) -> QuantumCircuit:
    """
    Args:
        n: Number of scales
        d: Depth at each scale
        pcc_circuit: DMERA past causal cone circuit of the observable
        support: Support of the observable
        theta_dict: Dictonary of Qiskit parameters for each layer
        resets: Qubits which are reset in each layer
        reset_map: Reset mappings for the qubits upon which the gates in each layer act
        reset_count: Number of reset operations to perform a reset
        barriers: Add barriers to the circuit for ease of readability
    Returns:
        quantum_circuit: A Qiskit circuit implementing the DMERA circuit with reset
    """
    # Determine the qubits from the reset map and set up the circuit
    L = n * d
    qubits = list(set(reset_map.values()))
    quantum_register = (QuantumRegister(1, "qubit" + str(qubit)) for qubit in qubits)
    classical_register = ClassicalRegister(len(support), "measure")
    quantum_circuit = QuantumCircuit(*(quantum_register), classical_register)
    assert len(qubits) <= 2**n
    assert L == len(pcc_circuit)
    # Populate the circuit with gates and reset
    for (layer_index, layer) in enumerate(pcc_circuit):
        for gate in layer:
            # Determine the appropriate gate
            if layer_index % d == 0:
                custom_gate = w(theta_dict[(layer_index, gate)]).to_gate()
            else:
                custom_gate = u(theta_dict[(layer_index, gate)]).to_gate()
            # Append the gate to the circuit
            quantum_circuit.append(
                custom_gate,
                [
                    inverse_map[reset_map[(layer_index, gate[0])]],
                    inverse_map[reset_map[(layer_index, gate[1])]],
                ],
            )
        # Add barriers to the circuit for ease of readability
        if barriers and layer_index != L - 1:
            quantum_circuit.barrier()
        # Reset the appropriate qubits
        for qubit in resets[layer_index]:
            for _ in range(reset_count):
                quantum_circuit.reset(inverse_map[qubit])
    return quantum_circuit


# %%


def generate_transverse_ising_circuits(
    n: int,
    d: int,
    reset_time: Optional[int] = None,
    reset_count: int = 1,
) -> tuple[
    dict[Union[tuple[int, int], tuple[int, int, int]], QuantumCircuit],
    dict[Union[tuple[int, int], tuple[int, int, int]], list[list[tuple[int, int]]]],
    dict[tuple[int, tuple[int, int]], Parameter],
    dict[Parameter, float],
]:
    """
    Args:
        n: Number of scales
        d: Depth at each scale
        reset_time: Time taken to perform a reset operation as a multiple of the time taken for a circuit layer
        reset_count: Number of reset operations to perform a reset
    Returns:
        quantum_circuits: A dictionary of Qiskit circuits implementing the DMERA circuit with reset for all of the different observables
        pcc_circuits: A dictionary of DMERA past causal cone circuits for all of the different observables
        theta_dict: Dictonary of Qiskit parameters for each gate in each layer of the circuit
        theta_values: Dictionary of Qiskit parameter values for each parameter
    """
    # Initialise the circuit and variational parameters
    L = n * d
    circuit = dmera(n, d)
    (theta_dict, theta_values) = generate_params(circuit)
    assert L == len(circuit)
    # Generate the supports
    two_qubit_supports = [list(range(i, i + 2)) for i in range(2**n - 1)]
    three_qubit_supports = [list(range(i, i + 3)) for i in range(2**n - 2)]
    supports: list[list[int]] = two_qubit_supports + three_qubit_supports
    qubit_numbers: list[int] = []
    quantum_circuits: dict[
        Union[tuple[int, int], tuple[int, int, int]], QuantumCircuit
    ] = {}
    pcc_circuits: dict[
        Union[tuple[int, int], tuple[int, int, int]], list[list[tuple[int, int]]]
    ] = {}
    # Generate the circuits
    for support in supports:
        # Generate the reset circuit parameters
        (pcc_circuit, resets, reset_map, inverse_map) = dmera_reset(
            n, d, circuit, support, reset_time, reset_count
        )
        # Generate the Qiskit circuit
        quantum_circuit = generate_dmera_reset(
            n,
            d,
            pcc_circuit,
            support,
            theta_dict,
            resets,
            reset_map,
            inverse_map,
            reset_count,
        )
        # Store the requisite number of qubits
        num_qubits = len(list(set(reset_map.values())))
        assert num_qubits == quantum_circuit.num_qubits
        qubit_numbers.append(num_qubits)
        # Add the appropriate measurements
        if len(support) == 2:
            quantum_circuit.h(inverse_map[reset_map[(L - 1, support[0])]])
            quantum_circuit.h(inverse_map[reset_map[(L - 1, support[1])]])
            quantum_circuit.measure(inverse_map[reset_map[(L - 1, support[0])]], 0)
            quantum_circuit.measure(inverse_map[reset_map[(L - 1, support[1])]], 1)
        elif len(support) == 3:
            quantum_circuit.h(inverse_map[reset_map[(L - 1, support[1])]])
            quantum_circuit.measure(inverse_map[reset_map[(L - 1, support[0])]], 0)
            quantum_circuit.measure(inverse_map[reset_map[(L - 1, support[1])]], 1)
            quantum_circuit.measure(inverse_map[reset_map[(L - 1, support[2])]], 2)
        else:
            raise ValueError
        # Add the circuit to the dictionary
        quantum_circuits[tuple(support)] = quantum_circuit.decompose()
        pcc_circuits[tuple(support)] = pcc_circuit
    # Print a diagnostic
    print(
        f"The maximum number of qubits required for any of the DMERA past causal cone circuits is {max(qubit_numbers)}, where the number of scales is {n}, the depth at each scale is {d}, the reset time is {reset_time}, and the reset count is {reset_count}.\n"
    )
    return (quantum_circuits, pcc_circuits, theta_dict, theta_values)


# %%

if MAIN:
    # Generate a set of quantum circuits
    n = 5
    d = 2
    reset_time = 1
    reset_count = 1
    (
        quantum_circuits,
        pcc_circuits,
        theta_dict,
        theta_values,
    ) = generate_transverse_ising_circuits(n, d, reset_time, reset_count)
    # Transpile an example circuit for a device and draw it
    provider = IBMQ.load_account()
    backend_name = "ibm_oslo"
    backend = provider.get_backend(backend_name)
    support_example = (0, 1, 2)
    quantum_example = quantum_circuits[support_example]
    pcc_example = pcc_circuits[support_example]
    transpiled_example = transpile(
        quantum_example.decompose(), backend=backend, optimization_level=3
    )
    print(
        f"This is a transpiled example circuit for the observable supported on {support_example} without bound parameters:"
    )
    display(transpiled_example.draw(output="mpl"))
    # Filter the dictionary for binding the parameters and draw the bound circuit
    theta_pcc_values: dict[Parameter, float] = {
        theta_dict[(layer_index, gate)]: theta_values[theta_dict[(layer_index, gate)]]
        for (layer_index, layer) in enumerate(pcc_example)
        for gate in layer
    }
    print(
        f"This is a transpiled example circuit for the observable supported on {support_example} with bound parameters:"
    )
    bound_example = transpile(
        transpiled_example.bind_parameters(theta_pcc_values),
        backend=backend,
        optimization_level=3,
    )
    display(bound_example.draw(output="mpl"))

# %%

# TODO: Reorganise this!


def dmera_protocol(
    quantum_circuits: dict[
        Union[tuple[int, int], tuple[int, int, int]], QuantumCircuit
    ],
    theta_dict: dict[tuple[int, tuple[int, int]], Parameter],
    theta_values: dict[Parameter, float],
    backend_name: str,
    optimization_level: int = 1,
):
    pass
    # # Get IBMQ backend
    # provider = IBMQ.load_account()
    # backend = provider.get_backend(backend_name)
    # # Cache previously transpiled circuits to save on transpiling costs
    # transpiled_circuits: dict[
    #     Union[tuple[int, int], tuple[int, int, int]], QuantumCircuit
    # ] = {}
    # # Transpile a circuit
    # transpiled_circuit: list[QuantumCircuit] = transpile(
    #     quantum_circuit, backend=backend, optimization_level=optimization_level
    # )
    # # Run a circuit
    # job = backend.run(transpiled_circuit.bind_parameters(theta_values))
    # retrieved_job = backend.retrieve_job(job.job_id())


# %%
