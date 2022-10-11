# %%

import math
import random
from copy import deepcopy
from time import time
from typing import Optional, Union
from qiskit import IBMQ, ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator

MAIN = __name__ == "__main__"


# %%


def dmera(n: int, d: int) -> tuple[list[list[tuple[int, int]]], list[list[int]]]:
    """
    Args:
        n: Number of scales
        d: Depth at each scale
    Returns:
        circuit: DMERA circuit
        sites_list: List of the sites at each scale
    """
    circuit: list[list[tuple[int, int]]] = []
    sites_list: list[list[int]] = []
    for s in range(n + 1):
        qubits = 3 * 2**s
        sites = [i * (1 << (n - s)) for i in range(qubits)]
        sites_list.append(sites)
        if s != 0:
            for j in range(d):
                if j % 2 == 0:
                    even_sites = [
                        (
                            sites[(2 * i + 0) % (qubits)],
                            sites[(2 * i + 1) % (qubits)],
                        )
                        for i in range(3 * 2 ** (s - 1))
                    ]
                    circuit.append(even_sites)
                else:
                    odd_sites = [
                        (
                            sites[(2 * i + 1) % (qubits)],
                            sites[(2 * i + 2) % (qubits)],
                        )
                        for i in range(3 * 2 ** (s - 1))
                    ]
                    circuit.append(odd_sites)
    return (circuit, sites_list)


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


def used(
    circuit: list[list[tuple[int, int]]], sites_list: list[list[int]], n: int
) -> tuple[list[int], list[int]]:
    """
    Args:
        circuit: Circuit
        sites_list: List of the sites at each scale
        n: Number of scales
    Returns:
        start_use: The circuit layer at which each qubit starts being used
        stop_use: The circuit layer at which each qubit stops being used
    """
    L = len(circuit)
    circuit_sorted: list[list[int]] = []
    circuit_sorted.append(sites_list[0])
    for layer in circuit:
        layer_sorted = sorted([qubit for gate in layer for qubit in gate])
        circuit_sorted.append(layer_sorted)
    # Mark unused qubits with the default value L
    qubits = 3 * 2**n
    start_use: list[int] = [L] * qubits
    stop_use: list[int] = [L] * qubits
    for qubit in range(qubits):
        # Determine the layers in which the qubit appears
        qubit_used: list[int] = [
            layer_index - 1
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
    sites_list: list[list[int]],
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
        sites_list: List of the sites at each scale
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
    qubits = 3 * 2**n
    assert L == len(pcc_circuit)
    if reset_time is None:
        reset_length = L
    else:
        reset_length = reset_time * reset_count
    (start_use, stop_use) = used(pcc_circuit, sites_list, n)
    qubit_map = list(range(qubits))
    reset_in_progress: list[int] = []
    # Store the resets and the mapping on the qubits
    resets: list[list[int]] = []
    reset_map: dict[tuple[int, int], int] = {}
    #################################################
    # Determine the resets for the initialisation
    #################################################
    layer_index = -1
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
    # Flatten the reset qubits
    reset_allocation = [
        qubit_index for layer in reset_allocation_layers for qubit_index in layer
    ]
    # Reset qubits and relabel them appropriately with the qubit map
    reset: list[int] = []
    for j in range(min(len(reset_qubits), len(reset_allocation))):
        reset.append(qubit_map[reset_qubits[j]])
        reset_in_progress.append(reset_allocation[j])
        qubit_map[reset_allocation[j]] = qubit_map[reset_qubits[j]]
        qubit_map[reset_qubits[j]] = qubits
    resets.append(reset)
    for qubit in sites_list[0]:
        reset_map[(-1, qubit)] = qubit_map[qubit]
    #################################################
    # Determine the resets for the circuit
    #################################################
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
            qubit_map[reset_qubits[j]] = qubits
        resets.append(reset)
    #################################################
    # Generate the inverse map
    #################################################
    qubit_reset_map = list(set(reset_map.values()))
    inverse_map: dict[int, int] = {}
    for (qubit_index, qubit) in enumerate(qubit_reset_map):
        inverse_map[qubit] = qubit_index
    return (pcc_circuit, resets, reset_map, inverse_map)


# %%


def get_backend(backend_name: str = "aer") -> AerSimulator:
    """
    Args:
        backend_name: A Qiskit backend name
    Returns:
        backend: A Qiskit backend
    """
    if backend_name == "aer":
        backend = AerSimulator(method="automatic")
    else:
        provider = IBMQ.load_account()
        backend = provider.get_backend(backend_name)
    return backend


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


def q(backend: AerSimulator) -> QuantumCircuit:
    """
    Args:
        backend: A Qiskit backend
    Returns:
        q_gate: A Qiskit circuit representing the q initialisation gate
    """
    q_gate = QuantumCircuit(3, name="q")
    init_state: list[float] = [
        1 / math.sqrt(109),
        0.0,
        0.0,
        6 / math.sqrt(109),
        0.0,
        6 / math.sqrt(109),
        6 / math.sqrt(109),
        0.0,
    ]
    q_gate.initialize(init_state, [0, 1, 2])
    q_gate = transpile(q_gate.decompose(), backend=backend, optimization_level=3)
    return q_gate


# %%


def generate_params(
    circuit: list[list[tuple[int, int]]]
) -> tuple[dict[tuple[int, tuple[int, int]], Parameter], dict[Parameter, float]]:
    """
    Args:
        circuit: Circuit
    Returns:
        theta_dict: Dictonary of Qiskit parameters for each gate in each layer of the circuit, indexed by the layer and gate
        theta_values: Dictionary of Qiskit parameter values for each parameter, indexed by the parameter
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
    sites_list: list[list[int]],
    support: list[int],
    theta_dict: dict[tuple[int, tuple[int, int]], Parameter],
    resets: list[list[int]],
    reset_map: dict[tuple[int, int], int],
    inverse_map: dict[int, int],
    q_gate: QuantumCircuit,
    reset_count: int = 1,
    barriers: bool = False,
    reverse_gate: bool = False,
) -> QuantumCircuit:
    """
    Args:
        n: Number of scales
        d: Depth at each scale
        pcc_circuit: DMERA past causal cone circuit of the observable
        sites_list: List of the sites at each scale
        support: Support of the observable
        theta_dict: Dictonary of Qiskit parameters for each layer, indexed by the layer and gate
        resets: Qubits which are reset in each layer
        reset_map: Reset mappings for the qubits upon which the gates in each layer act
        reset_count: Number of reset operations to perform a reset
        barriers: Add barriers to the circuit for ease of readability
        reverse_gate: For each gate, reverse which qubit is considered to be the 'first' on which the gate operates
    Returns:
        quantum_circuit: A Qiskit circuit implementing the DMERA circuit with reset
    """
    # Determine the qubits from the reset map and set up the circuit
    L = n * d
    qubits = list(inverse_map.keys())
    quantum_register = (QuantumRegister(1, "qubit" + str(qubit)) for qubit in qubits)
    classical_register = ClassicalRegister(len(support), "measure")
    quantum_circuit = QuantumCircuit(*(quantum_register), classical_register)
    assert len(qubits) <= 3 * 2**n
    assert L == len(pcc_circuit)
    # Initialise the quantum circuit
    quantum_circuit.append(
        q_gate.to_gate(),
        [
            inverse_map[reset_map[(-1, sites_list[0][0])]],
            inverse_map[reset_map[(-1, sites_list[0][1])]],
            inverse_map[reset_map[(-1, sites_list[0][2])]],
        ],
    )
    for qubit in resets[0]:
        for _ in range(reset_count):
            quantum_circuit.reset(inverse_map[qubit])
    # Populate the circuit with gates and reset
    for (layer_index, layer) in enumerate(pcc_circuit):
        for gate in layer:
            # Determine the appropriate gate
            if layer_index % d == 0:
                custom_gate = w(theta_dict[(layer_index, gate)]).to_gate()
            else:
                custom_gate = u(theta_dict[(layer_index, gate)]).to_gate()
            # Append the gate to the circuit
            if reverse_gate:
                quantum_circuit.append(
                    custom_gate,
                    [
                        inverse_map[reset_map[(layer_index, gate[1])]],
                        inverse_map[reset_map[(layer_index, gate[0])]],
                    ],
                )
            else:
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
        for qubit in resets[1 + layer_index]:
            for _ in range(reset_count):
                quantum_circuit.reset(inverse_map[qubit])
    return quantum_circuit


# %%


def generate_transverse_ising_circuits(
    n: int,
    d: int,
    backend: AerSimulator,
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
        backend: A Qiskit backend
        reset_time: Time taken to perform a reset operation as a multiple of the time taken for a circuit layer
        reset_count: Number of reset operations to perform a reset
    Returns:
        quantum_circuits: A dictionary of Qiskit circuits implementing the DMERA circuit with reset for all of the different observables, indexed by the support of the observables
        pcc_circuits: A dictionary of DMERA past causal cone circuits for all of the different observables, indexed by the support of the observables
        theta_dict: Dictonary of Qiskit parameters for each gate in each layer of the circuit, indexed by the layer and gate
        theta_values: Dictionary of Qiskit parameter values for each parameter, indexed by the parameter
    """
    # Initialise the circuit and variational parameters
    L = n * d
    qubits = 3 * 2**n
    (circuit, sites_list) = dmera(n, d)
    (theta_dict, theta_values) = generate_params(circuit)
    q_gate = q(backend)
    assert L == len(circuit)
    # Generate the supports
    two_qubit_supports = [[(i + 0) % qubits, (i + 1) % qubits] for i in range(qubits)]
    three_qubit_supports = [
        [(i - 1) % qubits, (i + 0) % qubits, (i + 1) % qubits] for i in range(qubits)
    ]
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
            n, d, circuit, sites_list, support, reset_time, reset_count
        )
        # Generate the Qiskit circuit
        quantum_circuit = generate_dmera_reset(
            n,
            d,
            pcc_circuit,
            sites_list,
            support,
            theta_dict,
            resets,
            reset_map,
            inverse_map,
            q_gate,
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
            quantum_circuit.h(inverse_map[reset_map[(L - 1, support[0])]])
            quantum_circuit.h(inverse_map[reset_map[(L - 1, support[2])]])
            quantum_circuit.measure(inverse_map[reset_map[(L - 1, support[0])]], 0)
            quantum_circuit.measure(inverse_map[reset_map[(L - 1, support[1])]], 1)
            quantum_circuit.measure(inverse_map[reset_map[(L - 1, support[2])]], 2)
        else:
            raise ValueError
        # Add the circuit to the dictionary
        quantum_circuits[tuple(support)] = quantum_circuit
        pcc_circuits[tuple(support)] = pcc_circuit
    # Print a diagnostic
    print(
        f"The maximum number of qubits required for any of the DMERA past causal cone circuits is {max(qubit_numbers)}, where the number of scales is {n}, the depth at each scale is {d}, the reset time is {reset_time}, and the reset count is {reset_count}.\n"
    )
    return (quantum_circuits, pcc_circuits, theta_dict, theta_values)


# %%


def get_theta_evenbly(d: int) -> list[float]:
    """
    Args:
        d: Depth at each scale
    Returns:
        theta_evenbly: Angles supplied in Entanglement renormalization and wavelets by Evenbly and White (2016)
    """
    theta_evenbly_2 = [math.pi / 12, math.pi / 12]  # -math.pi/6
    theta_evenbly_4 = [
        0.276143653403021,
        0.950326554644286,
        -0.111215262156182,
        -math.pi / 2,
    ]
    theta_evenbly_5 = [
        0.133662134988773,
        -1.311424155804674,
        -0.099557657512352,
        0.717592959416643,
        0.157462489552395,
    ]
    if d == 2:
        theta_evenbly = theta_evenbly_2
    elif d == 4:
        theta_evenbly = theta_evenbly_4
    elif d == 5:
        theta_evenbly = theta_evenbly_5
    else:
        print(
            f"No cached values for theta when d = {d}. Falling back and setting all thetas to be zero."
        )
        theta_evenbly = [0.0] * d
    return theta_evenbly


# %%


def estimate_site_energy(
    n: int,
    backend: AerSimulator,
    sample_number: int,
    shots: int,
    quantum_circuits: dict[
        Union[tuple[int, int], tuple[int, int, int]], QuantumCircuit
    ],
    pcc_circuits: dict[
        Union[tuple[int, int], tuple[int, int, int]], list[list[tuple[int, int]]]
    ],
    theta_dict: dict[tuple[int, tuple[int, int]], Parameter],
    theta_values: dict[Parameter, float],
):
    """
    Args:

    Returns:
        sites_energy:
    """
    # Generate random sites and determine the energies for each site
    qubits = 3 * 2**n
    sites = random.sample(range(qubits), sample_number)
    sites_energy: list[list[float]] = []
    for site in sites:
        site_supports: list[Union[tuple[int, int], tuple[int, int, int]]] = [
            ((site - 1) % qubits, (site + 0) % qubits, (site + 1) % qubits),
            ((site - 1) % qubits, (site + 0) % qubits),
            ((site + 0) % qubits, (site + 1) % qubits),
        ]
        # Bind the circuit parameters
        bound_circuits: list[QuantumCircuit] = []
        for site_support in site_supports:
            theta_values_pcc: dict[Parameter, float] = {
                theta_dict[(layer_index, gate)]: theta_values[
                    theta_dict[(layer_index, gate)]
                ]
                for (layer_index, layer) in enumerate(pcc_circuits[site_support])
                for gate in layer
            }
            bound_circuits.append(
                quantum_circuits[site_support]
                .decompose()
                .bind_parameters(theta_values_pcc)
            )
        # Transpile the circuits
        site_circuits = transpile(
            bound_circuits,
            backend=backend,
            optimization_level=0,
        )
        # Run the circuits
        start_time = time()
        job = backend.run(site_circuits, shots=shots)
        job_results = job.result().get_counts()
        counts: list[dict[str, int]] = [dict(count) for count in job_results]
        end_time = time()
        print(f"Running the circuits for site {site} took {end_time - start_time} s.")
        site_operator_values = [
            sum(
                ((-1) ** (sum(int(bit) for bit in key) % 2)) * count[key]
                for key in count.keys()
            )
            / shots
            for count in counts
        ]
        sites_energy.append(site_operator_values)
    return sites_energy


# %%

if MAIN:
    n = 2
    d = 2
    theta_evenbly = get_theta_evenbly(d)
    reset_time = None
    reset_count = 1
    backend = get_backend()
    (
        quantum_circuits,
        pcc_circuits,
        theta_dict,
        theta_values,
    ) = generate_transverse_ising_circuits(n, d, backend, reset_time, reset_count)
    # Set the theta values
    for (layer_index, gate) in theta_dict.keys():
        theta_values[theta_dict[(layer_index, gate)]] = theta_evenbly[layer_index % d]

# %%

if MAIN:
    sample_number = 5
    shots = 10**4
    sites_energy = estimate_site_energy(
        n,
        backend,
        sample_number,
        shots,
        quantum_circuits,
        pcc_circuits,
        theta_dict,
        theta_values,
    )
    print(sites_energy)
    abs_site_energy = [
        -abs(site[0]) - 0.5 * (abs(site[1]) + abs(site[2])) for site in sites_energy
    ]
    site_energy = [site[0] + 0.5 * (site[1] + site[2]) for site in sites_energy]
    site_sum = [sum(site) for site in sites_energy]
    print(site_energy)


# %%

if MAIN:
    # support_example = (2, 3, 4)
    support_example = (3, 4)
    quantum_example = transpile(
        quantum_circuits[support_example].decompose(),
        backend=backend,
        optimization_level=1,
    )
    pcc_example = pcc_circuits[support_example]
    # Filter the dictionary for binding the parameters
    theta_pcc_values: dict[Parameter, float] = {
        theta_dict[(layer_index, gate)]: theta_values[theta_dict[(layer_index, gate)]]
        for (layer_index, layer) in enumerate(pcc_example)
        for gate in layer
    }
    transpiled_example = transpile(
        quantum_example.bind_parameters(theta_pcc_values),
        backend=backend,
        optimization_level=1,
    )
    shots = 1000
    start = time()
    job = backend.run(transpiled_example, shots=shots)
    print(time() - start)
    counts = job.result().get_counts()
    print(time() - start)
    op_val = 0
    for key in counts.keys():
        if sum(int(bit) for bit in key) % 2 == 0:
            op_val += counts[key]
        else:
            op_val -= counts[key]
    op_val /= shots
    print(op_val)

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
