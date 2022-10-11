# %%

import math
import random
import numpy as np
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


def get_backend(backend_name: str = "aer"):
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
    w_gate.rz(param - math.pi / 2, 0)
    w_gate.rz(param, 1)
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
    u_gate.rz(param, 0)
    u_gate.rz(param, 1)
    u_gate.h([0, 1])
    u_gate.s([0, 1])
    u_gate.cz(0, 1)
    u_gate.h([0, 1])
    return u_gate


def q(backend) -> QuantumCircuit:
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
    reverse_gate: bool = True,
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
        inverse_map: Inverse mapping for the qubit reset mapping
        q_gate: Qiskit circuit that prepares the initial state
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
    backend,
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
    theta_evenbly_2 = [math.pi / 12, -math.pi / 6]
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
    backend,
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
) -> tuple[list[float], list[float]]:
    """
    Args:
        n: Number of scales
        backend: A Qiskit backend
        sample_number: Number of sites to sample the energy from
        shots: Number of shots to measure for each circuit
        quantum_circuits: A dictionary of Qiskit circuits implementing the DMERA circuit with reset for all of the different observables, indexed by the support of the observables
        pcc_circuits: A dictionary of DMERA past causal cone circuits for all of the different observables, indexed by the support of the observables
        theta_dict: Dictonary of Qiskit parameters for each gate in each layer of the circuit, indexed by the layer and gate
        theta_values: Dictionary of Qiskit parameter values for each parameter, indexed by the parameter
    Returns:
        sites_energy: Mean energy for each site
        sites_energy_sem: Energy standard error of the mean for each site
    """
    # Generate random sites and determine the supports for each site
    qubits = 3 * 2**n
    sites = sorted(random.sample(range(qubits), sample_number))
    sites_supports: list[list[Union[tuple[int, int], tuple[int, int, int]]]] = [
        [
            ((site - 1) % qubits, (site + 0) % qubits, (site + 1) % qubits),
            ((site - 1) % qubits, (site + 0) % qubits),
            ((site + 0) % qubits, (site + 1) % qubits),
        ]
        for site in sites
    ]
    flattened_supports: list[Union[tuple[int, int], tuple[int, int, int]]] = sorted(
        list(set([support for supports in sites_supports for support in supports]))
    )
    # Bind the parameters for the circuits
    bound_circuits: list[QuantumCircuit] = []
    for site_support in flattened_supports:
        theta_values_pcc: dict[Parameter, float] = {
            theta_dict[(layer_index, gate)]: theta_values[
                theta_dict[(layer_index, gate)]
            ]
            for (layer_index, layer) in enumerate(pcc_circuits[site_support])
            for gate in layer
        }
        bound_circuits.append(
            quantum_circuits[site_support].decompose().bind_parameters(theta_values_pcc)
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
    print(
        f"Running the circuits for {sample_number} sites took {end_time - start_time} s.\n"
    )
    # Calculate the value of the operators from the circuit data
    support_operators = [
        sum(
            (1 - 2 * (sum(int(bit) for bit in key) % 2)) * count[key]
            for key in count.keys()
        )
        / shots
        for count in counts
    ]
    # Determine the energy at each of the sites
    sites_energy: list[float] = []
    sites_energy_sem: list[float] = []
    for site_index in range(sample_number):
        site_operators = [
            support_operators[flattened_supports.index(site_support)]
            for site_support in sites_supports[site_index]
        ]
        energy = site_operators[0] - 0.5 * (site_operators[1] + site_operators[2])
        energy_sem = math.sqrt(
            (
                (1 + site_operators[0]) * (1 - site_operators[0])
                + 0.25
                * (
                    (1 + site_operators[1]) * (1 - site_operators[1])
                    + (1 + site_operators[2]) * (1 - site_operators[2])
                )
            )
            / shots
        )
        sites_energy.append(energy)
        sites_energy_sem.append(energy_sem)
    return (sites_energy, sites_energy_sem)


# %%


def estimate_energy(
    n: int,
    d: int,
    layer_theta: list[float],
    sample_number: int,
    shots: int,
    reset_time: Optional[int] = None,
    reset_count: int = 1,
    backend_name: str = "aer",
) -> tuple[float, float]:
    """
    Args:
        n: Number of scales
        d: Depth at each scale
        layer_theta: Theta values for the layers in each scale
        sample_number: Number of sites to sample the energy from
        shots: Number of shots to measure for each circuit
        reset_time: Time taken to perform a reset operation as a multiple of the time taken for a circuit layer
        reset_count: Number of reset operations to perform a reset
        backend_name: A Qiskit backend name
    Returns:
        energy_mean: Energy per site mean
        energy_sem: Energy per site standard error of the mean
    """
    # Get some parameters
    backend = get_backend(backend_name)
    # Generate the circuits
    (
        quantum_circuits,
        pcc_circuits,
        theta_dict,
        theta_values,
    ) = generate_transverse_ising_circuits(n, d, backend, reset_time, reset_count)
    # Set the theta values
    for (layer_index, gate) in theta_dict.keys():
        theta_values[theta_dict[(layer_index, gate)]] = layer_theta[layer_index % d]
    # Estimate the energy of the sites
    (sites_energy, sites_energy_sem) = estimate_site_energy(
        n,
        backend,
        sample_number,
        shots,
        quantum_circuits,
        pcc_circuits,
        theta_dict,
        theta_values,
    )
    # Calculate the mean energy and SEM
    energy_mean: float = np.mean(sites_energy).item()
    energy_sem_shots = np.sqrt(
        sum(np.array(sites_energy_sem) ** 2) / (len(sites_energy) ** 2)
    )
    energy_sem_sample_number = np.sqrt(np.var(sites_energy) / len(sites_energy))
    energy_sem: float = np.sqrt(
        energy_sem_shots**2 + energy_sem_sample_number**2
    ).item()
    print(
        f"The average energy per site is {energy_mean:.4f} with a standard error of the mean {energy_sem:.4f}; the shots component to the SEM is {energy_sem_shots:.4f} and the sample variance component is {energy_sem_sample_number:.4f}.\n"
    )
    return (energy_mean, energy_sem)


# %%

if MAIN:
    # Initialise parameters
    n = 3
    d_list = [2, 4, 5]
    reset_time = None
    reset_count = 1
    sample_number = 12
    shots = 10**5
    backend_name = "aer"
    # Energies supplied in Entanglement renormalization and wavelets by Evenbly and White (2016)
    energy_list = [-1.24212, -1.26774, -1.27297]
    # Estimate the energy for each depth
    energy_means: list[float] = []
    energy_sems: list[float] = []
    for d in d_list:
        layer_theta = get_theta_evenbly(d)
        (energy_mean, energy_sem) = estimate_energy(
            n,
            d,
            layer_theta,
            sample_number,
            shots,
            reset_time,
            reset_count,
            backend_name,
        )
        energy_means.append(energy_mean)
        energy_sems.append(energy_sem)
    # Store if the energy mean is within k SEMs of the true value
    k = 2
    d_passes: list[bool] = [
        True
        if (
            energy_means[index] - k * energy_sems[index] < energy_list[index]
            and energy_means[index] + k * energy_sems[index] > energy_list[index]
        )
        else False
        for index in range(len(d_list))
    ]
    print(
        f"The estimated energies are all within {k} standard errors of the mean of the true energies: {any(d_passes)}."
    )


# %%
