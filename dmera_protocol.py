# %%

import math
import random
import mthree
import numpy as np
import mapomatic as mm
from copy import deepcopy
from time import time
from typing import Optional, Union, Any
from qiskit import IBMQ, ClassicalRegister, QuantumRegister, QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit.compiler import transpile
from qiskit.transpiler import InstructionDurations, PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.visualization.timeline import draw, IQXSimple, IQXStandard, IQXDebugging
from qiskit.providers.fake_provider import FakeKolkata

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


def track_reset(
    qubits: int,
    L: int,
    reset_length: int,
    start_use: list[int],
    stop_use: list[int],
    layer_index: int,
    qubit_map: list[int],
    resets: list[list[int]],
    reset_in_progress: list[int],
):
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
            and qubit_index not in reset_in_progress
        ]
        for future_layer_index in range(layer_index + 1 + reset_length, L)
    ]
    # Randomly shuffle the layers
    for layer in reset_allocation_layers:
        random.shuffle(layer)
    reset_allocation = [
        qubit_index for layer in reset_allocation_layers for qubit_index in layer
    ]
    # Reset qubits and relabel them appropriately with the qubit map
    reset: list[int] = []
    for j in range(min(len(reset_qubits), len(reset_allocation))):
        reset_qubit = reset_qubits[j]
        paired_qubit = reset_allocation[j]
        reset.append(qubit_map[reset_qubit])
        reset_in_progress.append(paired_qubit)
        qubit_map[paired_qubit] = qubit_map[reset_qubit]
        qubit_map[reset_qubit] = qubits
    resets.append(reset)
    pass


# %%


def dmera_reset(
    n: int,
    d: int,
    circuit: list[list[tuple[int, int]]],
    sites_list: list[list[int]],
    support: list[int],
    reset_time: Optional[int],
    reset_count: int,
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
    # Track resets for initialisation
    track_reset(
        qubits,
        L,
        reset_length,
        start_use,
        stop_use,
        -1,
        qubit_map,
        resets,
        reset_in_progress,
    )
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
        # Track resets in the layer
        track_reset(
            qubits,
            L,
            reset_length,
            start_use,
            stop_use,
            layer_index,
            qubit_map,
            resets,
            reset_in_progress,
        )
    #################################################
    # Generate the inverse map
    #################################################
    qubit_reset_map = list(set(reset_map.values()))
    inverse_map: dict[int, int] = {}
    for (qubit_index, qubit) in enumerate(qubit_reset_map):
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


def q() -> QuantumCircuit:
    """
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
    q_gate = transpile(
        q_gate.decompose(),
        backend=Aer.get_backend("aer_simulator_statevector"),
        optimization_level=3,
    )
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
    reset_count: int,
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
    reset_time: Optional[int],
    reset_count: int,
    print_diagnostics: bool = False,
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
        print_diagnostics: Print diagnostics for the circuit generation
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
    q_gate = q()
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
        support_mapped = [inverse_map[reset_map[(L - 1, qubit)]] for qubit in support]
        if len(support) == 2:
            quantum_circuit.h(support_mapped[0])
            quantum_circuit.h(support_mapped[1])
            quantum_circuit.measure(support_mapped[0], 0)
            quantum_circuit.measure(support_mapped[1], 1)
        elif len(support) == 3:
            quantum_circuit.h(support_mapped[0])
            quantum_circuit.h(support_mapped[2])
            quantum_circuit.measure(support_mapped[0], 0)
            quantum_circuit.measure(support_mapped[1], 1)
            quantum_circuit.measure(support_mapped[2], 2)
        else:
            raise ValueError
        # Add the circuit to the dictionary
        quantum_circuits[tuple(support)] = quantum_circuit
        pcc_circuits[tuple(support)] = pcc_circuit
    # Print a diagnostic
    if print_diagnostics:
        print(
            f"The maximum number of qubits required for any of the DMERA past causal cone circuits is {max(qubit_numbers)}, where the number of scales is {n}, the depth at each scale is {d}, the reset time is {reset_time}, and the reset count is {reset_count}."
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
    transpile_kwargs: dict[str, Any],
) -> tuple[list[float], list[float]]:
    """
    Args:
        n: Number of scales
        sample_number: Number of sites to sample the energy from
        shots: Number of shots to measure for each circuit
        quantum_circuits: A dictionary of Qiskit circuits implementing the DMERA circuit with reset for all of the different observables, indexed by the support of the observables
        pcc_circuits: A dictionary of DMERA past causal cone circuits for all of the different observables, indexed by the support of the observables
        theta_dict: Dictonary of Qiskit parameters for each gate in each layer of the circuit, indexed by the layer and gate
        theta_values: Dictionary of Qiskit parameter values for each parameter, indexed by the parameter
        transpile_kwargs: Keyword arguments for circuit transpiling
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
    site_circuits = transpile(bound_circuits, **transpile_kwargs)
    # Run the circuits
    start_time = time()
    job = backend.run(site_circuits, shots=shots)
    job_results = job.result().get_counts()
    counts: list[dict[str, int]] = [dict(count) for count in job_results]
    end_time = time()
    print(
        f"Running the circuits for {sample_number} sites took {end_time - start_time} s."
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
    reset_time: Optional[int],
    reset_count: int,
    transpile_kwargs: dict[str, Any],
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
        transpile_kwargs: Keyword arguments for circuit transpiling
    Returns:
        energy_mean: Energy per site mean
        energy_sem: Energy per site standard error of the mean
    """
    # Generate the circuits
    (
        quantum_circuits,
        pcc_circuits,
        theta_dict,
        theta_values,
    ) = generate_transverse_ising_circuits(n, d, reset_time, reset_count)
    # Set the theta values
    for (layer_index, gate) in theta_dict.keys():
        theta_values[theta_dict[(layer_index, gate)]] = layer_theta[layer_index % d]
    # Estimate the energy of the sites
    (sites_energy, sites_energy_sem) = estimate_site_energy(
        n,
        sample_number,
        shots,
        quantum_circuits,
        pcc_circuits,
        theta_dict,
        theta_values,
        transpile_kwargs,
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
        f"The average energy per site is {energy_mean:.4f} with a standard error of the mean {energy_sem:.4f}; the shots component to the SEM is {energy_sem_shots:.4f} and the sample variance component is {energy_sem_sample_number:.4f}."
    )
    return (energy_mean, energy_sem)


# %%


class RemoveDelays(TransformationPass):
    """
    Return a circuit with any delays removed.
    This transformation is not semantics preserving.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the RemoveDelays pass on `dag`.
        """
        dag.remove_all_ops_named("delay")
        return dag


# %%

if MAIN:
    # Simulator parameters
    simulator_backend = Aer.get_backend("aer_simulator")
    simulator_kwargs: dict[str, Any] = {
        "backend": simulator_backend,
        "optimization_level": 3,
    }
    # Device parameters
    is_real = True
    if is_real:
        device_name = "ibm_oslo"
        provider = IBMQ.load_account()
        device_backend = provider.get_backend(device_name)
    else:
        device_backend = FakeKolkata()
    device_kwargs: dict[str, Any] = {
        "backend": device_backend,
        "scheduling_method": "alap",
        "layout_method": "sabre",
        "routing_method": "sabre",
        "optimization_level": 3,
    }

# %%


if MAIN:
    # Set the parameters
    n = 3
    d = 2
    reset_time = 1
    reset_count = 1
    shots = 10**4
    layer_theta = get_theta_evenbly(d)
    support = (3, 4, 5)
    k_reset = 10
    k_transpile = 20
    run_circuit = False
    # Generate circuits
    (
        quantum_circuits,
        pcc_circuits,
        theta_dict,
        theta_values,
    ) = generate_transverse_ising_circuits(n, d, reset_time, reset_count)
    # Set the theta values
    for (layer_index, gate) in theta_dict.keys():
        theta_values[theta_dict[(layer_index, gate)]] = layer_theta[layer_index % d]
    # Bind the parameters for the `support` pcc circuit
    theta_values_pcc: dict[Parameter, float] = {
        theta_dict[(layer_index, gate)]: theta_values[theta_dict[(layer_index, gate)]]
        for (layer_index, layer) in enumerate(pcc_circuits[support])
        for gate in layer
    }
    circuit = quantum_circuits[support].decompose().bind_parameters(theta_values_pcc)
    # Transpile the simulator circuit
    simulator_circuit = transpile(circuit, **simulator_kwargs)
    # Transpile the device circuit
    # device_circuits = transpile([circuit]*k_transpile, **device_kwargs)
    # Testing random circuit generation
    # If this is worthwhile I can rework my code to make it much more efficient
    circuit_list: list[QuantumCircuit] = []
    for idx in range(k_reset):
        (
            quantum_circuits,
            pcc_circuits,
            theta_dict,
            theta_values,
        ) = generate_transverse_ising_circuits(n, d, reset_time, reset_count)
        # Set the theta values
        for (layer_index, gate) in theta_dict.keys():
            theta_values[theta_dict[(layer_index, gate)]] = layer_theta[layer_index % d]
        # Bind the parameters for the `support` pcc circuit
        theta_values_pcc: dict[Parameter, float] = {
            theta_dict[(layer_index, gate)]: theta_values[
                theta_dict[(layer_index, gate)]
            ]
            for (layer_index, layer) in enumerate(pcc_circuits[support])
            for gate in layer
        }
        circuit_list.append(
            quantum_circuits[support].decompose().bind_parameters(theta_values_pcc)
        )
    device_circuits = transpile(circuit_list * k_transpile, **device_kwargs)
    # Choose the shortest of the k circuits
    durations: list[int] = [
        device_circuit.duration for device_circuit in device_circuits
    ]
    cx_counts: list[int] = [
        len(device_circuit.get_instructions("cx")) for device_circuit in device_circuits
    ]
    duration_order: np.ndarray = np.argsort(durations)
    min_duration_arg = np.argmin(durations)
    min_cx_arg = np.argmin(cx_counts)
    best_arg = duration_order[np.argmin(np.array(cx_counts)[duration_order][0:k_reset])]
    print(
        f"The circuit with the minimum duration {durations[min_duration_arg]} has {cx_counts[min_duration_arg]} CX gates, the circuit with the minimum CX gates {cx_counts[min_cx_arg]} has duration {durations[min_cx_arg]}, and the chosen best circuit has duration {durations[best_arg]} and {cx_counts[best_arg]} CX gates."
    )
    short_circuit = device_circuits[best_arg]
    # Remove delays
    remove_delays = PassManager(RemoveDelays())
    deflated_circuit = mm.deflate_circuit(remove_delays.run(short_circuit))
    # Use mapomatic to determine the optimal layout
    best_layout = mm.best_overall_layout(deflated_circuit, device_backend)
    device_circuit = transpile(
        deflated_circuit, initial_layout=best_layout[0], **device_kwargs
    )
    # Display the circuit
    display(
        draw(
            device_circuit,
            plotter="mpl",
            style=IQXSimple(**{"formatter.general.fig_width": 40}),
        )
    )
    print(
        f"The circuit duration is {device_circuit.duration}, which should not be much longer than the duration of the circuit before transpilation {short_circuit.duration}."
    )
    # Run the simulator circuit
    start_time = time()
    job = simulator_backend.run(simulator_circuit, shots=shots)
    simulator_counts = dict(job.result().get_counts())
    simulator_operator_value = (
        sum(
            (1 - 2 * (sum(int(bit) for bit in key) % 2)) * simulator_counts[key]
            for key in simulator_counts.keys()
        )
        / shots
    )
    end_time = time()
    print(
        f"Simulating 10^{np.log10(shots)} shots from the circuit took {end_time - start_time} s, and the operator's estimated value is {simulator_operator_value}."
    )
    if run_circuit:
        # Calibrate mthree
        mitigator = mthree.M3Mitigation(device_backend)
        measured_qubits = [
            measurement.qubits[0].index
            for measurement in device_circuit.get_instructions("measure")
        ]
        mitigator.cals_from_system(measured_qubits)
        # Run the device circuit
        start_time = time()
        job = device_backend.run(device_circuit, shots=shots)
        device_counts = job.result().get_counts()
        corrected_counts = mitigator.apply_correction(device_counts, measured_qubits)
        dict_counts = dict(corrected_counts)
        device_operator_value = (
            sum(
                (1 - 2 * (sum(int(bit) for bit in key) % 2)) * dict_counts[key]
                for key in dict_counts.keys()
            )
            / shots
        )
        end_time = time()
        print(
            f"Running 10^{np.log10(shots)} shots from the circuit took {end_time - start_time} s, and the operator's estimated value is {device_operator_value}."
        )

# %%

if MAIN:
    """
    Suggested parameter values:
        n: 3 (4 or even higher might be preferable)
        d_list: [2, 4, 5]
        reset_time: None or [The amount of time it takes to perform reset in comparison to a u(theta) or w(theta) gate]
            (to avoid waiting for reset operations to complete)
        reset_count: [Number of times to perform the reset operation]
            (more reset operations will produce a better reset)
        sample_number: 12
            (unsure if this value is good)
        shots: 10**5 or 10**2
            (the former gives accurate results, the latter is appropriate for simulations with reset)
        backend_name: "aer_simulator_statevector" or "ibm_oslo"
            (using a physical device like "ibm_oslo" may involve a very long wait, depending on queue times)
        Note that circuits with reset take much longer to simulate than would seem reasonable, drastically limiting the number of shots we can take.
        This should not be the case for actual quantum devices, however.
    """
    # Initialise parameters
    n = 3
    d_list = [2]
    reset_time = 1
    reset_count = 1
    sample_number = 12
    shots = 10**4
    backend_name = "aer"
    if backend_name[0:3] == "aer":
        backend = Aer.get_backend("aer_simulator")
        transpile_kwargs: dict[str, Any] = {
            "backend": backend,
            "optimization_level": 3,
        }
    else:
        provider = IBMQ.load_account()
        backend = provider.get_backend(backend_name)
        transpile_kwargs: dict[str, Any] = {
            "backend": backend,
            "scheduling_method": "alap",
            "optimization_level": 3,
        }
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
            transpile_kwargs,
        )
        energy_means.append(energy_mean)
        energy_sems.append(energy_sem)
    # Calculate the z scores for the energies
    energy_z_scores = [
        (energy_means[index] - energy_list[index]) / energy_sems[index]
        for index in range(len(d_list))
    ]
    print(
        f"For depths {d_list} at each scale, the estimated energies are {energy_z_scores} standard errors of the mean away from the true energies."
    )


# %%

# Use mapomatic to determine the optimal layout
# layouts = mm.matching_layouts(deflated_circuit, device_backend)
# scores = mm.evaluate_layouts(deflated_circuit, layouts, device_backend)
# trans_durations = []
# trans_scores = []
# for score in scores:
#     device_circuit = transpile(
#         deflated_circuit,
#         initial_layout=score[0],
#         **device_kwargs,
#     )
#     trans_durations.append(device_circuit.duration)
#     trans_scores.append(score[1])

# %%
