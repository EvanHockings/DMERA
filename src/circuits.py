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
