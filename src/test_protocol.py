# %%

from typing import Any
from protocol import theta_evenbly, estimate_energy
from qiskit import Aer  # type: ignore

MAIN = __name__ == "__main__"


# %%

if MAIN:
    """
    End-to-end test of the energy estimation and DMERA ground state preparation.
    Note that Qiskit is very slow to simulate circuits with reset operations, and consequently the circuits with reset have a factor of 100 fewer shots taken.
    In fact, this raises questions about how Qiskit is generating shots.
    Surely it shouldn't need to simulate the entire circuit for each shot!
    """
    # Initialise parameters
    n = 3
    d_list = [2, 4, 5]
    sample_number = 6
    reset_shots = 10**3
    reset_time = 1
    resetless_shots = 10**4
    resetless_time = None
    reset_count = 1
    reset_configs = None
    transpile_configs = None
    backend = Aer.get_backend("aer_simulator")
    transpile_kwargs: dict[str, Any] = {
        "backend": backend,
        "optimization_level": 3,
    }
    print_diagnostics = True
    # Energies supplied in Entanglement renormalization and wavelets by Evenbly and White (2016)
    true_energies = [-1.24212, -1.26774, -1.27297]
    # Estimate the energy for each depth with reset
    reset_energy_means: list[float] = []
    reset_energy_sems: list[float] = []
    for d in d_list:
        layer_theta = theta_evenbly(d)
        (energy_mean, energy_sem) = estimate_energy(
            n,
            d,
            layer_theta,
            sample_number,
            reset_shots,
            reset_time,
            reset_count,
            reset_configs,
            transpile_configs,
            transpile_kwargs,
            print_diagnostics,
        )
        print(energy_mean)
        reset_energy_means.append(energy_mean)
        reset_energy_sems.append(energy_sem)
    # Calculate the z-scores for the energies with reset
    reset_z_scores = [
        (reset_energy_means[index] - true_energies[index]) / reset_energy_sems[index]
        for index in range(len(d_list))
    ]
    # Assert that all the z-scores are within 3 standard deviations of the mean
    assert all([abs(z_score) < 3 for z_score in reset_z_scores])
    # Estimate the energy for each depth without reset
    resetless_energy_means: list[float] = []
    resetless_energy_sems: list[float] = []
    for d in d_list:
        layer_theta = theta_evenbly(d)
        (energy_mean, energy_sem) = estimate_energy(
            n,
            d,
            layer_theta,
            sample_number,
            resetless_shots,
            resetless_time,
            reset_count,
            reset_configs,
            transpile_configs,
            transpile_kwargs,
            print_diagnostics,
        )
        resetless_energy_means.append(energy_mean)
        resetless_energy_sems.append(energy_sem)
    # Calculate the z-scores for the energies without reset
    resetless_z_scores = [
        (resetless_energy_means[index] - true_energies[index])
        / resetless_energy_sems[index]
        for index in range(len(d_list))
    ]
    # Assert that all the z-scores are within 3 standard deviations of the mean
    assert all([abs(z_score) < 3 for z_score in resetless_z_scores])
    # Print that the tests passed
    print("Tests passed.")

# %%
