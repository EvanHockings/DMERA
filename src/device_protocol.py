# %%

from typing import Any
from protocol import theta_evenbly, estimate_energy
from qiskit import Aer  # type: ignore
from qiskit.providers.fake_provider import FakeNairobi  # type: ignore

MAIN = __name__ == "__main__"


# %%

if MAIN:
    """
    Simulation of the energy estimation and DMERA ground state preparation on FakeNairobi.
    The results aren't too much worse than the Aer simulator, but noise is still significant.
    """
    # Initialise parameters
    n = 3
    d = 2
    sample_number = 6
    shots = 10**2
    reset_time = 1
    reset_count = 1
    device_reset_configs = 4
    device_transpile_configs = 5
    device_backend = FakeNairobi()
    device_transpile_kwargs: dict[str, Any] = {
        "backend": device_backend,
        "optimization_level": 3,
        "scheduling_method": "alap",
        "layout_method": "sabre",
        "routing_method": "sabre",
    }
    simulator_reset_configs = None
    simulator_transpile_configs = None
    simulator_backend = Aer.get_backend("aer_simulator")
    simulator_transpile_kwargs: dict[str, Any] = {
        "backend": simulator_backend,
        "optimization_level": 3,
    }
    print_diagnostics = True
    # Energy supplied in Entanglement renormalization and wavelets by Evenbly and White (2016)
    true_energy = -1.24212
    layer_theta = theta_evenbly(d)
    # Estimate the energy on the device
    (device_energy_mean, device_energy_sem) = estimate_energy(
        n,
        d,
        layer_theta,
        sample_number,
        shots,
        reset_time,
        reset_count,
        device_reset_configs,
        device_transpile_configs,
        device_transpile_kwargs,
        print_diagnostics,
    )
    device_z_score = (device_energy_mean - true_energy) / device_energy_sem
    # Estimate the energy on the simulator
    (simulator_energy_mean, simulator_energy_sem) = estimate_energy(
        n,
        d,
        layer_theta,
        sample_number,
        shots,
        reset_time,
        reset_count,
        simulator_reset_configs,
        simulator_transpile_configs,
        simulator_transpile_kwargs,
        print_diagnostics,
    )
    simulator_z_score = (simulator_energy_mean - true_energy) / simulator_energy_sem
    # Print the results
    print(
        f"\nWhere the true energy is {true_energy}, running the circuits on '{device_backend.name()}' gave an energy estimate of {device_energy_mean:.5f} and a z-score {device_z_score:.3f}, given the SEM {device_energy_sem:.5f}.\nThe same parameters for '{simulator_backend.name()}' gave an energy estimate of {simulator_energy_mean:.5f} and a z-score {simulator_z_score:.3f}, given the SEM {simulator_energy_sem:.5f}."
    )

# %%
