# %%

from typing import Any
from protocol import theta_evenbly, estimate_energy
from qiskit import Aer, IBMQ  # type: ignore
from qiskit.providers.fake_provider import FakeNairobi  # type: ignore


MAIN = __name__ == "__main__"


# %%

if MAIN:
    """
    Simulation of the energy estimation and DMERA ground state preparation on FakeNairobi.
    The results aren't too much worse than the Aer simulator, but noise is still significant.
    The Aer simulator takes ~6 times longer to run the circuits than FakeNairobi, which is very strange.
    Reset also takes ~6 times longer on Nairobi than Oslo, which is also very strange and makes circuits with reset very awkward
    """
    # Initialise parameters
    n = 3
    d = 2
    sample_number = 6
    shots = 10**4
    reset_time = 1
    reset_count = 1
    device_reset_configs = 4
    device_transpile_configs = 5
    provider = IBMQ.load_account()
    device_backend = provider.get_backend("ibm_oslo")
    device_transpile_kwargs: dict[str, Any] = {
        "backend": device_backend,
        "optimization_level": 3,
        # "basis_gates": ["id", "rz", "sx", "x", "rzx", "cx", "reset"],
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
    theta_values = theta_evenbly(d)
    # Estimate the energy on the device
    (device_energy_mean, device_energy_sem) = estimate_energy(
        n,
        d,
        theta_values,
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
        theta_values,
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
        f"\nWhere the true energy is {true_energy}, running the circuits on '{device_backend.name()}' gave an energy estimate of {device_energy_mean:.5f} and a z-score of {device_z_score:.3f}, given the SEM of {device_energy_sem:.5f}.\nThe same parameters for '{simulator_backend.name()}' gave an energy estimate of {simulator_energy_mean:.5f} and a z-score of {simulator_z_score:.3f}, given the SEM of {simulator_energy_sem:.5f}."
    )

# %%
