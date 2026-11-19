"""A first Brian2 simulation: one spontaneously firing Hodgkin--Huxley cell.

The cell has two voltage-dependent conductances (sodium and potassium) and a
passive leak conductance.  The leak reversal potential is deliberately set to
-30 mV, so the leak depolarises a cell starting near -65 mV and drives regular
spiking without a separate injected current.
"""

# =============================================================================
# PART 1: IMPORTS AND IMPORTANT SETTINGS
# =============================================================================

import matplotlib.pyplot as plt
from brian2 import (
    Hz,
    NeuronGroup,
    SpikeMonitor,
    StateMonitor,
    cm,
    defaultclock,
    mV,
    ms,
    msiemens,
    prefs,
    run,
    second,
    start_scope,
    ufarad,
)

# Using NumPy keeps this introductory example portable (no C++ compiler needed).
prefs.codegen.target = "numpy"

TIME_STEP = 0.05 * ms
INITIAL_VOLTAGE = -65 * mV
FIRING_THRESHOLD = -55 * mV
FIGURE_FILE = "single_cell_spontaneous_spiking.png"


# =============================================================================
# PART 2: FUNCTIONS
# =============================================================================

def build_cell(cell_properties, equations):
    """Create one Hodgkin--Huxley neuron and initialise its state variables."""

    model_parameters = {
        **cell_properties,
        "firing_threshold": FIRING_THRESHOLD,
    }

    cell = NeuronGroup(
        1,
        equations,
        method="exponential_euler",
        threshold="v > firing_threshold",
        refractory="v > firing_threshold",  # count each spike only once
        namespace=model_parameters,
        name="single_hh_cell",
    )

    cell.v = INITIAL_VOLTAGE
    # Start each gate at its steady-state value for the initial voltage.
    cell.m = "alpha_m/(alpha_m + beta_m)"
    cell.h = "alpha_h/(alpha_h + beta_h)"
    cell.n = "alpha_n/(alpha_n + beta_n)"
    return cell


def simulate_cell(duration, cell_properties, equations):
    """Run a cell described by the three student-controlled inputs."""

    start_scope()  # makes repeated runs in an interactive lesson predictable
    defaultclock.dt = TIME_STEP
    cell = build_cell(cell_properties, equations)
    voltage_monitor = StateMonitor(cell, "v", record=True)
    spike_monitor = SpikeMonitor(cell)
    run(duration)
    return voltage_monitor, spike_monitor


def plot_results(voltage_monitor, spike_monitor, duration):
    """Plot the complete run and a short zoom that makes spikes easy to see."""

    time_ms = voltage_monitor.t / ms
    voltage_mv = voltage_monitor.v[0] / mV
    firing_rate = spike_monitor.num_spikes / duration

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    fig.suptitle(
        f"Spontaneous activity of one conductance-based neuron "
        f"({firing_rate / Hz:.1f} Hz)",
        fontsize=14,
    )

    axes[0].plot(time_ms, voltage_mv, color="#2457A6", linewidth=0.8)
    axes[0].set(title="Complete simulation", ylabel="Membrane voltage (mV)")

    zoom_length = 250 * ms
    zoom_start = max(0.0, (duration - zoom_length) / ms)
    axes[1].plot(time_ms, voltage_mv, color="#D1495B", linewidth=1.2)
    axes[1].set(
        title="Final 250 ms (zoom)",
        xlabel="Time (ms)",
        ylabel="Membrane voltage (mV)",
        xlim=(zoom_start, duration / ms),
    )

    for axis in axes:
        axis.axhline(0, color="black", linewidth=1.0, label="0 mV")
        axis.axhline(
            FIRING_THRESHOLD / mV,
            color="#D62728",
            linestyle="--",
            linewidth=1.2,
            label="Firing threshold (−55 mV)",
        )
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper right", frameon=False)

    fig.savefig(FIGURE_FILE, dpi=180)
    plt.show()


# =============================================================================
# PART 3: EXECUTE THE WORKFLOW
# =============================================================================

# These are the three inputs students can edit and explore.

# 1. How long should the simulation run?
duration = 1 * second

# 2. What electrical properties should the cell have?
# Try changing g_Na, g_K, or g_L and observe the spike shape and firing rate.
# E_L is more positive than the initial voltage, so the leak is depolarising.
# cell_properties = {
#     "C_m": 1 * ufarad / cm**2,
#     "g_Na": 120 * msiemens / cm**2,
#     "g_K": 36 * msiemens / cm**2,
#     "g_L": 0.3 * msiemens / cm**2,
#     "E_Na": 50 * mV,
#     "E_K": -77 * mV,
#     "E_L": -30 * mV,
# }




cell_properties = {
    "C_m": 1 * ufarad / cm**2,
    "g_Na": 120 * msiemens / cm**2,
    "g_K": 7 * msiemens / cm**2,
    "g_L": 0.5 * msiemens / cm**2,
    "E_Na": 50 * mV,
    "E_K": -77 * mV,
    "E_L": -30 * mV,
}

# 3. Which equations describe the cell?
# Currents use g*(E-v), so a positive current depolarises the membrane.
# exprel avoids a numerical 0/0 in the alpha_m and alpha_n rate equations.
equations = """
dv/dt = (g_Na*m**3*h*(E_Na-v)
         + g_K*n**4*(E_K-v)
         + g_L*(E_L-v))/C_m : volt

dm/dt = alpha_m*(1-m) - beta_m*m : 1
dh/dt = alpha_h*(1-h) - beta_h*h : 1
dn/dt = alpha_n*(1-n) - beta_n*n : 1

alpha_m = 1/ms/exprel(-(v+40*mV)/(10*mV)) : Hz
beta_m  = 4/ms*exp(-(v+65*mV)/(18*mV)) : Hz
alpha_h = 0.07/ms*exp(-(v+65*mV)/(20*mV)) : Hz
beta_h  = 1/ms/(1+exp(-(v+35*mV)/(10*mV))) : Hz
alpha_n = 0.1/ms/exprel(-(v+55*mV)/(10*mV)) : Hz
beta_n  = 0.125/ms*exp(-(v+65*mV)/(80*mV)) : Hz
"""





voltage, spikes = simulate_cell(duration, cell_properties, equations)
print(f"Detected {spikes.num_spikes} spikes in {duration / second:.1f} s")
print(f"Mean firing rate: {spikes.num_spikes / duration / Hz:.1f} Hz")
print(f"Saving figure as: {FIGURE_FILE}")
plot_results(voltage, spikes, duration)
