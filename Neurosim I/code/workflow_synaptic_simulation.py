"""A Brian2 simulation of one spontaneously firing cell receiving inhibition.

This lesson extends ``workflow_basic_simulation.py`` with a transient
conductance-based inhibitory synaptic current. During the event, the inhibitory
conductance pulls the voltage towards E_inh and briefly suppresses firing.
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

prefs.codegen.target = "numpy"  # no C++ compiler is required

TIME_STEP = 0.05 * ms
INITIAL_VOLTAGE = -65 * mV
FIRING_THRESHOLD = -55 * mV
FIGURE_FILE = "single_cell_transient_inhibition.png"


# =============================================================================
# PART 2: FUNCTIONS
# =============================================================================

def build_cell(cell_properties, equations):
    """Create and initialise one Hodgkin--Huxley neuron."""

    model_parameters = {
        **cell_properties,
        "firing_threshold": FIRING_THRESHOLD,
    }

    cell = NeuronGroup(
        1,
        equations,
        method="exponential_euler",
        threshold="v > firing_threshold",
        refractory="v > firing_threshold",
        namespace=model_parameters,
        name="inhibited_hh_cell",
    )
    cell.v = INITIAL_VOLTAGE
    cell.g_inh = 0 * msiemens / cm**2
    cell.m = "alpha_m/(alpha_m + beta_m)"
    cell.h = "alpha_h/(alpha_h + beta_h)"
    cell.n = "alpha_n/(alpha_n + beta_n)"
    return cell


def simulate_cell(duration, cell_properties, equations, synaptic_event):
    """Simulate the cell, switching inhibition on and off at specified times."""

    onset = synaptic_event["onset"]
    event_duration = synaptic_event["duration"]
    event_end = onset + event_duration

    if onset < 0 * second or event_duration < 0 * second or event_end > duration:
        raise ValueError("The synaptic event must fit within the simulation.")

    start_scope()
    defaultclock.dt = TIME_STEP
    cell = build_cell(cell_properties, equations)
    voltage_monitor = StateMonitor(cell, ["v", "g_inh"], record=True)
    spike_monitor = SpikeMonitor(cell)

    # Running the model in three segments makes the rectangular event explicit:
    # baseline -> inhibitory conductance on -> inhibitory conductance off.
    run(onset)
    cell.g_inh = cell_properties["g_inh_max"]
    run(event_duration)
    cell.g_inh = 0 * msiemens / cm**2
    run(duration - event_end)

    return voltage_monitor, spike_monitor


def plot_results(voltage_monitor, spike_monitor, duration, synaptic_event):
    """Plot voltage and inhibitory conductance, highlighting the event window."""

    time_ms = voltage_monitor.t / ms
    voltage_mv = voltage_monitor.v[0] / mV
    conductance = voltage_monitor.g_inh[0] / (msiemens / cm**2)
    onset_ms = synaptic_event["onset"] / ms
    event_end_ms = (synaptic_event["onset"] + synaptic_event["duration"]) / ms
    firing_rate = spike_monitor.num_spikes / duration

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    fig.suptitle(
        f"Transient inhibition of a spontaneously firing neuron "
        f"({firing_rate / Hz:.1f} Hz overall)",
        fontsize=14,
    )

    axes[0].plot(time_ms, voltage_mv, color="#2457A6", linewidth=0.9)
    axes[0].set(ylabel="Membrane voltage (mV)")
    axes[0].axhline(0, color="black", linewidth=1.0, label="0 mV")
    axes[0].axhline(
        FIRING_THRESHOLD / mV,
        color="#D62728",
        linestyle="--",
        linewidth=1.2,
        label="Firing threshold (−55 mV)",
    )

    axes[1].plot(time_ms, conductance, color="#7A5195", linewidth=1.5)
    axes[1].set(
        xlabel="Time (ms)",
        ylabel="$g_{inh}$\n(mS/cm²)",
        xlim=(0, duration / ms),
        ylim=(-0.05, max(0.5, 1.15 * conductance.max())),
    )

    for axis in axes:
        axis.axvspan(
            onset_ms,
            event_end_ms,
            color="#F4A261",
            alpha=0.25,
            label="Inhibitory event",
        )
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].legend(loc="upper right", frameon=False)
    fig.savefig(FIGURE_FILE, dpi=180)
    plt.show()


# =============================================================================
# PART 3: EXECUTE THE WORKFLOW
# =============================================================================

duration = 1 * second

# Students can change the onset and duration of the inhibitory event here.
synaptic_event = {
    "onset": 0.5 * second,
    "duration": 0.1 * second,
}

cell_properties = {
    "C_m": 1 * ufarad / cm**2,
    "g_Na": 120 * msiemens / cm**2,
    "g_K": 36 * msiemens / cm**2,
    "g_L": 0.3 * msiemens / cm**2,
    "E_Na": 50 * mV,
    "E_K": -77 * mV,
    "E_L": -30 * mV,
    # Synaptic strength and reversal potential are also easy to explore.
    "g_inh_max": 10 * msiemens / cm**2,
    "E_inh": -80 * mV,
}

# The added inhibitory current is g_inh*(E_inh-v). Because E_inh is below the
# usual membrane voltage, this term hyperpolarises the cell during the event.
equations = """
dv/dt = (g_Na*m**3*h*(E_Na-v)
         + g_K*n**4*(E_K-v)
         + g_L*(E_L-v)
         + g_inh*(E_inh-v))/C_m : volt

dg_inh/dt = 0*siemens/metre**2/second : siemens/metre**2

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

voltage, spikes = simulate_cell(
    duration, cell_properties, equations, synaptic_event
)
print(f"Detected {spikes.num_spikes} spikes in {duration / second:.1f} s")
print(f"Mean firing rate: {spikes.num_spikes / duration / Hz:.1f} Hz")
print(f"Saving figure as: {FIGURE_FILE}")
plot_results(voltage, spikes, duration, synaptic_event)
