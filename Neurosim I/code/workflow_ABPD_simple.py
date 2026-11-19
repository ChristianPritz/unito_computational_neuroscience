"""Simplified lobster STG AB/PD model with transient LP inhibition.

This student version preserves the core AB/PD pacemaker principle but uses a
reduced set of currents and simplified kinetics. Fast inward current generates
spikes, slow calcium sustains each burst, and potassium currents terminate it.
It is a teaching model, not an exact reproduction of the paper.
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
    log,
    mM,
    mV,
    ms,
    msiemens,
    nA,
    nF,
    nS,
    prefs,
    run,
    second,
    start_scope,
    uM,
)

prefs.codegen.target = "numpy"

TIME_STEP = 0.05 * ms
INITIAL_VOLTAGE = -50 * mV
INITIAL_CALCIUM = 0.05 * uM
FIRING_THRESHOLD = -20 * mV
FIGURE_FILE = "ABPD_simple_transient_LP_inhibition.png"


# =============================================================================
# PART 2: FUNCTIONS
# =============================================================================

def build_cell(cell_properties, equations):
    """Create the simplified AB/PD cell and initialise its dynamic gates."""

    parameters = {**cell_properties, "firing_threshold": FIRING_THRESHOLD}
    cell = NeuronGroup(
        1,
        equations,
        method="exponential_euler",
        threshold="v > firing_threshold",
        refractory="v > firing_threshold",
        namespace=parameters,
        name="simple_ABPD_cell",
    )

    cell.v = INITIAL_VOLTAGE
    cell.Ca = INITIAL_CALCIUM
    cell.v_LP = cell_properties["V_LP_rest"]
    cell.s = "s_inf"

    # Six channel gates are dynamic in this reduced model.
    for gate in ("h_Na", "m_CaS", "h_CaS", "m_KCa", "m_Kd", "m_H"):
        setattr(cell, gate, f"{gate}_inf")
    return cell


def simulate_cell(duration, cell_properties, equations, synaptic_event,
                  settling_time=3 * second):
    """Settle the cell, record it, and apply one timed LP inhibitory event."""

    onset = synaptic_event["onset"]
    event_duration = synaptic_event["duration"]
    event_end = onset + event_duration
    if onset < 0 * second or event_duration < 0 * second or event_end > duration:
        raise ValueError("The synaptic event must fit within the simulation.")

    start_scope()
    defaultclock.dt = TIME_STEP
    cell = build_cell(cell_properties, equations)

    # Let the slow gates approach the cell's natural rhythm before recording.
    run(settling_time)
    voltage_monitor = StateMonitor(cell, ["v", "s", "Ca"], record=True)
    spike_monitor = SpikeMonitor(cell)

    run(onset)
    cell.v_LP = cell_properties["V_LP_active"]
    run(event_duration)
    cell.v_LP = cell_properties["V_LP_rest"]
    run(duration - event_end)
    return voltage_monitor, spike_monitor


def plot_results(voltage_monitor, spike_monitor, duration, synaptic_event):
    """Plot membrane voltage, synaptic activation, and calcium."""

    time_s = (voltage_monitor.t - voltage_monitor.t[0]) / second
    voltage_mv = voltage_monitor.v[0] / mV
    onset_s = synaptic_event["onset"] / second
    event_end_s = (synaptic_event["onset"] + synaptic_event["duration"]) / second

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]}, constrained_layout=True,
    )
    fig.suptitle("Simplified AB/PD pacemaker with transient LP inhibition",
                 fontsize=14)

    axes[0].plot(time_s, voltage_mv, color="#2457A6", linewidth=0.9)
    axes[0].axhline(0, color="black", linewidth=1.0, label="0 mV")
    axes[0].axhline(
        FIRING_THRESHOLD / mV, color="#D62728", linestyle="--", linewidth=1.2,
        label="Spike-detection threshold (−20 mV)",
    )
    axes[0].set(ylabel="Voltage (mV)")

    axes[1].plot(time_s, voltage_monitor.s[0], color="#7A5195", linewidth=1.3)
    axes[1].set(ylabel="LP synapse\nactivation, $s$", ylim=(-0.03, 1.03))

    axes[2].plot(time_s, voltage_monitor.Ca[0] / uM,
                 color="#2A9D8F", linewidth=1.2)
    axes[2].set(xlabel="Recorded time (s)", ylabel="[Ca²⁺]\n(µM)")

    for axis in axes:
        axis.axvspan(onset_s, event_end_s, color="#F4A261", alpha=0.25,
                     label="LP inhibitory event")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].legend(loc="upper right", frameon=False, ncol=3, fontsize=8)
    fig.savefig(FIGURE_FILE, dpi=180)
    plt.show()


# =============================================================================
# PART 3: STUDENT SETTINGS AND EXECUTION
# =============================================================================

duration = 3 * second

synaptic_event = {
    "onset": 0.5 * second,
    "duration": 0.5 * second,
}

# Reduced conductance set inspired by paper AB/PD model #3. CaT and A currents
# are omitted: their roles are absorbed into the slow Ca and K currents.
cell_properties = {
    "area": 0.628e-3 * cm**2,
    "C": 0.628 * nF,
    "g_Na": 250 * msiemens / cm**2,
    # A stronger slow Ca current sustains a longer train within each burst.
    "g_CaS": 8 * msiemens / cm**2,
    # K(Ca) terminates each burst; Kd repolarises the individual spikes.
    "g_KCa": 20 * msiemens / cm**2,
    "g_Kd": 35 * msiemens / cm**2,
    "g_H": 0.02 * msiemens / cm**2,
    "g_leak": 0 * msiemens / cm**2,
    "g_syn": 30 * nS,
    "E_syn": -70 * mV,
    "V_LP_rest": -60 * mV,
    "V_LP_active": 20 * mV,
}

# Each current still has the familiar form g * gates * (E - v).
# Compared with the detailed model, activation gates marked "inf" respond
# instantly and all dynamic gates use a constant, easy-to-read time constant.
equations = """
dv/dt = (I_Na+I_CaS+I_KCa+I_Kd+I_H+I_leak+I_syn)/C : volt

I_Na   = g_Na*m_Na_inf**3*h_Na*(50*mV-v)*area : amp
I_CaS  = g_CaS*m_CaS**3*h_CaS*(E_Ca-v)*area : amp
I_KCa  = g_KCa*m_KCa**4*(-80*mV-v)*area : amp
I_Kd   = g_Kd*m_Kd**4*(-80*mV-v)*area : amp
I_H    = g_H*m_H*(-20*mV-v)*area : amp
I_leak = g_leak*(-50*mV-v)*area : amp
I_syn  = g_syn*s*(E_syn-v) : amp

# Simple calcium handling and a calcium-dependent reversal potential
dCa/dt = (14.96*uM/nA*I_CaS-Ca+0.05*uM)/(250*ms) : mole/metre**3
E_Ca = 12.2*mV*log(3*mM/Ca) : volt

# Instantaneous activation curves for the fast currents
m_Na_inf  = 1/(1+exp(-(v+25.5*mV)/(5.29*mV))) : 1

# Slow gates: one sigmoid target and one fixed time constant per gate
dh_Na/dt = (h_Na_inf-h_Na)/(3*ms) : 1
h_Na_inf = 1/(1+exp((v+48.9*mV)/(5.18*mV))) : 1

dm_CaS/dt = (m_CaS_inf-m_CaS)/(20*ms) : 1
m_CaS_inf = 1/(1+exp(-(v+33*mV)/(8.1*mV))) : 1
dh_CaS/dt = (h_CaS_inf-h_CaS)/(300*ms) : 1
h_CaS_inf = 1/(1+exp((v+60*mV)/(6.2*mV))) : 1

# Slow calcium-dependent potassium feedback terminates each burst.
dm_KCa/dt = (m_KCa_inf-m_KCa)/(120*ms) : 1
m_KCa_inf = (Ca/(Ca+3*uM))/(1+exp(-(v+28.3*mV)/(12.6*mV))) : 1

dm_Kd/dt = (m_Kd_inf-m_Kd)/(15*ms) : 1
m_Kd_inf = 1/(1+exp(-(v+12.3*mV)/(11.8*mV))) : 1

dm_H/dt = (m_H_inf-m_H)/(500*ms) : 1
m_H_inf = 1/(1+exp((v+75*mV)/(5.5*mV))) : 1

# Simplified first-order LP synapse: fast onset and slower offset
ds/dt = (s_inf-s)/tau_syn : 1
s_inf = 1/(1+exp((-35*mV-v_LP)/(5*mV))) : 1
tau_syn = (5*ms)*s_inf + (40*ms)*(1-s_inf) : second
dv_LP/dt = 0*volt/second : volt
"""

voltage, spikes = simulate_cell(
    duration, cell_properties, equations, synaptic_event
)
print(f"Detected {spikes.num_spikes} spikes in {duration / second:.1f} recorded s")
print(f"Mean spike rate: {spikes.num_spikes / duration / Hz:.1f} Hz")
print(f"Saving figure as: {FIGURE_FILE}")
plot_results(voltage, spikes, duration, synaptic_event)
