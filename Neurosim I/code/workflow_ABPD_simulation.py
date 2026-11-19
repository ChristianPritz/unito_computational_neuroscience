"""Brian2 tutorial: one lobster STG AB/PD pacemaker with LP inhibition.

The intrinsic model follows Prinz, Bucher & Marder (2004), using their AB/PD
model #3 conductance set.  It contains eight membrane currents and intracellular
calcium dynamics.  A transient LP-like glutamatergic synapse inhibits the cell.
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

TIME_STEP = 0.025 * ms  # approximate time step reported for the original model
INITIAL_VOLTAGE = -50 * mV
INITIAL_CALCIUM = 0.05 * uM
FIRING_THRESHOLD = -20 * mV
FIGURE_FILE = "ABPD_transient_LP_inhibition.png"


# =============================================================================
# PART 2: FUNCTIONS
# =============================================================================

def build_abpd_cell(cell_properties, equations):
    """Create one AB/PD model and initialise gates at steady state."""

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
        name="ABPD_cell",
    )

    cell.v = INITIAL_VOLTAGE
    cell.Ca = INITIAL_CALCIUM
    cell.v_LP = cell_properties["V_pre_rest"]
    cell.s = "s_inf"

    # Initialise every membrane gate to its voltage/calcium steady state.
    for gate in (
        "m_Na", "h_Na", "m_CaT", "h_CaT", "m_CaS", "h_CaS",
        "m_A", "h_A", "m_KCa", "m_Kd", "m_H",
    ):
        setattr(cell, gate, f"{gate}_inf")
    return cell


def simulate_abpd(duration, cell_properties, equations, synaptic_event,
                  settling_time=5 * second):
    """Settle the pacemaker, record it, and deliver one timed LP input."""

    onset = synaptic_event["onset"]
    event_duration = synaptic_event["duration"]
    event_end = onset + event_duration
    if onset < 0 * second or event_duration < 0 * second or event_end > duration:
        raise ValueError("The synaptic event must fit within the recorded simulation.")

    start_scope()
    defaultclock.dt = TIME_STEP
    cell = build_abpd_cell(cell_properties, equations)

    # The original study initialized cells on previously saved limit cycles.
    # A short unrecorded settling run is a transparent substitute for a lesson.
    run(settling_time)

    voltage_monitor = StateMonitor(cell, ["v", "s", "Ca"], record=True)
    spike_monitor = SpikeMonitor(cell)

    run(onset)
    cell.v_LP = cell_properties["V_pre_active"]
    run(event_duration)
    cell.v_LP = cell_properties["V_pre_rest"]
    run(duration - event_end)
    return voltage_monitor, spike_monitor


def plot_results(voltage_monitor, spike_monitor, duration, synaptic_event):
    """Plot AB/PD voltage, LP synaptic activation, and calcium concentration."""

    # StateMonitor keeps Brian's absolute clock (including the settling run).
    # Shift it so the displayed recording begins at t = 0 s.
    time_s = (voltage_monitor.t - voltage_monitor.t[0]) / second
    voltage_mv = voltage_monitor.v[0] / mV
    synaptic_activation = voltage_monitor.s[0]
    calcium_um = voltage_monitor.Ca[0] / uM
    onset_s = synaptic_event["onset"] / second
    event_end_s = (synaptic_event["onset"] + synaptic_event["duration"]) / second

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]}, constrained_layout=True,
    )
    fig.suptitle(
        "Lobster STG AB/PD model receiving transient LP inhibition",
        fontsize=14,
    )

    axes[0].plot(time_s, voltage_mv, color="#2457A6", linewidth=0.9)
    axes[0].axhline(0, color="black", linewidth=1.0, label="0 mV")
    axes[0].axhline(
        FIRING_THRESHOLD / mV, color="#D62728", linestyle="--", linewidth=1.2,
        label="Spike-detection threshold (−20 mV)",
    )
    axes[0].set(ylabel="Voltage (mV)")

    axes[1].plot(time_s, synaptic_activation, color="#7A5195", linewidth=1.3)
    axes[1].set(ylabel="LP synapse\nactivation, $s$", ylim=(-0.03, 1.03))

    axes[2].plot(time_s, calcium_um, color="#2A9D8F", linewidth=1.2)
    axes[2].set(xlabel="Recorded time (s)", ylabel="[Ca²⁺]\n(µM)")

    for axis in axes:
        axis.axvspan(
            onset_s, event_end_s, color="#F4A261", alpha=0.25,
            label="LP inhibitory event",
        )
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].legend(loc="upper right", frameon=False, ncol=3, fontsize=8)
    fig.savefig(FIGURE_FILE, dpi=180)
    plt.show()


# =============================================================================
# PART 3: STUDENT SETTINGS AND EXECUTION
# =============================================================================

duration = 5 * second

# Change these two values to move or lengthen the LP inhibitory input.
synaptic_event = {
    "onset": 1.5 * second,
    "duration": 0.5 * second,
}

# AB/PD model #3 from Table 2 of Prinz, Bucher & Marder (2004).
# Intrinsic conductances are densities; g_syn is a whole-synapse conductance.
cell_properties = {
    "area": 0.628e-3 * cm**2,
    "C": 0.628 * nF,
    "g_Na": 200 * msiemens / cm**2,
    "g_CaT": 2.5 * msiemens / cm**2,
    "g_CaS": 4 * msiemens / cm**2,
    "g_A": 50 * msiemens / cm**2,
    "g_KCa": 5 * msiemens / cm**2,
    "g_Kd": 50 * msiemens / cm**2,
    "g_H": 0.01 * msiemens / cm**2,
    "g_leak": 0 * msiemens / cm**2,
    "E_Na": 50 * mV,
    "E_K": -80 * mV,
    "E_H": -20 * mV,
    "E_leak": -50 * mV,
    "Ca_out": 3 * mM,
    "Ca_0": 0.05 * uM,
    "tau_Ca": 200 * ms,
    "f_Ca": 14.96 * uM / nA,
    "nernst_factor": 12.2 * mV,
    # LP is glutamatergic. The paper tested g_syn = 0, 3, 10, 30, 100 nS.
    "g_syn": 30 * nS,
    "E_syn": -70 * mV,
    "k_minus": 1 / (40 * ms),
    "V_syn_half": -35 * mV,
    "V_syn_slope": 5 * mV,
    "V_pre_rest": -60 * mV,
    "V_pre_active": 20 * mV,
}

equations = """
# Membrane currents (positive terms below are inward/depolarising)
dv/dt = (-I_Na-I_CaT-I_CaS-I_A-I_KCa-I_Kd-I_H-I_leak-I_syn)/C : volt

I_Na   = g_Na*m_Na**3*h_Na*(v-E_Na)*area : amp
I_CaT  = g_CaT*m_CaT**3*h_CaT*(v-E_Ca)*area : amp
I_CaS  = g_CaS*m_CaS**3*h_CaS*(v-E_Ca)*area : amp
I_A    = g_A*m_A**3*h_A*(v-E_K)*area : amp
I_KCa  = g_KCa*m_KCa**4*(v-E_K)*area : amp
I_Kd   = g_Kd*m_Kd**4*(v-E_K)*area : amp
I_H    = g_H*m_H*(v-E_H)*area : amp
I_leak = g_leak*(v-E_leak)*area : amp
I_syn  = g_syn*s*(v-E_syn) : amp

# Calcium concentration and calcium reversal potential
dCa/dt = (-f_Ca*(I_CaT+I_CaS)-Ca+Ca_0)/tau_Ca : mole/metre**3
E_Ca = nernst_factor*log(Ca_out/Ca) : volt

# Fast sodium current
dm_Na/dt = (m_Na_inf-m_Na)/tau_m_Na : 1
dh_Na/dt = (h_Na_inf-h_Na)/tau_h_Na : 1
m_Na_inf = 1/(1+exp((v/mV+25.5)/-5.29)) : 1
h_Na_inf = 1/(1+exp((v/mV+48.9)/5.18)) : 1
tau_m_Na = (-2.52/(1+exp((v/mV+120)/-25))+2.64)*ms : second
tau_h_Na = (1.34/(1+exp(-(v/mV+62.9)/10)))*(1/(1+exp((v/mV+34.9)/3.6))+1.5)*ms : second

# Transient and slow calcium currents
dm_CaT/dt = (m_CaT_inf-m_CaT)/tau_m_CaT : 1
dh_CaT/dt = (h_CaT_inf-h_CaT)/tau_h_CaT : 1
m_CaT_inf = 1/(1+exp((v/mV+27.1)/-7.2)) : 1
h_CaT_inf = 1/(1+exp((v/mV+32.1)/5.5)) : 1
tau_m_CaT = (-42.6/(1+exp((v/mV+68.1)/-20.5))+43.4)*ms : second
tau_h_CaT = (-179.6/(1+exp((v/mV+55)/-16.9))+210)*ms : second

dm_CaS/dt = (m_CaS_inf-m_CaS)/tau_m_CaS : 1
dh_CaS/dt = (h_CaS_inf-h_CaS)/tau_h_CaS : 1
m_CaS_inf = 1/(1+exp((v/mV+33)/-8.1)) : 1
h_CaS_inf = 1/(1+exp((v/mV+60)/6.2)) : 1
tau_m_CaS = (2.8+14/(exp((v/mV+27)/10)+exp((v/mV+70)/-13)))*ms : second
tau_h_CaS = (120+300/(exp((v/mV+55)/9)+exp((v/mV+65)/-16)))*ms : second

# A-type, calcium-dependent, delayed-rectifier, and H currents
dm_A/dt = (m_A_inf-m_A)/tau_m_A : 1
dh_A/dt = (h_A_inf-h_A)/tau_h_A : 1
m_A_inf = 1/(1+exp((v/mV+27.2)/-8.7)) : 1
h_A_inf = 1/(1+exp((v/mV+56.9)/4.9)) : 1
tau_m_A = (-20.8/(1+exp((v/mV+32.9)/-15.2))+23.2)*ms : second
tau_h_A = (-58.4/(1+exp((v/mV+38.9)/-26.5))+77.2)*ms : second

dm_KCa/dt = (m_KCa_inf-m_KCa)/tau_m_KCa : 1
m_KCa_inf = (Ca/(Ca+3*uM))/(1+exp((v/mV+28.3)/-12.6)) : 1
tau_m_KCa = (-150.2/(1+exp((v/mV+46)/-22.7))+180.6)*ms : second

dm_Kd/dt = (m_Kd_inf-m_Kd)/tau_m_Kd : 1
m_Kd_inf = 1/(1+exp((v/mV+12.3)/-11.8)) : 1
tau_m_Kd = (-12.8/(1+exp((v/mV+28.3)/-19.2))+14.4)*ms : second

dm_H/dt = (m_H_inf-m_H)/tau_m_H : 1
m_H_inf = 1/(1+exp((v/mV+75)/5.5)) : 1
tau_m_H = 2*ms/(exp((v/mV+169.7)/-11.6)+exp((v/mV-26.7)/14.3)) : second

# Graded LP glutamatergic synapse from the paper
ds/dt = (s_inf-s)/tau_s : 1
s_inf = 1/(1+exp((V_syn_half-v_LP)/V_syn_slope)) : 1
tau_s = (1-s_inf)/k_minus : second
dv_LP/dt = 0*volt/second : volt
"""

voltage, spikes = simulate_abpd(
    duration, cell_properties, equations, synaptic_event
)
print(f"Detected {spikes.num_spikes} spikes in {duration / second:.1f} recorded s")
print(f"Mean spike rate: {spikes.num_spikes / duration / Hz:.1f} Hz")
print(f"Saving figure as: {FIGURE_FILE}")
plot_results(voltage, spikes, duration, synaptic_event)
