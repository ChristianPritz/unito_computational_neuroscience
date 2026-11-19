"""Three-cell pyloric circuit: AB/PD pacemaker, LP and PY followers.

Reduced teaching model inspired by Prinz, Bucher & Marder (2004),
doi:10.1038/nn1352. Intrinsic kinetics are simplified and conductances tuned
for this reduction; they are not a reproduction of Table 2.
All communication is through voltage-dependent inhibitory synapses.
"""

# =============================================================================
# PART 1: IMPORTS AND SETTINGS
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from brian2 import (
    NeuronGroup, Synapses, Network, StateMonitor, SpikeMonitor,
    start_scope, defaultclock, prefs, cm, mV, ms, second, msiemens,
    nS, nF, nA, uM, mM, Hz,
)

prefs.codegen.target = "numpy"  # portable; no compiler needed
TIME_STEP = 0.05 * ms
FIRING_THRESHOLD = -20 * mV  # measurement only; never resets the voltage
CELL_NAMES = ("AB/PD", "LP", "PY")
COLORS = ("#2457A6", "#D1495B", "#2A9D8F")


# =============================================================================
# PART 2: FUNCTIONS
# =============================================================================
def build_cells(configurations):
    """One shared equation set, with an individual dictionary for each cell."""
    cells = NeuronGroup(
        len(configurations), equations, method="exponential_euler",
        threshold="v > firing_threshold", refractory="v > firing_threshold",
        namespace={"firing_threshold": FIRING_THRESHOLD},
    )
    for index, properties in enumerate(configurations):
        for key, value in properties.items():
            setattr(cells[index:index+1], key, value)
    cells.v = -50 * mV
    cells.Ca = 0.05 * uM
    for gate in ("h_Na", "m_CaS", "h_CaS", "m_KCa", "m_Kd", "m_H", "h_A"):
        setattr(cells, gate, gate + "_inf")
    cells.s_fast = 0
    cells.s_slow = 0
    return cells


def connect_cells(cells, matrix, kind):
    """Matrix rows are sources, columns are targets; entries are in nS."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (len(cells), len(cells)) or np.any(matrix < 0):
        raise ValueError("Use a non-negative, square connectivity matrix.")
    # Sum incoming conductances before computing their postsynaptic currents.
    connections = Synapses(cells, cells, model=f"""
        weight : siemens (constant)
        g_{kind}_post = weight*s_{kind}_pre : siemens (summed)
    """)
    sources, targets = np.nonzero(matrix)
    if len(sources):
        connections.connect(i=sources, j=targets)
        connections.weight = matrix[sources, targets] * nS
    else:
        connections.active = False
    return connections


def simulate_circuit(configurations, fast_matrix, slow_matrix, duration,
                     settling_time=5*second):
    """Run either isolated cells (zero matrices) or the recurrent circuit."""
    start_scope()
    defaultclock.dt = TIME_STEP
    cells = build_cells(configurations)
    fast = connect_cells(cells, fast_matrix, "fast")
    slow = connect_cells(cells, slow_matrix, "slow")
    network = Network(cells, fast, slow)
    network.run(settling_time)
    voltage = StateMonitor(cells, "v", record=True)
    spikes = SpikeMonitor(cells)
    network.add(voltage, spikes)
    network.run(duration)
    return voltage, spikes


def describe_rhythm(spikes):
    """Check the paper's burst order in complete AB/PD cycles.

    A gap longer than 100 ms separates pacemaker bursts. This simple analysis
    is intended for the default parameter set, not every possible experiment.
    """
    trains = [np.asarray(spikes.spike_trains()[i]/second) for i in range(3)]
    abpd = trains[0]
    if len(abpd) < 2:
        print("Too few pacemaker spikes to measure a rhythm.")
        return
    starts = abpd[np.r_[True, np.diff(abpd) > 0.1]][1:]
    passed = 0
    periods = np.diff(starts)
    for onset, next_onset in zip(starts[:-1], starts[1:]):
        bursts = [t[(t >= onset) & (t < next_onset)] for t in trains]
        if any(len(t) < 2 for t in bursts):
            continue
        ab, lp, py = bursts
        # AB/PD ends before LP begins; LP begins and ends before PY does.
        passed += int(ab[-1] < lp[0] < py[0] and lp[-1] < py[-1])
    if len(periods):
        print(f"Triphasic order: {passed}/{len(periods)} complete cycles; "
              f"mean period {periods.mean():.3f} s")
    else:
        print("Too few complete bursts to measure a cycle period.")


def plot_results(voltage, title, filename):
    """Aligned voltage panels make the firing order easy to compare."""
    time = (voltage.t-voltage.t[0])/second
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                             constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    for i, (name, color) in enumerate(zip(CELL_NAMES, COLORS)):
        axes[i].plot(time, voltage.v[i]/mV, color=color, lw=0.8)
        axes[i].axhline(0, color="black", lw=0.7, label="0 mV")
        axes[i].axhline(FIRING_THRESHOLD/mV, color="red", ls="--", lw=0.8,
                        label="Spike detection (−20 mV)")
        axes[i].set(ylabel=f"{name}\nVoltage (mV)", ylim=(-85, 60))
        axes[i].grid(alpha=0.15)
        axes[i].spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[-1].set(xlabel="Recorded time (s)", xlim=(0, time[-1]))
    fig.savefig(filename, dpi=180)
    return fig


# =============================================================================
# PART 3: INDIVIDUAL CONFIGURATIONS, EQUATIONS, AND EXECUTION
# =============================================================================
duration = 6 * second

ABPD_properties = {
    "area": 0.628e-3*cm**2, "C": 0.628*nF,
    "g_Na": 250*msiemens/cm**2, "g_CaS": 8*msiemens/cm**2,
    "g_KCa": 20*msiemens/cm**2, "g_Kd": 35*msiemens/cm**2,
    "g_H": 0.02*msiemens/cm**2, "g_leak": 0*msiemens/cm**2,
    "g_A": 0*msiemens/cm**2,
    # Slow the burst envelope without slowing the individual sodium spikes.
    "recovery_scale": 3.0,
}
# LP has less calcium drive than AB/PD and spikes slowly in isolation.
# Its lack of A-current allows it to rebound before PY.
LP_properties = {
    "area": 0.628e-3*cm**2, "C": 0.628*nF,
    "g_Na": 250*msiemens/cm**2, "g_CaS": 4*msiemens/cm**2,
    "g_KCa": 10*msiemens/cm**2, "g_Kd": 35*msiemens/cm**2,
    "g_H": 0.02*msiemens/cm**2, "g_leak": 0.02*msiemens/cm**2,
    "g_A": 0*msiemens/cm**2,
    "recovery_scale": 1.0,
}
PY_properties = {
    "area": 0.628e-3*cm**2, "C": 0.628*nF,
    "g_Na": 250*msiemens/cm**2, "g_CaS": 4*msiemens/cm**2,
    "g_KCa": 10*msiemens/cm**2, "g_Kd": 75*msiemens/cm**2,
    "g_H": 0.01*msiemens/cm**2, "g_leak": 0.01*msiemens/cm**2,
    "g_A": 20*msiemens/cm**2,  # transient outward current delays rebound
    "recovery_scale": 1.0,
}
# Rows = source, columns = target, order AB/PD, LP, PY. Values in nS.
# Fast glutamate: -70 mV reversal, 40 ms release decay.
# Slow acetylcholine: -80 mV reversal, 100 ms release decay.
# Preserve all seven connections from the template. Weak LP -> PY inhibition
# and stronger PY -> LP inhibition help establish the LP-before-PY sequence.
# These weights are tuned for the reduced model, not taken from one paper row.
fast_template = np.array([[0, 7, 1],
                          [1, 0, 0.1],
                          [0, 20, 0]], dtype=float)
slow_template = np.array([[0, 1, 10],
                          [0, 0, 0],
                          [0, 0, 0]], dtype=float)

# Each inward current is conductance * gating * (reversal - voltage).
# Calcium supports the burst, K(Ca) terminates it, and A delays rebound.
equations = """
dv/dt = (I_Na+I_CaS+I_KCa+I_Kd+I_H+I_A+I_leak+I_syn)/C : volt

I_Na   = g_Na*m_Na_inf**3*h_Na*(50*mV-v)*area : amp
I_CaS  = g_CaS*m_CaS**3*h_CaS*(E_Ca-v)*area : amp
I_KCa  = g_KCa*m_KCa**4*(-80*mV-v)*area : amp
I_Kd   = g_Kd*m_Kd**4*(-80*mV-v)*area : amp
I_H    = g_H*m_H*(-20*mV-v)*area : amp
I_leak = g_leak*(-50*mV-v)*area : amp
I_syn = g_fast*(-70*mV-v) + g_slow*(-80*mV-v) : amp
I_A = g_A*m_A_inf**3*h_A*(-80*mV-v)*area : amp
g_fast : siemens
g_slow : siemens

# Simple calcium handling and a calcium-dependent reversal potential
dCa/dt = (14.96*uM/nA*I_CaS-Ca+0.05*uM)/(250*ms*recovery_scale) : mole/metre**3
E_Ca = 12.2*mV*log(3*mM/Ca) : volt

# Instantaneous activation curves for the fast currents
m_Na_inf  = 1/(1+exp(-(v+25.5*mV)/(5.29*mV))) : 1

# Slow gates: one sigmoid target and one fixed time constant per gate
dh_Na/dt = (h_Na_inf-h_Na)/(3*ms) : 1
h_Na_inf = 1/(1+exp((v+48.9*mV)/(5.18*mV))) : 1

dm_CaS/dt = (m_CaS_inf-m_CaS)/(20*ms) : 1
m_CaS_inf = 1/(1+exp(-(v+33*mV)/(8.1*mV))) : 1
dh_CaS/dt = (h_CaS_inf-h_CaS)/(300*ms*recovery_scale) : 1
h_CaS_inf = 1/(1+exp((v+60*mV)/(6.2*mV))) : 1

# Slow calcium-dependent potassium feedback terminates each burst.
dm_KCa/dt = (m_KCa_inf-m_KCa)/(120*ms*recovery_scale) : 1
m_KCa_inf = (Ca/(Ca+3*uM))/(1+exp(-(v+28.3*mV)/(12.6*mV))) : 1

dm_Kd/dt = (m_Kd_inf-m_Kd)/(15*ms) : 1
m_Kd_inf = 1/(1+exp(-(v+12.3*mV)/(11.8*mV))) : 1

dm_H/dt = (m_H_inf-m_H)/(500*ms*recovery_scale) : 1
m_H_inf = 1/(1+exp((v+75*mV)/(5.5*mV))) : 1


# A-current delays rebound, especially in PY; zero in the AB/PD model.
m_A_inf = 1/(1+exp(-(v+27.2*mV)/(8.7*mV))) : 1
dh_A/dt = (h_A_inf-h_A)/(100*ms) : 1
h_A_inf = 1/(1+exp((v+56.9*mV)/(4.9*mV))) : 1

# Paper's graded release: ds/dt = k_minus*(exp((v-Vhalf)/slope)*(1-s)-s).
# This algebraic form avoids dividing by 1-s_inf near saturation.
ds_fast/dt = (exp((v+35*mV)/(5*mV))*(1-s_fast)-s_fast)/(40*ms) : 1
ds_slow/dt = (exp((v+35*mV)/(5*mV))*(1-s_slow)-s_slow)/(100*ms) : 1

area : metre**2 (constant)
recovery_scale : 1 (constant)
C : farad (constant)
g_Na : siemens/metre**2 (constant)
g_CaS : siemens/metre**2 (constant)
g_KCa : siemens/metre**2 (constant)
g_Kd : siemens/metre**2 (constant)
g_H : siemens/metre**2 (constant)
g_A : siemens/metre**2 (constant)
g_leak : siemens/metre**2 (constant)
"""

if __name__ == "__main__":
    configurations = [ABPD_properties, LP_properties, PY_properties]
    zero_connections = np.zeros((3, 3))
    print("Step 1: simulating the three isolated neurons...", flush=True)
    isolated, isolated_spikes = simulate_circuit(
        configurations, zero_connections, zero_connections, duration)
    print("Step 2: simulating the inhibitory circuit...", flush=True)
    connected, connected_spikes = simulate_circuit(
        configurations, fast_template, slow_template, duration)
    plot_results(isolated, "Intrinsic activity — no synaptic input",
                 "circuit_intrinsic_activity.png")
    plot_results(connected, "Pyloric circuit — AB/PD → LP → PY",
                 "circuit_triphasic_activity.png")
    for i, name in enumerate(CELL_NAMES):
        print(f"{name}: isolated {isolated_spikes.count[i]/(duration/second):.1f} Hz, "
              f"connected {connected_spikes.count[i]/(duration/second):.1f} Hz")
    describe_rhythm(connected_spikes)
    plt.show()
