# Three-cell pyloric circuit tutorial

Run in the existing environment:

```bash
conda activate neuro_sim
cd code
python workflow_circuit_simulation.py
```

The script first simulates three isolated neurons, then the recurrent circuit.
It displays and saves `circuit_intrinsic_activity.png` and
`circuit_triphasic_activity.png`. Each simulation includes five unrecorded
seconds for settling and six recorded seconds. NumPy execution requires no
compiler; the two simulations can take a few minutes.

## Model and interpretation

AB/PD represents the combined pacemaker kernel, as in Prinz, Bucher & Marder
(2004), *Similar network activity from disparate circuit parameters*,
[doi:10.1038/nn1352](https://doi.org/10.1038/nn1352). The supplied PDF's Methods
and Table 2 motivate the neuron types and connectivity:

- AB/PD bursts spontaneously.
- LP and PY spike slowly in isolation (the paper allows silence or tonic
  activity below 11.5 Hz).
- After inhibition, LP fires before PY.

This remains a reduced teaching model built from `workflow_ABPD_simple.py`.
The fitted conductances and fixed channel time constants do not reproduce the
paper's full eight-current models or its exact biological timing. Na, slow Ca,
K(Ca), Kd, H, leak, and an additional A-current are represented. The A-current
is active only in PY. AB/PD retains zero leak conductance. Calcium-dependent
potassium feedback ends bursts, and the A-current helps delay PY rebound.

All three cells share one equation set. Their parameters are separate
`ABPD_properties`, `LP_properties`, and `PY_properties` dictionaries.
`recovery_scale` multiplies calcium handling, slow-calcium inactivation,
K(Ca) activation, and H activation time constants. Its value is 3 for the
pacemaker and 1 for the followers; fast spike kinetics are unaffected.

| Maximal conductance (mS/cm²) | AB/PD | LP | PY |
|---|---:|---:|---:|
| Na | 250 | 250 | 250 |
| CaS | 8 | 4 | 4 |
| K(Ca) | 20 | 10 | 10 |
| Kd | 35 | 35 | 75 |
| H | 0.02 | 0.02 | 0.01 |
| leak | 0 | 0.02 | 0.01 |
| A | 0 | 0 | 20 |

Membrane area is 0.628 × 10⁻³ cm² and capacitance is 0.628 nF for each cell.
Intrinsic conductances are densities; synaptic weights are **whole-cell nS**.

## Connectivity

Both matrices use **row = presynaptic source, column = postsynaptic target**,
ordered AB/PD, LP, PY. Zeros preserve the template's absent connections.

| Connection | Fast (nS) | Slow (nS) |
|---|---:|---:|
| AB/PD → LP | 7 | 1 |
| AB/PD → PY | 1 | 10 |
| LP → AB/PD | 1 | 0 |
| LP → PY | 0.1 | 0 |
| PY → LP | 20 | 0 |

The combined pacemaker has both AB-like glutamatergic and PD-like cholinergic
outputs. The paper's release dynamics are used:

\[
I_{\rm syn}=g_f(E_f-V)+g_s(E_s-V),\qquad
\frac{ds}{dt}=k_-\left[e^{(V_{\rm source}-V_{1/2})/\Delta}(1-s)-s\right].
\]

Incoming effective conductances are summed as \(g_f=\sum_j w^f_{ji}s^f_j\)
and \(g_s=\sum_j w^s_{ji}s^s_j\). Fast synapses use −70 mV and
\(k_-=1/(40\,{\rm ms})\); slow synapses use −80 mV and
\(k_-=1/(100\,{\rm ms})\). Both use −35 mV half-activation and 5 mV slope.
This is algebraically equivalent to the paper's
\(\dot s=(s_\infty-s)/\tau_s\), avoiding a near-zero denominator.

There are no timed inputs, imposed burst windows, or voltage resets. Release
depends continuously on actual presynaptic voltage, including below spike
threshold. Setting both matrices to zero gives the isolated-cell experiment.
The dashed red −20 mV line is only a spike-detection convention, not a hard
sodium-channel threshold.

## Validation

With the final weights, a 20-second recording after 10 seconds of settling
passed the paper's ordering criteria in all 22 complete measured cycles:
AB/PD ended before LP began, LP began before PY, and LP ended before PY.
At 0.05 ms integration steps, mean cycle period was 0.833 s and mean onset
phases were approximately 0.457 (LP) and 0.706 (PY), relative to AB/PD onset.
AB/PD produced 17 spikes per complete cycle; followers generally produced
about 5 LP and 6 PY spikes. These are model results, not experimental values.

Halving the time step to 0.025 ms preserved the ordering in 22/22 cycles;
mean period was 0.834 s. The circuit is faster than the biological rhythms
used to classify the paper's full models. It demonstrates their qualitative
triphasic organization, without claiming quantitative reproduction.

The script prints isolated/connected mean spike rates and checks complete
cycles using a 100 ms interspike-gap criterion for AB/PD burst detection.
Students changing parameters may obtain other patterns, so the diagnostic
prints the number of passing cycles rather than assuming success.
