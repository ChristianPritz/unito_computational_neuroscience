#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:28:46 2026

@author: dominik
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Experimental data
# ============================================================
# Replace these with your measurements from the balloon video

t_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

V_data = np.array([
    1.00,
    0.88,
    0.77,
    0.68,
    0.60,
    0.53,
    0.47,
    0.42,
    0.38,
    0.34,
    0.31
])


# ============================================================
# 2. Estimate k at each time interval
# ============================================================
#
# dV/dt ≈ (V[n+1] - V[n]) / dt
#
# From
#
# dV/dt = -k V
#
# we get
#
# k = -(1/V) dV/dt
#

dt_data = np.diff(t_data)

k_data = -(np.diff(V_data) / dt_data) / V_data[:-1]

# Volume associated with each k
V_for_k = V_data[:-1]


# ============================================================
# 3. Fit k as a function of V
# ============================================================
#
# Here we assume:
#
# k(V) = a + b V
#
# This is just an example. You could try other functions.
#

a, b = np.polyfit(V_for_k, k_data, 1)

print("Fitted k(V):")
print(f"k(V) = {b:.4f} + {a:.4f} V")

# NOTE:
# np.polyfit returns slope, intercept
# so:
# k(V) = a*V + b


# ============================================================
# 4. Define the nonlinear ODE
# ============================================================
#
# dV/dt = -k(V) V
#
# with
#
# k(V) = a V + b
#
# therefore:
#
# dV/dt = -(a V + b)V
#

def dVdt(V):
    k = a * V + b
    return -k * V


# ============================================================
# 5. Euler discretization
# ============================================================
#
# V[n+1] = V[n] + dV/dt * dt
#

dt = 0.01
T = t_data[-1]

t = np.arange(0, T + dt, dt)

V_model = np.zeros(len(t))

# Initial condition
V_model[0] = V_data[0]

# Euler loop
for n in range(len(t) - 1):

    # Calculate derivative at current volume
    rate = dVdt(V_model[n])

    # Euler step
    V_model[n + 1] = V_model[n] + rate * dt


# ============================================================
# 6. Plot experimental data and nonlinear model
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    t_data,
    V_data,
    label="Experimental data"
)

plt.plot(
    t,
    V_model,
    label="Nonlinear ODE"
)

plt.xlabel("Time")
plt.ylabel("Volume")
plt.title("Balloon deflation: nonlinear ODE")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 7. Plot the estimated k(V)
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    V_for_k,
    k_data,
    label="Estimated k"
)

V_range = np.linspace(
    V_for_k.min(),
    V_for_k.max(),
    100
)

k_fit = a * V_range + b

plt.plot(
    V_range,
    k_fit,
    label="Fitted k(V)"
)

plt.xlabel("Volume")
plt.ylabel("k")
plt.title("How k changes with balloon volume")
plt.legend()
plt.grid(True)

plt.show()