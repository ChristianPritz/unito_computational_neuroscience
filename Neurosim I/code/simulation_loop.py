#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-neuron circuit with bursting neuron 0 and tonic firing for remaining neurons
"""


from brian2 import *
import numpy as np
from score import scoring_output

def permute_synapses(matrix,values,mode="nonzero"):
    perm_mat = np.zeros(matrix.shape)
    nonzero = matrix>0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if mode == 'nonzero':
                if nonzero[i,j]:
                    perm_mat[i,j] = np.random.choice(values,size=1)
            elif mode == 'all': 
                if i != j:
                    perm_mat[i,j] = np.random.choice(values,size=1)
    return perm_mat

def detect_blocks(spike_times,block_thresh=50):
    blocks = []
    starts = []
    ends = []
    durs = []
    current_block = [spike_times[0]]
    isi = np.diff(spike_times)
    for i in range(1, len(spike_times)):
        if isi[i-1] < block_thresh:
            current_block.append(spike_times[i])
        else:
            if len(current_block) > 1:  # at least 2 spikes to count as a block
                blocks.append(current_block)
            current_block = [spike_times[i]]
        
        
    
    # append the last block
    if len(current_block) > 1:
        blocks.append(current_block)
    for i in blocks:
        starts.append(i[0])
        ends.append(i[-1])
        durs.append(i[-1]-i[0])
        
    return starts,ends,durs
                


def sim_loop(fast_template,slow_template,synaptic_weights,
             num_sims = 500,plotting=False):
    
    scores = []
    coordinates =  []
    b_stats = []
    spike_stats = []

    
    for i in range(num_sims):
    
    
        
        
        # ======================
        # CONNECTIVITY MATRICES
        # ======================
        
        
        fast_syn_strengths = permute_synapses(fast_template,synaptic_weights)
        
        slow_syn_strengths = permute_synapses(slow_template,synaptic_weights)
        
        # ======================
        # BURST PARAMETERS
        # ======================
        burstfreq = 2     # Hz
        burstdur = 100    # ms
        burstamp = 0.6    # nA
        
        fast_tau = 20 * ms
        slow_tau = 100 * ms
        
        #tonic properties
        tonicCurrent = 0.6
        
        
        # ======================
        # SETUP
        # ======================
        start_scope()
        duration = 2 * second
        defaultclock.dt = 0.1 * ms
        
        # Neuron parameters
        N = 3
        tau_m = 10 * ms
        v_rest = -65 * mV
        v_reset = -70 * mV
        v_thresh = -50 * mV
        refractory_period = 3 * ms
        E_inh = -80 * mV
        C_m = 200 * pF
        
        # ======================
        # NEURON MODEL
        # ======================
        eqs = '''
        dv/dt = ((v_rest - v)/tau_m + g_inh*(E_inh - v)/C_m + I_burst/C_m + I_tonic/C_m) : volt (unless refractory)
        dg_fast/dt = -g_fast/fast_tau : siemens
        dg_slow/dt = -g_slow/slow_tau : siemens
        g_inh = g_fast + g_slow : siemens
        I_burst : amp
        I_tonic : amp
        '''
        
        # ======================
        # CREATE NEURONS
        # ======================
        neurons = NeuronGroup(
            N, eqs,
            threshold='v > v_thresh',
            reset='v = v_reset',
            refractory=refractory_period,
            method='euler',
            namespace=dict(
                v_rest=v_rest,
                v_reset=v_reset,
                v_thresh=v_thresh,
                tau_m=tau_m,
                E_inh=E_inh
            )
        )
        
        neurons.v = v_reset
        neurons.I_burst = 0 * nA
        neurons.I_tonic = 0 * nA
        
        # ======================
        # BURSTING NEURON 0
        # ======================
        bursting = neurons[0:1]
        rest = neurons[1:]
        
        burst_freq = burstfreq * Hz
        burst_duration = burstdur * ms
        burst_amp = burstamp * nA
        cycle = 1 / burst_freq
        
        bursting.run_regularly(
            'I_burst = burst_amp * int((t % cycle) < burst_duration)',
            dt=defaultclock.dt
        )
        
        # ======================
        # TONIC FIRING FOR REST NEURONS
        # ======================
        # Adjust this to tune tonic firing rate (~100 Hz)
        rest.I_tonic = tonicCurrent * nA
        
        # ======================
        # SYNAPSES
        # ======================
        fast = Synapses(neurons, neurons, 'w_fast : siemens', on_pre='g_fast_post += w_fast')
        pre, post = np.nonzero(fast_syn_strengths)
        fast.connect(i=pre, j=post)
        fast.w_fast = (fast_syn_strengths[pre, post] * nS)
        
        slow = Synapses(neurons, neurons, 'w_slow : siemens', on_pre='g_slow_post += w_slow')
        pre, post = np.nonzero(slow_syn_strengths)
        slow.connect(i=pre, j=post)
        slow.w_slow = (slow_syn_strengths[pre, post] * nS)
        
        # ======================
        # MONITORS
        # ======================
        spikes = SpikeMonitor(neurons)
        state = StateMonitor(neurons, ['v', 'g_inh'], record=True)
        
        try: 
            # ======================
            # RUN SIMULATION
            # ======================
            print("Running simulation...")
            run(duration)
            
            # ======================
            # ANALYSIS
            # ======================
            print("\nActual firing rates:")
            for i in range(N):
                print(f"Neuron {i}: {spikes.count[i]/duration:.2f} Hz")
            
            # ======================
            # PLOTS
            # ======================
            if plotting:
                figure(figsize=(14, 10))
                colors = ['b', 'g', 'r']
                samples = int(duration/defaultclock.dt)
                
                # Voltage traces
                for i in range(N):
                    subplot(N+2,1,i+1)
                    plot(state.t[:samples]/second, state.v[i][:samples]/mV, color=colors[i])
                    axhline(v_thresh/mV, ls='--', color='k')
                    ylabel("V (mV)")
                    if i==0:
                        title("Voltages")
                    if i<N-1:
                        xticks([])
                
                # Raster plot
                subplot(N+2,1,N+1)
                for i in range(N):
                    t_sp = spikes.t[spikes.i==i]
                    plot(t_sp/second, np.ones_like(t_sp)*i, '|', ms=8, color=colors[i])
                ylabel("Neuron")
                xlabel("Time (s)")
                title("Raster")
                
                # g_inh traces
                subplot(N+2,1,N+2)
                for i in range(N):
                    plot(state.t[:samples]/second, state.g_inh[i][:samples]/nS, lw=1, color=colors[i])
                ylabel("g_inh (nS)")
                xlabel("Time (s)")
                title("Total Inhibitory Conductance")
                
                tight_layout()
                show()
 
            


            s = []
            e = []
            d = []
            spks = []
            for i in range(3):
                si,ei,di = detect_blocks(spikes.t[spikes.i==i],block_thresh=100/1000*second) 
                s.append(si)
                e.append(ei)
                d.append(di)
                spks.append(spikes.t[spikes.i==i])
            block_stats = [s,e,d]
            

     
            colorcode =  scoring_output(block_stats)
            
            print("The score is: ",colorcode)
            
            scores.append(colorcode)
            b_stats.append(block_stats)
            spike_stats.append(spks)
           
            if colorcode > 1:
                figure(figsize=(14, 10))
                colors = ['b', 'g', 'r']
                samples = int(duration/defaultclock.dt)
                
                # Voltage traces
                for i in range(N):
                    subplot(N+2,1,i+1)
                    plot(state.t[:samples]/second, state.v[i][:samples]/mV, color=colors[i])
                    axhline(v_thresh/mV, ls='--', color='k')
                    ylabel("V (mV)")
                    if i==0:
                        title("Voltages")
                    if i<N-1:
                        xticks([])
                
                # Raster plot
                subplot(N+2,1,N+1)
                for i in range(N):
                    t_sp = spikes.t[spikes.i==i]
                    plot(t_sp/second, np.ones_like(t_sp)*i, '|', ms=8, color=colors[i])
                ylabel("Neuron")
                xlabel("Time (s)")
                title("Raster")
                
                # g_inh traces
                subplot(N+2,1,N+2)
                for i in range(N):
                    plot(state.t[:samples]/second, state.g_inh[i][:samples]/nS, lw=1, color=colors[i])
                ylabel("g_inh (nS)")
                xlabel("Time (s)")
                title("Total Inhibitory Conductance")
                
                tight_layout()
                show()
            print('--------------------------')
        except:
            
            
            scores.append(0)
            b_stats.append(np.nan)
            spike_stats.append(np.nan)
        
        # this flattens all the synaptic weights into a line and records them 
        # for analysis
        coordinates.append(np.hstack((fast_syn_strengths.flatten(),slow_syn_strengths.flatten())))
        
    
    #return scores, coordinates        
     
        
    coords = np.array(coordinates)
    scrs = np.array(scores)

    np.savez("sim_scores.npz", coords=coords, scrs=scrs)#,spikes=spikes)
    #np.save("b_stats.npy", b_stats, allow_pickle=True)
    #np.save("spikes.npy", spike_stats, allow_pickle=True)

    return coords,scrs
