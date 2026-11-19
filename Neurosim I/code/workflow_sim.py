#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 15:39:27 2025

@author: christian
"""
from brian2 import *
import numpy as np
from simulation_loop import sim_loop

def synapse_histogram(values,score,threshold,bins=30, xlabel="", ylabel="", title="", 
                   xlim=None, ylim=None, log=False,):
    idx = score>=threshold
    vals = values[idx]
    plt.figure(figsize=(7, 5))

    plt.hist(vals, bins=bins, edgecolor="black", alpha=0.7)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if log:
        plt.yscale("log")

    plt.tight_layout()
    plt.show()
    


def plot_phase_diagram(
        X, Y, scores,
        labels=["X", "Y"],
        limits=None,
        log=None,
        #cmap="gist_heat"
    ):


    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.cm.copper
    cmap= cmap.reversed()

    sc = ax.scatter(
        X, Y,
        c=scores,
        cmap=cmap,
        alpha=0.5,
        edgecolors='none'
    )

    # Colorbar outside
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Score")

    # Labels
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    # Optional limits
    if limits is not None:
        if limits[0] is not None:
            ax.set_xlim(limits[0])
        if limits[1] is not None:
            ax.set_ylim(limits[1])

    # Optional log scale
    if log == "x":
        ax.set_xscale("log")
    elif log == "y":
        ax.set_yscale("log")
    elif log == "both":
        ax.set_xscale("log")
        ax.set_yscale("log")

    #ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def display_synapses(coordinate,nrns = ['AB','LP','PY'],c_range=None):
    fast = coordinate[0:9].reshape((3,3))
    slow = coordinate[9:18].reshape((3,3))
    
    mats = [slow,fast]
    titles = ["Slow connections","Fast connections"]
    for adj_matrix,title in zip(mats,titles): 

       # Basic checks
       if adj_matrix.shape[0] != adj_matrix.shape[1]:
           raise ValueError("adj_matrix must be square (n x n).")
       if len(nrns) != adj_matrix.shape[0]:
           raise ValueError("Length of nrns must match matrix dimension.")
       
       plt.figure(figsize=(7, 6))
       if c_range is None:
           im = plt.imshow(adj_matrix, interpolation='nearest', aspect='equal',
                           cmap='Reds')
           
       else:
           im = plt.imshow(adj_matrix, interpolation='nearest', aspect='equal',
                           cmap='Reds',vmin=c_range[0], vmax=c_range[1])
    
       # Title
       plt.title(title, fontsize=22)
    
       # Tick labels
       plt.xticks(ticks=np.arange(len(nrns)), labels=nrns, rotation=90,
                  fontsize=22)
       plt.yticks(ticks=np.arange(len(nrns)), labels=nrns,fontsize=22)
    
       # Add gridlines for clarity
       #plt.grid(color='white', linestyle='-', linewidth=0.5)
       
       # Nicely annotated colorbar
       cbar = plt.colorbar(im)

       cbar.set_label("Synaptic Strength [nS]", fontsize=14)
    
       plt.tight_layout()
       plt.show()
    



#working coordinates

fast_template = np.array([[0, 7, 10],
                          [10, 0, 5],
                          [0, 20, 0]])  # in nS

slow_template = np.array([[0, 5, 10],
                               [0, 0, 0],
                               [0, 0, 0]])  # in nS

#synaptic_weights = [0.0001,1,2,5,10,20,30,50,100,150]
synaptic_weights = np.arange(0,300)
synaptic_weights[0] = 0.01



num_sims = 20000



coords,scrs = sim_loop(fast_template,slow_template,synaptic_weights,num_sims,plotting=False)

plot_phase_diagram(coords[:,1],coords[:,2],scrs,labels=['fast PD>>LP','fast PD>PYC']) 
plot_phase_diagram(coords[:,3],coords[:,5],scrs,labels=['fast LP>>PD','fast LP>>PY']) 
plot_phase_diagram(coords[:,7],coords[:,5],scrs,labels=['fast C>>LP','fast LP>>PY']) 
plot_phase_diagram(coords[:,10],coords[:,11],scrs,labels=['slow PD>>LP','PD>>PY'])     


thrshld = 6
bins = np.arange(0,np.max(synaptic_weights),10)
synapse_histogram(coords[:,1],scrs,thrshld,bins=bins, xlabel="fast PD>>LP", ylabel="", title="", 
                   ylim=None, log=False,xlim=[min(synaptic_weights),max(synaptic_weights)])
synapse_histogram(coords[:,2],scrs,thrshld,bins=bins, xlabel="fast PD>>PY", ylabel="", title="", 
                   ylim=None, log=False,xlim=[min(synaptic_weights),max(synaptic_weights)])
synapse_histogram(coords[:,3],scrs,thrshld,bins=bins, xlabel="fast LP>>PD", ylabel="", title="", 
                   ylim=None, log=False,xlim=[min(synaptic_weights),max(synaptic_weights)])
synapse_histogram(coords[:,5],scrs,thrshld,bins=bins, xlabel="fast LP>>PY", ylabel="", title="", 
                    ylim=None, log=False,xlim=[min(synaptic_weights),max(synaptic_weights)])
synapse_histogram(coords[:,7],scrs,thrshld,bins=bins, xlabel="fast PY>>LP", ylabel="", title="", 
                   ylim=None, log=False,xlim=[min(synaptic_weights),max(synaptic_weights)])
synapse_histogram(coords[:,10],scrs,thrshld,bins=bins, xlabel="slow PD>>LP", ylabel="", title="", 
                   ylim=None, log=False,xlim=[min(synaptic_weights),max(synaptic_weights)])
synapse_histogram(coords[:,11],scrs,thrshld,bins=bins, xlabel="slow PD>>PY", ylabel="", title="", 
                   ylim=None, log=False,xlim=[min(synaptic_weights),max(synaptic_weights)])


print('----------------------------------------------------------------------')
print("TRIPHASIC: ", np.sum(scrs>1) )

print("PYLORIC COMPLIANT: ", np.sum(scrs>4) )
print('----------------------------------------------------------------------')


positives = scrs > 4

p_idx = np.where(positives == True)
for i in p_idx[0]:
    print(i)
    display_synapses(coords[i,:],c_range=[min(synaptic_weights),max(synaptic_weights)])
    





