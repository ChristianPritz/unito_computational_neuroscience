#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:16:11 2026

@author: wormulon
"""


import matplotlib.pyplot as plt 
import numpy as np
import csv
import networkx as nx
from itertools import pairwise
from itertools import permutations
from connectomics_functions import *



def visulize_graph(G,pos):
    # Visualize network
    plt.figure(figsize=(8, 6))
    
    # Use manually defined coordinates
    
    nx.draw_networkx(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=700,
        edge_color="gray",
        arrows=True,
        arrowsize=15,
        font_size=10
    )
    
    plt.title("Network")
    plt.axis("off")
    plt.show()


# =============================================================================
#  Cell 1
# =============================================================================
# Here we gonna explore some basic concepts about graphs in connectomics: 
# Let's load an adjacency matrix and see what the graph looks like and 
# the let's examine some basic statistics. 


A_machine, nodes = load_adjacency_matrix("/home/wormulon/Downloads/testConn.csv")
G_machine = nx.from_numpy_array(A_machine, create_using=nx.DiGraph)
G_machine = nx.relabel_nodes(G, dict(enumerate(nodes)))
# Let us plot the graph 
# this variable sorts the position of the neurons in the graph
positions = {"A": [0,0],
             "B": [1,0],
             "C": [2,0],
             "D": [3,0],
             "E": [0.5,-1],
             "F": [1.5,-1],
             "G": [2.5,-1],
             "H": [1,-2],
             "I": [2,-2],
             "J": [-0.5,-3],
             "K": [0.5,-3],
             "L": [1.5,-3],
             "M": [2.5,-3],
             "N": [3.5,-3],}


plot_network(G_machine,positions=positions)
plot_adjacency_matrix(G_machine,sort_nodes=False, figsize=(3,3))
basic_stats(G_machine)

# =============================================================================
#  Cell 2
# =============================================================================
# This doesn't look like a graph from biological network. Let's rework the graph 
# so that the information can flow bi-directionally. 
# Try to edit it until the reachability is 100% and graph is fully connected. 


e,A_edit = edit_adjacency_matrix(A_machine, nodes)
G_edit = nx.from_numpy_array(A_edit, create_using=nx.DiGraph)
G_edit = nx.relabel_nodes(G, dict(enumerate(nodes)))
plot_network(G_machine,positions=positions)
plot_adjacency_matrix(G_machine,sort_nodes=False, figsize=(3,3))
basic_stats(G_edit)







A_bio, nodes2 = load_adjacency_matrix("/home/wormulon/Downloads/testConn2.csv")
A_weighted, nodes = load_adjacency_matrix("/home/wormulon/Downloads/testConn_weighted.csv") 

G_bio = nx.from_numpy_array(A_bio, create_using=nx.DiGraph)
G_bio = nx.from_numpy_array(A_bio, create_using=nx.DiGraph)
G_bio = nx.relabel_nodes(G_bio, dict(enumerate(nodes)))
G_weighted = nx.from_numpy_array(A_weighted, create_using=nx.DiGraph)
G_weighted = nx.relabel_nodes(G_weighted, dict(enumerate(nodes)))

basic_stats(G)
basic_stats(G_bio)
basic_stats(G_weighted)












positions = {"A": [0,0],
             "B": [1,0],
             "C": [2,0],
             "D": [3,0],
             "E": [0.5,-1],
             "F": [1.5,-1],
             "G": [2.5,-1],
             "H": [1,-2],
             "I": [2,-2],
             "J": [-0.5,-3],
             "K": [0.5,-3],
             "L": [1.5,-3],
             "M": [2.5,-3],
             "N": [3.5,-3],}

# positions = {0: [0,0],
#              1: [1,0],
#              2: [2,0],
#              3: [3,0],
#              4: [0.5,-1],
#              5: [1.5,-1],
#              6: [2.5,-1],
#              7: [1,-2],
#              8: [2,-2],
#              9: [-0.5,-3],
#              10: [0.5,-3],
#              11: [1.5,-3],
#              12: [2.5,-3],
#              13: [3.5,-3],}


plot_network(G,positions=positions)
plot_network(G_bio,positions=positions)
plot_network(G_weighted,positions=positions)



#Now showing the difference between degree preserving and erdos-renyi.........
#........................................................................#
g_preserved = randomize_graph(G_bio,method="config_preserving",n_type="directed")
g_erdos = randomize_graph(G_bio,method="erdos_renyi",n_type="directed")

plot_network(G_bio,positions=positions)
plot_network(g_preserved,positions=positions)
plot_network(g_erdos,positions=positions)


print("Functional connectivity:")
basic_stats(G_bio)
print("Randomized, with preserved degrees:")
basic_stats(g_preserved)
print("Erdos Renyi:")
basic_stats(g_erdos)


#Now basically Rich clubbing the fake networks ...........................
#........................................................................#



# ======== COMPUTE RICH-CLUB ========
rc_real, rc_rand_avg, rc_rand_std, rc_norm =  rich_club(G_weighted, method="config_preserving",n_rand=10)
rc_real, rc_rand_avg, rc_rand_std, rc_norm =  rich_club(G_weighted, method="erdos_renyi",n_rand=10)


# Convert to undirected and get degrees
G_und = G_weighted.to_undirected()
degrees = dict(G_und.degree())


# ======== DETERMINE RICH-CLUB THRESHOLD ========
k_vals = list(rc_real.keys())
significant_k = [k for k in k_vals if rc_real[k] > rc_rand_avg[k] + rc_rand_std[k]]

if not significant_k:
    print("No significant rich-club degree found")
    k_rich = None
    rich_club_neurons = []
else:
    k_rich = 4 # This is the value from the paper............
    rich_club_neurons = [n for n, d in degrees.items() if d >= k_rich]
    print(f"Rich-club threshold degree: {k_rich}")
    print(f"Number of neurons in rich club: {len(rich_club_neurons)}")
    print("Rich-club neurons:", rich_club_neurons)

plot_rich_club(rc_real,rc_rand_avg,rc_rand_std,rc_norm,k_vals,k_rich)
