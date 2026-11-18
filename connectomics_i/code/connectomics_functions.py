import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import IFrame, display,HTML
import tempfile, webbrowser, os
from itertools import permutations
import csv
from joblib import Parallel, delayed



def load_adjacency_matrix(filepath):
    """
    Load a weighted adjacency matrix from a CSV file.

    Empty cells are interpreted as zero.

    Returns
    -------
    adjacency_matrix : numpy.ndarray
    node_names : list
    """

    with open(filepath, 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Extract node names
    node_names = [name.strip() for name in data[0][1:]]

    # Extract weights, replacing empty cells with zero
    adjacency_matrix = np.array(
        [
            [float(value) if value.strip() else 0.0
             for value in row[1:]]
            for row in data[1:]
        ],
        dtype=float
    )

    return adjacency_matrix, node_names


def basic_stats(G, show_degrees=False, n_random=10):
    """
    Calculate basic graph statistics for a neuronal network.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed neuronal connectivity graph.
    show_degrees : bool
        If True, print the in-degree and out-degree of each neuron.
    n_random : int
        Number of random networks used to estimate small-worldness.

    Notes
    -----
    Clustering coefficient and small-worldness are calculated on
    the undirected version of G.
    """

    # Ensure that the graph is a copy, so the original is not modified.
    G = G.copy()

    # In-degree and out-degree for each neuron.
    if show_degrees:
        for node in G.nodes():
            print(f"{node}: In-degree = {G.in_degree(node)}, "
                  f"Out-degree = {G.out_degree(node)}")

    # ---------------------------------------------------------
    # 1. Shortest paths and reachability
    # ---------------------------------------------------------

    distances = []
    disconnected = 0
    total = 0

    # Consider all ordered pairs of distinct neurons.
    pairs = permutations(G.nodes, 2)

    for source, target in pairs:
        total += 1

        try:
            # Calculate shortest directed path length.
            distance = nx.shortest_path_length(
                G, source=source, target=target
            )
            distances.append(distance)

        except nx.NetworkXNoPath:
            # No directed path exists between these neurons.
            disconnected += 1

    # Percentage of neuron pairs connected by a directed path.
    reachability = (1 - disconnected / total) * 100 if total else 0

    # ---------------------------------------------------------
    # 2. Basic degree statistics
    # ---------------------------------------------------------

    avg_in_degree = np.mean([d for _, d in G.in_degree()])
    avg_out_degree = np.mean([d for _, d in G.out_degree()])

    # ---------------------------------------------------------
    # 3. Clustering coefficient
    # ---------------------------------------------------------

    # Convert to an undirected graph for clustering analysis.
    # Self-connections are removed because they do not contribute
    # to the standard clustering coefficient.
    G_undirected = G.to_undirected()
    G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))

    # Average clustering coefficient across all neurons.
    clustering = nx.average_clustering(G_undirected)

    # ---------------------------------------------------------
    # 4. Small-world coefficient
    # ---------------------------------------------------------

    # Small-worldness compares the network with random networks.
    #
    # sigma = (C / C_random) / (L / L_random)
    #
    # C        = clustering coefficient of the real network
    # C_random = average clustering of random networks
    # L        = average shortest path length of real network
    # L_random = average shortest path length of random networks

    # Use the largest connected component to calculate path lengths.
    # This avoids problems caused by disconnected nodes.
    if G_undirected.number_of_nodes() > 1:
        largest_component = max(
            nx.connected_components(G_undirected), key=len
        )
        G_lcc = G_undirected.subgraph(largest_component).copy()

        L = nx.average_shortest_path_length(G_lcc)

        # Generate random graphs with the same number of nodes
        # and approximately the same number of edges.
        random_clustering = []
        random_path_length = []

        for _ in range(n_random):
            G_random = nx.gnm_random_graph(
                G_lcc.number_of_nodes(),
                G_lcc.number_of_edges()
            )

            # Random graphs may be disconnected.
            if nx.is_connected(G_random):
                random_clustering.append(
                    nx.average_clustering(G_random)
                )
                random_path_length.append(
                    nx.average_shortest_path_length(G_random)
                )

        if random_clustering and random_path_length:
            C_random = np.mean(random_clustering)
            L_random = np.mean(random_path_length)

            if C_random > 0 and L_random > 0 and L > 0:
                sigma = (clustering / C_random) / (L / L_random)
            else:
                sigma = np.nan
        else:
            C_random = np.nan
            L_random = np.nan
            sigma = np.nan
    else:
        L = np.nan
        C_random = np.nan
        L_random = np.nan
        sigma = np.nan

    # ---------------------------------------------------------
    # 5. Print results
    # ---------------------------------------------------------

    print("==============================================================")
    print("NETWORK STATISTICS")
    print("==============================================================")

    print("Number of neurons:", G.number_of_nodes())
    print("Total number of connections:", G.number_of_edges())

    print("--------------------------------------------------------------")
    print("DEGREE STATISTICS")
    print("Average in-degree:", avg_in_degree)
    print("Average out-degree:", avg_out_degree)

    print("--------------------------------------------------------------")
    print("PATH LENGTH AND REACHABILITY")
    print("Average directed shortest path length:",
          np.mean(distances) if distances else np.nan)
    print("Reachability:", round(reachability, 2), "%")

    print("--------------------------------------------------------------")
    print("CLUSTERING AND SMALL-WORLDNESS")
    print("Average clustering coefficient:", clustering)
    print("Random network clustering:", C_random)
    print("Real network path length (largest component):", L)
    print("Random network path length:", L_random)
    print("Small-world coefficient (sigma):", sigma)

    print("==============================================================")

    return {
        "average_in_degree": avg_in_degree,
        "average_out_degree": avg_out_degree,
        "average_path_length": np.mean(distances) if distances else np.nan,
        "reachability": reachability,
        "clustering_coefficient": clustering,
        "random_clustering": C_random,
        "path_length_lcc": L,
        "random_path_length": L_random,
        "small_world_sigma": sigma
    }



def plot_adjacency_matrix(G, sort_nodes=True, cmap="jet", 
                          step=1, figsize=(2,2),show_labels=True):
    """
    Plot adjacency matrix with zeros shown as white.
    - Automatically detects 'Weight' or 'weight' edge attributes
    - Optionally sorts nodes by degree
    """
    # Optionally sort nodes by degree
    if sort_nodes:
        nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    else:
        nodes = list(G.nodes())

    # Create adjacency matrix manually to ensure correct weights
    n = len(nodes)
    A = np.zeros((n, n))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if G.has_edge(u, v):
                # Prefer 'Weight', then 'weight', default 1
                w = G[u][v].get("Weight", G[u][v].get("weight", 1))
                A[i, j] = w

    # Mask zeros so they appear white
    A_masked = np.ma.masked_where(A == 0, A)

    # Load colormap
    cmap_mod = plt.colormaps[cmap].copy()
    cmap_mod.set_bad(color="white")  # masked = white

    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=600)
    im = ax.imshow(A_masked, interpolation="nearest", cmap=cmap_mod)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Edge weight")

   # Ticks
    ax.set_xticks(np.arange(0, n, step))
    ax.set_yticks(np.arange(0, n, step))
    if show_labels:
        ax.set_xticklabels(nodes[::step], rotation=90, fontsize=4)
        ax.set_yticklabels(nodes[::step], fontsize=4)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.set_xlabel("Neurons", fontsize=6)
    ax.set_ylabel("Neurons", fontsize=6)
    ax.set_title("Adjacency Matrix - Weights", fontsize=7)

    plt.tight_layout()
    plt.show()


def average_connection_distance(G, pos):
    dist = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            dists = [np.linalg.norm(np.array(pos[node]) - np.array(pos[n]))
                     for n in neighbors if n in pos]
            dist[node] = np.mean(dists) if dists else np.nan
        else:
            dist[node] = np.nan
    return dist



def nodal_efficiency(G):
    eff = {}
    length_dict = dict(nx.all_pairs_shortest_path_length(G))
    for node in G.nodes():
        L = [length_dict[node][other]
             for other in G.nodes() if other != node and other in length_dict[node]]
        eff[node] = np.mean([1/l for l in L if l > 0]) if L else 0
    return eff


def average_connection_distance(G, pos):
    dist = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            dists = [np.linalg.norm(np.array(pos[node]) - np.array(pos[n]))
                     for n in neighbors if n in pos]
            dist[node] = np.mean(dists) if dists else np.nan
        else:
            dist[node] = np.nan
    return dist


def nodal_efficiency(G):
    eff = {}
    length_dict = dict(nx.all_pairs_shortest_path_length(G))
    for node in G.nodes():
        L = [length_dict[node][other]
             for other in G.nodes() if other != node and other in length_dict[node]]
        eff[node] = np.mean([1/l for l in L if l > 0]) if L else 0
    return eff

def participation_coefficient(G, modules):
    pc = {}
    for node in G.nodes():
        ki = G.degree(node)
        if ki == 0:
            pc[node] = 0
            continue
        sum_frac_sq = 0
        for m in set(modules.values()):
            kis = sum(1 for nbr in G.neighbors(node) if modules[nbr] == m)
            sum_frac_sq += (kis / ki) ** 2
        pc[node] = 1 - sum_frac_sq
    return pc


def metric_set(G):
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    if nx.is_connected(G):
        Gc = G
    else:
        Gc = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    metrics = {}
    degs = np.array([d for _, d in G.degree()])
    metrics['mean_degree'] = degs.mean()
    metrics['deg_var'] = degs.var()
    metrics['max_degree'] = degs.max()
    metrics['transitivity'] = nx.transitivity(G)
    metrics['triangle_count'] = sum(nx.triangles(G).values()) / 3
    metrics['assortativity'] = nx.degree_pearson_correlation_coefficient(G)
    try:
        metrics['avg_path_length'] = nx.average_shortest_path_length(Gc)
    except Exception:
        metrics['avg_path_length'] = np.nan

    # spectral radius (largest eigenvalue)
    from scipy.sparse.linalg import eigs
    A = nx.to_scipy_sparse_array(G, format='csr', dtype=float)  # updated call
    try:
        vals, _ = eigs(A, k=1, which='LR')
        metrics['spectral_radius'] = np.real(vals[0])
    except Exception:
        metrics['spectral_radius'] = np.nan

    return metrics




import networkx as nx
import random


def directed_edge_swap(G, nswap, max_tries, seed=None):
    """Perform directed edge swaps preserving in- and out-degree."""

    rng = random.Random(seed)
    G = G.copy()

    edges = list(G.edges())
    swaps = 0
    tries = 0

    while swaps < nswap and tries < max_tries:

        tries += 1

        if len(edges) < 2:
            break

        (u, v), (x, y) = rng.sample(edges, 2)

        # Avoid self-loops and existing edges
        if u == y or x == v:
            continue

        if G.has_edge(u, y) or G.has_edge(x, v):
            continue

        # Perform swap
        G.remove_edge(u, v)
        G.remove_edge(x, y)

        G.add_edge(u, y)
        G.add_edge(x, v)

        # Update edge list
        edges.remove((u, v))
        edges.remove((x, y))
        edges.extend([(u, y), (x, v)])

        swaps += 1

    if swaps < nswap:
        print(f"Warning: performed only {swaps} of {nswap} requested swaps.")

    return G


def randomize_graph(
    G,
    n_type="undirected",
    method="config_preserving",
    nswap=10,
    swap_factor=100,
    max_tries_factor=10,
    max_tries=10000,
    seed=None
):

    if n_type not in ["directed", "undirected"]:
        raise ValueError("n_type must be 'directed' or 'undirected'")

    # Convert input graph
    if n_type == "directed":
        G0 = nx.DiGraph(G)
    else:
        G0 = nx.Graph(G)

    G0.remove_edges_from(nx.selfloop_edges(G0))

    n = G0.number_of_nodes()
    E = G0.number_of_edges()

    if method == "config_preserving":

        nswap = swap_factor * E
        max_tries = max(nswap * max_tries_factor, nswap + 1000)

        if n_type == "undirected":
            G_rand = G0.copy()
            nx.double_edge_swap(
                G_rand,
                nswap=nswap,
                max_tries=max_tries,
                seed=seed
            )

        else:
            G_rand = directed_edge_swap(
                G0,
                nswap=nswap,
                max_tries=max_tries,
                seed=seed
            )

    elif method == "erdos_renyi":

        if n_type == "undirected":
            p = E / (n * (n - 1) / 2) if n > 1 else 0
            G_rand = nx.gnp_random_graph(n, p, seed=seed)

        else:
            p = E / (n * (n - 1)) if n > 1 else 0
            G_rand = nx.gnp_random_graph(
                n, p, directed=True, seed=seed
            )

        G_rand = nx.relabel_nodes(
            G_rand, dict(enumerate(G0.nodes()))
        )

    else:
        raise ValueError(
            "method must be 'config_preserving' or 'erdos_renyi'"
        )

    return G_rand



def rich_club(G, method="config_preserving", k_max=None,
              nswap=10, max_tries=100000, n_rand=100,
              plot_rand_stats=False):

    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    G_und = G.to_undirected()

    # real RC
    rc_real = nx.rich_club_coefficient(G_und, normalized=False, Q=None)

    if k_max is None:
        k_max = max(dict(G_und.degree()).values())

    # storage
    rc_rand_list = []
    deg_counts_rand = []    # number of nodes with degree > k
    mean_deg_rand = []       # variance of degrees above k

    # randomize
    for _ in range(n_rand):

        G_rand = randomize_graph(G_und, method=method, nswap=nswap, max_tries=max_tries)

        if G_rand.is_directed() or G_rand.is_multigraph():
            G_rand = nx.Graph(G_rand)

        G_rand.remove_edges_from(nx.selfloop_edges(G_rand))

        # RC
        rc_rand = nx.rich_club_coefficient(G_rand, normalized=False, Q=None)
        rc_rand_list.append(rc_rand)

        # degrees
        degs_r = dict(G_rand.degree())

        # vector of node counts > k
        count_vec = np.zeros(k_max + 1)
        var_vec = np.zeros(k_max + 1)

        for k in range(k_max + 1):
            # nodes above threshold
            above_k = [d for d in degs_r.values() if d > k]

            # how many nodes above k?
            count_vec[k] = len(above_k)

            # variance in degree among these nodes
            if len(above_k) > 1:
                var_vec[k] = np.mean(above_k)
            else:
                var_vec[k] = np.nan

        deg_counts_rand.append(count_vec)
        mean_deg_rand.append(var_vec)

    # RC mean and std
    rc_rand_mean = {}
    rc_rand_std = {}

    for k in range(k_max + 1):
        values = [rc[k] for rc in rc_rand_list if k in rc]
        rc_rand_mean[k] = np.mean(values) if values else np.nan
        rc_rand_std[k] = np.std(values) if values else np.nan

    rc_norm = {
        k: (rc_real.get(k, np.nan) / rc_rand_mean[k]
            if rc_rand_mean[k] and rc_rand_mean[k] > 0 else np.nan)
        for k in range(k_max + 1)
    }

    # -----------------------------
    # PLOTTING
    # -----------------------------
    if plot_rand_stats:

        ks = np.arange(k_max + 1)

        # first plot: counts of nodes with deg > k
        deg_counts_rand = np.array(deg_counts_rand)
        mean_deg = deg_counts_rand.mean(axis=0)
        std_deg = deg_counts_rand.std(axis=0)

        plt.figure(figsize=(6, 4))
        plt.plot(ks, mean_deg, label="Mean # nodes with degree > k")
        plt.fill_between(ks, mean_deg - std_deg, mean_deg + std_deg,
                         alpha=0.3, label="±1 std")
        plt.xlabel("Degree threshold k")
        plt.ylabel("# Nodes with degree > k")
        plt.title("Random Networks: Degree Abundance vs k")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.show()

        # second: rich-club stdev
        plt.figure(figsize=(6, 4))
        plt.plot(ks, list(rc_rand_std.values()), label="Std Dev of RC(k)")
        plt.xlabel("Degree threshold k")
        plt.ylabel("Std deviation of RC coefficient")
        plt.title("Random Networks: RC coefficient variability")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.show()

    return rc_real, rc_rand_mean, rc_rand_std, rc_norm



def plot_rich_club(rc_real,rc_rand_avg,rc_rand_std,rc_norm,k_vals,k_rich):
    # ======== PLOT REAL vs RANDOMIZED ========
    fig, ax = plt.subplots(dpi=300)
    ax.plot(k_vals, list(rc_real.values()), '.-', color='red', label='Real φ(k)')

    rand_vals = np.array([rc_rand_avg[k] for k in k_vals])
    rand_std = np.array([rc_rand_std[k] for k in k_vals])
    ax.plot(k_vals, rand_vals, '.-', color='grey', label='Randomized φ(k)')
    ax.fill_between(k_vals, rand_vals - rand_std, rand_vals + rand_std, color='grey', alpha=0.3, label='Rand ± STD')

    if k_rich is not None:
        ax.axvline(k_rich, color='blue', linestyle='--', label='Rich-club threshold')

    ax.set_xlabel('Degree (k)')
    ax.set_ylabel('Rich-Club Coefficient')
    ax.set_title('Rich-Club Coefficient (C. elegans)')
    ax.legend()
    ax.grid(True)
    plt.show()

    # ======== PLOT NORMALIZED φ(k) ========
    fig, ax = plt.subplots(dpi=600)
    norm_vals = np.array([rc_norm[k] for k in k_vals])
    rand_std_norm = np.array([rc_rand_std[k] / rc_rand_avg[k] if rc_rand_avg[k] > 0 else 0 for k in k_vals])

    ax.plot(k_vals, norm_vals, '.-', color='green', label='Normalized φ(k)')
    ax.fill_between(k_vals, norm_vals - rand_std_norm, norm_vals + rand_std_norm, color='green', alpha=0.3, label='Rand ± STD')

    if k_rich is not None:
        ax.axvline(k_rich, color='blue', linestyle='--', label='Rich-club threshold')

    ax.axhline(1.0, linestyle='--', color='red')
    ax.set_xlabel('Degree (k)')
    ax.set_ylabel('Normalized Rich-Club Coefficient')
    ax.set_title('Normalized Rich-Club Coefficient (C. elegans)')
    ax.legend()
    ax.grid(True)
    plt.show()

def plot_weight_distribution(G, bins=50, fit_powerlaw=True):

    # Extract all weights (skip zeros and missing)
    weights = np.array([
        d.get("Weight", d.get("weight"))
        for _, _, d in G.edges(data=True)
        if d.get("Weight", d.get("weight")) is not None
    ])
    weights = weights[weights > 0]

    if len(weights) == 0:
        print("No positive weights found.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # --- Histogram (linear scale)
    counts, bin_edges, _ = ax[0].hist(weights, bins=bins, edgecolor="black")
    ax[0].set_title("Synaptic Weight Histogram")
    ax[0].set_xlabel("Weight")
    ax[0].set_ylabel("Count")

    # --- Log-log distribution using histogram counts
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    nonzero = counts > 0
    ax[1].scatter(bin_centers[nonzero], counts[nonzero], s=20, alpha=0.7)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_title("Log-Log Weight Distribution")
    ax[1].set_xlabel("Weight")
    ax[1].set_ylabel("Count")

    # --- Fit power law
    if fit_powerlaw and np.any(nonzero):
        try:
            import powerlaw
            # Fit only positive counts
            fit = powerlaw.Fit(weights, verbose=False)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            x_fit = np.linspace(xmin, weights.max(), 200)
            y_fit = x_fit ** -alpha * len(weights)  # approximate counts
            ax[1].plot(x_fit, y_fit, 'r--', label=f"Power-law fit (α={alpha:.2f})")
            ax[1].legend()
        except ImportError:
            # fallback: linear fit in log-log space using histogram
            log_x = np.log10(bin_centers[nonzero])
            log_y = np.log10(counts[nonzero])
            coeff = np.polyfit(log_x, log_y, 1)
            y_fit = 10 ** (coeff[1] + coeff[0] * log_x)
            ax[1].plot(bin_centers[nonzero], y_fit, 'r--', label=f"Log-log fit slope={coeff[0]:.2f}")
            ax[1].legend()

    plt.tight_layout()
    plt.show()


# def plot_network(G, node_size=300, edge_width_factor=2, cmap="viridis",
#                  figsize=(5,5), show_labels=True):

#     # Extract weights
#     weights = np.array([
#         d.get("Weight", d.get("weight", 1))
#         for _, _, d in G.edges(data=True)
#     ])

#     # Normalize edge colors to [0,1]
#     if len(weights) == 0:
#         weights_norm = np.ones(len(G.edges()))
#     else:
#         w_min, w_max = weights.min(), weights.max()
#         weights_norm = (weights - w_min) / (w_max - w_min + 1e-9)

#     # Layout
#     pos = nx.spring_layout(G, seed=42)

#     plt.figure(figsize=figsize)

#     # Draw nodes
#     nx.draw_networkx_nodes(
#         G, pos,
#         node_size=node_size,
#         node_color="skyblue",
#         edgecolors="black",
#         linewidths=1.5,
#         alpha=0.9
#     )

#     # Draw edges with weight scaling
#     nx.draw_networkx_edges(
#         G, pos,
#         width=weights * edge_width_factor,
#         edge_color=weights_norm,
#         edge_cmap=plt.get_cmap(cmap),
#         alpha=0.7,
#         arrows=True,
#         connectionstyle='arc3,rad=0.1'
#     )

#     # Draw labels only if requested
#     if show_labels:
#         nx.draw_networkx_labels(
#             G, pos,
#             labels={n: str(n) for n in G.nodes()},
#             font_size=8,
#             font_color="black",
#             font_weight="bold"
#         )

#     plt.axis("off")
#     plt.title("Network Graph (weighted)")
#     plt.show()

def plot_network(
    G,
    node_size=300,
    cmap="viridis",
    figsize=(5,5),
    show_labels=True,
    edge_width_factor=(0.5, 5),   # (min_width, max_width)
    scaling="auto"                # "auto" or (min_weight, max_weight)
):
    """
    Plot weighted network with:
    - edge_width_factor = (min_width, max_width)
    - scaling = "auto" OR (w_low, w_high) to map weights consistently across graphs.
    - Legend for edge widths in lower-right corner.
    """

    # ---------------------------
    # Extract weights
    # ---------------------------
    weights = np.array([
        d.get("Weight", d.get("weight", 1.0))
        for _, _, d in G.edges(data=True)
    ])

    if len(weights) == 0:
        print("Warning: Graph has no edges.")
        weights = np.array([1.0])

    # Weight domain for normalization
    if scaling == "auto":
        w_low, w_high = weights.min(), weights.max()
    else:
        w_low, w_high = scaling

    # Clip weights to scaling range
    weights_clipped = np.clip(weights, w_low, w_high)

    # Normalize weights → [0,1]
    weights_norm = (weights_clipped - w_low) / (w_high - w_low + 1e-12)

    # ---------------------------
    # Edge width scaling
    # ---------------------------
    wmin, wmax = edge_width_factor
    edge_widths = wmin + weights_norm * (wmax - wmin)

    # ---------------------------
    # Layout
    # ---------------------------
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=figsize)

    # ---------------------------
    # Draw nodes
    # ---------------------------
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color="skyblue",
        edgecolors="black",
        linewidths=1.5,
        alpha=0.9
    )

    # ---------------------------
    # Draw edges
    # ---------------------------
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color=weights_norm,
        edge_cmap=plt.get_cmap(cmap),
        alpha=0.7,
        arrows=True,
        connectionstyle='arc3,rad=0.1'
    )

    # ---------------------------
    # Draw labels
    # ---------------------------
    if show_labels:
        nx.draw_networkx_labels(
            G, pos,
            labels={n: str(n) for n in G.nodes()},
            font_size=8,
            font_color="black",
            font_weight="bold"
        )

    # ---------------------------
    # Add width legend
    # ---------------------------
    import matplotlib.lines as mlines

    lw1 = wmin
    lw2 = (wmin + wmax) / 2
    lw3 = wmax

    legend_lines = [
        mlines.Line2D([], [], linewidth=lw1, color="black", label=f"{w_low:.2f}"),
        mlines.Line2D([], [], linewidth=lw2, color="black", label=f"{(w_low+w_high)/2:.2f}"),
        mlines.Line2D([], [], linewidth=lw3, color="black", label=f"{w_high:.2f}"),
    ]

    plt.legend(
        handles=legend_lines,
        title="Edge width = weight",
        loc="lower right",
        frameon=True,
        fontsize=8,
        title_fontsize=9
    )

    plt.axis("off")
    plt.title("Network Graph (weighted)")
    plt.show()



def random_walk_propagation(
    G_dir,
    starting_neurons,
    n_walks=1000,
    max_steps=100,
    weighted=True,
    seed=42
):

    rng = np.random.default_rng(seed)

    # Check that all starting neurons exist in the graph
    missing = [
        n for n in starting_neurons
        if n not in G_dir
    ]

    if missing:
        raise ValueError(
            f"Starting neurons not found in graph: {missing}"
        )

    if len(starting_neurons) == 0:
        raise ValueError("No starting neurons provided.")

    # Store number of unique neurons visited
    # for every walk at every time step
    results = np.zeros((n_walks, max_steps + 1))

    for walk in range(n_walks):

        # Randomly select one olfactory starting neuron
        current = rng.choice(starting_neurons)

        # Keep track of all neurons visited
        visited = {current}

        results[walk, 0] = len(visited)

        for step in range(1, max_steps + 1):

            # Get all downstream neurons
            neighbours = list(
                G_dir.successors(current)
            )

            # Stop if no downstream connections exist
            if len(neighbours) == 0:
                results[walk, step:] = len(visited)
                break

            # Determine transition probabilities
            if weighted:

                weights = np.array([
                    G_dir[current][n].get("weight", 1)
                    for n in neighbours
                ], dtype=float)

                # Ignore non-positive weights
                weights = np.maximum(weights, 0)

                if weights.sum() == 0:
                    probabilities = None
                else:
                    probabilities = weights / weights.sum()

            else:
                probabilities = None

            # Choose next neuron
            current = rng.choice(
                neighbours,
                p=probabilities
            )

            # Add neuron to visited set
            visited.add(current)

            # Record cumulative number of unique neurons
            results[walk, step] = len(visited)

    # Average number of unique neurons visited
    mean_visited = results.mean(axis=0)

    # Standard deviation across walks
    std_visited = results.std(axis=0)

    # Convert to percentage of total network
    percentage_visited = (
        mean_visited / G_dir.number_of_nodes()
    ) * 100

    return mean_visited, std_visited, percentage_visited

def calculate_network_metrics(G_dir, name):

    # Louvain community detection requires an
    # undirected graph.
    G_undir = G_dir.to_undirected()

    # Detect communities using synaptic weights.
    communities = nx.community.louvain_communities(
        G_undir,
        weight="weight",
        seed=42
    )

    # Calculate modularity.
    modularity = nx.community.modularity(
        G_undir,
        communities,
        weight="weight"
    )

    # Calculate betweenness centrality.
    # Using the directed graph preserves connection direction.
    # Here we use unweighted shortest paths.
    betweenness = nx.betweenness_centrality(
        G_dir,
        normalized=True,
        weight=None
    )

    # Identify the neurons with the highest betweenness.
    top_neurons = sorted(
        betweenness.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    print(f"\n--- {name} ---")
    print(f"Number of communities: {len(communities)}")
    print(f"Modularity: {modularity:.4f}")
    print("\nTop 10 neurons by betweenness centrality:")

    for neuron, value in top_neurons:
        print(f"{neuron}: {value:.5f}")

    return modularity, communities, betweenness

def plot_node_neighborhood(G, start_node, max_depth, 
                           node_size=300,edge_width_factor = [0.5,5], 
                           cmap="viridis", figsize=(5,5),
                           show_labels=False,scaling='auto'):
    """
    Extracts and plots the subgraph containing all nodes within
    max_depth hops from start_node.
    """

    if start_node not in G:
        raise ValueError(f"Node {start_node} not in graph.")

    # BFS to radius = max_depth
    nodes_in_radius = nx.single_source_shortest_path_length(G, start_node, cutoff=max_depth)
    nodes_in_radius = list(nodes_in_radius.keys())

    # Extract induced subgraph
    subG = G.subgraph(nodes_in_radius).copy()

    # Highlight the central node
    subG.nodes[start_node]["color"] = "red"

    # Use your existing plotter
    plot_network(
        subG,
        node_size=node_size,
        edge_width_factor = edge_width_factor,
        cmap=cmap,
        figsize=figsize,show_labels=show_labels,
        scaling=scaling
    )

    return subG


def indegree_outdegree_ratio(G, use_weights=True):

    weight = "weight" if use_weights else None

    ratios = {}
    nodes = []
    vals = []
    indi = []
    outdi = []
    for node in G.nodes():
        indeg = G.in_degree(node, weight=weight)
        outdeg = G.out_degree(node, weight=weight)
        indi.append(indeg)
        outdi.append(outdeg)
        if outdeg == 0:
            ratios[node] = float("inf") if indeg > 0 else 0.0
            
        else:
            ratios[node] = indeg / outdeg
        nodes.append(node)
        vals.append(ratios[node])
    
    return np.array(nodes), np.array(vals),np.array(outdi),np.array(indi)


def clustering_coefficient(G, node):
    neighbors = list(G.neighbors(node))
    k = len(neighbors)
    
    if k < 2:
        return 0.0
    
    # Count edges between neighbors
    links = 0
    for i in range(k):
        for j in range(i + 1, k):
            if G.has_edge(neighbors[i], neighbors[j]):
                links += 1
                
    # Clustering coefficient formula
    return (2 * links) / (k * (k - 1))

def all_clustering_coeffs(G):
    return {node: clustering_coefficient(G, node) for node in G.nodes()}

def prune_and_track_sccs_parallel(G, order="largest", n_jobs=-1, return_constituents_at=None):
    sizes1, sizes2 = [], []
    nodes = list(G.nodes())
    
    # Sort nodes by degree
    if order == "largest":
        nodes_sorted = sorted(nodes, key=lambda n: G.degree(n), reverse=True)
    elif order == "smallest":
        nodes_sorted = sorted(nodes, key=lambda n: G.degree(n))
    else:
        raise ValueError("order must be 'largest' or 'smallest'")
    
    G_copy = G.copy()
    snapshots = []
    
    # Take snapshots of node sets after each removal
    for step, node in enumerate(nodes_sorted, 1):
        G_copy.remove_node(node)
        if G_copy.number_of_nodes() == 0:
            break
        snapshots.append((step, set(G_copy.nodes())))
    
    # Run SCC computations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_scc_sizes)(G, nodes_remaining)
        for _, nodes_remaining in snapshots
    )
    
    # Collect results
    for size1, size2 in results:
        sizes1.append(size1)
        sizes2.append(size2)
    
    if return_constituents_at is not None:
        # Find the snapshot that corresponds to the requested step
        match = next(((step, nodes_remaining) for step, nodes_remaining in snapshots 
                      if step == return_constituents_at), None)
        if match is not None:
            step, nodes_remaining = match
            H = G.subgraph(nodes_remaining).copy()
            comps = sorted(nx.strongly_connected_components(H), key=len, reverse=True)
            scc1_nodes = comps[0] if len(comps) > 0 else set()
            scc2_nodes = comps[1] if len(comps) > 1 else set()
            return sizes1, sizes2, scc1_nodes, scc2_nodes
    
    return sizes1, sizes2


def _compute_scc_sizes(G, nodes_remaining):
    H = G.subgraph(nodes_remaining).copy()
    comps = sorted(nx.strongly_connected_components(H), key=len, reverse=True)
    size1 = len(comps[0]) if len(comps) > 0 else 0
    size2 = len(comps[1]) if len(comps) > 1 else 0
    return size1, size2


def small_worldness(G, n_rand=5, seed=None):
    # Helper: get largest connected component if graph is disconnected
    def largest_connected_subgraph(graph):
        if nx.is_connected(graph):
            return graph
        else:
            largest_cc = max(nx.connected_components(graph), key=len)
            return graph.subgraph(largest_cc).copy()
    
    Gc = largest_connected_subgraph(G)
    
    # Clustering coefficient and average shortest path length
    C = nx.average_clustering(Gc)
    L = nx.average_shortest_path_length(Gc)
    
    # Random graph comparison
    C_rand_list = []
    L_rand_list = []
    n = Gc.number_of_nodes()
    m = Gc.number_of_edges()
    
    for i in range(n_rand):
        G_rand = nx.gnm_random_graph(n, m, seed=(None if seed is None else seed+i))
        G_rand_c = largest_connected_subgraph(G_rand)
        C_rand_list.append(nx.average_clustering(G_rand_c))
        L_rand_list.append(nx.average_shortest_path_length(G_rand_c))
    
    C_rand = sum(C_rand_list) / n_rand
    L_rand = sum(L_rand_list) / n_rand
    
    # Small-worldness
    S = (C / C_rand) / (L / L_rand)
    return S

print("Imports are sucessfull #######################################")





