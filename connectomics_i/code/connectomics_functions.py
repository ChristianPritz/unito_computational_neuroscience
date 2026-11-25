import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import IFrame, display,HTML
import tempfile, webbrowser, os
from itertools import permutations
import csv, joblib
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


def basic_stats(
    G,
    show_degrees=False,
    n_random=10,
    random_method="config_preserving",
    seed=None
):
    """
    Calculate basic graph statistics for a neuronal network.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed neuronal connectivity graph.

    show_degrees : bool
        If True, print the in-degree and out-degree of each neuron.

    n_random : int
        Number of randomized networks used to estimate small-worldness.
        Default = 10.

    random_method : str
        Randomization method:
        - "config_preserving": Degree-preserving edge rewiring.
        - "erdos_renyi": Erdős-Rényi random graph.

        Default = "config_preserving".

    seed : int or None
        Random seed for reproducibility.

    Notes
    -----
    Clustering coefficient and small-worldness are calculated
    on the undirected version of G.

    Random networks are generated using randomize_graph().
    Random network metrics are averaged across valid realizations.
    """

    # Validate parameters
    if n_random < 1:
        raise ValueError("n_random must be at least 1.")

    if random_method not in ["config_preserving", "erdos_renyi"]:
        raise ValueError(
            "random_method must be 'config_preserving' "
            "or 'erdos_renyi'."
        )

    # Ensure that the graph is a copy
    G = G.copy()

    # ---------------------------------------------------------
    # 1. Degree statistics
    # ---------------------------------------------------------

    if show_degrees:
        for node in G.nodes():
            print(
                f"{node}: In-degree = {G.in_degree(node)}, "
                f"Out-degree = {G.out_degree(node)}"
            )

    avg_in_degree = (
        np.mean([d for _, d in G.in_degree()])
        if G.number_of_nodes() else np.nan
    )

    avg_out_degree = (
        np.mean([d for _, d in G.out_degree()])
        if G.number_of_nodes() else np.nan
    )

    # ---------------------------------------------------------
    # 2. Shortest paths and reachability
    # ---------------------------------------------------------

    distances = []
    disconnected = 0
    total = 0

    pairs = permutations(G.nodes, 2)

    for source, target in pairs:
        total += 1

        try:
            distance = nx.shortest_path_length(
                G,
                source=source,
                target=target
            )

            distances.append(distance)

        except nx.NetworkXNoPath:
            disconnected += 1

    reachability = (
        (1 - disconnected / total) * 100
        if total else 0
    )

    average_directed_path_length = (
        np.mean(distances) if distances else np.nan
    )

    # ---------------------------------------------------------
    # 3. Clustering coefficient
    # ---------------------------------------------------------

    G_undirected = G.to_undirected()

    G_undirected.remove_edges_from(
        nx.selfloop_edges(G_undirected)
    )

    clustering = nx.average_clustering(G_undirected)

    # ---------------------------------------------------------
    # 4. Small-world coefficient
    # ---------------------------------------------------------

    def largest_connected_subgraph(graph):

        if graph.number_of_nodes() == 0:
            return graph.copy()

        largest_cc = max(
            nx.connected_components(graph),
            key=len
        )

        return graph.subgraph(largest_cc).copy()

    # Use largest connected component of the real network
    G_lcc = largest_connected_subgraph(G_undirected)

    if G_lcc.number_of_nodes() > 1:

        # Real network metrics
        C_real = nx.average_clustering(G_lcc)

        L_real = nx.average_shortest_path_length(G_lcc)

        # Store metrics from randomized networks
        random_clustering = []
        random_path_length = []

        # Generate multiple randomized networks
        for i in range(n_random):

            random_seed = (
                None if seed is None else seed + i
            )

            try:

                G_random = randomize_graph(
                    G_undirected,
                    n_type="undirected",
                    method=random_method,
                    seed=random_seed
                )

                # Extract largest connected component
                G_random_lcc = largest_connected_subgraph(
                    G_random
                )

                # Path length requires at least two nodes
                if G_random_lcc.number_of_nodes() < 2:
                    continue

                # Calculate random network metrics
                C_rand = nx.average_clustering(
                    G_random_lcc
                )

                L_rand = nx.average_shortest_path_length(
                    G_random_lcc
                )

                random_clustering.append(C_rand)
                random_path_length.append(L_rand)

            except (nx.NetworkXError, nx.NetworkXException):
                continue

        # Average metrics across valid random networks
        if random_clustering and random_path_length:

            C_random = np.mean(random_clustering)
            L_random = np.mean(random_path_length)

            C_random_std = np.std(random_clustering)
            L_random_std = np.std(random_path_length)

            n_random_used = len(random_clustering)

            # Small-world coefficient
            if C_random > 0 and L_random > 0 and L_real > 0:

                gamma = C_real / C_random
                lambda_ = L_real / L_random

                sigma = gamma / lambda_

            else:
                gamma = np.nan
                lambda_ = np.nan
                sigma = np.nan

        else:

            C_random = np.nan
            L_random = np.nan

            C_random_std = np.nan
            L_random_std = np.nan

            gamma = np.nan
            lambda_ = np.nan
            sigma = np.nan

            n_random_used = 0

    else:

        C_real = np.nan
        L_real = np.nan

        C_random = np.nan
        L_random = np.nan

        C_random_std = np.nan
        L_random_std = np.nan

        gamma = np.nan
        lambda_ = np.nan
        sigma = np.nan

        n_random_used = 0

    
    # ---------------------------------------------------------
    # 5. Print results
    # ---------------------------------------------------------
    
    WIDTH = 66
    # Ah now we are nesting our own print functions, fancy, fancy
    def print_section(title):
        print("\n" + "-" * WIDTH)
        print(f"  {title}")
        print("-" * WIDTH)
    
    
    def print_stat(label, value, unit="", precision=2):
        if isinstance(value, (int, np.integer)):
            formatted = f"{value:,}"
        elif isinstance(value, (float, np.floating)):
            formatted = f"{value:,.{precision}f}"
        else:
            formatted = str(value)
    
        print(f"  {label:<43} {formatted:>16}{unit}")
    
    
    print("\n" + "=" * WIDTH)
    print(f"{'NETWORK STATISTICS':^{WIDTH}}")
    print("=" * WIDTH)
    
    print_stat("Number of neurons:", G.number_of_nodes())
    print_stat("Total number of connections:", G.number_of_edges())
    
    
    # DEGREE STATISTICS
    print_section("DEGREE STATISTICS")
    
    print_stat("Average in-degree:", avg_in_degree)
    print_stat("Average out-degree:", avg_out_degree)
    
    
    # PATH LENGTH AND REACHABILITY
    print_section("PATH LENGTH AND REACHABILITY")
    
    print_stat(
        "Average directed shortest path length:",
        average_directed_path_length
    )
    
    print_stat("Reachability:", reachability, unit=" %")
    
    
    # CLUSTERING AND SMALL-WORLDNESS
    print_section("CLUSTERING AND SMALL-WORLDNESS")
    
    print_stat("Average clustering coefficient:", clustering)
    
    print_stat("Randomization method:", random_method)
    
    print_stat("Requested random networks:", n_random)
    print_stat("Valid random networks:", n_random_used)
    
    print_stat("Real network clustering (LCC):", C_real)
    print_stat("Real network path length (LCC):", L_real)
    
    # Optional statistics
    # print_stat("Random network clustering:", C_random)
    # print_stat("Random clustering SD:", C_random_std)
    # print_stat("Random network path length:", L_random)
    # print_stat("Random path length SD:", L_random_std)
    # print_stat("Gamma (clustering ratio):", gamma)
    # print_stat("Lambda (path length ratio):", lambda_)
    
    print_stat("Small-world coefficient (sigma):", sigma)
    
    print("\n" + "=" * WIDTH + "\n")


    # ---------------------------------------------------------
    # 6. Return results
    # ---------------------------------------------------------

    return {
        "average_in_degree": avg_in_degree,
        "average_out_degree": avg_out_degree,

        "average_path_length": average_directed_path_length,
        "reachability": reachability,

        "clustering_coefficient": clustering,

        "path_length_lcc": L_real,

        "random_method": random_method,
        "n_random_requested": n_random,
        "n_random_used": n_random_used,

        "random_clustering": C_random,
        "random_clustering_std": C_random_std,

        "random_path_length": L_random,
        "random_path_length_std": L_random_std,

        "gamma": gamma,
        "lambda": lambda_,

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

# Erdos renyi clones
def directed_erdos_renyi_weighted(
    G,
    seed=None,
    weight_distribution="uniform",
    weight_params=None
):
    rng = np.random.default_rng(seed)

    # Remove self-loops
    G0 = nx.DiGraph(G)
    G0.remove_edges_from(nx.selfloop_edges(G0))

    nodes = list(G0.nodes())
    N = len(nodes)
    E = G0.number_of_edges()

    # Original connection probability
    p = E / (N * (N - 1)) if N > 1 else 0

    # Generate standard directed ER topology
    G_rand = nx.gnp_random_graph(
        N, p, directed=True, seed=seed
    )

    G_rand = nx.relabel_nodes(
        G_rand, dict(enumerate(nodes))
    )

    # Default distribution parameters
    if weight_params is None:
        weight_params = {}

    edges = list(G_rand.edges())
    n_edges = len(edges)

    # Independently generate edge weights
    if weight_distribution == "uniform":
        low = weight_params.get("low", 1)
        high = weight_params.get("high", 10)

        weights = rng.uniform(low, high, n_edges)

    elif weight_distribution == "normal":
        mean = weight_params.get("mean", 5)
        std = weight_params.get("std", 1)

        weights = rng.normal(mean, std, n_edges)
        weights = np.maximum(weights, 0)

    elif weight_distribution == "lognormal":
        mean = weight_params.get("mean", 1)
        sigma = weight_params.get("sigma", 0.5)

        weights = rng.lognormal(mean, sigma, n_edges)

    elif weight_distribution == "powerlaw":
        alpha = weight_params.get("alpha", 2.5)
        xmin = weight_params.get("xmin", 1)

        weights = xmin * (
            1 + rng.pareto(alpha - 1, n_edges)
        )

    else:
        raise ValueError("Unknown weight distribution")

    # Assign weights
    for (u, v), w in zip(edges, weights):
        G_rand[u][v]["weight"] = float(w)

    return G_rand


 
def directed_fixed_edge_weighted(G, seed=None):

    rng = np.random.default_rng(seed)

    # Initialize graph
    G_rand = nx.DiGraph()
    G_rand.add_nodes_from(G.nodes(data=True))

    # Remove self-loops from consideration
    G0 = nx.DiGraph(G)
    G0.remove_edges_from(nx.selfloop_edges(G0))

    nodes = list(G0.nodes())
    n = len(nodes)
    E = G0.number_of_edges()

    # Store original weights (default = 1)
    weights = [
        data.get("weight", 1.0)
        for _, _, data in G0.edges(data=True)
    ]

    # Generate all possible directed edges (excluding self-loops)
    possible_edges = [
        (u, v)
        for u in nodes
        for v in nodes
        if u != v
    ]

    # Select exactly E edges randomly
    if E > len(possible_edges):
        raise ValueError("Too many edges for a simple directed graph.")

    selected = rng.choice(
        len(possible_edges),
        size=E,
        replace=False
    )

    random_edges = [possible_edges[i] for i in selected]

    # Shuffle original weights
    rng.shuffle(weights)

    # Add edges and redistribute weights
    for (u, v), w in zip(random_edges, weights):
        G_rand.add_edge(u, v, weight=w)

    return G_rand



def randomize_graph(
    G,
    n_type="undirected",
    method="config_preserving",
    nswap=10,
    swap_factor=100,
    max_tries_factor=10,
    max_tries=10000,
    seed=None,
    weighted=False
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

    # Store original edge weights
    if weighted:
        weights = [
            data.get("weight", 1.0)
            for _, _, data in G0.edges(data=True)
        ]

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

    # Redistribute original weights across randomized edges
    if weighted:
        if len(weights) != G_rand.number_of_edges():
            raise RuntimeError("Number of edges changed unexpectedly.")

        rng = np.random.default_rng(seed)
        rng.shuffle(weights)

        for (u, v), weight in zip(G_rand.edges(), weights):
            G_rand[u][v]["weight"] = weight

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
    cmap="viridis",
    figsize=(5, 5),
    show_labels=True,
    edge_width_factor=(0.5, 5),
    scaling="auto",
    positions=None,
    show_legend=True,
    settings=None
):
    """
    Plot a weighted directed or undirected network.

    Edge coloring:
        edge_color_mode = "weight": colors represent edge weights.
        edge_color_mode = "uniform": all edges have the same color.

    Available settings:
        node_size, node_color, node_edgecolor, node_linewidth, node_alpha
        edge_alpha, edge_curvature, edge_style, arrow_size, arrow_style
        edge_color_mode, edge_color
        label_font_size, label_font_color, label_font_weight
        title, title_font_size
    """

    # ---------------------------
    # Default settings
    # ---------------------------
    default_settings = {
        "node_size": 300,
        "node_color": "skyblue",
        "node_edgecolor": "black",
        "node_linewidth": 1.5,
        "node_alpha": 1.0,

        "edge_alpha": 1.0,
        "edge_curvature": 0.1,
        "edge_style": "solid",
        "arrow_size": 15,
        "arrow_style": "-|>",

        "edge_color_mode": "weight",
        "edge_color": "gray",

        "label_font_size": 8,
        "label_font_color": "black",
        "label_font_weight": "bold",

        "title": "Network Graph (weighted)",
        "title_font_size": 12
    }

    settings = settings or {}
    cfg = {**default_settings, **settings}

    # Validate edge color mode
    if cfg["edge_color_mode"] not in ("weight", "uniform"):
        raise ValueError(
            "edge_color_mode must be 'weight' or 'uniform'."
        )

    # ---------------------------
    # Extract weights
    # ---------------------------
    edges = list(G.edges(data=True))

    weights = np.array([
        d.get("Weight", d.get("weight", 1.0))
        for _, _, d in edges
    ], dtype=float)

    if len(weights) == 0:
        print("Warning: Graph has no edges.")
        weights = np.array([1.0])

    # ---------------------------
    # Weight scaling
    # ---------------------------
    if scaling == "auto":
        w_low, w_high = weights.min(), weights.max()
    else:
        w_low, w_high = scaling

    weights_clipped = np.clip(weights, w_low, w_high)

    if w_high == w_low:
        weights_norm = np.ones(len(weights)) * 0.5
    else:
        weights_norm = (
            (weights_clipped - w_low) / (w_high - w_low)
        )

    # ---------------------------
    # Edge width scaling
    # ---------------------------
    wmin, wmax = edge_width_factor
    edge_widths = wmin + weights_norm * (wmax - wmin)

    # ---------------------------
    # Layout
    # ---------------------------
    if positions is None:
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = positions

    plt.figure(figsize=figsize)

    # ---------------------------
    # Draw nodes
    # ---------------------------
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=cfg["node_size"],
        node_color=cfg["node_color"],
        edgecolors=cfg["node_edgecolor"],
        linewidths=cfg["node_linewidth"],
        alpha=cfg["node_alpha"]
    )

    # ---------------------------
    # Draw edges
    # ---------------------------
    if len(edges) > 0:

        edgelist = [(u, v) for u, v, _ in edges]

        if cfg["edge_color_mode"] == "weight":
            edge_colors = weights_norm
            edge_cmap = plt.get_cmap(cmap)
        else:
            edge_colors = cfg["edge_color"]
            edge_cmap = None

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist,
            width=edge_widths,
            edge_color=edge_colors,
            edge_cmap=edge_cmap,
            alpha=cfg["edge_alpha"],
            style=cfg["edge_style"],
            arrows=G.is_directed(),
            arrowsize=cfg["arrow_size"],
            arrowstyle=cfg["arrow_style"],
            connectionstyle=(
                f'arc3,rad={cfg["edge_curvature"]}'
            )
        )

    # ---------------------------
    # Draw labels
    # ---------------------------
    if show_labels:
        nx.draw_networkx_labels(
            G,
            pos,
            labels={n: str(n) for n in G.nodes()},
            font_size=cfg["label_font_size"],
            font_color=cfg["label_font_color"],
            font_weight=cfg["label_font_weight"]
        )

    # ---------------------------
    # Width legend
    # ---------------------------
    if show_legend and len(edges) > 0:
        import matplotlib.lines as mlines

        legend_lines = [
            mlines.Line2D(
                [], [], linewidth=wmin,
                color="black", label=f"{w_low:.2f}"
            ),
            mlines.Line2D(
                [], [], linewidth=(wmin + wmax) / 2,
                color="black", label=f"{(w_low + w_high) / 2:.2f}"
            ),
            mlines.Line2D(
                [], [], linewidth=wmax,
                color="black", label=f"{w_high:.2f}"
            )
        ]

        plt.legend(
            handles=legend_lines,
            title="Edge width = weight",
            loc="upper right",
            frameon=True,
            fontsize=8,
            title_fontsize=9
        )

    # ---------------------------
    # Final formatting
    # ---------------------------
    plt.axis("off")
    plt.title(cfg["title"], fontsize=cfg["title_font_size"])
    plt.tight_layout()
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





