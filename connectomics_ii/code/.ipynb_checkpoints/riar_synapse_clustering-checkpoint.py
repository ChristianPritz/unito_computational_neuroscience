"""Extract and cluster RIAR synapses from Ruach et al. Dataset S1.

The MAT file contains the N2U reconstruction used to calculate contour
distances.  This is a different coordinate system/reconstruction from the
small NeuroMorpho RIAR.swc file in this directory, so the two must not be
overlaid without an explicit registration transform.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matio
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


HERE = Path(__file__).resolve().parent
MAT_PATH = HERE / "data" / "Supplementary_File1all_neurons.mat"


def silhouette_precomputed(distances, labels):
    """Mean silhouette coefficient for a precomputed distance matrix."""
    values = []
    for i, label in enumerate(labels):
        same = labels == label
        same[i] = False
        if not same.any():
            values.append(0.0)
            continue
        within = distances[i, same].mean()
        nearest_other = min(
            distances[i, labels == other].mean()
            for other in np.unique(labels)
            if other != label
        )
        values.append((nearest_other - within) / max(within, nearest_other))
    return float(np.mean(values))


def text_values(values):
    return ";".join(sorted({str(v) for v in values if pd.notna(v)}))


def main():
    neurons = matio.load_from_mat(MAT_PATH)["all_neurons"].ravel()
    names = [str(neuron["NameStr"].ravel()[0]) for neuron in neurons]
    riar = neurons[names.index("RIAR")]

    contacts = riar["all_synapses"].reset_index(drop=True).copy()
    registered = riar["synapsesObjs"].reset_index(drop=True)
    contacts[["riar_x", "riar_y", "riar_z"]] = registered[["x1", "y1", "z1"]]
    contacts["synapse_id"] = contacts["idx"].astype(int)
    contacts["site_id"] = contacts["adjMatInd"].astype(int)
    contacts = contacts.rename(
        columns={"x": "density_x", "y": "density_y", "z": "density_z"}
    )
    contacts.to_csv(HERE / "data" / "RIAR_synapse_contacts.csv", index=False)

    # adjMatInd identifies the registered position on RIAR. Repeated rows are
    # partner-specific copies of polyadic synapses and have zero contour distance.
    groups = contacts.groupby("site_id", sort=False)
    sites = groups.first().reset_index()
    sites["synapse_ids"] = groups["synapse_id"].apply(text_values).to_numpy()
    sites["directions"] = groups["dierection"].apply(text_values).to_numpy()
    sites["presynaptic_neurons"] = groups["pre"].apply(text_values).to_numpy()
    sites["postsynaptic_neurons"] = groups["post"].apply(text_values).to_numpy()

    first_rows = np.flatnonzero(~contacts["site_id"].duplicated())
    contour = np.asarray(riar["dist_map"], dtype=float)[np.ix_(first_rows, first_rows)]
    if not np.allclose(contour, contour.T) or not np.allclose(np.diag(contour), 0):
        raise ValueError("RIAR contour-distance matrix is not a valid symmetric matrix")

    tree = linkage(squareform(contour, checks=False), method="average")
    scores = {}
    candidate_labels = {}
    for n_clusters in range(2, min(13, len(sites))):
        labels = fcluster(tree, n_clusters, criterion="maxclust")
        candidate_labels[n_clusters] = labels
        scores[n_clusters] = silhouette_precomputed(contour, labels)
    best_k = max(scores, key=scores.get)
    sites["cluster"] = candidate_labels[best_k]
    sites.to_csv(HERE / "data" / "RIAR_synaptic_sites_clustered.csv", index=False)

    summary = (
        sites.groupby("cluster")
        .agg(
            n_sites=("site_id", "size"),
            mean_x=("riar_x", "mean"),
            mean_y=("riar_y", "mean"),
            mean_z=("riar_z", "mean"),
        )
        .reset_index()
    )
    summary["selected_k"] = best_k
    summary["silhouette"] = scores[best_k]
    summary.to_csv(HERE / "data" / "RIAR_cluster_summary.csv", index=False)

    rows = riar["rows"]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")
    for row in rows.itertuples(index=False):
        ax.plot([row.x1, row.x2], [row.y1, row.y2], [row.z1, row.z2], color="0.75", lw=0.7)
    scatter = ax.scatter(
        sites["riar_x"], sites["riar_y"], sites["riar_z"],
        c=sites["cluster"], cmap="tab10", s=24, depthshade=False,
    )
    ax.set(xlabel="x (µm)", ylabel="y (µm)", zlabel="z (µm)")
    ax.set_title(f"RIAR synaptic sites: contour-distance clustering (k={best_k})")
    ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "data" / "RIAR_synaptic_clusters.png", dpi=180)
    plt.close(fig)

    print(f"RIAR: {len(contacts)} contact rows, {len(sites)} unique registered sites")
    print(f"Selected {best_k} clusters; silhouette={scores[best_k]:.3f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
