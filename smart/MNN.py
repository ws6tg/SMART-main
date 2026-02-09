import os
import random
import numba as nb
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from torch.backends import cudnn
from tqdm import tqdm


@nb.njit('int32[:,::1](float32[:,::1])', parallel=True)
def fastSort32(a):
    """
    Perform fast argsort for float32 arrays using Numba parallelization.

    Parameters
    ----------
    a : np.ndarray (float32)
        2D array of distances.

    Returns
    -------
    b : np.ndarray (int32)
        Indices that would sort each row of `a`.
    """
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i, :] = np.argsort(a[i, :])
    return b


@nb.njit('int32[:,::1](float64[:,::1])', parallel=True)
def fastSort64(a):
    """
    Perform fast argsort for float64 arrays using Numba parallelization.

    Parameters
    ----------
    a : np.ndarray (float64)
        2D array of distances.

    Returns
    -------
    b : np.ndarray (int32)
        Indices that would sort each row of `a`.
    """
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i, :] = np.argsort(a[i, :])
    return b


def Mutual_Nearest_Neighbors(adata, key=None, n_nearest_neighbors=1, farthest_ratio=0.5, max_samples=20000):
    """
    Find mutual nearest neighbors (MNNs) and construct triplets with optional sampling.

    Parameters
    ----------
    adata : AnnData
        Input dataset.
    key : str, optional
        Key in `adata.obsm` to use as features (default: use `adata.X`).
    n_nearest_neighbors : int, default=1
        Number of nearest neighbors to consider.
    farthest_ratio : float, default=0.5
        Fraction of farthest neighbors to consider when sampling negatives.
    max_samples : int, default=20000
        Maximum number of cells to process. If dataset is larger, random sampling is applied.

    Returns
    -------
    anchors : list[int]
        Indices of anchor points in original `adata`.
    positives : list[int]
        Indices of positive samples (MNNs).
    negatives : list[int]
        Indices of negative samples (randomly sampled farthest neighbors).
    """
    original_indices = np.arange(adata.shape[0])  # Store original indices
    l = adata.shape[0]

    # Apply sampling if dataset is too large
    if l > max_samples:
        print(f"Dataset size {l} exceeds max_samples {max_samples}, performing random sampling...")
        np.random.seed(42)
        sample_idx = np.random.choice(l, max_samples, replace=False)
        adata_sampled = adata[sample_idx].copy()
        original_indices = original_indices[sample_idx]
        l = max_samples
        print(f"Working with sampled {l} cells")
    else:
        adata_sampled = adata.copy()

    X = adata_sampled.X if key is None else adata_sampled.obsm[key]
    distances = pairwise_distances(X)
    same_count = (distances == 0).sum(axis=1)
    print('Distances calculation completed!')

    nearest_neighbors_index = []
    farthest_neighbors_index = []

    if distances.dtype == "float64":
        sorted_neighbors_index = fastSort64(distances)
    elif distances.dtype == "float32":
        sorted_neighbors_index = fastSort32(distances)

    for i, j in enumerate(same_count):
        nearest_neighbors_index.append(sorted_neighbors_index[i, j:j + n_nearest_neighbors])
        farthest_neighbors_index.append(
            sorted_neighbors_index[i, np.random.choice(
                np.arange(-int((l - j) * farthest_ratio), 0),
                n_nearest_neighbors ** 2
            )]
        )

    nn_dict = {i: set(nearest_neighbors_index[i]) for i in range(l)}

    anchors, positives, negatives = [], [], []
    for i, (nearest_neighbors, farthest_neighbors) in enumerate(zip(nearest_neighbors_index, farthest_neighbors_index)):
        if not np.all(X[i] == 0):
            for j in nearest_neighbors:
                if i in nn_dict[j]:
                    anchors.append(original_indices[i])
                    positives.append(original_indices[j])
                    negatives.append(original_indices[np.random.choice(farthest_neighbors, 1)[0]])

    print(f"The data using feature '{key if key else 'X'}' contains {len(anchors)} mnn_anchors")
    return anchors, positives, negatives


def MMN_batch(X, batches, far_frac=0.6, top_k=2, random_state=None, verbose=True):
    """
    Compute mutual nearest neighbors across batches and construct triplets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_cells, n_features).
    batches : np.ndarray
        Batch labels of length `n_cells`.
    far_frac : float, default=0.6
        Fraction of farthest same-batch neighbors to consider as negatives.
    top_k : int, default=2
        Number of nearest neighbors to consider between batches.
    random_state : int, optional
        Seed for reproducibility.
    verbose : bool, default=True
        Whether to print progress.

    Returns
    -------
    anchors : np.ndarray
        Anchor indices.
    positives : np.ndarray
        Positive (MNN) indices.
    negatives : np.ndarray
        Negative sample indices.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    X = np.asarray(X)
    batches = np.asarray(batches)

    anchors, positives, negatives = [], [], []
    unique_batches = np.unique(batches)
    batch_to_indices = {b: np.where(batches == b)[0] for b in unique_batches}

    for batch_a, batch_b in combinations(unique_batches, 2):
        idx_a = batch_to_indices[batch_a]
        idx_b = batch_to_indices[batch_b]
        Xa, Xb = X[idx_a], X[idx_b]

        # Nearest neighbor search (a -> b)
        nn_ab = NearestNeighbors(n_neighbors=top_k).fit(Xb)
        _, indices_ab = nn_ab.kneighbors(Xa)
        a2b = {idx_a[i]: set(idx_b[j] for j in indices_ab[i]) for i in range(len(idx_a))}

        # Nearest neighbor search (b -> a)
        nn_ba = NearestNeighbors(n_neighbors=top_k).fit(Xa)
        _, indices_ba = nn_ba.kneighbors(Xb)
        b2a = {idx_b[i]: set(idx_a[j] for j in indices_ba[i]) for i in range(len(idx_b))}

        # Build MNN pairs
        mnn_pairs = [(a, b) for a in a2b for b in a2b[a] if a in b2a.get(b, set())]

        # Construct triplets
        for anchor_idx, pos_idx in mnn_pairs:
            same_batch_idxs = batch_to_indices[batches[anchor_idx]]
            same_batch_idxs = same_batch_idxs[same_batch_idxs != anchor_idx]
            if same_batch_idxs.size == 0:
                continue

            dists = np.linalg.norm(X[same_batch_idxs] - X[anchor_idx], axis=1)
            num_far = max(1, int(len(dists) * far_frac))
            far_indices = same_batch_idxs[np.argpartition(dists, -num_far)[-num_far:]]
            neg_idx = np.random.choice(far_indices)

            anchors.append(anchor_idx)
            positives.append(pos_idx)
            negatives.append(neg_idx)

        if verbose:
            print(f"Batch {batch_a} vs {batch_b}: {len(mnn_pairs)} MNN (top-{top_k}) triplets")

    return np.array(anchors), np.array(positives), np.array(negatives)