import os
import random
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from torch.backends import cudnn
from harmony import harmonize
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


def set_seed(seed=2024):
    """
    Set random seed for reproducibility across Python, NumPy, PyTorch, and CUDA.

    Parameters
    ----------
    seed : int, default=2024
        Random seed value.

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def harmony(adata, feature_labels, batch_labels, use_gpu=True):
    """
    Perform batch correction using Harmony.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing features in `.obsm`.
    feature_labels : str
        Key in `adata.obsm` for feature representation.
    batch_labels : str
        Key in `adata.obs` for batch labels.
    use_gpu : bool, default=True
        Whether to use GPU acceleration.

    Returns
    -------
    None
        Updates `adata.obsm` with corrected representation: `{feature_labels}_harmony`.
    """
    df_batches = pd.DataFrame(np.reshape(adata.obs[batch_labels], (-1, 1)), columns=['batch'])
    bc_latent = harmonize(
        adata.obsm[feature_labels],
        df_batches,
        batch_key="batch",
        use_gpu=use_gpu,
        verbose=True,
        max_iter_harmony=20,
        theta=10,
    )
    adata.obsm[f"{feature_labels}_harmony"] = bc_latent


def pca(adata, use_reps=None, n_comps=10):
    """
    Perform dimensionality reduction using PCA.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object.
    use_reps : str, optional
        Key in `adata.obsm` to use as input features.
        If None, use `adata.X`.
    n_comps : int, default=10
        Number of principal components.

    Returns
    -------
    np.ndarray
        PCA-reduced features of shape [n_samples, n_comps].
    """
    from sklearn.decomposition import PCA
    from scipy.sparse import csc_matrix, csr_matrix

    pca = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, (csc_matrix, csr_matrix)):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    return feat_pca


def mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=2020):
    """
    Perform clustering using R package `mclust`.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing representation in `.obsm`.
    num_cluster : int
        Number of clusters.
    modelNames : str, default="EEE"
        Model type in `mclust`.
    used_obsm : str, default="emb_pca"
        Key in `adata.obsm` to use for clustering.
    random_seed : int, default=2020
        Random seed for reproducibility.

    Returns
    -------
    adata : anndata.AnnData
        Updated AnnData with `adata.obs['mclust']`.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri

    robjects.r.library("mclust")
    rpy2.robjects.numpy2ri.activate()

    r_random_seed = robjects.r["set.seed"]
    r_random_seed(random_seed)
    rmclust = robjects.r["Mclust"]

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(
    adata,
    n_clusters=7,
    key="emb",
    add_key="SMART",
    method="SMART",
    start=0.1,
    end=3.0,
    increment=0.01,
    use_pca=False,
    n_comps=20,
):
    """
    Perform clustering on latent representations with multiple supported methods.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object of scanpy.
    n_clusters : int, default=7
        Number of clusters.
    key : str, default="emb"
        Key of input representation in `adata.obsm`.
    add_key : str, default="SMART"
        Key to store clustering results in `adata.obs`.
    method : str, default="SMART"
        Clustering method. Options: ["mclust", "leiden", "louvain", "gmm", "kmeans"].
    start : float, default=0.1
        Start resolution for search (used in leiden/louvain).
    end : float, default=3.0
        End resolution for search (used in leiden/louvain).
    increment : float, default=0.01
        Step size for resolution search.
    use_pca : bool, default=False
        Whether to reduce dimensions using PCA.
    n_comps : int, default=20
        Number of components for PCA if `use_pca=True`.

    Returns
    -------
    None
        Updates `adata.obs[add_key]` with clustering results.
    """
    if use_pca:
        adata.obsm[key + "_pca"] = pca(adata, use_reps=key, n_comps=n_comps)

    if method == "mclust":
        adata = mclust_R(adata, used_obsm=(key + "_pca" if use_pca else key), num_cluster=n_clusters)
        adata.obs[add_key] = adata.obs["mclust"]

    elif method in ["leiden", "louvain"]:
        res = search_res(adata, n_clusters, use_rep=(key + "_pca" if use_pca else key),
                         method=method, start=start, end=end, increment=increment)
        if method == "leiden":
            sc.tl.leiden(adata, random_state=0, resolution=res)
            adata.obs[add_key] = adata.obs["leiden"]
        else:
            sc.tl.louvain(adata, random_state=0, resolution=res)
            adata.obs[add_key] = adata.obs["louvain"]

    elif method == "gmm":
        X = adata.obsm[key + "_pca"] if use_pca else adata.obsm[key]
        gmm = GaussianMixture(n_components=n_clusters, random_state=0)
        adata.obs[add_key] = gmm.fit_predict(X).astype("category")

    elif method == "kmeans":
        X = adata.obsm[key + "_pca"] if use_pca else adata.obsm[key]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        adata.obs[add_key] = kmeans.fit_predict(X).astype("category")

    else:
        raise ValueError("Clustering method must be one of ['mclust', 'leiden', 'louvain', 'gmm', 'kmeans'].")


def search_res(adata, n_clusters, method="leiden", use_rep="emb", start=0.1, end=3.0, increment=0.01):
    """
    Search for resolution value that yields the desired number of clusters.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object.
    n_clusters : int
        Target number of clusters.
    method : str, default="leiden"
        Clustering method. Options: ["leiden", "louvain"].
    use_rep : str, default="emb"
        Representation key for clustering.
    start : float, default=0.1
        Start resolution.
    end : float, default=3.0
        End resolution.
    increment : float, default=0.01
        Resolution step size.

    Returns
    -------
    res : float
        Resolution value that yields `n_clusters` clusters.
    """
    print("Searching resolution...")
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)

    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == "leiden":
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = adata.obs["leiden"].nunique()
        elif method == "louvain":
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = adata.obs["louvain"].nunique()
        print(f"resolution={res}, cluster number={count_unique}")
        if count_unique == n_clusters:
            return res

    raise ValueError("Resolution not found. Try a bigger range or smaller step size.")


def getcolordict(adata, my_cluster, true_cluster, colordict):
    """
    Map predicted clusters to true clusters using color dictionary.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with clustering results.
    my_cluster : str
        Column name of predicted clusters in `adata.obs`.
    true_cluster : str
        Column name of true clusters in `adata.obs`.
    colordict : dict
        Dictionary mapping true clusters to colors.

    Returns
    -------
    dict
        Mapping from predicted cluster IDs to colors.
    """
    v = adata.obs[[my_cluster, true_cluster]].value_counts()
    colordict1 = {}
    for a in v.index:
        if a[0] not in colordict1.keys() and colordict[a[1]] not in colordict1.values():
            colordict1[a[0]] = colordict[a[1]]

    for a in adata.obs[my_cluster].unique():
        if a not in colordict1.keys():
            print(a)
            for b in colordict.values():
                if b not in colordict1.values():
                    colordict1[a] = b
    return colordict1
