import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

def Cal_Spatial_Net(adata, radius=None, n_neighbors=None, model='KNN', verbose=True, include_self=False):
    """
    Construct spatial neighbor graph from spatial coordinates.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing cell-level spatial coordinates in `adata.obsm['spatial']`.
    radius : float, optional
        Radius for neighborhood search (used when `model='Radius'`).
    n_neighbors : int, optional
        Number of neighbors (used when `model='KNN'`).
    model : {'KNN', 'Radius'}, default='KNN'
        Type of graph construction method.
    verbose : bool, default=True
        If True, print summary of constructed graph.
    include_self : bool, default=False
        Whether to include self-loops in the adjacency matrix.

    Returns
    -------
    None
        The adjacency matrix is stored in `adata.uns['adj']`, and edges in `adata.uns['edgeList']`.
    """
    spatial = adata.obsm['spatial']
    if model == 'KNN':
        adata.uns['adj'] = kneighbors_graph(spatial, n_neighbors=n_neighbors, mode='connectivity', include_self=include_self)
    elif model == 'Radius':
        adata.uns['adj'] = radius_neighbors_graph(spatial, radius=radius, mode='connectivity', include_self=include_self)

    edgeList = np.nonzero(adata.uns['adj'])
    adata.uns['edgeList'] = np.array([edgeList[0], edgeList[1]])

    if verbose:
        print('The graph contains %d edges, %d cells.' % (adata.uns['edgeList'].shape[1], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (adata.uns['edgeList'].shape[1] / adata.n_obs))
