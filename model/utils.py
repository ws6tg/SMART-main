import os
import random

import numba as nb
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import kneighbors_graph,radius_neighbors_graph
import scanpy as sc
from sklearn.metrics.pairwise import pairwise_distances
from torch.backends import cudnn
from tqdm import tqdm


def Cal_Spatial_Net(adata, radius=None, n_neighbors=None,model='KNN', verbose=True):
    spatial=adata.obsm['spatial']
    if model=='KNN':
        adata.uns['adj']=kneighbors_graph(spatial,n_neighbors=n_neighbors,mode='connectivity')
    elif model=='Radius':
        adata.uns['adj']=radius_neighbors_graph(spatial,radius=radius,mode='connectivity')
    edgeList=np.nonzero(adata.uns['adj'])
    adata.uns['edgeList'] = np.array([edgeList[0],edgeList[1]])
    if verbose:
        print('The graph contains %d edges, %d cells.' % (adata.uns['edgeList'].shape[1], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (adata.uns['edgeList'].shape[1] / adata.n_obs))
        


@nb.njit('int32[:,::1](float32[:,::1])', parallel=True)
def fastSort32(a):
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i,:] = np.argsort(a[i,:])
    return b
    
@nb.njit('int32[:,::1](float64[:,::1])', parallel=True)
def fastSort64(a):
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i,:] = np.argsort(a[i,:])
    return b
    
def Mutual_Nearest_Neighbors(adata, key=None, n_nearest_neighbors=1, farthest_ratio=0.5):
    anchors = []
    positives = []
    negatives = []
    
    l = adata.shape[0]    
    if key is None:
        X=adata.X
        
    else:
        X=adata.obsm[key]
        
    distances = pairwise_distances(X)
    
    same_count=(distances==0).sum(axis=1)    
    print(f'distances calculation completed!')
    
    nearest_neighbors_index=[]
    farthest_neighbors_index=[]
    
    #sorted_neighbors_index=np.argsort(distances, axis=1)
    if distances.dtype=="float64":
        sorted_neighbors_index=fastSort64(distances)
    elif distances.dtype=="float32":
        sorted_neighbors_index=fastSort32(distances)
        
    for i,j in enumerate(same_count):
        nearest_neighbors_index.append(sorted_neighbors_index[i, j:j + n_nearest_neighbors])
        farthest_neighbors_index.append(sorted_neighbors_index[i, np.random.choice(np.arange(-int((l-j) * farthest_ratio), 0), n_nearest_neighbors**2)])

    for i, (nearest_neighbors, farthest_neighbors) in enumerate(zip(nearest_neighbors_index, farthest_neighbors_index)):
        if sum(X[i])!=0:
            for j in nearest_neighbors:
                if i in nearest_neighbors_index[j]:
                    anchors.append(i)
                    positives.append(j)
                    negatives.append(np.random.choice(farthest_neighbors, 1)[0])
    
    print(f'The data use feature \'{key if key else "X"}\' contains {len(anchors)} mnn_anchors')

    return anchors, positives, negatives


def pca(adata, use_reps=None, n_comps=10):
    """Dimension reduction with PCA algorithm"""

    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    return feat_pca

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(adata, n_clusters=7, key='emb', add_key='SpatialGlue', method='mclust', start=0.1, end=3.0,
               increment=0.01, use_pca=False, n_comps=20):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'.
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """

    if use_pca:
        adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)

    if method == 'mclust':
        if use_pca:
            adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
        else:
            adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
        adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
        if use_pca:
            res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end,
                             increment=increment)
        else:
            res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
        if use_pca:
            res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end,
                             increment=increment)
        else:
            res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['louvain']


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res
    
def getcolordict(adata,my_cluster,true_cluster,colordict):
    v=adata.obs[[my_cluster,true_cluster]].value_counts()
    colordict1={}
    for a in v.index:
        if a[0] not in colordict1.keys() and colordict[a[1]] not in colordict1.values():
            colordict1[a[0]]=colordict[a[1]]
    for a in adata.obs[my_cluster].unique():
        if a not in colordict1.keys():
            print(a)
            for b in colordict.values():
                if b not in colordict1.values():
                    colordict1[a]=b
    return colordict1
    
def fix_seed(seed):
    # seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'