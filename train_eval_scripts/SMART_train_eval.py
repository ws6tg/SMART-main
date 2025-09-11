#!/usr/bin/env python
# coding: utf-8

"""
SMART_train_eval.py

Converted from Tutorial Notebook for training and evaluation.
Usage:
    python SMART_train_eval.py --data_dir /path/to/dataset [options]
"""

import argparse
import os
import torch
import pandas as pd
import scanpy as sc
import warnings
from muon import prot as pt
from muon import atac as ac
from smart.train import train_SMART
from smart.utils import set_seed
from smart.utils import pca
from smart.utils import clustering
from smart.build_graph import Cal_Spatial_Net
from smart.MNN import Mutual_Nearest_Neighbors
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

set_seed(2024)

# -------------------------------
# 1. Parse input arguments
# -------------------------------
parser = argparse.ArgumentParser(description="SMART training and evaluation script.")
parser.add_argument("--data_dir", type=str, required=True,
                    help="Path to the dataset folder containing h5ad and annotation files.")
parser.add_argument("--emb_dim", type=int, default=64, help="Embedding dimension (default: 64)")
parser.add_argument("--n_epochs", type=int, default=300, help="Number of training epochs (default: 300)")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
parser.add_argument("--window_size", type=int, default=10, help="Triplet window size (default: 10)")
args = parser.parse_args()

data_dir = args.data_dir
emb_dim = args.emb_dim
n_epochs = args.n_epochs
lr = args.lr
window_size = args.window_size

# -------------------------------
# 2. Environment & Seed
# -------------------------------
set_seed(2024)
warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = '/public/home/cit_wlhuang/.conda/envs/smart/lib/R'

# -------------------------------
# 3. Load data
# -------------------------------
print("Loading RNA, Protein (ADT), and ATAC data...")
adata_omics1 = sc.read_h5ad(os.path.join(data_dir, 'adata_RNA.h5ad'))
adata_omics2 = sc.read_h5ad(os.path.join(data_dir, 'adata_ADT.h5ad'))
adata_omics3 = sc.read_h5ad(os.path.join(data_dir, 'adata_ATAC.h5ad'))
print("Data loaded successfully.\n")

adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()
adata_omics3.var_names_make_unique()

adata_omics1.obs["anno"] = pd.read_table(
    os.path.join(data_dir, "anno.txt"), header=None
).loc[adata_omics1.obs.index.astype("int")].values[:, 0].astype("str")

# -------------------------------
# 4. Data preprocessing
# -------------------------------
# RNA
print("Preprocessing RNA data...")
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)
adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=30)
print("RNA preprocessing done.\n")

# Protein
print("Preprocessing Protein (ADT) data...")
adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
pt.pp.clr(adata_omics2)
sc.pp.scale(adata_omics2)
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=30)
print("Protein preprocessing done.\n")

# ATAC
print("Preprocessing ATAC data...")
adata_omics3 = adata_omics3[adata_omics1.obs_names].copy()
ac.pp.tfidf(adata_omics3, scale_factor=1e4)
sc.pp.normalize_per_cell(adata_omics3, counts_per_cell_after=1e4)
sc.pp.log1p(adata_omics3)
adata_omics3.obsm['feat'] = pca(adata_omics3, n_comps=30)
print("ATAC preprocessing done.\n")

# -------------------------------
# 5. Spatial neighbour graph
# -------------------------------
print("Building spatial neighbor graphs...")
for adata in [adata_omics1, adata_omics2, adata_omics3]:
    Cal_Spatial_Net(adata, model="KNN", n_neighbors=4)
print("Spatial graphs constructed.\n")

# -------------------------------
# 6. MNN triplet samples
# -------------------------------
print("Computing Mutual Nearest Neighbor (MNN) triplets...")
adata_list = [adata_omics1, adata_omics2, adata_omics3]
x = [torch.FloatTensor(adata.obsm["feat"]).to(device) for adata in adata_list]
edges = [torch.LongTensor(adata.uns["edgeList"]).to(device) for adata in adata_list]
triplet_samples_list = [
    Mutual_Nearest_Neighbors(adata, key="feat", n_nearest_neighbors=3, farthest_ratio=0.6)
    for adata in adata_list
]
print("Triplet samples ready.\n")

# -------------------------------
# 7. Model training
# -------------------------------
print(f"Training SMART model: emb_dim={emb_dim}, n_epochs={n_epochs}, lr={lr}, window_size={window_size}...")
model = train_SMART(
    features=x,
    edges=edges,
    triplet_samples_list=triplet_samples_list,
    weights=[1, 1, 1, 1, 1, 1],
    emb_dim=emb_dim,
    n_epochs=n_epochs,
    lr=lr,
    weight_decay=1e-6,
    device=device,
    window_size=window_size,
    slope=1e-4
)
print("Model training completed.\n")

adata_omics1.obsm["SMART"] = model(x, edges)[0].cpu().detach().numpy()

# -------------------------------
# 8. Clustering and evaluation
# -------------------------------
print("Performing clustering and evaluating ARI...")
tool = 'mclust'
clustering(adata_omics1, key='SMART', add_key='SMART', n_clusters=5, method=tool, use_pca=True)
ari = adjusted_rand_score(adata_omics1.obs['anno'], adata_omics1.obs['SMART'])
print("ARI score:", ari)

# -------------------------------
# 9. Visualization
# -------------------------------
print("Generating UMAP and spatial plots...")
fig, ax_list = plt.subplots(1, 3, figsize=(10, 3))
sc.pp.neighbors(adata_omics1, use_rep='SMART', n_neighbors=10)
sc.tl.umap(adata_omics1)
sc.pl.umap(adata_omics1, color='SMART', ax=ax_list[0], title='SMART', s=60, show=False)
sc.pl.embedding(adata_omics1, basis='spatial', color='SMART', ax=ax_list[1], title='SMART', s=90, show=False)
sc.pl.embedding(adata_omics1, basis='spatial', color='anno', ax=ax_list[2], title='anno', s=90, show=False)
plt.tight_layout(w_pad=0.3)
plt.show()
print("Plots generated.\n")

# Save figure and processed data
print("Saving processed data and figure...")
fig.savefig(os.path.join(data_dir, "SMART_training_result.png"))
adata_omics1.write_h5ad(os.path.join(data_dir, "SMART_training_processed.h5ad"))
print("All outputs saved successfully.")
