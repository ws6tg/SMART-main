# SMART: Spatial multi-omic aggregation using graph neural networks and metric learning
This repository contains SMART model and jupyter notebooks essential for reproducing the benchmarking outcomes shown in the paper.
![图片2](https://github.com/user-attachments/assets/eee3641a-ba5c-4453-9862-442916f443a5)


## Overview
Spatial multi-omics enables the exploration of tissue microenvironments and heterogeneity from the perspective of different omics modalities across distinct spatial domains within tissues. To jointly analyze the spatial multi-omics data, computational methods are desired to integrate multiple omics with spatial information into a unified space. Here, we present SMART (Spatial Multi-omic Aggregation using gRaph neural networks and meTric learning), a computational framework for spatial multi-omic integration. SMART leverages a modality-independent modular and stacking framework with spatial coordinates and adjusts the aggregation using triplet relationships. SMART excels at accurately identifying spatial regions of anatomical structures, compatible with spatial datasets of any type and number of omics layers, while demonstrating exceptional computational efficiency and scalability on large datasets. Moreover, a variant of SMART, SMART-MS, expands its capabilities to integrate spatial multi-omics data across multiple tissue sections. In summary, SMART provides a versatile, efficient, and scalable solution for integrating spatial multi-omics data.

## Requirements
- python==3.9
- muon==0.1.6
- rpy2==3.5.12
- scanpy==1.10.2
- scikit-learn==1.5.1
- torch==2.4.1
- torch_geometric==2.3.0
- anndata==0.10.8
- matplotlib==3.9.2
- tqdm==4.66.5
- numba==0.60.0
- R==4.3.0

## Datasets
The 10X Visium Human Lymph Node data can be accessed from the Gene Expression Omnibus (GEO) with accession code GSE263617 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263617). The MISAR-seq mouse brain data can be accessed from National Genomics Data Center with accession number OEP003285 (https://www.biosino.org/node/project/detail/OEP003285). The Stereo-CITE-seq data can be accessed from BGI STOmics Cloud (https://cloud.stomics.tech/). The spatial CUT&Tag–RNA-seq and spatial ATAC–RNA-seq mouse brain data can be accessed at GEO with accession code GSE205055 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055) or UCSC Cell and Genome Browser (https://brain-spatial-omics.cells.ucsc.edu). The STARmap and RIBOmap mouse brain data can be accessed from Zenodo (https://zenodo.org/record/8041114) or Single Cell Portal (SCP) (https://singlecell.broadinstitute.org/single_cell/study/SCP1835). The SPOTS mouse spleen data can be accessed at GEO with accession code GSE198353 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198353). The 10X Visium Human Tonsil data can be accessed from https://zenodo.org/records/12654113/preview/data_imputation.zip?include_deleted=0#tree_item0.
