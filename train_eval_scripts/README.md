# SMART Training and Evaluation

Follow the **Environment Setup Guide**  in [SMART-main/README.md at main · ws6tg/SMART-main](https://github.com/ws6tg/SMART-main/blob/main/README.md). The script `SMART_train_eval.py` integrates both training and evaluation workflows for multi-omics spatial data.

## Dataset Requirements

The dataset folder must contain:

- `adata_RNA.h5ad` – RNA modality data
- `adata_ADT.h5ad` – Protein/ADT modality data
- `adata_ATAC.h5ad` – ATAC modality data
- `anno.txt` – Cell type annotations

Specify the dataset folder using the `--data_dir` argument.

------

## Usage

Run the script:

```bash
python SMART_train_eval.py --data_dir /path/to/dataset
```

The script will:

1. Load and preprocess RNA, Protein, and ATAC data
2. Extract features via PCA
3. Build spatial neighbor graphs
4. Compute Mutual Nearest Neighbor (MNN) triplets
5. Train the SMART model
6. Perform clustering and calculate ARI (Adjusted Rand Index)
7. Generate UMAP and spatial embeddings
8. Save processed data and figures in the dataset folder

------

## Optional Parameters

You can adjust the training configuration using the following command-line options:

| Argument        | Type  | Default | Description                 |
| --------------- | ----- | ------- | --------------------------- |
| `--emb_dim`     | int   | 64      | Embedding dimension         |
| `--n_epochs`    | int   | 300     | Number of training epochs   |
| `--lr`          | float | 0.001   | Learning rate               |
| `--window_size` | int   | 10      | Triplet window size for MNN |

Example with custom parameters:

```bash
python SMART_train_eval.py --data_dir ~/SMART/SMART_data/simulated_data \
                           --emb_dim 128 \
                           --n_epochs 500 \
                           --lr 0.0005 \
                           --window_size 15
```

------

## Output

After execution, the following files are saved in the dataset folder:

- `SMART_training_processed.h5ad` – Processed AnnData object with SMART embeddings
- `SMART_training_result.png` – UMAP and spatial embeddings visualization
- ARI score is printed to the console.