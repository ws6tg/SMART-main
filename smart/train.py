from scipy import stats
import torch
from tqdm import tqdm
from smart.model import SMART
from smart.layer import SAGEConv_Decoder, SAGEConv_Encoder
import torch.nn.functional as F
import numpy as np


def laplacian_regularization(x, edge_index):
    """
    Compute Laplacian regularization loss.

    This loss enforces smoothness by penalizing differences between
    embeddings of adjacent nodes.

    Parameters
    ----------
    x : torch.Tensor
        Node embeddings of shape [num_nodes, emb_dim].
    edge_index : torch.LongTensor
        Graph connectivity in COO format with shape [2, num_edges].

    Returns
    -------
    torch.Tensor
        Scalar Laplacian regularization loss.
    """
    row, col = edge_index
    diff = x[row] - x[col]
    loss = (diff ** 2).sum(dim=1).mean()
    return loss


def train_SMART(
    features,
    edges,
    triplet_samples_list,
    weights=[1, 1],
    emb_dim=64,
    n_epochs=500,
    lr=0.0001,
    weight_decay=1e-5,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    window_size=20,
    slope=0.0001,
    Conv_Encoder=SAGEConv_Encoder,
    Conv_Decoder=SAGEConv_Decoder,
    margin=0.5,
    return_loss=False,
    laplacian_alpha=0,
):
    """
    Train the SMART model with reconstruction, triplet, and optional Laplacian loss.

    Parameters
    ----------
    features : list of torch.Tensor
        Node feature matrices for each modality. Each element has shape [num_nodes, in_dim].
    edges : list of torch.LongTensor
        Graph connectivity for each modality. Each element has shape [2, num_edges].
    triplet_samples_list : list of tuple
        Each tuple contains (anchors, positives, negatives) indices for triplet loss.
    weights : list of float, default=[1, 1, 1, 1]
        Loss weights in the following order:
        [triplet_loss_modality1, triplet_loss_modality2,
         reconstruction_loss_modality1, reconstruction_loss_modality2].
        - triplet_loss_modalityX: weight for triplet loss of modality X
        - reconstruction_loss_modalityX: weight for reconstruction loss of modality X
    emb_dim : int, default=64
        Dimension of shared latent embedding.
    n_epochs : int, default=500
        Number of training epochs.
    lr : float, default=0.0001
        Learning rate for Adam optimizer.
    weight_decay : float, default=1e-5
        Weight decay for optimizer.
    device : torch.device, optional
        Device to train on (default: GPU if available).
    window_size : int, default=20
        Window size for early stopping slope detection.
    slope : float, default=0.0001
        Minimum absolute slope threshold for continuing training.
    Conv_Encoder : class, default=SAGEConv_Encoder
        Graph encoder class.
    Conv_Decoder : class, default=SAGEConv_Decoder
        Graph decoder class.
    margin : float, default=0.5
        Margin for triplet loss.
    return_loss : bool, default=False
        Whether to return loss history along with trained model.
    laplacian_alpha : float, default=0
        Weight for Laplacian regularization term. Disabled if set to 0.

    Returns
    -------
    model : SMART
        Trained SMART model.
    loss_list : list of tuple, optional
        Only returned if `return_loss=True`. Contains (total_loss, tri_loss, rec_loss) per epoch.
    """
    hidden_dims = [x.shape[1] for x in features] + [emb_dim]
    model = SMART(hidden_dims=hidden_dims, device=device,
                  Conv_Encoder=Conv_Encoder, Conv_Decoder=Conv_Decoder)

    features, edges = [x.to(device) for x in features], [edge.to(device) for edge in edges]
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_list = []

    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        z, x_rec = model(features, edges)

        # Triplet loss
        triplet_loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_loss = 0
        for i, (anchors, positives, negatives) in enumerate(triplet_samples_list):
            anchor_arr = z[anchors]
            positive_arr = z[positives]
            negative_arr = z[negatives]

            tri_output = triplet_loss_fn(anchor_arr, positive_arr, negative_arr)
            w = weights[len(weights) // 2 + i]
            tri_loss += w * tri_output

        # Reconstruction loss
        rec_loss = 0
        for i, (feature, x_r) in enumerate(zip(features, x_rec)):
            rec_output = F.mse_loss(feature, x_r)
            w = weights[i]
            rec_loss += w * rec_output

        # Total loss
        loss = rec_loss + tri_loss

        # Add Laplacian regularization if enabled
        if laplacian_alpha != 0:
            loss += laplacian_alpha * laplacian_regularization(z, edges[0])

        # Early stopping based on slope of recent loss trend
        if epoch > window_size and epoch % 10 == 0:
            x_axis = np.arange(window_size)
            res1 = stats.linregress(x_axis, [i[1] for i in loss_list[-window_size:]])  # tri_loss trend
            res2 = stats.linregress(x_axis, [i[2] for i in loss_list[-window_size:]])  # rec_loss trend
            if abs(res1.slope) < slope or abs(res2.slope) < slope:
                if res1.slope != 0 and res2.slope != 0:
                    print("Early stopping: flat trend detected.")
                    break

        # Backward & optimize
        loss_list.append((loss.item(), tri_loss.item(), rec_loss.item()))
        loss.backward()
        optimizer.step()

    return model if not return_loss else (model, loss_list)

def train_SMART_MS(
    features,
    edges,
    triplet_samples_list,
    weights=[1, 1],
    emb_dim=64,
    n_epochs=500,
    lr=0.0001,
    weight_decay=1e-5,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    window_size=20,
    slope=0.0001,
    Conv_Encoder=SAGEConv_Encoder,
    Conv_Decoder=SAGEConv_Decoder,
    margin=0.5,
    return_loss=False,
    laplacian_alpha=0,
):
    """
    Train the SMART-MS model with reconstruction, triplet, and optional Laplacian loss.

    Parameters
    ----------
    features : list of torch.Tensor
        Node feature matrices for each modality. Each element has shape [num_nodes, in_dim].
    edges : list of torch.LongTensor
        Graph connectivity for each modality. Each element has shape [2, num_edges].
    triplet_samples_list : list of tuple
        Each tuple contains (anchors, positives, negatives) indices for triplet loss.
    weights : list of float, default=[1, 1, 1, 1]
        Loss weights in the following order:
        [triplet_loss_modality1, triplet_loss_modality2,
         reconstruction_loss_modality1, reconstruction_loss_modality2].
        - triplet_loss_modalityX: weight for triplet loss of modality X
        - reconstruction_loss_modalityX: weight for reconstruction loss of modality X
    emb_dim : int, default=64
        Dimension of shared latent embedding.
    n_epochs : int, default=500
        Number of training epochs.
    lr : float, default=0.0001
        Learning rate for Adam optimizer.
    weight_decay : float, default=1e-5
        Weight decay for optimizer.
    device : torch.device, optional
        Device to train on (default: GPU if available).
    window_size : int, default=20
        Window size for early stopping slope detection.
    slope : float, default=0.0001
        Minimum absolute slope threshold for continuing training.
    Conv_Encoder : class, default=SAGEConv_Encoder
        Graph encoder class.
    Conv_Decoder : class, default=SAGEConv_Decoder
        Graph decoder class.
    margin : float, default=0.5
        Margin for triplet loss.
    return_loss : bool, default=False
        Whether to return loss history along with trained model.
    laplacian_alpha : float, default=0
        Weight for Laplacian regularization term. Disabled if set to 0.

    Returns
    -------
    model : SMART
        Trained SMART model.
    loss_list : list of tuple, optional
        Only returned if `return_loss=True`. Contains (total_loss, tri_loss, rec_loss) per epoch.
    """
    hidden_dims = [x.shape[1] for x in features] + [emb_dim]
    model = SMART(hidden_dims=hidden_dims, device=device,
                  Conv_Encoder=Conv_Encoder, Conv_Decoder=Conv_Decoder)

    features, edges = [x.to(device) for x in features], [edge.to(device) for edge in edges]
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_list = []

    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        z, x_rec = model(features, edges)

        # Triplet loss
        triplet_loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_loss = 0
        for i, (anchors, positives, negatives) in enumerate(triplet_samples_list):
            anchor_arr = z[anchors]
            positive_arr = z[positives]
            negative_arr = z[negatives]

            tri_output = triplet_loss_fn(anchor_arr, positive_arr, negative_arr)
            w = weights[len(weights) // 2 + i]
            tri_loss += w * tri_output

        # Reconstruction loss
        rec_loss = 0
        for i, (feature, x_r) in enumerate(zip(features, x_rec)):
            rec_output = F.mse_loss(feature, x_r)
            w = weights[i]
            rec_loss += w * rec_output

        # Total loss
        loss = rec_loss + tri_loss

        # Add Laplacian regularization if enabled
        if laplacian_alpha != 0:
            loss += laplacian_alpha * laplacian_regularization(z, edges[0])

        # Early stopping based on slope of recent loss trend
        if epoch > window_size and epoch % 10 == 0:
            x_axis = np.arange(window_size)
            res1 = stats.linregress(x_axis, [i[1] for i in loss_list[-window_size:]])  # tri_loss trend
            res2 = stats.linregress(x_axis, [i[2] for i in loss_list[-window_size:]])  # rec_loss trend
            if abs(res1.slope) < slope or abs(res2.slope) < slope:
                if res1.slope != 0 and res2.slope != 0:
                    print("Early stopping: flat trend detected.")
                    break

        # Backward & optimize
        loss_list.append((loss.item(), tri_loss.item(), rec_loss.item()))
        loss.backward()
        optimizer.step()

    return model if not return_loss else (model, loss_list)
