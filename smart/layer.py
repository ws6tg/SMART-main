
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv, GCNConv, GPSConv, GINConv,
    SAGEConv, LGConv, GATv2Conv, GCN2Conv, GraphConv
)

class SAGEConv_Encoder(torch.nn.Module):
    """
    Encoder based on GraphSAGE convolution.

    Parameters
    ----------
    in_channels : int
        Input feature dimension.
    out_channels : int
        Output feature dimension.
    """
    def __init__(self, in_channels, out_channels):
        super(SAGEConv_Encoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels, normalize=True)
        self.conv2 = SAGEConv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, in_channels].
        edge_index : torch.LongTensor
            Graph connectivity in COO format with shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Encoded node features of shape [num_nodes, out_channels].
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class SAGEConv_Decoder(torch.nn.Module):
    """
    Decoder based on GraphSAGE convolution for reconstruction.
    """
    def __init__(self, in_channels, out_channels):
        super(SAGEConv_Decoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, in_channels, normalize=True)
        self.conv2 = SAGEConv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        """
        Forward pass of the decoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GCNConv_Encoder(torch.nn.Module):
    """
    Encoder based on GCN convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(GCNConv_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels, normalize=True)
        self.conv2 = GCNConv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        """
        Forward pass of the encoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GCNConv_Decoder(torch.nn.Module):
    """
    Decoder based on GCN convolution for reconstruction.
    """
    def __init__(self, in_channels, out_channels):
        super(GCNConv_Decoder, self).__init__()
        self.conv1 = GCNConv(in_channels, in_channels, normalize=True)
        self.conv2 = GCNConv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        """
        Forward pass of the decoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GCN2Conv_Encoder(torch.nn.Module):
    """
    Encoder based on GCNII (GCN2Conv).
    """
    def __init__(self, in_channels, out_channels):
        super(GCN2Conv_Encoder, self).__init__()
        self.conv1 = GCN2Conv(in_channels, out_channels, normalize=True)
        self.conv2 = GCN2Conv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        """
        Forward pass of the encoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GCN2Conv_Decoder(torch.nn.Module):
    """
    Decoder based on GCNII (GCN2Conv) for reconstruction.
    """
    def __init__(self, in_channels, out_channels):
        super(GCN2Conv_Decoder, self).__init__()
        self.conv1 = GCN2Conv(in_channels, in_channels, normalize=True)
        self.conv2 = GCN2Conv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        """
        Forward pass of the decoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GATConv_Encoder(torch.nn.Module):
    """
    Encoder based on Graph Attention Networks (GAT).
    """
    def __init__(self, in_channels, out_channels):
        super(GATConv_Encoder, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=2, concat=False)
        self.conv2 = GATConv(out_channels, out_channels, heads=2, concat=False)

    def forward(self, x, edge_index):
        """
        Forward pass of the encoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GATConv_Decoder(torch.nn.Module):
    """
    Decoder based on GAT convolution for reconstruction.
    """
    def __init__(self, in_channels, out_channels):
        super(GATConv_Decoder, self).__init__()
        self.conv1 = GATConv(in_channels, in_channels, heads=2, concat=False)
        self.conv2 = GATConv(in_channels, out_channels, heads=2, concat=False)

    def forward(self, x, edge_index):
        """
        Forward pass of the decoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GraphConv_Encoder(torch.nn.Module):
    """
    Encoder based on GraphConv.
    """
    def __init__(self, in_channels, out_channels):
        super(GraphConv_Encoder, self).__init__()
        self.conv1 = GraphConv(in_channels, out_channels)
        self.conv2 = GraphConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Forward pass of the encoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GraphConv_Decoder(torch.nn.Module):
    """
    Decoder based on GraphConv for reconstruction.
    """
    def __init__(self, in_channels, out_channels):
        super(GraphConv_Decoder, self).__init__()
        self.conv1 = GraphConv(in_channels, in_channels)
        self.conv2 = GraphConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Forward pass of the decoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x