

import torch
from torch import nn
from torch.nn import Parameter

from torch_geometric.nn import GATConv, GCNConv, GPSConv, GINConv, SAGEConv, LGConv, GATv2Conv,GCN2Conv


import torch.nn.functional as F

class SAGEConv_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv_Encoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels, normalize=True)
        self.conv2 = SAGEConv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class SAGEConv_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv_Decoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, in_channels, normalize=True)
        self.conv2 = SAGEConv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

class GCNConv_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels, normalize=True)
        self.conv2 = GCNConv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GCNConv_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_Decoder, self).__init__()
        self.conv1 = GCNConv(in_channels, in_channels, normalize=True)
        self.conv2 = GCNConv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

class GCN2Conv_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN2Conv_Encoder, self).__init__()
        self.conv1 = GCN2Conv(in_channels, out_channels, normalize=True)
        self.conv2 = GCN2Conv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GCN2Conv_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN2Conv_Decoder, self).__init__()
        self.conv1 = GCN2Conv(in_channels, in_channels, normalize=True)
        self.conv2 = GCN2Conv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x
        
class GATConv_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATConv_Encoder, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels,heads=2,concat=False)
        self.conv2 = GATConv(out_channels, out_channels,heads=2,concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GATConv_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATConv_Decoder, self).__init__()
        self.conv1 = GATConv(in_channels, in_channels,heads=2,concat=False)
        self.conv2 = GATConv(in_channels, out_channels,heads=2,concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x





class SMART_1m(torch.nn.Module):
    def __init__(self, hidden_dims,Conv_Encoder=SAGEConv_Encoder,Conv_Decoder=SAGEConv_Decoder):
        super(SMART_1m, self).__init__()

        [in_dim1, out_dim] = hidden_dims
        self.conv1_enc = Conv_Encoder(in_dim1, out_dim)
        self.fc=nn.Linear(out_dim, out_dim)
        self.conv1_dec = Conv_Decoder(out_dim, in_dim1)

    def forward(self, features1, edge_index1):
        x1 = self.conv1_enc(features1, edge_index1)
        x=self.fc(x1)
        x1_rec = self.conv1_dec(x, edge_index1)

        return x1, x1_rec


class SMART_2m(torch.nn.Module):
    def __init__(self, hidden_dims,Conv_Encoder=SAGEConv_Encoder,Conv_Decoder=SAGEConv_Decoder):
        super(SMART_2m, self).__init__()

        [in_dim1, in_dim2, out_dim] = hidden_dims
        self.conv1_enc = Conv_Encoder(in_dim1, out_dim)
        self.conv2_enc = Conv_Encoder(in_dim2, out_dim)
        self.fc=nn.Linear(2*out_dim, out_dim)
        self.conv1_dec = Conv_Decoder(out_dim, in_dim1)
        self.conv2_dec = Conv_Decoder(out_dim, in_dim2)

    def forward(self, features1, edge_index1, features2, edge_index2):
        x1 = self.conv1_enc(features1, edge_index1)
        x2 = self.conv2_enc(features2, edge_index2)
        x=self.fc(torch.cat([x1, x2], dim=1))
        x1_rec = self.conv1_dec(x, edge_index1)
        x2_rec = self.conv2_dec(x, edge_index2)

        return x, x1_rec, x2_rec


class SMART_3m(torch.nn.Module):
    def __init__(self, hidden_dims,Conv_Encoder=SAGEConv_Encoder,Conv_Decoder=SAGEConv_Decoder):
        super(SMART_3m, self).__init__()

        [in_dim1, in_dim2, in_dim3,out_dim] = hidden_dims
        self.conv1_enc = Conv_Encoder(in_dim1, out_dim)
        self.conv2_enc = Conv_Encoder(in_dim2, out_dim)
        self.conv3_enc = Conv_Encoder(in_dim3, out_dim)
        #self.atten = AttentionLayer(out_dim, out_dim)
        self.fc=nn.Linear(3*out_dim, out_dim)
        self.conv1_dec = Conv_Decoder(out_dim, in_dim1)
        self.conv2_dec = Conv_Decoder(out_dim, in_dim2)
        self.conv3_dec = Conv_Decoder(out_dim, in_dim3)

    def forward(self, features1, edge_index1, features2, edge_index2, features3, edge_index3):
        x1 = self.conv1_enc(features1, edge_index1)
        x2 = self.conv2_enc(features2, edge_index2)
        x3 = self.conv3_enc(features3, edge_index3)
        #x = self.atten(torch.stack([x1, x2,x3], dim=1))
        x=self.fc(torch.cat([x1, x2,x3],dim=-1))
        x1_rec = self.conv1_dec(x, edge_index1)
        x2_rec = self.conv2_dec(x, edge_index2)
        x3_rec = self.conv3_dec(x, edge_index3)

        return x, x1_rec, x2_rec,x3_rec