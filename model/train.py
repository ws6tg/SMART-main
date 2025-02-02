import torch
from tqdm import tqdm

from model.model import SMART_1m, SMART_3m, SMART_2m,SAGEConv_Encoder,SAGEConv_Decoder
import torch.nn.functional as F


def train_SMART(adata_list, triplet_samples_list, feature_key="feat", edge_key="edgeList", weights=None,emb_dim=64, n_epochs=500,
               lr=0.0001,weight_decay=1e-5,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),Conv_Encoder=SAGEConv_Encoder,Conv_Decoder=SAGEConv_Decoder):
    if len(adata_list) == 1:
        x1, edge_index1 = torch.FloatTensor(adata_list[0].obsm[feature_key]), torch.LongTensor(
            adata_list[0].uns[edge_key])
        
        model=train_SMART_1m(feature=[x1],
                            edge=[edge_index1],
                            triplet_samples_list=triplet_samples_list,
                            weights=weights,
                            emb_dim=emb_dim, 
                            n_epochs=n_epochs,
                            lr=lr,
                            weight_decay=weight_decay,
                            device = device)
        
        model.eval()
        x1, edge_index1 = x1.to(device), edge_index1.to(device)
        z = model(x1, edge_index1)[0]
        SMART_feature = z.to('cpu').detach().numpy()

        return SMART_feature
    
    if len(adata_list) == 2:
        x1, edge_index1, x2, edge_index2 = torch.FloatTensor(adata_list[0].obsm[feature_key]), torch.LongTensor(
            adata_list[0].uns[edge_key]), torch.FloatTensor(adata_list[1].obsm[feature_key]), torch.LongTensor(
            adata_list[1].uns[edge_key])
        
        model=train_SMART_2m(feature=[x1,x2],
                            edge=[edge_index1,edge_index2],
                            triplet_samples_list=triplet_samples_list,
                            weights=weights,
                            emb_dim=emb_dim, 
                            n_epochs=n_epochs,
                            lr=lr,
                            weight_decay=weight_decay,
                            device = device)
        
        model.eval()
        x1, edge_index1, x2, edge_index2 = x1.to(device), edge_index1.to(device), x2.to(device), edge_index2.to(device)
        z = model(x1, edge_index1, x2, edge_index2)[0]
        SMART_feature = z.to('cpu').detach().numpy()

        return SMART_feature

    if len(adata_list) == 3:
        x1, edge_index1, x2, edge_index2, x3, edge_index3 = torch.FloatTensor(
            adata_list[0].obsm[feature_key]), torch.LongTensor(
            adata_list[0].uns[edge_key]), torch.FloatTensor(adata_list[1].obsm[feature_key]), torch.LongTensor(
            adata_list[1].uns[edge_key]), torch.FloatTensor(adata_list[2].obsm[feature_key]), torch.LongTensor(
            adata_list[2].uns[edge_key])

        model=train_SMART_3m(feature=[x1,x2,x3],
                            edge=[edge_index1,edge_index2,edge_index3],
                            triplet_samples_list=triplet_samples_list,
                            weights=weights,
                            emb_dim=emb_dim, 
                            n_epochs=n_epochs,
                            lr=lr,
                            weight_decay=weight_decay,
                            device = device,
                            Conv_Encoder=Conv_Encoder,
                            Conv_Decoder=Conv_Decoder)
        
        model.eval()
        x1, edge_index1, x2, edge_index2, x3, edge_index3 = x1.to(device), edge_index1.to(device), x2.to(device), edge_index2.to(device), x3.to(device), edge_index3.to(device)
        z = model(x1, edge_index1, x2, edge_index2, x3, edge_index3)[0]
        SMART_feature = z.to('cpu').detach().numpy()

        return SMART_feature
        
def train_SMART_1m(feature,edge,triplet_samples_list,weights=[1, 1],emb_dim=64, n_epochs=500,lr=0.0001,weight_decay=1e-5,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')): 
    
    x1=feature[0]
    edge_index1=edge[0]
    anchors1, positives1, negatives1 = triplet_samples_list[0]
    
    model = SMART_1m(hidden_dims=[x1.shape[1], emb_dim])
  
    x1, edge_index1= x1.to(device), edge_index1.to(device)
    model.to(device)
    
    n_epochs = n_epochs
    loss_list = []
    w1, w2 = weights
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    
    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()
        z, x1_rec= model(x1, edge_index1)
    
        anchor_arr1 = x1_rec[anchors1,]
        positive_arr1 = x1_rec[positives1,]
        negative_arr1 = x1_rec[negatives1,]
    
        triplet_loss = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')
        tri_output1 = triplet_loss(anchor_arr1, positive_arr1, negative_arr1)

    
        loss = w1 * F.mse_loss(x1, x1_rec)  + w2 * tri_output1
        loss_list.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    return model
    
def train_SMART_2m(feature,edge,triplet_samples_list,weights=[1, 1, 1, 1],emb_dim=64, n_epochs=500,lr=0.0001,weight_decay=1e-5,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):     
    x1,x2=feature
    edge_index1,edge_index2=edge
    anchors1, positives1, negatives1 = triplet_samples_list[0]
    anchors2, positives2, negatives2 = triplet_samples_list[1]
    
    model = SMART_2m(hidden_dims=[x1.shape[1], x2.shape[1], emb_dim])
       
    x1, edge_index1, x2, edge_index2 = x1.to(device), edge_index1.to(device), x2.to(device), edge_index2.to(device)
    model.to(device)
    
    n_epochs = n_epochs
    loss_list = []
    w1, w2, w3, w4 = weights
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    
    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()
        z, x1_rec, x2_rec = model(x1, edge_index1, x2, edge_index2)
    
        anchor_arr1 = x1_rec[anchors1,]
        positive_arr1 = x1_rec[positives1,]
        negative_arr1 = x1_rec[negatives1,]
    
        anchor_arr2 = x2_rec[anchors2,]
        positive_arr2 = x2_rec[positives2,]
        negative_arr2 = x2_rec[negatives2,]
    
        triplet_loss = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')
        tri_output1 = triplet_loss(anchor_arr1, positive_arr1, negative_arr1)
        tri_output2 = triplet_loss(anchor_arr2, positive_arr2, negative_arr2)
    
        loss = w1 * F.mse_loss(x1, x1_rec) + w2 * F.mse_loss(x2, x2_rec) + w3 * tri_output1 + w4 * tri_output2
    
        loss_list.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    return model
    
def train_SMART_3m(feature,edge,triplet_samples_list,weights=[1, 1, 1, 1,1,1],emb_dim=64, n_epochs=500,lr=0.0001,weight_decay=1e-5,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),Conv_Encoder=SAGEConv_Encoder,Conv_Decoder=SAGEConv_Decoder):

    x1,x2,x3=feature
    edge_index1,edge_index2,edge_index3=edge
    anchors1, positives1, negatives1 = triplet_samples_list[0]
    anchors2, positives2, negatives2 = triplet_samples_list[1]
    anchors3, positives3, negatives3 = triplet_samples_list[2]
    
    model = SMART_3m(hidden_dims=[x1.shape[1], x2.shape[1], x3.shape[1], emb_dim],Conv_Encoder=Conv_Encoder,Conv_Decoder=Conv_Decoder)
    
    x1, edge_index1, x2, edge_index2, x3, edge_index3 = x1.to(device), edge_index1.to(device), x2.to(
        device), edge_index2.to(device), x3.to(device), edge_index3.to(device)

    model.to(device)

    n_epochs = n_epochs
    loss_list = []
    w1, w2, w3, w4,w5,w6 = weights
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()
        z, x1_rec, x2_rec, x3_rec = model(x1, edge_index1, x2, edge_index2, x3, edge_index3)

        anchor_arr1 = x1_rec[anchors1,]
        positive_arr1 = x1_rec[positives1,]
        negative_arr1 = x1_rec[negatives1,]

        anchor_arr2 = x2_rec[anchors2,]
        positive_arr2 = x2_rec[positives2,]
        negative_arr2 = x2_rec[negatives2,]

        anchor_arr3 = x3_rec[anchors3,]
        positive_arr3 = x3_rec[positives3,]
        negative_arr3 = x3_rec[negatives3,]

        triplet_loss = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')
        tri_output1 = triplet_loss(anchor_arr1, positive_arr1, negative_arr1)
        tri_output2 = triplet_loss(anchor_arr2, positive_arr2, negative_arr2)
        tri_output3 = triplet_loss(anchor_arr3, positive_arr3, negative_arr3)

        loss_rec=w1 * F.mse_loss(x1, x1_rec) + w2 * F.mse_loss(x2, x2_rec) + w3 * F.mse_loss(x3,
                                                                                           x3_rec)
        loss_tri=w4 * tri_output1 + w5 * tri_output2 + w6 * tri_output3
        loss = loss_rec+loss_tri

        loss_list.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    return model