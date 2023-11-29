import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from self_attention import *
from edge_features import EdgeFeatures


class GraphTrans(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, num_encoder_layers=4, k_neighbors=30, augment_eps=0., dropout=0.2):
        super(GraphTrans, self).__init__()

        # Hyperparameters
        self.augment_eps = augment_eps

        # Edge featurization layers
        self.EdgeFeatures = EdgeFeatures(edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.W_out = nn.Linear(hidden_dim, 1, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, X, V, mask):
        # Prepare node and edge embeddings
        # X is the alpha-C coordinate matrix; V is the pre-computed and normalized features ProtTrans+DSSP
        E, E_idx = self.EdgeFeatures(X, mask) # X [B, L, 3] mask [B, L] => E [B, L, K, d_edge]; E_idx [B, L, K]

        # Data augmentation
        if self.training and self.augment_eps > 0:
            V = V + 0.1 * self.augment_eps * torch.randn_like(V)

        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend # mask_attend [B, L, K] 
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx) 
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        logits = self.W_out(h_V).squeeze(-1) # [B, L]
        return logits


class MetalSite(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, num_encoder_layers=4, k_neighbors=30, augment_eps=0., dropout=0.2):
        super(MetalSite, self).__init__()

        # Hyperparameters
        self.augment_eps = augment_eps

        # Edge featurization layers
        self.EdgeFeatures = EdgeFeatures(edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.FC_ZN1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_ZN2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_CA1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_CA2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_MG1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_MG2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_MN1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_MN2 = nn.Linear(hidden_dim, 1, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, X, V, mask):
        # Prepare node and edge embeddings
        # X is the alpha-C coordinate matrix; V is the pre-computed and normalized features ProtTrans+DSSP
        E, E_idx = self.EdgeFeatures(X, mask) # X [B, L, 3] mask [B, L] => E [B, L, K, d_edge]; E_idx [B, L, K]

        # Data augmentation
        if self.training and self.augment_eps > 0:
            V = V + 0.2 * self.augment_eps * torch.randn_like(V)

        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend # mask_attend [B, L, K] 
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx) 
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        logits_ZN = self.FC_ZN2(F.elu(self.FC_ZN1(h_V))).squeeze(-1) # [B, L]
        logits_CA = self.FC_CA2(F.elu(self.FC_CA1(h_V))).squeeze(-1) # [B, L]
        logits_MG = self.FC_MG2(F.elu(self.FC_MG1(h_V))).squeeze(-1) # [B, L]
        logits_MN = self.FC_MN2(F.elu(self.FC_MN1(h_V))).squeeze(-1) # [B, L]

        logits = torch.cat((logits_ZN, logits_CA, logits_MG, logits_MN), 1)
        return logits
