
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling
import copy
from scipy.spatial.distance import cosine
from torch_geometric.utils  import dense_to_sparse 

class TemporalConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding= kernel_size//2)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.PReLU()
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, hidden_dim)
        return x
class NodeEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim, num_layers):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            TemporalConv(input_dim, hidden_dim),
            nn.Dropout(0.3),
            TemporalConv(hidden_dim, hidden_dim),
        )    
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim//4, heads=4, dropout=0.2)
            )
        self.adapt_adj = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.skip_conn = nn.Linear(input_dim, hidden_dim)
    def forward(self, x, edge_index):
        identity = self.skip_conn(x)
        x_temporal = self.temporal_conv(x.unsqueeze(1)).squeeze(1)
        x_gat = x_temporal 
        for layer in self.gat_layers: 
            x_gat = layer(x_gat, edge_index)
            x_gat = F.elu(x_gat) 
        x_final = x_gat + identity
        return x_final, self.projection(x_gat)

class Scattering_Consistency_Fuse(nn.Module):
    def __init__(self, encoder, hidden_dim, momentum=0.5):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.PReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.center = nn.Parameter(torch.randn(hidden_dim))
        with torch.no_grad():
            self.center.data = self.center.data / self.center.data.norm(dim=1, keepdim=True) * torch.sqrt(torch.rand(1))        
        self.momentum = momentum
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

    def update_target(self):
        with torch.no_grad():
            for o_param, t_param in zip(self.online_encoder.parameters(),
                                      self.target_encoder.parameters()):
                t_param.data = self.momentum * t_param.data + (1 - self.momentum) * o_param.data
        

    def forward(self, x, edge_index):
        h_online, proj_online = self.online_encoder(x, edge_index)
        h_pred = self.predictor(proj_online)

        with torch.no_grad():
            h_target, _ = self.target_encoder(x, edge_index)
            proj_target = F.normalize(h_target,p=2,dim=1)
        return h_pred, proj_target

    def compute_loss(self, x, h_pred, h_target, edge_index):
        # L_Scatter
        center_sim = F.cosine_similarity(h_target.unsqueeze(1), 
                                       self.center.unsqueeze(0), 
                                       dim=-1) 
        Scatter_loss = -torch.logsumexp(center_sim, dim=1).mean()
        # L_Time
        time_loss = F.mse_loss(h_pred[1:], h_pred[:-1])
        # L_constrct
        pos_src, pos_dst = edge_index
        pos_sim = F.cosine_similarity(h_pred[pos_src], h_target[pos_dst])
        constrct_loss = -F.logsigmoid(pos_sim).mean()
        total_loss = Scatter_loss + constrct_loss + time_loss
        return total_loss
    
    def get_anomaly_scores(self, x, edge_index):
        with torch.no_grad():
            h, _ = self.online_encoder(x, edge_index)
            time_inconsistency = F.mse_loss(h[1:], h[:-1], reduction='none').mean(dim=1)
            time_inconsistency = F.pad(time_inconsistency,  (0,1))
            dists = torch.cdist( h.unsqueeze(0), self.center.unsqueeze(0)).squeeze(0) 
            Scatter_diviation = 1./torch.min(dists, dim=1)[0]
            return Scatter_diviation  + time_inconsistency
                
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.encoder = NodeEmbedding(input_dim, hidden_dim, hidden_dim, num_layers)
        self.model = Scattering_Consistency_Fuse(self.encoder, hidden_dim)
        
    def forward(self, data):
        return self.model(data.x, data.edge_index)
    
    def update_target(self):
        self.model.update_target()
        
    def compute_loss(self, x, h_pred, h_target, edge_index):
        return self.model.compute_loss(x, h_pred, h_target, edge_index)
    
    def get_anomaly_scores(self, x, edge_index):
        return self.model.get_anomaly_scores( x, edge_index)
