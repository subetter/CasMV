import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as scatter
from modules.utils import MergeLayer_output, Feat_Process_Layer, drop_edge
from modules.embedding_module import get_embedding_module
from modules.time_encoding import TimeEncode



class TGAT(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.cfg = config

        self.nodes_dim = self.cfg.input_dim
        self.edge_dim = self.cfg.input_dim
        self.dims = self.cfg.hidden_dim

        self.n_heads = self.cfg.n_heads
        self.dropout = self.cfg.drop_out
        self.n_layers = self.cfg.n_layer

        self.mode = self.cfg.mode

        self.time_encoder = TimeEncode(dimension=self.dims)
        self.embedding_module_type = self.cfg.module_type
        self.embedding_module = get_embedding_module(module_type=self.embedding_module_type,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     node_features_dims=self.dims,
                                                     edge_features_dims=self.dims,
                                                     time_features_dim=self.dims,
                                                     hidden_dim=self.dims,
                                                     n_heads=self.n_heads, dropout=self.dropout)

        self.node_preocess_fn = Feat_Process_Layer(self.nodes_dim, self.dims)
        self.edge_preocess_fn = Feat_Process_Layer(self.edge_dim, self.dims)
        self.affinity_score = MergeLayer_output(self.dims, self.dims, drop_out=0.2)



    def forward(self,batch):
        # apply tgat
        for i in batch:
            source_node_embedding = self.compute_temporal_embeddings(i["src_neigh_edge"], i["src_edge_to_time"],
                                                                                    i["src_edge_feature"], i["src_node_features"])

            root_embedding = source_node_embedding[i["src_center_node_idx"], :]
        
        return root_embedding



    def compute_temporal_embeddings(self, neigh_edge, edge_to_time, edge_feat, node_feat):
        node_feat = self.node_preocess_fn(node_feat)
        edge_feat = self.edge_preocess_fn(edge_feat)

        node_embedding = self.embedding_module.compute_embedding(neigh_edge, edge_to_time,
                                                                 edge_feat, node_feat)
        return node_embedding
