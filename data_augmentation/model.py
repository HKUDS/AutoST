import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from loader import BioDataset
from dataloader import DataLoaderFinetune
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import dill

import torch.nn as nn
import math


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr = "add", input_layer = False):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(4, emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        self.linear_1 = torch.nn.Linear(emb_dim*emb_dim, emb_dim)
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Linear(emb_dim,emb_dim)
            # self.input_node_embeddings = torch.nn.Embedding(emb_dim, emb_dim)
            # torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        # print("edge_index:", edge_index)
        # println()
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index[0].device)
        row, col = edge_index
        
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        # print("x_forward:", x)
        # print("x.size(0):", x.size(0))
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))
        # print("edge_attr:", edge_attr.size())
        ad_s = edge_attr.size()[1]
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 4)
        self_loop_attr[:,1] = 1 # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # print("self_loop_attr:", self_loop_attr.size())
        # print("__self_attr:", edge_attr.size())
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
        # print("self_attr__:", edge_attr.size())
        # print("&&&:",edge_attr)
        # println()
        edge_embeddings = self.edge_encoder(edge_attr.float())
    
        # print("111")
        # print("x.size()[0]:",x.size()[0])
        if self.input_layer:
            # print("x.to(torch.int64):", x)
            x = self.input_node_embeddings(x)
            # print("x_shape:", x.size())
            # x_ = torch.reshape(x, (x.size()[0],-1))
            # print("%%%%x_:",x_.size())
            # x = self.linear_1(x_)
        # print("^^:", x.size())
        # println()
        norm = self.norm(edge_index[0].long(), x.size(0), x.dtype)

        # print("&&&:", x.size())
        x = self.linear(x)
        # print("x in model:", x)
        # print("edge_index:",edge_index)
        # println()
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr = "add", input_layer = False):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(4, heads * emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))
# 
        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 4)
        self_loop_attr[:,1] = 1 # attribute for self-loop edge
        # print("edge_attr.device:", edge_attr.device)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_encoder(edge_attr)

        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1,))

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out





def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    # print("bound:", bound, size, math.sqrt(size))
    if tensor is not None:
        # print("tensor.data.uniform_(-bound, bound):", tensor.data.uniform_(-bound, bound))
        return tensor.data.uniform_(-bound, bound)

class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, num_bases,input_layer, 
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))
        self.edge_encoder = torch.nn.Linear(4, in_channels)
        self.input_layer = input_layer
        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Linear(in_channels,out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        # uniform(size, self.basis)
        # uniform(size, self.att)
        # uniform(size, self.root)
        # uniform(size, self.bias)
        zeros(uniform(size, self.basis))
        glorot(uniform(size, self.att))
        glorot(uniform(size, self.root))
        zeros(uniform(size, self.bias))
        # glorot(self.att)
        # zeros(self.bias)
        # print("uniform(size, self.basis):", uniform(size, self.basis))
        # print("uniform(size, self.att):", uniform(size, self.att))
        # print("uniform(size, self.root):", uniform(size, self.root))
        # print("uniform(size, self.bias):", uniform(size, self.bias))
        # println()

    def forward(self, x,edge_index,edge_attr, edge_type,edge_norm=None, size=None):
        # edge_index = add_self_loops(edge_index, num_nodes = x.size(0))[0].long()
        # print(edge_index[:2])
        # print("edge_attr:", edge_attr.size())
        # print("edge_type:", edge_type.size())
        # ad_s = edge_attr.size()[1]
        # # add features corresponding to self-loop edges.
        # self_loop_attr = torch.zeros(x.size(0), 4)
        # self_loop_attr[:,1] = 1 # attribute for self-loop edge
        # self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # print("self_loop_attr:", self_loop_attr.size())
        # # print("__self_attr:", edge_attr.size())
        # edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
        # print("self_attr__:", edge_attr.size())
        # print("&&&:",edge_attr)
        # println()
        # edge_attr = self_loop_attr
        # print("x:", x)
        edge_embeddings = self.edge_encoder(edge_attr.float())
        # print("edge_embdedding:", edge_embeddings.size())
        # print("edge_embeddings:", edge_embeddings)
        # print("x:", x)
        # tmp = self.propagate(edge_index = edge_index.long(), size=size, x=x, edge_type=edge_type.long(),edge_attr=edge_embeddings, edge_norm=edge_norm)
        # print("tmp:", tmp)
        # print("edge_type:", edge_type)
        return self.propagate(edge_index = edge_index.long(), size=size, x=x, edge_type=edge_type.long(),edge_attr=edge_embeddings, edge_norm=edge_norm)
        # return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,edge_norm=edge_norm)


    def message(self, x_j, edge_index_j, edge_type, edge_norm):
       
        # print("self.att:", self.att)
        # print("self.basis:", self.basis)
        # print("self.basis shape:", self.basis.size())
        # print("self.numbe_bases:", self.num_bases)
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        # print("start w:", w)
        # print("self.in_channels:",self.in_channels)
        # print("self.out_channels:", self.out_channels)
        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, torch.squeeze(index, 1).int())
            # print("1 w:", w)
            # print("1:edge_type:", edge_type)
            # print("1 index:", index)
            # print("1 out:", out)
        else:
            # w = w.view(self.num_relations, self.in_channels, self.out_channels)
            # w = torch.index_select(w, 0, edge_type)
            # out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
            # print("message w :", w)
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            # print("w original:", w.size())
            # print("edge_type:", edge_type.size())
            # print("x_j:", x_j.size())
            # print("2 w before:", w)
            # print("w:", w)
            # print("edge_type:", edge_type)
            w = torch.index_select(w.to(edge_type.device), 0, torch.squeeze(edge_type, 1).int())
            # w = torch.gather(input=w, dim=0, index=torch.squeeze(edge_type, 1).long())
            # print("x_j.unsqueeze(1):", x_j.unsqueeze(1).size())
            # print("w after:", w.size())
            # print(w.size()[0])
            # print("w:", w)
            # print("x_j:", x_j)
            # w_ = (nn.Linear(96*96, 30*30).to(x_j.device)(w.view(w.size()[0], -1))).view(w.size()[0], 30, 30)
            # x_j_ = nn.Linear(96, 30).to(x_j.device)(x_j)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
            # out = nn.Linear(30, 96).to(x_j.device)(out)
            # print("out:", out)
            # print("out:", out.size())
            # println()
            # print("2 w after:", w)
            # print("2:edge_type:", edge_type)
            # print("2 out:", out)
        # print("message first out:", out)
        # print("message passing output:", out if edge_norm is None else out * edge_norm.view(-1, 1))
        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)    

class GNN(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gat, graphsage, gcn
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin", out_channels = 96):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of message-passing GNN convs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if layer == 0:
                input_layer = True
            else:
                input_layer = False

            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add", input_layer = input_layer))
            elif gnn_type == "gcn":
                # self.gnns.append(GCNConv(emb_dim, input_layer = input_layer))
                self.gnns.append(RGCNConv(emb_dim, emb_dim, 5, 4, input_layer = input_layer))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim, input_layer = input_layer))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim, input_layer = input_layer))

    #def forward(self, x, edge_index, edge_attr):
   
    def forward(self, x, edge_index, edge_attr, edge_type):
        h_list = [x]
        # print("h_list:", h_list)
        for layer in range(self.num_layer):
            # print("x_f:",x[0])
            # print("e_f:", edge_index[:10])
            # print(layer,h_list[layer].size())
            # print("self.gnns:", self.gnns[0])
            # print(h_list[0])
            # println()
            # print("input edge type:", edge_type)
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr, edge_type)+ 1e-6  # x, edge_index, edge_type, edge_norm=None, size=None
            # print("h_1:", h.size())
            # println()
            # print("h:", h, layer)
            if layer == self.num_layer - 1:
                #remove relu from the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
                # print("h_3:", h.size())
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
                # print("h_2:", h.size())
            h_list.append(h)
            # print("h_ist:", h_list)

        if self.JK == "last":
            # print("h_list[-1]:", h_list[-1].size())
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            tmp = torch.sum(torch.cat(h_list[1:], dim = 0), dim = 0)
            # print("tmp:", tmp.size())
            node_representation = torch.sum(torch.cat(h_list[1:], dim = 0), dim = 0)[0]
        # print("node_shape:", node_representation.size())
        # print("node:", node_representation.size())
        # println()
        return node_representation




class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases)

        self.dropout_ratio = dropout
        # self.input_layer = input_layer
        # if self.input_layer:
        #     self.input_node_embeddings = torch.nn.Linear(emb_dim,emb_dim)

    def forward(self, entity, edge_index, edge_attr, edge_type,  edge_norm):
        # x = self.entity_embedding(entity)
        x = entity
        # print("sp:",x.size())
        # print("edge_attr:", edge_attr.size())
        # print("input conv1:", edge_type)

        x = F.relu(self.conv1(x, edge_index,edge_attr,  edge_type, edge_norm))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        # print("x:", x.size())
        # print("input conv2:", edge_type)
        x = self.conv2(x, edge_index, edge_attr,edge_type,  edge_norm)
        
        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.relation_embedding[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))
class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_representation = self.gnn(x, edge_index, edge_attr)
        
        pooled = self.pool(node_representation, batch)
        center_node_rep = node_representation[data.center_node_idx]

        graph_rep = torch.cat([pooled, center_node_rep], dim = 1)

        return self.graph_pred_linear(graph_rep)
if __name__ == "__main__":
    pass



