import inspect
import warnings
from collections.abc import Sequence

from torch_sparse import spmm
from torch_scatter import scatter_mean, scatter_add, scatter_max
import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import data, utils, layers
from torchdrug.layers import functional
import numpy as np

class ContinuousFilterConv(nn.Module):
    """
    Continuous filter operator from
    `SchNet: A continuous-filter convolutional neural network for modeling quantum interactions`_.

    .. _SchNet\: A continuous-filter convolutional neural network for modeling quantum interactions:
        https://arxiv.org/pdf/1706.08566.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        hidden_dim (int, optional): hidden dimension. By default, same as :attr:`output_dim`
        cutoff (float, optional): maximal scale for RBF kernels
        num_gaussian (int, optional): number of RBF kernels
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, hidden_dim=None, cutoff=5, num_gaussian=100,
                 batch_norm=False, activation="shifted_softplus"):
        super(ContinuousFilterConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        if hidden_dim is None:
            hidden_dim = output_dim
        self.hidden_dim = hidden_dim
        self.rbf = layers.RBF(stop=cutoff, num_kernel=num_gaussian)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if activation == "shifted_softplus":
            self.activation = functional.shifted_softplus
        elif isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
    
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.rbf_layer = nn.Linear(num_gaussian, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, hidden_dim)
        else:
            self.edge_linear = None

    def message(self, edge_list, edge_feature, node_position, input):
        # node_in, node_out = graph.edge_list.t()[:2]
        node_in = edge_list[:,0]
        node_out = edge_list[:,1]
        # position = graph.node_position
        position = node_position
        message = self.input_layer(input)[node_in]
        if self.edge_linear:
            message = message + self.edge_linear(edge_feature.float())
        weight = self.rbf_layer(self.rbf(position[node_in], position[node_out]))
        message = message * weight
        return message


    def aggregate(self, edge_list, edge_weight, num_node, message):
        node_out = edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=num_node)
        return update

    def message_and_aggregate(self, edge_list, edge_weight, edge_feature, num_node, node_position, input):
        # node_in, node_out = graph.edge_list.t()[:2]
        node_in = edge_list[:,0]
        node_out = edge_list[:,1]
        position = node_position
        
        num_edge = len(edge_list)
        

        rbf_weight = self.rbf_layer(self.rbf(position[node_in], position[node_out]))
        indices = torch.stack([node_out, node_in, torch.arange(num_edge).cuda(0)])
        # print(edge_weight.shape, indices.shape)
        adjacency = utils.sparse_coo_tensor(indices, edge_weight, (num_node, num_node, num_edge))
#         print(adjacency.shape, rbf_weight.shape, self.input_layer(input).shape)
#         print(adjacency.shape, rbf_weight.shape,input.shape)
#         input1 = input.clone()
        update = functional.generalized_rspmm(adjacency, rbf_weight, self.input_layer(input))
       
#         print(update.shape)
        # update = self.my_rspmm(indices, edge_weight, self.input_layer(input), rbf_weight, num_node)
#         rbf_weight1 = self.rbf_layer1(self.rbf(position[node_in], position[node_out]))
        if self.edge_linear:
            edge_input = edge_feature.float()
            edge_input = self.edge_linear(edge_input)
#             print(edge_weight.shape, rbf_weight.shape)
            edge_weight = edge_weight.unsqueeze(-1) * rbf_weight
#             print(edge_input.shape, edge_weight.shape)
            edge_update = scatter_add(edge_input * edge_weight, edge_list[:, 1], dim=0,
                                      dim_size=num_node)
#             print(edge_update.shape)
            update = update + edge_update

        return update


    def combine(self, input, update):
        output = self.output_layer(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    
    def forward(self, edge_list, edge_weight, edge_feature, num_node, node_position, input):
        """
        Parameters:
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        """
        update = self.message_and_aggregate(edge_list, edge_weight, edge_feature, num_node, node_position, input)
        output = self.combine(input, update)
        return output
    
class SchNet(nn.Module):
    """
    SchNet from `SchNet: A continuous-filter convolutional neural network for modeling quantum interactions`_.

    .. _SchNet\: A continuous-filter convolutional neural network for modeling quantum interactions:
        https://arxiv.org/pdf/1706.08566.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        edge_input_dim (int, optional): dimension of edge features
        cutoff (float, optional): maximal scale for RBF kernels
        num_gaussian (int, optional): number of RBF kernels
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
    """

    def __init__(self, input_dim, hidden_dims, edge_input_dim=None, cutoff=5, num_gaussian=100, short_cut=True,
                 batch_norm=False, activation="shifted_softplus", concat_hidden=False):
        super(SchNet, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(ContinuousFilterConv(self.dims[i], self.dims[i + 1], edge_input_dim, None, cutoff,
                                                           num_gaussian, batch_norm, activation))

        self.readout = layers.SumReadout()

    def forward(self, edge_list, edge_weight, edge_feature, num_node, node_position, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Require the graph(s) to have node attribute ``node_position``.

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(edge_list, edge_weight, edge_feature, num_node, node_position, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
#         graph_feature = self.readout(graph, node_feature)

        return {
#             "graph_feature": graph_feature,
            "node_feature": node_feature
        }