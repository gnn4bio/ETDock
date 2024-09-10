import torch
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.utils import to_dense_batch
from torch import nn
from torch.nn import Linear
import sys
import torch.nn as nn
from gvp import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from GATv2 import GAT
from GINv2 import GIN

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x



class GVP_embedding(nn.Module):
    '''
    Modified based on https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py
    GVP-GNN for Model Quality Assessment as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.1):

        super(GVP_embedding, self).__init__()
        
        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def forward(self, h_V, edge_index, h_E, seq):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        seq = self.W_s(seq)
        h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        return out


def get_pair_dis_one_hot(d, bin_size=2, bin_min=-1, bin_max=30):
    # without compute_mode='donot_use_mm_for_euclid_dist' could lead to wrong result.
    pair_dis = torch.cdist(d, d, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=16)
    return pair_dis_one_hot

class TriangleProteinToCompound(torch.nn.Module):
    def __init__(self, embedding_channels=256, c=128, hasgate=True):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.layernorm_c = torch.nn.LayerNorm(c)
        self.hasgate = hasgate
        if hasgate:
            self.gate_linear = Linear(embedding_channels, c)
        self.linear = Linear(embedding_channels, c)
        self.ending_gate_linear = Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = Linear(c, embedding_channels)
    def forward(self, z, protein_pair, compound_pair, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        # z_mask of shape b, i, j, 1
        z = self.layernorm(z)
        if self.hasgate:
            ab = self.gate_linear(z).sigmoid() * self.linear(z) * z_mask
        else:
            ab = self.linear(z) * z_mask
        g = self.ending_gate_linear(z).sigmoid()
        block1 = torch.einsum("bikc,bkjc->bijc", protein_pair, ab)
        block2 = torch.einsum("bikc,bjkc->bijc", ab, compound_pair)
        z = g * self.linear_after_sum(self.layernorm_c(block1+block2)) * z_mask
        return z

class TriangleProteinToCompound_v2(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, c=128):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.layernorm_c = torch.nn.LayerNorm(c)

        self.gate_linear1 = Linear(embedding_channels, c)
        self.gate_linear2 = Linear(embedding_channels, c)

        self.linear1 = Linear(embedding_channels, c)
        self.linear2 = Linear(embedding_channels, c)

        self.ending_gate_linear = Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = Linear(c, embedding_channels)
    def forward(self, z, protein_pair, compound_pair, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        protein_pair = self.layernorm(protein_pair)
        compound_pair = self.layernorm(compound_pair)
 
        ab1 = self.gate_linear1(z).sigmoid() * self.linear1(z) * z_mask
        ab2 = self.gate_linear2(z).sigmoid() * self.linear2(z) * z_mask
        protein_pair = self.gate_linear2(protein_pair).sigmoid() * self.linear2(protein_pair)
        compound_pair = self.gate_linear1(compound_pair).sigmoid() * self.linear1(compound_pair)

        g = self.ending_gate_linear(z).sigmoid()
        block1 = torch.einsum("bikc,bkjc->bijc", protein_pair, ab1)
        block2 = torch.einsum("bikc,bjkc->bijc", ab2, compound_pair)
        # print(g.shape, block1.shape, block2.shape)
        z = g * self.linear_after_sum(self.layernorm_c(block1+block2)) * z_mask
        return z

class Self_Attention(nn.Module):
    def __init__(self, hidden_size,num_attention_heads=8,drop_rate=0.5):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dp = nn.Dropout(drop_rate)
        self.ln = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,q,k,v,attention_mask=None,attention_weight=None):
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs = self.dp(attention_probs)
        if attention_weight is not None:
            attention_weight_sorted_sorted = torch.argsort(torch.argsort(-attention_weight,axis=-1),axis=-1)
            # if self.training:
            #     top_mask = (attention_weight_sorted_sorted<np.random.randint(28,45))
            # else:
            top_mask = (attention_weight_sorted_sorted<32)
            attention_probs = attention_probs * top_mask
            # attention_probs = attention_probs * attention_weight
            attention_probs = attention_probs / (torch.sum(attention_probs,dim=-1,keepdim=True) + 1e-5)
        # print(attention_probs.shape,v.shape)
        # attention_probs = self.dp(attention_probs)
        outputs = torch.matmul(attention_probs, v)

        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
        outputs = outputs.view(*new_output_shape)
        outputs = self.ln(outputs)
        return outputs

class TriangleSelfAttentionRowWise(torch.nn.Module):
    # use the protein-compound matrix only.
    def __init__(self, embedding_channels=128, c=32, num_attention_heads=4):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = c
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.embedding_channels = embedding_channels
        # self.dp = nn.Dropout(drop_rate)
        # self.ln = nn.LayerNorm(hidden_size)

        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.layernorm1 = torch.nn.LayerNorm(64)
        # self.layernorm = torch.nn.BatchNorm1d(embedding_channels)
        # self.layernorm1 = torch.nn.BatchNorm3d(64)
        # self.layernorm_c = torch.nn.LayerNorm(c)
        self.gate_linear1 = Linear(embedding_channels, 128)
        self.gate_linear2 = Linear(embedding_channels, 128)

        self.linear1 = Linear(embedding_channels, 128)
        self.linear2 = Linear(embedding_channels, 128)
        
        self.linear_q = Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_k = Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_v = Linear(embedding_channels, self.all_head_size*3, bias=False)
        # self.b = Linear(embedding_channels, h, bias=False)
        self.g = Linear(embedding_channels, self.all_head_size)
        self.final_linear = Linear(self.all_head_size, embedding_channels)
        self.vec_proj = nn.Linear(64, 64 * 3, bias=False)
        self.dk_proj = nn.Linear(50, embedding_channels)
        self.dv_proj = nn.Linear(50, embedding_channels)
        # self.mapping = nn.Linear(50, 32)
        self.o_proj = nn.Linear(embedding_channels, embedding_channels * 3)
        self.act = nn.SiLU()
        # self.mlp2lig = nn.Linear(32,32)
        ######## learning interactive embedding z
        self.q = nn.Linear(128, 128,bias=False)
        self.k = nn.Linear(128, 128,bias=False)
        self.v = nn.Linear(128, 128,bias=False)
        # self.mlp0 = nn.Linear(129,128)
        # self.mlp1 = nn.Linear(64,64)
        # self.mlp2 = nn.Linear(128,3)
        # self.g_vec = Linear(embedding_channels, embedding_channels)
        # self.mlp2weight = nn.Linear(128,128)
        self.map128 = nn.Linear(64,128,bias=False)
        self.map64 = nn.Linear(128,64,bias=False)
        self.dd = nn.Linear(32,32*3,bias=False)
        
    def reshape_last_dim(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # x = x.view(*new_x_shape)
        return x.view(*new_x_shape)
    def reshape_last_dimX3(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size*3)
        # print(new_x_shape)
        # x = x.view(*new_x_shape)
        return x.view(*new_x_shape)

    def forward(self, z, z_mask, vec):
        
        z = self.layernorm(z)
        vec = self.layernorm1(vec) 
        # print(z.shape, vec.shape)
        
        # p_length = z.shape[1]
        # batch_n = z.shape[0]
        # l_length = z.shape[2]
        # new_z = torch.zeros(z.shape, device=z.device)
        z_i = z
        # print(z_mask)
        # print(z_mask.shape)
        # z_mask_i = z_mask.view((batch_n, p_length, 1, 1, -1))
        # z_mask_i1= z_mask.view((batch_n, l_length, 1, 1, -1))
        # attention_mask_i = (1e9 * (z_mask_i.float() - 1.))
        # attention_mask_i1 = (1e9 * (z_mask_i1.float() - 1.))
        # q, k, v of shape b, j, h, c
       
        q = self.reshape_last_dim(self.linear_q(z_i)) #  * (self.attention_head_size**(-0.5))
        k = self.reshape_last_dim(self.linear_k(z_i))
  
        v = self.reshape_last_dimX3(self.linear_v(z_i))

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), 64, dim=-1)
        

        vec_dot = self.map128((vec1 * vec2).sum(dim=-2))

        vec_sub = vec1 - vec2

#         dk = self.act(self.reshape_last_dim(self.dk_proj(dis_compound)))
#         dv = self.act(self.reshape_last_dim(self.dv_proj(dis_protein)))

#         dd = torch.einsum("biihc,bjjhc->bijhc",dv,dk)
        dd = self.reshape_last_dim(z)
        v *= self.dd(dd)
      
        x, vec1, vec2= torch.split(v, self.attention_head_size, dim=-1)
        
        x = nn.Softmax(dim=-1)(q*k*dd+z_mask.unsqueeze(-1).unsqueeze(-1))*x
      
        x = x.reshape(*x.size()[:-2], self.embedding_channels)
        vec1 = vec1.reshape(*vec1.size()[:-2], self.embedding_channels)
        vec2 = vec2.reshape(*vec2.size()[:-2], self.embedding_channels)
       
        o1, o2, o3 = torch.split(self.o_proj(x), self.embedding_channels, dim=-1)
        
        o2 = self.gate_linear1(o2).sigmoid() * self.linear1(o2)
        o3 = self.gate_linear2(o3).sigmoid() * self.linear2(o3)
        
        o2 *= vec1
        o3 *= vec2
       
        # dvec = self.map64((o1 * o2 + o1 * o3).unsqueeze(dim=-2)).relu()*vec_sub + vec
        dx = self.final_linear(o1 * vec_dot+o1 * o2  * vec_dot + o1 * o3 * vec_dot) * z_mask.unsqueeze(-1)
        # dx = self.final_linear(vec1 * vec_dot+ vec2  * vec_dot + x* vec_dot+x) * z_mask.unsqueeze(-1)
        return dx

class Transition(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, n=4):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.linear1 = Linear(embedding_channels, n*embedding_channels)
        self.linear2 = Linear(n*embedding_channels, embedding_channels)
    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        z = self.linear2((self.linear1(z)).relu())
        return z



class IaBNet_with_affinity(torch.nn.Module):
    def __init__(self, hidden_channels=128, embedding_channels=128, c=128, mode=0, protein_embed_mode=1, compound_embed_mode=1, n_trigonometry_module_stack=5, protein_bin_max=30, readout_mode=2):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.protein_bin_max = protein_bin_max
        self.mode = mode
        self.protein_embed_mode = protein_embed_mode
        self.compound_embed_mode = compound_embed_mode
        self.n_trigonometry_module_stack = n_trigonometry_module_stack
        self.readout_mode = readout_mode

        if protein_embed_mode == 0:
            self.conv_protein = GNN(hidden_channels, embedding_channels)
            self.conv_compound = GNN(hidden_channels, embedding_channels)
            # self.conv_protein = SAGEConv((-1, -1), embedding_channels)
            # self.conv_compound = SAGEConv((-1, -1), embedding_channels)
        if protein_embed_mode == 1:
            self.conv_protein = GVP_embedding((6, 3), (embedding_channels, 16), 
                                              (32, 1), (32, 1), seq_in=True)
            

        if compound_embed_mode == 0:
            self.conv_compound = GNN(hidden_channels, embedding_channels)
        elif compound_embed_mode == 1:
            self.conv_compound = GIN(input_dim = 56, hidden_dims = [128,56,embedding_channels], edge_input_dim = 19, concat_hidden = False)

        if mode == 0:
            self.protein_pair_embedding = Linear(16, c)
            self.compound_pair_embedding = Linear(16, c)
            self.protein_to_compound_list = []
            self.protein_to_compound_list = nn.ModuleList([TriangleProteinToCompound_v2(embedding_channels=embedding_channels, c=c) for _ in range(n_trigonometry_module_stack)])
            self.triangle_self_attention_list = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=embedding_channels) for _ in range(n_trigonometry_module_stack)])
            self.tranistion = Transition(embedding_channels=embedding_channels, n=4)

        self.linear = Linear(embedding_channels, 1)
        self.linear_energy = Linear(embedding_channels, 1)
        if readout_mode == 2:
            self.gate_linear = Linear(embedding_channels, 1)
        # self.gate_linear = Linear(embedding_channels, 1)
        self.bias = torch.nn.Parameter(torch.ones(1))
        self.leaky = torch.nn.LeakyReLU()
        self.dropout = nn.Dropout2d(p=0.25)
        self.mlp64 = nn.Linear(1, 64, bias=False)
        self.layernorm12 = torch.nn.LayerNorm(64)
        self.mlp2128 = nn.Linear(56, 128,bias=False)
        self.mlp256 = nn.Linear(128, 56,bias=False)
        self.mlp22128 = nn.Linear(56, 128,bias=False)
        self.layern = nn.LayerNorm(56)
        self.layern1 = nn.LayerNorm(128)
        self.layern2 = nn.LayerNorm(128)
        self.layern3 = nn.LayerNorm(56)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.mlpweight128 = nn.Linear(50,128,bias=False)
        # self.layernorm12 = nn.LayerNorm(64)
        self.mlp64 = nn.Linear(1,64,bias=False)
        self.bachn = nn.BatchNorm1d(512)
        self.bachn1 = nn.BatchNorm1d(256)
        self.bachn2 = nn.BatchNorm1d(56)
        
        self.morgan_mlp0 = nn.Linear(1024, 512,bias=False)
        self.morgan_mlp1 = nn.Linear(512, 256,bias=False)
        self.morgan_mlp2 = nn.Linear(256, 128,bias=False)
        
        self.mlp99 = nn.Linear(128,256, bias=False)
        self.mlp999 = nn.Linear(128,256, bias=False)
        self.mlp991 = nn.Linear(256,128, bias=False)
        self.mlp9991 = nn.Linear(256,128, bias=False)
        
    def forward(self, data):
        if self.protein_embed_mode == 0:
            x = data['protein'].x.float()
            edge_index = data[("protein", "p2p", "protein")].edge_index
            protein_batch = data['protein'].batch
            protein_out = self.conv_protein(x, edge_index)
        if self.protein_embed_mode == 1:
            nodes = (data['protein']['node_s'], data['protein']['node_v'])
            edges = (data[("protein", "p2p", "protein")]["edge_s"], data[("protein", "p2p", "protein")]["edge_v"])
            protein_batch = data['protein'].batch
            protein_out = self.conv_protein(nodes, data[("protein", "p2p", "protein")]["edge_index"], edges, data.seq)

        if self.compound_embed_mode == 0:
            compound_x = data['compound'].x.float()
            compound_edge_index = data[("compound", "c2c", "compound")].edge_index
            compound_batch = data['compound'].batch
            compound_out = self.conv_compound(compound_x, compound_edge_index)
        elif self.compound_embed_mode == 1:
            compound_x = data['compound'].x.float()
            compound_edge_index = data[("compound", "c2c", "compound")].edge_index.T
            compound_edge_feature = data[("compound", "c2c", "compound")].edge_attr
            edge_weight = data[("compound", "c2c", "compound")].edge_weight
            compound_batch = data['compound'].batch
            compound_out = self.conv_compound(compound_edge_index,edge_weight,compound_edge_feature,compound_x.shape[0],compound_x)['node_feature']

        
        
        morgan = data.morgan
        # morgan = torch.tensor(morgan, dtype=torch.float)
        # print(morgan.dtype)
        morgan = self.morgan_mlp2(self.morgan_mlp1(self.morgan_mlp0(morgan).relu()).relu()).relu()
        # morgan = self.morgan_mlp2((self.morgan_mlp1((self.morgan_mlp0(morgan)))))
        # print(morgan.shape, compound_out.shape)
        # morgan = self.layern(morgan)
        
        # morgan_up = self.mlp2128(morgan)
        # morgan_up = self.layern1(morgan_up)
        # compound_out = self.layern2(compound_out)
        
        compound_out = compound_out + torch.matmul(torch.matmul(compound_out, morgan.T), morgan)
        
        # compound_out_down = self.mlp256(compound_out)
        # compound_out_down = self.layern3(compound_out_down)
        
        # compound_out_down = compound_out_down + torch.matmul(torch.matmul(compound_out_down, morgan.T), morgan)
        
        # compound_out = self.mlp22128(compound_out_down)
        
        
        
        
        
        
        # protein_batch version could further process b matrix. better than for loop.
        # protein_out_batched of shape b, n, c
        protein_out_batched, protein_out_mask = to_dense_batch(protein_out, protein_batch)
        compound_out_batched, compound_out_mask = to_dense_batch(compound_out, compound_batch)

        node_xyz = data.node_xyz

        p_coords_batched, p_coords_mask = to_dense_batch(node_xyz, protein_batch)
        c_rdkit_coords_batched, c_rdkit_coords_mask = to_dense_batch(data.rdkit_coord, compound_batch)
        # c_rdkit_coords_batched, c_rdkit_coords_mask = to_dense_batch(data.coords, compound_batch)
        
        protein_pair = get_pair_dis_one_hot(p_coords_batched, bin_size=2, bin_min=-1, bin_max=self.protein_bin_max)
        # compound_pair = get_pair_dis_one_hot(c_coords_batched, bin_size=1, bin_min=-0.5, bin_max=15)
        compound_pair_batched, compound_pair_batched_mask = to_dense_batch(data.compound_pair, data.compound_pair_batch)
        batch_n = compound_pair_batched.shape[0]
        max_compound_size_square = compound_pair_batched.shape[1]
        max_compound_size = int(max_compound_size_square**0.5)
        assert (max_compound_size**2 - max_compound_size_square)**2 < 1e-4
        compound_pair = torch.zeros((batch_n, max_compound_size, max_compound_size, 16)).to(data.compound_pair.device)
        for i in range(batch_n):
            one = compound_pair_batched[i]
            compound_size_square = (data.compound_pair_batch == i).sum()
            compound_size = int(compound_size_square**0.5)
            compound_pair[i,:compound_size, :compound_size] = one[:compound_size_square].reshape(
                                                                (compound_size, compound_size, -1))
        protein_pair = self.protein_pair_embedding(protein_pair.float())
        compound_pair = self.compound_pair_embedding(compound_pair.float())
        # b = torch.einsum("bik,bjk->bij", protein_out_batched, compound_out_batched).flatten()

        protein_out_batched = self.layernorm(self.mlp991(self.mlp99(protein_out_batched)))
        compound_out_batched = self.layernorm(self.mlp9991(self.mlp999(compound_out_batched)))
        # z of shape, b, protein_length, compound_length, channels.
        z = torch.einsum("bik,bjk->bijk", protein_out_batched, compound_out_batched)
        z_mask = torch.einsum("bi,bj->bij", protein_out_mask, compound_out_mask)
        # z = z * z_mask.unsqueeze(-1)
  
        new_x_shape = z.size()[:-1] + (3, 64)
        # vec = vec.view(*new_x_shape)
        # print(new_x_shape,p_coords_batched.shape,c_rdkit_coords_batched.shape)
        vec = torch.zeros(new_x_shape).cuda(2)
        # vec = torch.einsum("bick,bjck->bijck", self.layernorm12(self.mlp64(p_coords_batched.unsqueeze(-1))), self.layernorm12(self.mlp64(c_rdkit_coords_batched.unsqueeze(-1))))
        # vec = torch.einsum("bick,bjck->bijck", self.layernorm12(self.mlp64(p_coords_batched.unsqueeze(-1))), self.layernorm12(self.mlp64(c_rdkit_coords_batched.unsqueeze(-1))))
        # vec_mask = torch.einsum("bi,bj->bij", p_coords_mask, c_rdkit_coords_mask)
        
        
        if self.mode == 0:
            for _ in range(1):
                for i_module in range(self.n_trigonometry_module_stack):
                    z = z + self.dropout(self.protein_to_compound_list[i_module](z, protein_pair, compound_pair, z_mask.unsqueeze(-1)))
                    z = z + self.dropout(self.triangle_self_attention_list[i_module](z, z_mask,vec))
                    z = self.tranistion(z)
        # batch_dim = z.shape[0]

        b = self.linear(z).squeeze(-1)
        y_pred = b[z_mask]
        y_pred = y_pred.sigmoid() * 10   # normalize to 0 to 10.
        if self.readout_mode == 0:
            pair_energy = self.linear_energy(z).squeeze(-1) * z_mask
            affinity_pred = self.leaky(self.bias + ((pair_energy).sum(axis=(-1, -2))))
        if self.readout_mode == 1:
            # valid_interaction_z = (z * z_mask.unsqueeze(-1)).mean(axis=(1, 2))
            valid_interaction_z = (z * z_mask.unsqueeze(-1)).sum(axis=(1, 2)) / z_mask.sum(axis=(1, 2)).unsqueeze(-1)
            affinity_pred = self.linear_energy(valid_interaction_z).squeeze(-1)
            # print("z shape", z.shape, "z_mask shape", z_mask.shape,   "valid_interaction_z shape", valid_interaction_z.shape, "affinity_pred shape", affinity_pred.shape)
        if self.readout_mode == 2:
            pair_energy = (self.gate_linear(z).sigmoid() * self.linear_energy(z)).squeeze(-1) * z_mask
            affinity_pred = self.leaky(self.bias + ((pair_energy).sum(axis=(-1, -2))))
        return y_pred, affinity_pred



def get_model(mode, logging, device):
    if mode == 0:
        logging.info("5 stack, readout2, pred dis map add self attention and GVP embed, compound model GIN")
        model = IaBNet_with_affinity().to(device)
    return model
