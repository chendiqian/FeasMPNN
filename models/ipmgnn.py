import torch
from models.hetero_gnn import BipartiteHeteroGNN
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_batch


class BipartiteIPMGNN(BipartiteHeteroGNN):
    def __init__(self,
                 conv,
                 head,
                 concat,
                 hid_dim,
                 num_conv_layers,
                 num_pred_layers,
                 hid_pred,
                 num_mlp_layers,
                 norm):
        super().__init__(conv,
                         head,
                         concat,
                         hid_dim,
                         num_conv_layers,
                         num_pred_layers,
                         hid_pred,
                         num_mlp_layers,
                         norm,
                         False, False)

        self.predictor = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.predictor.append(MLP([hid_dim] * num_pred_layers + [1]))

    def forward(self, data):
        cons_embedding = self.b_encoder(data.b[:, None])
        vals_embedding = self.obj_encoder(data.c[:, None])

        hiddens = []
        edge_norms = data.norm if hasattr(data, 'norm') else None
        cons_embedding_0 = cons_embedding
        vals_embedding_0 = vals_embedding
        for i in range(self.num_layers):
            vals_embedding, cons_embedding = self.gcns[i](cons_embedding,
                                                          vals_embedding,
                                                          cons_embedding_0,
                                                          vals_embedding_0,
                                                          data['vals', 'to', 'cons'].edge_index,
                                                          data['cons', 'to', 'vals'].edge_index,
                                                          data['vals', 'to', 'cons'].edge_attr,
                                                          data['cons', 'to', 'vals'].edge_attr,
                                                          data['cons'].batch,
                                                          data['vals'].batch,
                                                          edge_norms)
            hiddens.append(vals_embedding)

        vals = torch.cat([self.predictor[i](hiddens[i]) for i in range(self.num_layers)], dim=1)
        return vals

    @torch.no_grad()
    def evaluation(self, data):
        opt_obj = data.obj_solution
        batched_c, _ = to_dense_batch(data.c, data['vals'].batch)  # batchsize x max_nnodes

        # prediction
        pred_x = self.forward(data)[0][:, -1]
        pred_x, real_node_mask = to_dense_batch(pred_x, data['vals'].batch)  # batchsize x max_nnodes
        batch_obj = (pred_x * batched_c).sum(1)
        return pred_x[real_node_mask], torch.abs((opt_obj - batch_obj) / opt_obj), None, None
