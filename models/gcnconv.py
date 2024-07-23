import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, MLP, Linear


class GCNConv(MessagePassing):
    def __init__(self, edge_dim, hid_dim, num_mlp_layers, norm):
        super(GCNConv, self).__init__(aggr='add')

        self.lin_src = Linear(hid_dim, hid_dim)
        self.lin_dst = Linear(hid_dim, hid_dim)
        self.lin_edge = Linear(edge_dim, hid_dim)
        self.mlp = MLP([hid_dim] * (num_mlp_layers + 1), norm=norm, plain_last=False)

    def forward(self, x, edge_index, edge_attr, batch, norm):
        x = (self.lin_src(x[0]), x[1])

        if edge_attr is not None and hasattr(self, 'lin_edge'):
            edge_attr = self.lin_edge(edge_attr)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

        x_dst = x[1]
        x_dst = self.lin_dst(x_dst)
        out = out + x_dst

        return self.mlp(out, batch)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
