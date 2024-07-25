import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, Linear


class SGCNConv(MessagePassing):
    def __init__(self, edge_dim, hid_dim):
        super(SGCNConv, self).__init__(aggr='add')
        self.lin_edge = Linear(edge_dim, hid_dim)

    def forward(self, x, edge_index, edge_attr, batch, norm):
        if edge_attr is not None and hasattr(self, 'lin_edge'):
            edge_attr = self.lin_edge(edge_attr)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)
        out = (out + x[1]) / 2 ** 0.5

        return out

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu((x_j + edge_attr) / 2 ** 0.5)

    def update(self, aggr_out):
        return aggr_out
