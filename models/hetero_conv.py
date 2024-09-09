import inspect
from typing import Dict
from typing import Optional

import torch
from torch_geometric.typing import EdgeType, NodeType


class TripartiteConv(torch.nn.Module):
    def __init__(
            self,
            v2h_conv: torch.nn.Module,
            h2v_conv: torch.nn.Module,
            v2c_conv: torch.nn.Module,
            c2v_conv: torch.nn.Module,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleDict(
            {'vals_hids': v2h_conv,
             'hids_vals': h2v_conv,
             'vals_cons': v2c_conv,
             'cons_vals': c2v_conv}
        )
        self.has_skip = 'x_0' in inspect.signature(v2c_conv.forward).parameters.keys()

    def forward(
            self,
            x_dict: Dict[NodeType, torch.FloatTensor],
            x0_dict: Dict[NodeType, torch.FloatTensor],
            batch_dict: Dict[NodeType, torch.LongTensor],
            edge_index_dict: Dict[EdgeType, torch.LongTensor],
            edge_attr_dict: Dict[EdgeType, torch.FloatTensor],
            norm_dict: Dict[EdgeType, Optional[torch.FloatTensor]]
    ) -> Dict[NodeType, torch.FloatTensor]:

        for src, rel, dst in [('vals', 'to', 'hids'),
                              ('hids', 'to', 'vals'),
                              ('vals', 'to', 'cons'),
                              ('cons', 'to', 'vals')]:
            args = [(x_dict[src], x_dict[dst])]
            if self.has_skip:
                args.append(x0_dict[dst])
            args = args + [edge_index_dict[(src, rel, dst)],
                           edge_attr_dict[(src, rel, dst)],
                           batch_dict[dst]]
            if norm_dict[(src, rel, dst)] is not None:
                args.append(norm_dict[(src, rel, dst)])

            x_dict[dst] = self.convs['_'.join([src, dst])](*args)

        return x_dict
