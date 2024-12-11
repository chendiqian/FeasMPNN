import inspect
from typing import Dict
from typing import Optional

import torch
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.nn.conv.hetero_conv import group


class TripartiteConv(torch.nn.Module):
    def __init__(
            self,
            v2v_conv: torch.nn.Module,
            v2c_conv: torch.nn.Module,
            c2v_conv: torch.nn.Module,
            v2o_conv: torch.nn.Module,
            o2v_conv: torch.nn.Module,
            c2o_conv: torch.nn.Module,
            o2c_conv: torch.nn.Module,
            sync_conv: bool = False
    ):
        super().__init__()

        self.convs = torch.nn.ModuleDict(
            {'vals_vals': v2v_conv,
             'vals_cons': v2c_conv,
             'cons_vals': c2v_conv,
             'vals_obj': v2o_conv,
             'obj_vals': o2v_conv,
             'cons_obj': c2o_conv,
             'obj_cons': o2c_conv}
        )
        self.has_skip = 'x_0' in inspect.signature(v2c_conv.forward).parameters.keys()
        # we use c -> o -> v setting as is the best in IPMGNN
        self.conv_sequence = [('vals_cons', 'obj_cons'),
                              ('cons_obj', 'vals_obj'),
                              ('vals_vals', 'cons_vals', 'obj_vals')]
        self.sync_conv = sync_conv

    def forward(
            self,
            x_dict: Dict[NodeType, torch.FloatTensor],
            x0_dict: Dict[NodeType, torch.FloatTensor],
            batch_dict: Dict[NodeType, torch.LongTensor],
            edge_index_dict: Dict[EdgeType, torch.LongTensor],
            edge_attr_dict: Dict[EdgeType, torch.FloatTensor],
            norm_dict: Dict[EdgeType, Optional[torch.FloatTensor]]
    ) -> Dict[NodeType, torch.FloatTensor]:

        new_dict = {}
        for conv_group in self.conv_sequence:
            current_results = []
            dst = conv_group[0].split('_')[1]
            for conv in conv_group:
                src, dst = conv.split('_')
                args = [(x_dict[src], x_dict[dst])]
                if self.has_skip:
                    args.append(x0_dict[dst])
                args = args + [edge_index_dict[(src, 'to', dst)],
                               edge_attr_dict[(src, 'to', dst)],
                               batch_dict[dst]]
                if norm_dict[(src, 'to', dst)] is not None:
                    args.append(norm_dict[(src, 'to', dst)])

                current_results.append(self.convs[conv](*args))

            if self.sync_conv:
                new_dict[dst] = group(current_results, 'mean')
            else:
                x_dict[dst] = group(current_results, 'mean')

        return new_dict if self.sync_conv else x_dict


class BipartiteConv(torch.nn.Module):
    def __init__(
            self,
            v2v_conv: torch.nn.Module,
            v2c_conv: torch.nn.Module,
            c2v_conv: torch.nn.Module,
            sync_conv: bool = False
    ):
        super().__init__()

        self.convs = torch.nn.ModuleDict(
            {'vals_vals': v2v_conv,
             'vals_cons': v2c_conv,
             'cons_vals': c2v_conv}
        )
        self.has_skip = 'x_0' in inspect.signature(v2c_conv.forward).parameters.keys()
        self.sync_conv = sync_conv

    def forward(
            self,
            x_dict: Dict[NodeType, torch.FloatTensor],
            x0_dict: Dict[NodeType, torch.FloatTensor],
            batch_dict: Dict[NodeType, torch.LongTensor],
            edge_index_dict: Dict[EdgeType, torch.LongTensor],
            edge_attr_dict: Dict[EdgeType, torch.FloatTensor],
            norm_dict: Dict[EdgeType, Optional[torch.FloatTensor]]
    ) -> Dict[NodeType, torch.FloatTensor]:

        new_dict = {}
        for src, rel, dst in [('vals', 'to', 'vals'),
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

            if self.sync_conv:
                new_dict[dst] = self.convs['_'.join([src, dst])](*args)
            else:
                x_dict[dst] = self.convs['_'.join([src, dst])](*args)

        return new_dict if self.sync_conv else x_dict
