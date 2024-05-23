import warnings
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from torch.nn import ModuleList
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.hetero_conv import group
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import Adj, EdgeType, NodeType
from torch_geometric.utils.hetero import check_add_self_loops


class HeteroConv(torch.nn.Module):
    def __init__(
        self,
        convs: Dict[EdgeType, Tuple[MessagePassing, int]],
        aggr: Optional[str] = "sum",
    ):
        super().__init__()

        for edge_type, (module, _) in convs.items():
            check_add_self_loops(module, [edge_type])

        src_node_types = set([key[0] for key in convs.keys()])
        dst_node_types = set([key[-1] for key in convs.keys()])
        if len(src_node_types - dst_node_types) > 0:
            warnings.warn(
                f"There exist node types ({src_node_types - dst_node_types}) "
                f"whose representations do not get updated during message "
                f"passing as they do not occur as destination type in any "
                f"edge type. This may lead to unexpected behavior.")

        self.convs = ModuleDict({'__'.join(k): v[0] for k, v in convs.items()})
        conv_rank = {'__'.join(k): v[1] for k, v in convs.items()}
        sorted_rank_value = sorted(set(conv_rank.values()))  # small value has high priority

        self.ranked_convs = ModuleList([])
        for i, rank in enumerate(sorted_rank_value):
            module_dict = ModuleDict({})
            for k, v in conv_rank.items():
                if v == rank:
                    module_dict[k] = self.convs[k]
            self.ranked_convs.append(module_dict)

        self.aggr = aggr

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        # for conv in self.convs.values():
        #     conv.reset_parameters()
        raise NotImplementedError

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        edge_attr_dict: Dict[EdgeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        for cur_rank, cur_convs in enumerate(self.ranked_convs):
            out_dict = defaultdict(list)
            for edge_type, edge_index in edge_index_dict.items():
                src, rel, dst = edge_type

                str_edge_type = '__'.join(edge_type)
                if str_edge_type not in cur_convs:
                    continue

                conv = cur_convs[str_edge_type]

                if src == dst:
                    out = conv(x_dict[src], edge_index, edge_attr_dict[edge_type], batch_dict[dst])
                else:
                    out = conv((x_dict[src], x_dict[dst]), edge_index, edge_attr_dict[edge_type], batch_dict[dst])

                out_dict[dst].append(out)

            for key, value in out_dict.items():
                x_dict[key] = group(value, self.aggr)

        return x_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'
