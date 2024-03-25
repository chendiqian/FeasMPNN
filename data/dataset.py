import gzip
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import torch
from torch_geometric.data import Batch, HeteroData, InMemoryDataset
from tqdm import tqdm

from scipy.optimize._linprog_util import _clean_inputs, _get_Abc
from collections import namedtuple


_LPProblem = namedtuple('_LPProblem',
                        'c A_ub b_ub A_eq b_eq bounds x0 integrality')
_LPProblem.__new__.__defaults__ = (None,) * 7  # make c the only required arg


class LPDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, 'data.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return ['instance_0.pkl.gz']   # there should be at least one pkg

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def process(self):
        num_instance_pkg = len([n for n in os.listdir(self.raw_dir) if n.endswith('pkl.gz')])

        data_list = []
        for i in range(num_instance_pkg):
            # load instance
            print(f"processing {i}th package, {num_instance_pkg} in total")
            with gzip.open(os.path.join(self.raw_dir, f"instance_{i}.pkl.gz"), "rb") as file:
                ip_pkgs = pickle.load(file)

            for ip_idx in tqdm(range(len(ip_pkgs))):
                (A, b, c, x) = ip_pkgs[ip_idx]

                lp = _LPProblem(c, A, b, None, None, (0., 1.), None, None)
                lp = _clean_inputs(lp)
                A_full, b_full, c_full, *_ = _get_Abc(lp, 0.)

                A = torch.from_numpy(A).to(torch.float)
                b = torch.from_numpy(b).to(torch.float)
                c = torch.from_numpy(c).to(torch.float)

                A_row, A_col = torch.where(A)
                A_full = torch.from_numpy(A_full)
                A_full_row, A_full_col = torch.where(A_full)

                data = HeteroData(
                    cons={'x': torch.cat([A.mean(1, keepdims=True),
                                          A.std(1, keepdims=True)], dim=1)},
                    vals={'x': torch.cat([A.mean(0, keepdims=True),
                                          A.std(0, keepdims=True)], dim=0).T},
                    obj={'x': torch.cat([c.mean(0, keepdims=True),
                                         c.std(0, keepdims=True)], dim=0)[None]},

                    cons__to__vals={'edge_index': torch.vstack(torch.where(A)),
                                    'edge_attr': A[torch.where(A)][:, None]},
                    vals__to__cons={'edge_index': torch.vstack(torch.where(A.T)),
                                    'edge_attr': A.T[torch.where(A.T)][:, None]},
                    vals__to__obj={'edge_index': torch.vstack([torch.arange(A.shape[1]),
                                                               torch.zeros(A.shape[1], dtype=torch.long)]),
                                   'edge_attr': c[:, None]},
                    obj__to__vals={'edge_index': torch.vstack([torch.zeros(A.shape[1], dtype=torch.long),
                                                               torch.arange(A.shape[1])]),
                                   'edge_attr': c[:, None]},
                    cons__to__obj={'edge_index': torch.vstack([torch.arange(A.shape[0]),
                                                               torch.zeros(A.shape[0], dtype=torch.long)]),
                                   'edge_attr': b[:, None]},
                    obj__to__cons={'edge_index': torch.vstack([torch.zeros(A.shape[0], dtype=torch.long),
                                                               torch.arange(A.shape[0])]),
                                   'edge_attr': b[:, None]},

                    x_solution=torch.from_numpy(x).to(torch.float),
                    c=c,
                    b=b,
                    c_full=torch.from_numpy(c_full),
                    b_full=torch.from_numpy(b_full),
                    A_row=A_row,
                    A_col=A_col,
                    A_val=A[A_row, A_col],
                    A_full_row=A_full_row,
                    A_full_col=A_full_col,
                    A_full_val=A_full[A_full_row, A_full_col]
                    )

                if self.pre_filter is not None:
                    raise NotImplementedError

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            torch.save(Batch.from_data_list(data_list), osp.join(self.processed_dir, f'batch{i}.pt'))
            data_list = []

        data_list = []
        for i in range(num_instance_pkg):
            data_list.extend(Batch.to_data_list(torch.load(osp.join(self.processed_dir, f'batch{i}.pt'))))
        torch.save(self.collate(data_list), osp.join(self.processed_dir, 'data.pt'))
