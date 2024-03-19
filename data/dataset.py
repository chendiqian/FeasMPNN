import gzip
import os
import os.path as osp
import pickle
from collections import namedtuple
from typing import Callable, List, Optional

import torch
from scipy.optimize._linprog_util import _clean_inputs, _get_Abc
from torch_geometric.data import Batch, HeteroData, InMemoryDataset
from torch_sparse import SparseTensor
from tqdm import tqdm

# https://github.com/scipy/scipy/blob/e574cbcabf8d25955d1aafeed02794f8b5f250cd/scipy/optimize/_linprog_util.py#L15
_LPProblem = namedtuple('_LPProblem',
                        'c A_ub b_ub A_eq b_eq bounds x0 integrality')
_LPProblem.__new__.__defaults__ = (None,) * 7  # make c the only required arg


class LPDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        lappe: int = 0,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.extra_path = f'{lappe}lap_'
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, 'data.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return ['instance_0.pkl.gz']   # there should be at least one pkg

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed_' + self.extra_path)

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
                (A_original, b_original, c_original, obj, x) = ip_pkgs[ip_idx]

                A = torch.from_numpy(A_original).to(torch.float)
                b = torch.from_numpy(b_original).to(torch.float)
                c = torch.from_numpy(c_original).to(torch.float)

                lp = _LPProblem(c_original, A_original, b_original, None, None, (0, 1), None, None)
                lp = _clean_inputs(lp)
                A_full, b_full, c_full, *_ = _get_Abc(lp, 0.)  # standard from Ax = b

                sp_a = SparseTensor.from_dense(A, has_value=True)
                sp_a_full = SparseTensor.from_dense(torch.from_numpy(A_full).to(torch.float), has_value=True)

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

                    obj_val=obj,
                    solution=torch.from_numpy(x).to(torch.float),
                    c=c,
                    b=b,
                    A_row=sp_a.storage._row,
                    A_col=sp_a.storage._col,
                    A_val=sp_a.storage._value,
                    b_full=torch.from_numpy(b_full).to(torch.float),
                    c_full=torch.from_numpy(c_full).to(torch.float),
                    A_full_row=sp_a_full.storage._row,
                    A_full_col=sp_a_full.storage._col,
                    A_full_val=sp_a_full.storage._value,
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
