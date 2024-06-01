import gzip
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

from numpy import zeros
import torch
from torch_geometric.data import Batch, HeteroData, InMemoryDataset
from tqdm import tqdm

from solver.linprog_ip import _ip_hsd_feas


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
        return ['instance_0.pkl.gz']  # there should be at least one pkg

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def process(self):
        # num_instance_pkg = len([n for n in os.listdir(self.raw_dir) if n.endswith('pkl.gz')])

        # data_list = []
        # for i in range(num_instance_pkg):
        #     # load instance
        #     print(f"processing {i}th package, {num_instance_pkg} in total")
        #     with gzip.open(os.path.join(self.raw_dir, f"instance_{i}.pkl.gz"), "rb") as file:
        #         ip_pkgs = pickle.load(file)
        #
        #     for ip_idx in tqdm(range(len(ip_pkgs))):
        #         (A, b, c, x, proj_matrix) = ip_pkgs[ip_idx]
        #
        #         # find some random, feasible initial point
        #         x_feasible, *_ = _ip_hsd_feas(A, b, zeros(A.shape[1]), 0.,
        #                                       alpha0=0.99995, beta=0.1,
        #                                       maxiter=100, tol=1.e-6, sparse=False,
        #                                       lstsq=False, sym_pos=True, cholesky=None,
        #                                       pc=True, ip=True, permc_spec='MMD_AT_PLUS_A',
        #                                       rand_start=True)
        #
        #         A = torch.from_numpy(A).to(torch.float)
        #         b = torch.from_numpy(b).to(torch.float)
        #         c = torch.from_numpy(c).to(torch.float)
        #         x = torch.from_numpy(x).to(torch.float)
        #         x_feasible = torch.from_numpy(x_feasible).to(torch.float)
        #
        #         A_row, A_col = torch.where(A)
        #
        #         # proj_matrix @ x projects onto the nullspace of A
        #         proj_matrix = torch.from_numpy(proj_matrix).to(torch.float)
        #
        #         data = HeteroData(
        #             cons={'x': torch.cat([A.mean(1, keepdims=True),
        #                                   A.std(1, keepdims=True)], dim=1)},
        #             vals={'x': torch.cat([A.mean(0, keepdims=True),
        #                                   A.std(0, keepdims=True)], dim=0).T},
        #             obj={'x': torch.cat([c.mean(0, keepdims=True),
        #                                  c.std(0, keepdims=True)], dim=0)[None]},
        #
        #             # assign half of the symmetry, need to finish it in the collate function
        #             cons__to__vals={'edge_index': torch.vstack(torch.where(A)),
        #                             'edge_attr': A[torch.where(A)][:, None]},
        #             # vals__to__cons={'edge_index': torch.vstack(torch.where(A.T)),
        #             #                 'edge_attr': A.T[torch.where(A.T)][:, None]},
        #             vals__to__obj={'edge_index': torch.vstack([torch.arange(A.shape[1]),
        #                                                        torch.zeros(A.shape[1], dtype=torch.long)]),
        #                            'edge_attr': c[:, None]},
        #             # obj__to__vals={'edge_index': torch.vstack([torch.zeros(A.shape[1], dtype=torch.long),
        #             #                                            torch.arange(A.shape[1])]),
        #             #                'edge_attr': c[:, None]},
        #             cons__to__obj={'edge_index': torch.vstack([torch.arange(A.shape[0]),
        #                                                        torch.zeros(A.shape[0], dtype=torch.long)]),
        #                            'edge_attr': b[:, None]},
        #             # obj__to__cons={'edge_index': torch.vstack([torch.zeros(A.shape[0], dtype=torch.long),
        #             #                                            torch.arange(A.shape[0])]),
        #             #                'edge_attr': b[:, None]},
        #
        #             x_solution=x,
        #             x_feasible=x_feasible,
        #             obj_solution=c.dot(x),
        #             c=c,
        #             b=b,
        #             A_row=A_row,
        #             A_col=A_col,
        #             A_val=A[A_row, A_col],
        #             proj_matrix=proj_matrix.reshape(-1),
        #         )
        #
        #         if self.pre_filter is not None:
        #             raise NotImplementedError
        #
        #         if self.pre_transform is not None:
        #             data = self.pre_transform(data)
        #
        #         data_list.append(data)
        #
        #     torch.save(Batch.from_data_list(data_list), osp.join(self.processed_dir, f'batch{i}.pt'))
        #     data_list = []

        num_instance_pkg = len([n for n in os.listdir(self.processed_dir) if n.endswith('.pt')])
        data_list = []
        for i in range(num_instance_pkg):
            data_list.extend(Batch.to_data_list(torch.load(osp.join(self.processed_dir, f'batch{i}.pt'))))
        torch.save(self.collate(data_list), osp.join(self.processed_dir, 'data.pt'))
