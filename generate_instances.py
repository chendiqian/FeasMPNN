from collections import namedtuple

import numpy as np
import scipy
import scipy.sparse
from scipy.optimize._linprog_util import _clean_inputs, _get_Abc, _presolve
from sklearn.datasets import make_sparse_spd_matrix

_LPProblem = namedtuple('_LPProblem',
                        'c A_ub b_ub A_eq b_eq bounds x0 integrality')
_LPProblem.__new__.__defaults__ = (None,) * 7  # make c the only required arg


def normalize_cons(A, b):
    if A is None or b is None:
        return A, b
    Ab = np.concatenate([A, b[:, None]], axis=1)
    max_logit = np.abs(Ab).max(axis=1)
    max_logit[max_logit == 0] = 1.
    Ab = Ab / max_logit[:, None]
    A = Ab[:, :-1]
    b = Ab[:, -1]
    return A, b


def postprocess(A, b):
    A, b = normalize_cons(A, b)

    # process LP into standard form Ax=b, x>=0
    lp = _LPProblem(np.ones(A.shape[1]), A, b, None, None, (0., 1.), None, None)
    # https://github.com/scipy/scipy/blob/v1.14.1/scipy/optimize/_linprog_util.py#L213
    lp = _clean_inputs(lp)
    # https://github.com/scipy/scipy/blob/v1.14.1/scipy/optimize/_linprog_util.py#L477
    lp, _, _, _, _, status, _ = _presolve(lp, True, None, tol=1e-9)
    if status != 0:
        success = False
    else:
        success = True

    A, b, *_ = _get_Abc(lp, 0.)
    return A, b, success


def generic(nrows, ncols, A_density, P_density, rng):
    nnzrs = int(nrows * ncols * A_density)
    assert nnzrs >= nrows  # at least 1 col per row
    assert nnzrs >= 2 * ncols  # at leats 2 rows per col

    # compute number of rows per column
    indices = rng.choice(ncols, size=nnzrs)  # random column indexes
    indices[:2 * ncols] = np.repeat(np.arange(ncols), 2)  # force at leats 2 rows per col
    _, col_nrows = np.unique(indices, return_counts=True)

    # for each column, sample random rows
    indices[:nrows] = rng.permutation(nrows)  # force at least 1 column per row
    i = 0
    indptr = [0]
    for n in col_nrows:

        # empty column, fill with random rows
        if i >= nrows:
            indices[i:i + n] = rng.choice(nrows, size=n, replace=False)

        # partially filled column, complete with random rows among remaining ones
        elif i + n > nrows:
            remaining_rows = np.setdiff1d(np.arange(nrows), indices[i:nrows],
                                          assume_unique=True)
            indices[nrows:i + n] = rng.choice(remaining_rows, size=i + n - nrows,
                                              replace=False)

        i += n
        indptr.append(i)

    # sparce CSC to sparse CSR matrix
    A = scipy.sparse.csc_matrix(
        (rng.randn(len(indices)), indices, indptr),
        shape=(nrows, ncols)).toarray().T
    b = rng.rand(A.shape[0])
    A, b, success = postprocess(A, b)

    if success:
        q = rng.randn(A.shape[1]).astype(np.float64)

        # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_spd_matrix.html
        P = make_sparse_spd_matrix(n_dim=A.shape[1], alpha=1 - P_density,
                                   smallest_coef=0.1, largest_coef=0.9, random_state=rng).astype(np.float64)
    else:
        q = P = None

    return A.astype(np.float64), b.astype(np.float64), P, q, success
