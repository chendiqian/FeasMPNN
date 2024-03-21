import warnings
from collections import namedtuple

import numpy as np
from scipy.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve, lstsq

# https://github.com/scipy/scipy/blob/e574cbcabf8d25955d1aafeed02794f8b5f250cd/scipy/optimize/_linprog_util.py#L15
_LPProblem = namedtuple('_LPProblem',
                        'c A_ub b_ub A_eq b_eq bounds x0 integrality')
_LPProblem.__new__.__defaults__ = (None,) * 7  # make c the only required arg
SMALL_EPS = 1.e-7


def mu(x, s):
    return x.dot(s) / len(x)


def smart_start(A, b, c, option='smart'):
    if option == 'smart':
        a_at_inv = cho_factor(A @ A.T)
        x_tilde = A.T @ cho_solve(a_at_inv, b)
        lambda_tilde = cho_solve(a_at_inv, A @ c)
        s_tilde = c - A.T @ lambda_tilde

        delta_x = max(0, - 1.5 * np.min(x_tilde))
        delta_s = max(0, - 1.5 * np.min(s_tilde))

        x_cap = x_tilde + delta_x
        s_cap = s_tilde + delta_s

        delta_x_cap = 0.5 * np.dot(x_cap, s_cap) / np.sum(s_cap)
        delta_s_cap = 0.5 * np.dot(x_cap, s_cap) / np.sum(x_cap)

        x0 = x_cap + delta_x_cap
        lambda0 = lambda_tilde
        s0 = s_cap + delta_s_cap
    elif option == 'dumb':
        x0 = np.ones(A.shape[1])
        lambda0 = np.zeros(A.shape[0])
        s0 = np.ones(A.shape[1])
    elif option == 'rand':
        x0 = np.random.rand(A.shape[1])
        lambda0 = np.random.rand(A.shape[0])
        s0 = np.random.rand(A.shape[1])
    else:
        raise ValueError
    return x0, lambda0, s0


def ipm_overleaf(c,
                 A,
                 b,
                 init='smart',
                 lin_solver='cho',
                 max_iter=10000,
                 tol=1.e-6,
                 sigma=0.3):
    x, lambd, s = smart_start(A, b, c, init)
    _mu = mu(x, s)
    last_x = x

    pbar = range(max_iter)
    for iteration in pbar:
        try:
            s_inv = (s + SMALL_EPS) ** -1
            xs_inv = x * s_inv
            A_XS_inv = A * xs_inv[None]
            M = A_XS_inv @ A.transpose()
            rhs = b - A @ x + A_XS_inv @ c - M @ lambd - A @ s_inv * sigma * _mu

            # solve M @ x = rhs
            if lin_solver == 'cho':
                c_and_lower = cho_factor(M)
                grad_lambda = cho_solve(c_and_lower, rhs)
            elif lin_solver == 'lstsq':
                grad_lambda = lstsq(M, rhs)[0]
            else:
                raise NotImplementedError

            AT_lambda_plut_dlambda = A.transpose() @ (lambd + grad_lambda)
            grad_s = - AT_lambda_plut_dlambda - s + c
            grad_x = s_inv * sigma * _mu + xs_inv * (AT_lambda_plut_dlambda - c)

            alpha = 1.
            gradx_mask = grad_x < 0
            if np.any(gradx_mask):
                alpha = min(alpha, (-x[gradx_mask] / grad_x[gradx_mask]).min())
            grads_mask = grad_s < 0
            if np.any(grads_mask):
                alpha = min(alpha, (-s[grads_mask] / grad_s[grads_mask]).min())
            alpha_l = alpha_s = alpha_x = alpha

            x = x + alpha_x * grad_x
            lambd = lambd + alpha_l * grad_lambda
            s = s + alpha_s * grad_s
            _mu = mu(x, s)

            if np.abs(x - last_x).max() < tol:
                break
            last_x = x
        except (LinAlgError, FloatingPointError, ValueError, ZeroDivisionError):
            warnings.warn(f'Instability occured at iter {iteration}, turning to lstsq')
            lin_solver = 'lstsq'

    sol = {
        'x': x,
        'lambd': lambd,
        's': s,
        'nit': iteration}
    return sol
