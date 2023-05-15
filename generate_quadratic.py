import numpy as np
from scipy import sparse


def generate_quadratic_objective(n, density=0.1, alpha = 0.1):
    """
    Generate a quadratic objective function with a sparse positive semi-definite Q.

    :param n: The size of the input vector x.
    :param density: The density of the sparse matrix.
    :return: A function that calculates the value of the objective function and its gradient.
    """
    # Generate a random sparse matrix
    np.random.seed(0)
    A = sparse.random(n, n, density=density)

    # Generate Q as the product of A and its transpose
    Q = A.T @ A

    Q = Q.toarray() + alpha * np.eye(n)

    # Generate a random vector b
    b = np.random.randn(n)

    # dict non zero elements
    dict_nonzeros = {i: np.nonzero(Q[:, i])[0] for i in range(Q.shape[1])}

    # Define the objective function
    def f(x):
        return 0.5 * x.T @ Q @ x + b.T @ x, Q @ x + b

    def grad_f(x):
        return Q @ x + b

    def fast_updates_f(fx, gx, stepsize, idx):
        list_nonzeros = dict_nonzeros[idx]
        fx = fx - stepsize * gx[idx]**2 + 0.5 * stepsize**2 * Q[idx, idx] * gx[idx]**2
        gx[list_nonzeros] = gx[list_nonzeros] - stepsize * Q[list_nonzeros, idx] * gx[idx]
        return fx, gx, list_nonzeros

    def safe_updates_f(x):
        Qx = Q @ x
        bx = b.T @ x
        gx = Qx + b
        fx = 0.5 * x.T @ Qx + bx
        return fx, gx
    return fast_updates_f, safe_updates_f, Q, b
