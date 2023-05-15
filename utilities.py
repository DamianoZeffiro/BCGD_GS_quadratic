import numpy as np

# stepsize_strategies
def stepsize_creator(stepsize_strat, Q):
    if stepsize_strat == 'L':
        L = np.linalg.norm(Q, ord=2)
        def stepsize_strat(grad_i, i):
            return 1 / L
    elif stepsize_strat == 'block_L':
        vec_norms = np.linalg.norm(Q, ord=2, axis=0)
        def stepsize_strat(grad_i, i):
            return 1 / vec_norms[i]
    elif stepsize_strat == 'exact':
        vec_curvatures = np.diag(Q)
        def stepsize_strat(grad_i, i):
            return 1 / vec_curvatures[i]
    return stepsize_strat

# priority_strategies
def max_priority_generator(priority_type, Q):
    if priority_type == 'abs':
        def abs_val(val, idx):
            return abs(val)
        return abs_val
    elif priority_type == 'abs_L':
        vec_norms = np.linalg.norm(Q, ord=2, axis=0)
        def abs_val_scaled(val, idx):
            return abs(val) / vec_norms[idx]
        return abs_val_scaled
    elif priority_type == 'max_improvement':
        vec_curvatures = np.diag(Q)
        def improvement(val, idx):
            return val**2/vec_curvatures[idx]
        return improvement
