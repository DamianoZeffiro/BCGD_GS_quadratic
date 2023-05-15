import numpy as np
from heap_class import Heap

def gauss_southwell(fast_updates_f, safe_updates_f, x0, num_iters=1000, tol=1e-6, f_priorities=None, stepsize_strat = None):
    # Initialize the solution
    x = x0.copy()
    n = len(x)
    # Create a max heap for storing the gradients
    fx, grad = safe_updates_f(x)
    heap_gradient = Heap(grad, f_priorities)
    f_history = [fx]
    for _ in range(num_iters):
        # Extract the feature with the largest gradient
        max_grad_val, idx = heap_gradient.get_max()

        # Perform a line search along the direction of the largest gradient
        step_size = stepsize_strat(grad[idx], idx)
        x[idx] -= step_size * grad[idx]
        # update the function information
        fx, grad, list_updated = fast_updates_f(fx, grad, step_size, idx)
        f_history.append(fx)
        # Push the updated gradient components to the heap
        for i in list_updated:
            heap_gradient.update_priority(i, grad[i])
        # Check the convergence criterion
        if abs(max_grad_val) < tol:
            break

    return x, f_history