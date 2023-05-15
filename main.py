import matplotlib.pyplot as plt
from generate_quadratic import generate_quadratic_objective
from algorithm import gauss_southwell
from utilities import *

# block coordinate Gauss-Southwell for the optimization of a quadratic function
# objective f(x) = 0.5 * x^T Q x + b^T x

n = 1000 # problem dimension
density = 0.1 # density of the matrix A such that Q = A^T A
num_iters = 3000 # max number of iterations

# Generate a quadratic objective function f(x) = 0.5 * x^T Q x + b^T x
# fast_updates = function to update gradient and objective quickly
# safe_updates = function to update gradient and objective safely but more slowly
fast_updates_f,  safe_updates_f, Q, b = generate_quadratic_objective(n, density=density)

# Compute the optimal solution and the optimal objective value
x_optimal = np.linalg.solve(Q, -b)
f_optimal = 0.5 * x_optimal.T @ Q @ x_optimal + b.T @ x_optimal

# Initial guess
np.random.seed(1)
x0 = np.random.randn(n)

# stepsize strategies
# L: Lipschitz constant stepsize
# block_L: block coordinate Lipschitz constant stepsize
# exact: exact linesearch stepsize
vec_stepsize_strats = ['L', 'block_L', 'exact']

# priority strategies
# abs: component with largest absolute value
# abs_L: component with largest absolute value divided by block Lipschitz constant
# max_improvement: component that maximizes the improvement in the objective function with linesearch
vec_priority_strats = ['abs', 'abs_L', 'max_improvement']

# dictionary to store function history for each strategy
dict_f_history = {}
for i in range(len(vec_stepsize_strats)):
    # choose stepsize strategy
    stepsize_strat = vec_stepsize_strats[i]
    # choose priority strategy
    priority = vec_priority_strats[i]
    strategy = stepsize_strat + '+' + priority
    # run gauss-southwell
    x_opt, dict_f_history[strategy] = gauss_southwell(fast_updates_f, safe_updates_f, x0,
                                                            f_priorities=max_priority_generator(priority, Q),
                                       stepsize_strat = stepsize_creator(stepsize_strat, Q), num_iters=num_iters)

# Start creating the plot
plt.figure()

# plot each strategy
for i in range(len(vec_stepsize_strats)):
    stepsize_strat = vec_stepsize_strats[i]
    priority = vec_priority_strats[i]
    strategy = stepsize_strat + '+' + priority
    # get the function history for the strategy
    f_history = dict_f_history[strategy]
    # Subtract the optimal function value from all function values
    dist_to_optimal = [abs(val - f_optimal) for val in f_history]
    x_values = list(range(len(f_history)))
    plt.semilogy(x_values, dist_to_optimal, label=strategy)

# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Comparison of Step Size and Priority Strategies')

# Add a legend
plt.legend()

# Show the plot
plt.show()
