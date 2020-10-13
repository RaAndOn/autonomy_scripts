# /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def differentiate_polynomial(poly_deg, diff_deg):
    """
    """
    # Initialize the variable exponents for each term in the polynomial
    polynomial_exponential = np.reshape(
        np.arange(poly_deg + 1, dtype=int), (poly_deg + 1, 1))
    polynomial_constants = np.reshape(
        np.ones(poly_deg + 1, dtype=int), (poly_deg + 1, 1))

    # Differentiate polynomial
    for diff in range(diff_deg):
        polynomial_constants *= polynomial_exponential
        polynomial_exponential = np.where(
            polynomial_exponential - 1 < 0, 0, polynomial_exponential - 1)
    return polynomial_constants, polynomial_exponential


def polynomial_cost_function(poly_deg, diff_deg):
    """

    """
    # Differentiate polynomial
    polynomial_constants, polynomial_exponential = differentiate_polynomial(poly_deg,
                                                                            diff_deg)

    # square polynomial to make cost function quadratic
    polynomial_exponential_matrix = polynomial_exponential @ polynomial_exponential.T
    polynomial_constants_matrix = polynomial_constants @ polynomial_constants.T

    # Integrate cost function integrand
    nonzero = np.nonzero(polynomial_constants_matrix)
    polynomial_exponential_matrix[nonzero] += 1
    polynomial_constants_matrix[nonzero] = np.divide(
        polynomial_constants_matrix[nonzero], polynomial_exponential_matrix[nonzero])
    # cost_function_matrix = polynomial_constants_matrix * \
    #     (np.power(tf, polynomial_exponential_matrix) -
    #      np.power(ti, polynomial_exponential_matrix))

    return polynomial_constants_matrix, polynomial_exponential_matrix


def full_cost_matrix(poly_deg, diff_deg, task_space_size):
    """
    """
    polynomial_constants, polynomial_exponential = polynomial_cost_function(
        poly_deg, diff_deg)
    cost_matrix_size = (poly_deg + 1) * task_space_size
    cost_constants_matrix = np.zeros([cost_matrix_size, cost_matrix_size])
    cost_exponential_matrix = np.zeros([cost_matrix_size, cost_matrix_size])

    for i in range(task_space_size):
        lb = (poly_deg + 1) * i
        ub = (poly_deg + 1) * (i + 1)
        cost_constants_matrix[lb: ub, lb: ub] = polynomial_constants
        cost_exponential_matrix[lb: ub, lb: ub] = polynomial_exponential

    return cost_constants_matrix, cost_exponential_matrix


def polynomial_constraint_matrix(poly_deg, constraint_deg):
    """
    """
    # Create constraints matrix
    constraint_constants = np.array(
        [], dtype=np.int64).reshape(0, poly_deg + 1)
    constraint_exponentials = np.array(
        [], dtype=np.int64).reshape(0, poly_deg + 1)
    for deg in range(constraint_deg):
        constant_vec, exponential_vec = differentiate_polynomial(poly_deg, deg)
        constraint_constants = np.vstack(
            (constraint_constants, constant_vec.T))
        constraint_exponentials = np.vstack(
            (constraint_exponentials, exponential_vec.T))

    return constraint_constants, constraint_exponentials


def full_constraints_matrix(poly_deg, constraint_deg, task_space_size):
    """
    """
    constraint_constants, constraint_exponentials = polynomial_constraint_matrix(
        poly_deg, constraint_deg)
    constraint_matrix_cols = (poly_deg + 1) * task_space_size
    constraint_matrix_rows = constraint_deg * task_space_size
    constraint_constants_matrix = np.zeros(
        [constraint_matrix_rows, constraint_matrix_cols])
    constraint_exponential_matrix = np.zeros(
        [constraint_matrix_rows, constraint_matrix_cols])

    for i in range(task_space_size):
        lb = (poly_deg + 1) * i
        ub = (poly_deg + 1) * (i + 1)
        for deg in range(constraint_deg):
            row = constraint_deg * i + deg
            constraint_constants_matrix[row,
                                        lb:ub] = constraint_constants[deg, :]
            constraint_exponential_matrix[row,
                                          lb:ub] = constraint_exponentials[deg, :]

    return constraint_constants_matrix, constraint_exponential_matrix


def trajectory(polynomial_degree, differentiation_degree,
               time_waypoints, oos_waypoints):
    """
    """
    num_waypoints = time_waypoints.shape[0]
    task_space_size = pos_waypoints.shape[1]

    cost_constants_matrix, cost_exponential_matrix = full_cost_matrix(
        polynomial_degree, differentiation_degree, task_space_size)

    constraint_constants_matrix, constraint_exponentials_matrix = full_constraints_matrix(
        polynomial_degree, 3, task_space_size)

    tf = time_waypoints[1]
    ti = time_waypoints[0]
    cost_function_matrix = cost_constants_matrix * (np.power(tf, cost_exponential_matrix) -
                                                    np.power(ti, cost_exponential_matrix))
    constraint_start_matrix = constraint_constants_matrix * \
        np.power(ti, constraint_exponentials_matrix)
    constraint_goal_matrix = constraint_constants_matrix * \
        np.power(tf, constraint_exponentials_matrix)
    constraint_function_matrix = np.vstack(
        (constraint_start_matrix, constraint_goal_matrix))
    constraint_start = np.zeros(constraint_constants_matrix.shape[0])
    constraint_goal = np.zeros(constraint_constants_matrix.shape[0])
    constraint_start[::3] = pos_waypoints[0, :]
    constraint_goal[::3] = pos_waypoints[1, :]
    rhs = np.hstack(
        (np.zeros(cost_function_matrix.shape[0]), constraint_start, constraint_goal))
    solution = qp_solve(cost_function_matrix,
                        constraint_function_matrix, np.reshape(rhs, (rhs.shape[0], 1)))
    coeffs = np.reshape(solution[:-constraint_function_matrix.shape[0]],
                        (task_space_size, polynomial_degree+1))

    print(coeffs)
    return coeffs


def qp_solve(cost, constraint_eqn, rhs):
    """
    """
    block1 = np.vstack((2*cost, constraint_eqn))
    block2 = np.vstack((constraint_eqn.T, np.zeros(
        (constraint_eqn.shape[0], constraint_eqn.shape[0]))))

    bigblock = np.hstack((block1, block2))
    return np.linalg.pinv(bigblock) @ rhs


if __name__ == "__main__":
    pos_waypoints = np.array([[0, 0, 0],  # x1, y1, z1
                              [5, 3, 1]])  # x2, y2, z2
    time_waypoints = np.array([0, 5])  # t1, t2, t3
    polynomial_degree = 7
    differentiation_degree = 4
    coeffs = trajectory(polynomial_degree, differentiation_degree,
                        time_waypoints, pos_waypoints)

    times = np.linspace(0, 5, 11)
    x = np.array([np.polyval(np.flip(coeffs[0, :], axis=0), t) for t in times])
    y = np.array([np.polyval(np.flip(coeffs[1, :], axis=0), t) for t in times])
    z = np.array([np.polyval(np.flip(coeffs[2, :], axis=0), t) for t in times])

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.plot3D(x, y, z, 'gray')
    plt.show()
