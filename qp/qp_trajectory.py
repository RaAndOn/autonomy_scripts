# /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def differentiate_polynomial(polynomial_degree, differentiation_degree):
    """
    This function creates a representation of polynomial with no coefficients
    and then differentiates the polynomial returning the new polynomial with
    constant multipliers

    Parameters
    polynomial_degree - Degree of the polynomial
    differentiation_degree - Number of degrees to differentiate the polynomial

    Return
    polynomial_constants - Vector where each cell representing the polynomial coefficients
    polynomial_exponential - Vector where each cell represents the power to which the polynomial is raised
    """
    # Initialize the constant and exponent vectors representing only the variable terms with no coefficients
    # Exponential Example: [0 1 2] represents the exponents x^0 + x^1 + x^2
    # Constants Example: [4 0 4] would represent the constants: [4]x^0 + [0]x^1 +[4]x^2
    polynomial_exponential = np.reshape(
        np.arange(polynomial_degree + 1, dtype=int), (polynomial_degree + 1, 1))
    polynomial_constants = np.reshape(
        np.ones(polynomial_degree + 1, dtype=int), (polynomial_degree + 1, 1))

    # Differentiate polynomial representation
    for degree in range(differentiation_degree):
        polynomial_constants *= polynomial_exponential
        polynomial_exponential = np.where(
            polynomial_exponential - 1 < 0, 0, polynomial_exponential - 1)

    # Return the differentiated polynomial representation
    return polynomial_constants, polynomial_exponential


def time_dependent_cost(polynomial_degree, differentiation_degrees):
    """
    This function creates a matrix representing the portions of the cost function dependent
    on time. It does this by creating a polynomial and differentiates it according to the parameters.
    It then takes the outer product of the polynomial with itself and integrates the outer product.

    Parameters
    polynomial_degree - Degree of the polynomial
    differentiation_degree - Number of degrees to differentiate the polynomial

    Return
    Outer product of the differentiated polynomial
    """
    # Get differentiated polynomial
    polynomial_constants, polynomial_exponential = differentiate_polynomial(
        polynomial_degree, differentiation_degrees)

    # Take polynomial outer product for the cost function quadratic
    cost_exponentials = polynomial_exponential @ polynomial_exponential.T
    cost_constants = polynomial_constants @ polynomial_constants.T

    # Integrate outer product symbolically
    nonzero_indices = np.nonzero(cost_constants)
    cost_exponentials[nonzero_indices] += 1
    cost_constants[nonzero_indices] = np.divide(
        cost_constants[nonzero_indices], cost_exponentials[nonzero_indices])

    # Return integrated outer product of differentiated polynomial
    return cost_constants, cost_exponentials


def full_task_space_time_dependent_cost(polynomial_degree, differentiation_degree, task_space_size):
    """
    The trajectory can be solved for simultaneously across all task space variables
    by placing all of the cost terms in a giant sparse matrix.

    Parameters
    polynomial_degree - Degree of the polynomial
    differentiation_degree - Number of degrees to differentiate the polynomial
    task_space_size - Size of the task space

    Return
    Matrix with cost matrices for all task space variables
    """
    # Calculate the cost terms dependent on time
    cost_constants, cost_exponentials = time_dependent_cost(
        polynomial_degree, differentiation_degree)

    # Initialize full cost matrix as zeros
    cost_matrix_size = (polynomial_degree + 1) * task_space_size
    full_cost_matrix_constants = np.zeros([cost_matrix_size, cost_matrix_size])
    full_cost_matrix_exponentials = np.zeros(
        [cost_matrix_size, cost_matrix_size])

    # Populate full cost matrix, with each task variable getting its own rows an columns
    for i in range(task_space_size):
        lb = (polynomial_degree + 1) * i
        ub = (polynomial_degree + 1) * (i + 1)
        full_cost_matrix_constants[lb: ub, lb: ub] = cost_constants
        full_cost_matrix_exponentials[lb: ub, lb: ub] = cost_exponentials

    # Return full cost matrix
    return full_cost_matrix_constants, full_cost_matrix_exponentials


def polynomial_constraints(polynomial_degree, constraint_degree):
    """
    This function creates constraint functions for differentiating a
    polynomial to different degrees

    Parameters
    polynomial_degree - Degree of the polynomial
    constraint_degree - Number of degrees of differentiation to which the polynomial is constrained

    Return
    Polynomial constraints
    """
    # Initialize constraints polynomials
    constraint_constants = np.array(
        [], dtype=np.int64).reshape(0, polynomial_degree + 1)
    constraint_exponentials = np.array(
        [], dtype=np.int64).reshape(0, polynomial_degree + 1)

    # Add to the constraints for each degree of differentiation
    for degree in range(constraint_degree):
        constant_vec, exponential_vec = differentiate_polynomial(
            polynomial_degree, degree)
        constraint_constants = np.vstack(
            (constraint_constants, constant_vec.T))
        constraint_exponentials = np.vstack(
            (constraint_exponentials, exponential_vec.T))

    # Return constraints for each degree of polynomial differentiation
    return constraint_constants, constraint_exponentials


def full_task_space_constraints(polynomial_degree, constraint_degree, task_space_size):
    """
    The solutions to the full task space can be found simultaneously by placing all of
    the constraints in a giant matrix

    Parameters
    polynomial_degree - Degree of the polynomial
    constraint_degree - Number of degrees of differentiation to which the polynomial is constrained
    task_space_size - Size of the task space

    Return
    Full task space constraints
    """
    # Get generic constraint for a differentiated polynomial
    constraint_constants, constraint_exponentials = polynomial_constraints(
        polynomial_degree, constraint_degree)

    # Initialize full task space constrains as zeros
    num_constraint_cols = (polynomial_degree + 1) * task_space_size
    num_constraint_rows = constraint_degree * task_space_size
    full_constraint_constants = np.zeros(
        [num_constraint_rows, num_constraint_cols])
    full_constraint_exponential = np.zeros(
        [num_constraint_rows, num_constraint_cols])

    # Populate full constraint matrix
    for i in range(task_space_size):
        lb = (polynomial_degree + 1) * i
        ub = (polynomial_degree + 1) * (i + 1)
        for deg in range(constraint_degree):
            row = constraint_degree * i + deg
            full_constraint_constants[row,
                                      lb:ub] = constraint_constants[deg, :]
            full_constraint_exponential[row,
                                        lb:ub] = constraint_exponentials[deg, :]

    # Return full constraint matrix
    return full_constraint_constants, full_constraint_exponential


def qp_trajectory(polynomial_degree, differentiation_degree,
                  time_waypoints, pos_waypoints, vel_waypoints):
    """
    This function determines an Nth degree continuous polynomial for each variable
    in the task space that minimizes the cost of the polynomial against the Mth derivative
    of the polynomial. The polynomials are also constrained by the time and waypoints passed in

    Parameters
    polynomial_degree - Degree of the polynomial
    constraint_degree - Number of degrees of differentiation to which the polynomial is constrained
    time_waypoints - Times at which each waypoint occur
    pos_waypoints - Positions defining each waypoints

    Return
    Coefficients defining trajectory polynomials
    """
    # Initialize variables
    num_waypoints = time_waypoints.shape[0]
    task_space_size = pos_waypoints.shape[1]

    cost_constants_matrix, cost_exponential_matrix = full_task_space_time_dependent_cost(
        polynomial_degree, differentiation_degree, task_space_size)

    constraint_constants_matrix, constraint_exponentials_matrix = full_task_space_constraints(
        polynomial_degree, 3, task_space_size)

    coefficients = {}
    # Iterate through each waypoint
    for time in range(time_waypoints.shape[0] - 1):
        # Numerically integrate cost function
        ti = time_waypoints[time]
        tf = time_waypoints[time + 1]
        cost_function_matrix = cost_constants_matrix * (np.power(tf, cost_exponential_matrix) -
                                                        np.power(ti, cost_exponential_matrix))

        # Create constraint matrix for time bounds
        constraint_start_matrix = constraint_constants_matrix * \
            np.power(ti, constraint_exponentials_matrix)
        constraint_goal_matrix = constraint_constants_matrix * \
            np.power(tf, constraint_exponentials_matrix)
        constraint_function_matrix = np.vstack(
            (constraint_start_matrix, constraint_goal_matrix))

        # Build a right hand-side vector using the constraints
        constraint_start = np.zeros(constraint_constants_matrix.shape[0])
        constraint_goal = np.zeros(constraint_constants_matrix.shape[0])
        constraint_start[::3] = pos_waypoints[time, :]
        constraint_start[1::3] = vel_waypoints[time, :]
        constraint_goal[::3] = pos_waypoints[time+1, :]
        constraint_goal[1::3] = vel_waypoints[time+1, :]
        rhs = np.hstack(
            (np.zeros(cost_function_matrix.shape[0]), constraint_start, constraint_goal))

        # Solution vector from lagrange multipliers [ lagrange multipliers, coefficients ].T
        lagrange_multipliers_and_coefficients = lagrange_multipliers_solve(cost_function_matrix,
                                                                           constraint_function_matrix, np.reshape(rhs, (rhs.shape[0], 1)))

        # Strip lagrange multipliers from the solution and reshape coefficients so each row corresponds to a task space
        coefficients[time] = np.reshape(lagrange_multipliers_and_coefficients[:-constraint_function_matrix.shape[0]],
                                        (task_space_size, polynomial_degree+1))

    # Return coefficients
    return coefficients


def lagrange_multipliers_solve(cost_matrix, constraint_matrix, rhs):
    """
    This function solves for the coefficients and lagrange
    multipliers of the cost matrix and constraints

    Parameters
    cost_matrix - Time dependent part of the cost function
    constraint_matrix - Constraint equations
    rhs - [ 0 ; constraint values ]

    Return

    """
    # Assemble matrix
    block1 = np.vstack((2*cost_matrix, constraint_matrix))
    block2 = np.vstack((constraint_matrix.T, np.zeros(
        (constraint_matrix.shape[0], constraint_matrix.shape[0]))))
    bigblock = np.hstack((block1, block2))
    # Solve for [ lagrange multipliers ; coefficients ]
    return np.linalg.inv(bigblock) @ rhs


if __name__ == "__main__":
    pos_waypoints = np.array([[0, 0, 0],  # x1, y1, z1
                              [5, 5, 5],  # x2, y2, z2
                              [10, 0, 5],  # x3, y3, z3
                              [5, -5, 5],  # x4, y4, z4
                              [0, 0, 0]])  # x5, y5, z5
    vel_waypoints = np.array([[0, 0, 0],  # vx1, vy1, vz1
                              [1, 0, 0],  # vx2, vy2, vz2
                              [0, -1, 0],  # vx3, vy3, vz3
                              [-1, 0, 0],  # vx4, vy4, vz4
                              [0, 0, 0]])  # vx5, vy5, vz5
    time_waypoints = np.array([0, 5, 10, 15, 20])  # t1, t2, t3
    polynomial_degree = 7
    differentiation_degree = 4
    coeffs = qp_trajectory(polynomial_degree, differentiation_degree,
                           time_waypoints, pos_waypoints, vel_waypoints)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    for i in range(time_waypoints.shape[0]-1):
        times = np.linspace(time_waypoints[i], time_waypoints[i+1], 11)
        x = np.array([np.polyval(np.flip(coeffs[i][0, :], axis=0), t)
                      for t in times])
        y = np.array([np.polyval(np.flip(coeffs[i][1, :], axis=0), t)
                      for t in times])
        z = np.array([np.polyval(np.flip(coeffs[i][2, :], axis=0), t)
                      for t in times])

        ax.plot3D(x, y, z)
    plt.show()
