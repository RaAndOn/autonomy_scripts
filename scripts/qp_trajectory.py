# /usr/bin/python3

import numpy as np


def trajectory(poly_deg, diff_deg, start, goal, ti=0, tf=2):
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
    integral = polynomial_constants_matrix * \
        (np.power(tf, polynomial_exponential_matrix) -
         np.power(ti, polynomial_exponential_matrix))

    # Create constraints matrix
    pos_constants, pos_exponential = differentiate_polynomial(poly_deg, 0)
    vel_constants, vel_exponential = differentiate_polynomial(poly_deg, 1)

    pos_initial = pos_constants * np.power(ti, pos_exponential)
    vel_initial = vel_constants * np.power(ti, vel_exponential)
    pos_final = pos_constants * np.power(tf, pos_exponential)
    vel_final = vel_constants * np.power(tf, vel_exponential)

    H = np.vstack((pos_initial.T, vel_initial.T,
                   pos_final.T, vel_final.T))

    block1 = np.vstack((2*integral, H))
    block2 = np.vstack((H.T, np.zeros((H.shape[0], H.shape[0]))))

    d = np.array([start["X"], start["VX"], goal["X"], goal["VX"]])
    b = np.hstack((np.zeros(integral.shape[0]), d))
    b = np.reshape(b, (b.shape[0], 1))

    bigblock = np.hstack((block1, block2))
    print(np.linalg.pinv(bigblock) @ b)


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


def constraint_matrix(pi, pf, vi, vf):
    """

    """


if __name__ == "__main__":
    start = {"X": 0.0, "Y": 0.0, "VX": 0.0, "VY": 0.0}
    goal = {"X": 5.0, "Y": 7.0, "VX": 10.0, "VY": 10.0}
    trajectory(7, 4, start, goal)
