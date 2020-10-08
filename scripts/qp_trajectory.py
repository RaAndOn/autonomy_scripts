# /usr/bin/python3

from sympy import *
from sympy.abc import x
from sympy.utilities.lambdify import lambdify

import numpy as np


def trajectory(poly_deg, diff_deg, ti=0, tf=2):
    """

    """
    # initialize the variable exponents for each term in the polynomial
    polynomial_exponential = np.reshape(
        np.arange(poly_deg + 1, dtype=int), (poly_deg + 1, 1))
    polynomial_constants = np.reshape(
        np.ones(poly_deg + 1, dtype=int), (poly_deg + 1, 1))

    # Differentiate polynomial
    for diff in np.arange(diff_deg):
        polynomial_constants *= polynomial_exponential
        polynomial_exponential = np.where(
            polynomial_exponential - 1 < 0, 0, polynomial_exponential - 1)

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

    # build constraint_matrix

    # np.linalg.pinv()


def coefficient_symbols(num_coeff):
    coeff_list = []
    for coeff in range(num_coeff):
        i = Symbol('i')
        ai = Indexed('a', i)
        ai = ai.subs(i, coeff)
        coeff_list.append(ai)
    return np.array(coeff_list)


if __name__ == "__main__":
    #     from sympy.abc import x

    # from sympy.utilities.lambdify import lambdify, implemented_function

    # from sympy import Function

    # f = implemented_function('diff', lambda x: x+1)

    trajectory(7, 4)
