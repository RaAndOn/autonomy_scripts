#!/usr/bin/python3

from lqr_base import *

import numpy as np
import math

class LQRPendulum(LQRBase):

  def __init__ (self):
    # X = [x, y, theta, v, curvature]
    # u = [acceleration, curvature rate (1/turning radius)]

    self.gravity = 9.81
    self.length = 0.5
    self.damping = 0.1


    super().__init__()


  def A_func(self, state):
    """
    A_func creates and returns the state jacobian for a car
    :param state: state of the system
    :return: state jacobian
    """
    return np.array([[0.0,                                              1.0],
                     [-self.gravity * self.length * math.sin(state[0]), -self.damping]])

  def B_func(self, state):
    """
    B_func creates and returns the control jacobian for a car
    :param state: state of the system
    :return: control jacobian
    """
    return np.array([[0.0],
                     [1.0]])

  def Q_func(self):
    """
    Q_func returns the state weight matrix
    :return: state weight matrix
    """
    return np.array([[1.0, 0.0],
                     [0.0, 1.0]])

  def R_func(self):
    """
    R_func returns the control weight matrix
    :return: control weight matrix
    """
    return np.array([[1.0]])

if __name__ == "__main__":
  lqr = LQRPendulum()
  print(lqr.solveDARE(np.array([[0.0],[0.0]])))
