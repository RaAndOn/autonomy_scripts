#!/usr/bin/python3

import numpy as np

import abc

class LQRBase(metaclass=abc.ABCMeta):
  def __init__(self):
    pass

  @abc.abstractmethod
  def A_func(self, state):
    """
    A_func should be used to create and return the state jacobian
    :param state: state of the system
    """
    pass

  @abc.abstractmethod
  def B_func(self, state):
    """
    B_func should be used to create and return the control jacobian
    :param state: state of the system
    """
    pass

  @abc.abstractmethod
  def Q_func(self):
    """
    Q_func should return the state weight matrix
    """
    pass

  @abc.abstractmethod
  def R_func(self):
    """
    R_func should return the control weight matrix
    """
    pass

  def solveDARE(self, state):
    """
    Discrete Time Algebraic Riccati equation (DARE)
    https://en.wikipedia.org/wiki/Algebraic_Riccati_equation
    """
    B = self.B_func(state)
    A = self.A_func(state)
    AT = A.T
    BT = B.T
    R = self.R_func()
    Q = self.Q_func()

    P = self.Q_func()
    iter = 0
    diff = 1
    while iter < 1000 and diff > .001:
      R_BT_P_B = R + (BT @ P @ B)
      if not np.linalg.det(R_BT_P_B):
        break
      Inv_R_BT_P_B = np.linalg.inv(R_BT_P_B)
      P_next = AT @ P @ A - AT @ P @ B @ Inv_R_BT_P_B @ (BT @ P @ A) + Q
      diff = np.max(np.abs(P_next - P))
      P = P_next
      iter += 1

    # K = tmp *BT * P * A
    K = np.linalg.inv(R_BT_P_B) @ BT @ P @ A
    print(K)
    return K if diff <= .001 else False
