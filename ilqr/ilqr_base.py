#!/usr/bin/python3

import abc
import numpy as np
from datetime import datetime
from helpers import *

class iLQRBase(metaclass=abc.ABCMeta):

  def __init__ (self, max_iterations, max_execution_time_seconds, dt_sec, control_lower_limit, control_upper_limit):

    self.max_iterations = max_iterations
    self.max_execution_time_seconds = max_execution_time_seconds
    self.dt_sec = dt_sec
    self.control_lower_limit = control_lower_limit
    self.control_upper_limit = control_upper_limit
    self.Q = self.Q_func()
    self.Qf = self.Qf_func()
    self.R = self.R_func()
    self.R_inv = np.linalg.inv(self.R)

    self.state_space_size = self.Q.shape[0]
    self.control_space_size = self.R.shape[0]

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

  @abc.abstractmethod
  def Qf_func(self):
    """
    Q_func should return the final state weight matrix
    """
    pass

  @abc.abstractmethod
  def add_states(self, state_1, state_2):
    """
    add_states is used to add two states together in a correct way
    :param state_1: first state to add
    :param state_2: second state to add
    """
    pass

  def dynamics(self, state, control):
    """
    dynamics performs the dynamics on the system for a single time step
    :param state: state of the vehicle
    :param control: control to be performed at the time step
    :return: state after the dynamics have been performed
    """
    state_dot = self.A_func(state) @ state + self.B_func(state) @ control
    return self.add_states(state, state_dot * self.dt_sec)

  def ilqr(self, trajectory_nominal, controls_nominal, trajectory_desired, controls_desired, start_state):
    """
    ilqr performs MPC to match a trajectory to a desired trajectory
    :param trajectory_nominal: Current nominal trajectory the vehicle is performing
    :param controls_nominal: Current controls corresponding to the nominal trajectory
    :param trajectory_desired: Desired trajectory to perform, this does not need to be feasible
    :param controls_desired: Current controls corresponding to the desired trajectory
    :param start_state: State of the vehicle at this moment in time
    :return: New nominal trajectory
    :return: New nominal control
    """
    steps = trajectory_desired.shape[0]
    start_time = datetime.now()
    for t in np.arange(self.max_iterations):
      time_delta = datetime.now() - start_time
      if time_delta.total_seconds() >= self.max_execution_time_seconds:
        print("iLQR has exceeded max execution time of {}".format(self.max_execution_time_seconds))
        break
      # initialize Values
      S2 = np.zeros((steps, self.state_space_size, self.state_space_size))
      S1 = np.zeros((steps, self.state_space_size))
      S2[-1] = self.Qf
      S1[-1] = -2 * self.Qf @ ( self.add_states(trajectory_desired[-1], -trajectory_nominal[-1]) )

      # Loop backwards throught time to find the optimal S
      for ind in np.flip(np.arange(steps - 1)):
        prev_ind = ind + 1

        state_des_prev = trajectory_desired[prev_ind]
        control_des_prev = controls_desired[prev_ind]
        state_nom_prev = trajectory_nominal[prev_ind]
        control_nom_prev = controls_nominal[prev_ind]
        state_bar_des_prev = self.add_states(state_des_prev, -state_nom_prev)
        control_bar_des_prev = control_des_prev - control_nom_prev

        A_nom_prev = self.A_func(state_nom_prev)
        B_nom_prev = self.B_func(state_nom_prev)

        S2_prev = S2[prev_ind]
        S2_dot = -(self.Q - S2_prev @ B_nom_prev @ self.R_inv @ B_nom_prev.T @ S2_prev + S2_prev @ A_nom_prev + A_nom_prev.T @ S2_prev)
        S2[ind] = S2_prev - S2_dot * self.dt_sec

        S1_prev = S1[prev_ind]
        S1_dot = -( -2 * self.Q @ state_bar_des_prev + (A_nom_prev.T - S2_prev @ B_nom_prev @ self.R_inv @ B_nom_prev.T ) @ S1_prev + 2 * S2_prev @ B_nom_prev @ control_bar_des_prev)
        S1[ind] = S1_prev - S1_dot * self.dt_sec

      trajectory = np.zeros(trajectory_nominal.shape)
      controls = np.zeros(controls_nominal.shape)
      trajectory[0] = start_state
      # Iterate forward through time to find the optimal policy
      for ind in np.arange(steps-1):
        state_bar = trajectory[ind] - trajectory_nominal[ind]
        B = self.B_func(trajectory[ind])
        controls[ind] = controls_desired[ind] - self.R_inv @ B.T @ (S2[ind] @ state_bar + 0.5 * S1[ind])
        # Clip the control output
        controls[ind] = np.clip(controls[ind], self.control_lower_limit, self.control_upper_limit)
        trajectory[ind + 1] = self.dynamics(trajectory[ind], controls[ind])

      trajectory_nominal = np.copy(trajectory)
      controls_nominal = np.copy(controls)

    return trajectory_nominal, controls_nominal
