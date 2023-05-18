from ilqr_base import *
from helpers import *

import matplotlib.pyplot as plt

class iLQRCar(iLQRBase):

  def __init__ (self, max_iterations=5, max_execution_time_seconds=.2, dt_sec=.1, control_lower_limit=[-0.5,-.2], control_upper_limit=np.array([1,.2])):
    # X = [x, y, theta, v, curvature]
    # u = [acceleration, curvature rate (1/turning radius)]

    super().__init__(max_iterations, max_execution_time_seconds, dt_sec, control_lower_limit, control_upper_limit)


  def A_func(self, state):
    """
    A_func creates and returns the state jacobian for a car
    :param state: state of the system
    :return: state jacobian
    """
    return np.array([[0.0, 0.0, 0.0, np.cos(state[2]),       0.0],
                     [0.0, 0.0, 0.0, np.sin(state[2]),       0.0],
                     [0.0, 0.0, 0.0,              0.0, -state[3]],
                     [0.0, 0.0, 0.0,              0.0,       0.0],
                     [0.0, 0.0, 0.0,              0.0,       0.0]])

  def B_func(self, state):
    """
    B_func creates and returns the control jacobian for a car
    :param state: state of the system
    :return: control jacobian
    """
    return np.array([[0.0,   0.0],
                     [0.0,   0.0],
                     [0.0,   0.0],
                     [1.0,   0.0],
                     [0.0,   1.0]])

  def Q_func(self):
    """
    Q_func returns the state weight matrix
    :return: state weight matrix
    """
    return np.array([[1.0, 0.0, 0.0 ,0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 1.0]])

  def R_func(self):
    """
    R_func returns the control weight matrix
    :return: control weight matrix
    """
    return np.array([[1.0, 0.0],
                     [0.0, 1.0]])

  def Qf_func(self):
    """
    Q_func returns the final state weight matrix
    """
    return np.array([[10.0, 0.0, 0.0 ,0.0, 0.0],
                     [0.0, 10.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 10.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 1.0]])

  def add_states(self, state_1, state_2):
    """
    add_states is used to add two vehicle states together in a correct way
    (e.g. wrapping angles)
    :param state_1: first state to add
    :param state_2: second state to add
    :return: combined vehicle state
    """
    new_state = state_1 + state_2
    new_state[2] = wrap_to_pi(new_state[2])
    return new_state

  def simple_trajectory(self, start_state, end_state, time):
    """
    simple_trajectory creates a simple in-feasible trajectory
    :param start_state: first state in the trajectory
    :param end_state: final state in the trajectory
    :param time: time over which the trajectory is performed
    :return: Simple trajectory
    """
    steps = int(time / self.dt_sec)
    control = np.zeros((steps, self.control_space_size))

    x = np.linspace(start_state[0], end_state[0], steps)
    y = np.linspace(start_state[1], end_state[1], steps)
    orientation = wrap_to_pi(np.arctan2(end_state[1] - start_state[1], end_state[0] - start_state[0]) * np.ones((steps)))
    velocity = np.zeros((steps))
    curvature = np.zeros((steps))

    trajectory = np.vstack((x,y,orientation,velocity,curvature)).T

    return trajectory, control

  def plot_simulation(self, fig, trajectory, desired_trajectory, state, show_time=1, plt_block=False):
    """
    plot_simulation plots the current state of the simulation on a figure
    :param fig: figure on which to plot
    :param trajectory: current trajectory the vehicle is following
    :param desired_trajectory: the desired trajectory
    :param state: the current state of the vehicle
    :param show_time: duration over which to show the figure
    :param plt_block: if true holds the figure indefinitely
    """
    fig.clear(True)

    plot_2d_trajectory(trajectory)
    plot_2d_trajectory(desired_trajectory, color='k')
    draw_car(state[0], state[1], state[2])

    plt.axis('equal')
    plt.show(block=plt_block)
    plt.pause(show_time)

  def create_ilqr_input(self, desired_trajectory, desired_control, trajectory, control):
    additional_steps = desired_trajectory.shape[0] - trajectory.shape[0]
    for step in np.arange(additional_steps):
      trajectory = np.vstack((trajectory, trajectory[-1]))
      control = np.vstack((control, np.zeros(control[-1].shape)))

    return trajectory, control

  def simulate(self, start_state, end_state, time):
    """
    simulate performs a simple simulation of a system using ilqr to follow a desired trajectory
    :param start_state: first state in the trajectory
    :param end_state: final state in the trajectory
    :param time: time over which the trajectory is performed
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    desired_trajectory, desired_control = ilqr.simple_trajectory(start_state, end_state, time)
    trajectory = np.repeat(np.reshape(start_state,(1,start_state.shape[0])),desired_trajectory.shape[0],axis=0)
    control = np.zeros(desired_control.shape)
    state = start_state
    curr_ind = 0

    while desired_trajectory[curr_ind:].shape[0]:
      curr_ind = calc_nearest_index(state, desired_trajectory, curr_ind)
      if desired_trajectory[curr_ind:].shape[0] == 1:
        return

      while desired_trajectory[curr_ind:].shape[0] > trajectory.shape[0]:
        control = np.vstack((control, np.zeros(ilqr.control_space_size)))
        trajectory = np.vstack((trajectory, trajectory[-1])) if trajectory.shape[0] else np.vstack((trajectory, state))
      trajectory, control = ilqr.ilqr(trajectory, control, desired_trajectory[curr_ind:], desired_control[curr_ind:], state)

      for _ in np.arange(25):
        if not trajectory.size:
          break
        ilqr.plot_simulation(fig, trajectory, desired_trajectory, state, show_time=0.1)
        state = self.dynamics(state, control[0,:])
        trajectory = trajectory[1:]
        control = control[1:]


if __name__ == "__main__":
  ilqr = iLQRCar(max_iterations=10, max_execution_time_seconds=.5)
  start_state = np.array([0.,0.,0.,0.,0.])
  end_state = np.array([-20,-20.,0.0,0.,0.])
  time = 5
  ilqr.simulate(start_state, end_state, time)
