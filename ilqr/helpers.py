import numpy as np

import matplotlib.pyplot as plt

def wrap_to_pi(angles):
  """
  wrap_to_pi takes in a numpy array of angles in radians
  and wraps it from pi to -pi

  :param: angles: numpy array of angles
  :return: wrapped angles from pi to -pi
  """
  indices = np.where(angles >= np.pi, 1, 0)
  while np.any(indices):
    angles -= 2 * np.pi * indices
    indices = np.where(angles >= np.pi, 1, 0)

  indices = np.where(angles < -np.pi, 1, 0)
  while np.any(indices):
    angles += 2 * np.pi * indices
    indices = np.where(angles < -np.pi, 1, 0)

  return angles

def transform_point2d(point, d_x, d_y, theta):
  """
  transform_point2d takes in a point and the parameters defining
  a 2d transform. It then performs a 2d transform on the point

  :param point: point to transform
  :param d_x: transform delta x
  :param d_y: transform delta y
  :param theta: transform rotation
  :return: transformed point
  """
  transform = np.array([[np.cos(theta), -np.sin(theta), d_x],
                        [np.sin(theta),  np.cos(theta), d_y],
                        [          0.0,            0.0, 1.0]])
  return transform @ point

def draw_arrow(x, y, theta, length=1, color='b'):
  """
  draw_arrow plots an arrow on a matplotlib figure

  :param x: x location of the arrow
  :param y: y location of the arrow
  :param theta: orientation of the arrow
  :param length: length of the arrow
  :param color: color of the arrow
  """
  bottom = transform_point2d(np.array([0.0, 0.0, 1.0]), x, y, theta)
  top = transform_point2d(np.array([length, 0.0, 1.0]), x, y, theta)
  left = transform_point2d(np.array([0.75 * length, -0.25 * length, 1.0]), x, y, theta)
  right = transform_point2d(np.array([0.75  *length, 0.25 * length, 1.0]), x, y, theta)
  x = [bottom[0], top[0], left[0], right[0], top[0]]
  y = [bottom[1], top[1], left[1], right[1], top[1]]

  plt.plot(x, y, color)

def plot_2d_trajectory(trajectory, length=1, color='b'):

  length = np.linalg.norm(trajectory[0,:2] - trajectory[-1,:2]) / trajectory.shape[0]

  for ind in range(trajectory.shape[0]):
    x = trajectory[ind,0]
    y = trajectory[ind,1]
    theta = trajectory[ind,2]
    draw_arrow(x, y, theta, length, color)

def draw_car(x, y, theta, length=1.0, width=0.4, color='g'):
  """
  draw_car plots a car on a matplotlib figure

  :param x: x location of the car
  :param y: y location of the car
  :param theta: orientation of the car
  :param length: length of the car
  :param width: width of the car
  :param color: color of the car
  """

  center = transform_point2d(np.array([0.0, 0.0, 1.0]), x, y, theta)

  back_left = transform_point2d(np.array([-0.25 * length, 0.5 * width, 1.0]), x, y, theta)
  back_right = transform_point2d(np.array([-0.25 * length, -0.5 * width, 1.0]), x, y, theta)
  front_right = transform_point2d(np.array([0.75 * length, -0.5 * width, 1.0]), x, y, theta)
  front_left = transform_point2d(np.array([0.75 * length, 0.5 * width, 1.0]), x, y, theta)

  x = [back_left[0], back_right[0], front_right[0], front_left[0], back_left[0]]
  y = [back_left[1], back_right[1], front_right[1], front_left[1], back_left[1]]

  plt.plot(x, y, color)
  plt.scatter(center[0], center[1], c=color, marker="X")
  draw_arrow(center[0], center[1], theta, length=length*0.5, color='r')

def calc_nearest_index(state, trajectory, curr_ind=0):
  """
  calc_nearest_index determines the index on a trajectory which corresponds
  to the vehicles current position

  :param state: Current state of the vehicle
  :param trajectory: Trajectory which the vehicle is following
  :param curr_ind: Current index in the trajectory from which to start searching
  """
  d_pose = np.linalg.norm(state[:2] - trajectory[curr_ind:, :2], axis=1)
  d_theta = wrap_to_pi(state[2] - trajectory[curr_ind:, 2])
  indices = np.where(np.abs(d_theta) < np.pi/4)

  if not d_pose[indices].shape[0]:
    return 0

  ind = np.argmin(d_pose[indices]) + curr_ind

  return ind
