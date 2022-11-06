# import packages

import logging
import math
import random
import numpy as np
import time
import torch as t
import torch.nn as nn
from torch import optim
from torch.nn import utils
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# setting up environment parameters
# normalized environment parameters
FRAME_TIME = 1    # time interval
GRAVITY_ACCEL = 9.81/1000    # gravity constant
BOOST_ACCEL = 18/1000  # thrust constant
ROTATION_VEL = 10/1000     # rotation constant


# Define System Dynamics
# Chose "1. More realistic definition of state and action spaces: Rocket orientation, angular velocity, etc." for better problem formulation
class Dynamics(nn.Module):

    def __init__(self):
        super(Dynamics, self).__init__()

    @staticmethod
    def forward(state, action):

        """
        action[0]: thrust controller
        action[1]: omega controller
        state[0] = x
        state[1] = x_dot
        state[2] = y
        state[3] = y_dot
        state[4] = theta
        """

        # Apply gravity
        delta_state_gravity = t.tensor([0., 0., -0.5 * FRAME_TIME**2 * GRAVITY_ACCEL, -GRAVITY_ACCEL * FRAME_TIME, 0.])
        # Gravity affects both y and y_dot
        # Gravity term in y: -0.5*t^2*g
        # Gravity term in y_dot: -g*t


        # Thrust
        state_tensor = t.tensor([-0.5 * FRAME_TIME * t.sin(state[4]),
                                 -t.sin(state[4]),
                                 0.5 * FRAME_TIME * t.cos(state[4]),
                                 t.cos(state[4]),
                                 0])
        delta_state = BOOST_ACCEL * FRAME_TIME * t.mul(state_tensor, action[0])
        # Part of the dynamics that is affected by the thrust controller
        # Thrust term in x: -0.5*t^2*thrust*sin(theta)
        # Thrust term in x_dot: -t*thrust*sin(theta)
        # Thrust term in y: -0.5*t^2*thrust*cos(theta)
        # Thrust term in y_dot: t*cos(theta)
        # No thrust term in theta


        # Theta
        delta_state_theta = FRAME_TIME * ROTATION_VEL * t.mul(t.tensor([0., 0., 0., 0., 1.]), action[1])
        # Part of the dynamics that is affected by the omega controller which only involves theta
        # Omega term in theta: t*omega_controller

        # Matrix that include the relationship between the state variables
        step_mat = t.tensor([[1., FRAME_TIME, 0., 0., 0.],
                             [0., 1., 0., 0., 0.],
                             [0., 0., 1., FRAME_TIME, 0.],
                             [0., 0., 0., 1., 0.],
                             [0., 0., 0., 0., 1.]])        

        # Update state
        state = t.matmul(step_mat, state)
        state = state + delta_state + delta_state_gravity + delta_state_theta

        return state

# Create a deterministic controller
# nn.Sigmoid outputs values from 0 to 1, nn.Tanh from -1 to 1

class Controller(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states (5)
        dim_hidden: up to me (chosen to be 20)
        dim_output: # of actions (2)
        """
        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            nn.Sigmoid()
        )

    def forward(self, state):
        action = self.network(state)
        return action


# Execute Simulation
# the simulator that rolls out state(1), state(2), ..., state(T)
# self.action_trajectory and self.state_trajectory stores the action and state trajectories along time

class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.action_trajectory = []
        self.state_trajectory = []

    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(T):
            action = self.controller.forward(state)
            state = self.dynamics.forward(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error(state)

    @staticmethod
    def initialize_state():
        state = np.array([10., 0., 500., -4., math.pi*0.5])     # chose this set of initial state
        state = np.multiply(state, 1/1000)      # normalize initial state
        return t.tensor(state, requires_grad=False).float()

    def error(self, state):
        return (state[0]**2 + state[1]**2 + state[2]**2 + state[3]**2 + state[4]**2)*1000
        # multiplied by the normalized factor
        # included all the state variables to compute the error


# Set up optimizer
# LBFGS is chosen as the optimizer
# loss.backard is where the gradient is calculated (d_loss/d_variables)
# self.optimizer.step(closure) is where gradient descent is done

class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.05)  # chose a smaller learning rate to converge to a solution

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss
        self.optimizer.step(closure)
        return closure()

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            print('[%d] loss: %.3f' % (epoch + 1, loss))
            self.visualize()

    def visualize(self):
        data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        x = data[:, 0]
        x_dot = data[:, 1]
        y = data[:, 2]
        y_dot = data[:, 3]
        theta = data[:, 4]

        fig, axs = plt.subplots(3, 1)

        # Vertical distance vs horizontal distance
        axs[0].plot(y, x)
        axs[0].set_xlabel('Height from Ground')
        axs[0].set_ylabel('Horizontal Distance')

        # Speed vs. Horizontal distance
        axs[1].plot(y, x_dot, label='Horizontal Speed')
        axs[1].plot(y, y_dot, label='Vertical Speed')
        axs[1].set_xlabel('Height from Ground')
        axs[1].set_ylabel('Speed')
        axs[1].legend(loc='best')

        # Orientation vs. horizontal distance
        axs[2].plot(y, theta)
        axs[2].set_xlabel('Height from Ground')
        axs[2].set_ylabel('Rocket Orientation')

        plt.show()


# Run the code

T = 20    # number of time steps
dim_input = 5    # state space dimensions
dim_hidden = 20  # latent dimensions
dim_output = 2  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)   # define controller
s = Simulation(c, d, T)     # define simulation
o = Optimize(s)     # define optimizer
o.train(40)     # solve the optimization problem

# Analysis of the results:
# The horizontal distance converged to zero since it was set to a small number of 10 and a non-zero orientation.
# Because of this, the horizontal distance can change by applying thrust.
# Horizontal speed remained zero. This may be because only one step was needed to get the horizontal distance to 0.
# The initial rocket orientation was pi/2, so the horizontal speed is zero after the first step.
# Vertical speed converged to zero as it should.
# Orientation converged to a number close to pi which is essentially zero.
# There is still some loss but extremely small.
# This could be because the controller can't recognize extremely small differences.

