# import modules

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

FRAME_TIME = 1    # time interval
GRAVITY_ACCEL = 9.81/1000    # gravity constant
BOOST_ACCEL = 18/1000  # thrust constant
PLATFORM_WIDTH = 0.25    # landing platform width
PLATFORM_HEIGHT = 0.06  # landing platform height
ROTATION_ACCEL = 0.2     # rotation constant


# Define System Dynamics

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
        delta_state_gravity = t.tensor([0., 0., 0., -GRAVITY_ACCEL * FRAME_TIME, 0.])
        # delta_state_gravity = t.tensor([0., GRAVITY_ACCEL * FRAME_TIME])

        # Thrust
        """
        state_tensor = t.zeros((1, 5))
        state_tensor[0, 0] = -0.5 * FRAME_TIME * t.sin(state[4])
        state_tensor[0, 1] = -t.sin(state[4])
        state_tensor[0, 2] = 0.5 * FRAME_TIME * t.cos(state[4])
        state_tensor[0, 3] = t.cos(state[4])
        delta_state = BOOST_ACCEL * FRAME_TIME * t.mul(state_tensor, action[0])        
        """
        state_tensor = t.tensor([-0.5 * FRAME_TIME * t.sin(state[4]),
                                 -t.sin(state[4]),
                                 0.5 * FRAME_TIME * t.cos(state[4]),
                                 t.cos(state[4]),
                                 0])
        delta_state = BOOST_ACCEL * FRAME_TIME * t.mul(state_tensor, action[0])
        # delta_state = BOOST_ACCEL * FRAME_TIME * t.tensor([0., -1.]) * action


        # Theta
        delta_state_theta = FRAME_TIME * ROTATION_ACCEL * t.mul(t.tensor([0., 0., 0., 0., 1.]), action[1])

        state = state + delta_state + delta_state_gravity + delta_state_theta


        step_mat = t.tensor([[1., FRAME_TIME, 0., 0., 0.],
                             [0., 1., 0., 0., 0.],
                             [0., 0., 1., FRAME_TIME, 0.],
                             [0., 0., 0., 1., 0.],
                             [0., 0., 0., 0., 1.]])        

        # Update state
        #step_mat = t.tensor([[1., FRAME_TIME],
        #                     [0., 1.]])

        state = t.matmul(step_mat, state.T)
        #state = state.T

        return state.T

# Create controller

class Controller(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states
        dim_hidden: up to you
        dim_output: # of actions
        """
        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            # You can add more layers here
            nn.Sigmoid()
        )

    def forward(self, state):
        action = self.network(state)
        return action

# Execute Simulation

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
        state = [10./1000, 18./1000, 200./1000, -12./1000, 1./1000]
                 #[3, 2, 8, 4, 3.141*(3/4)],
                 #[8, 4, 14, 1, 3.141*(2/3)]]    # TODO: need batch of initial states
        return t.tensor(state, requires_grad=False).float()

    def error(self, state):
        return state[0]**2 + state[1]**2

# Set up optimizer

class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.01)

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
        y = data[:, 2]
        plt.plot(x, y)
        plt.show()

 # Run the code


T = 20    # number of time steps
dim_input = 5    # state space dimensions
dim_hidden = 6  # latent dimensions
dim_output = 2  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)   # define controller
s = Simulation(c, d, T)     # define simulation
o = Optimize(s)     # define optimizer
o.train(40)     # solve the optimization problem
