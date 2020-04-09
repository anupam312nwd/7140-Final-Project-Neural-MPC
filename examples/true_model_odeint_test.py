from utils.gym_utils import generate_training_data
from utils.acrobot_env import AcrobotEnv, wrap, bound

import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi

import torch
import torch.nn as nn

from torchdiffeq import odeint

env = AcrobotEnv()

horizon = 100

# Setup controller
##################################################
def controller(actions):
    def discrete_controller(t):
        if torch.round(t) <= len(actions) - 1:
            return torch.unsqueeze(actions[torch.round(t).long()], dim=0)
        
        return torch.Tensor([0.0])
    
    return discrete_controller

actions = torch.randint(0, env.action_space.n, (horizon,))

t = torch.linspace(0, horizon, 1000)
c = controller(actions)

c_t = torch.zeros_like(t)
for i, v in enumerate(t):
    c_t[i] = c(v)
    
plt.figure()
plt.plot(t, c_t - 1)
plt.title("Selected action sequence")
plt.savefig("plots/controller_test.png")


# Setup true model
##################################################
def model(env):
    def env_model(t, s_augmented):
        m1 = env.LINK_MASS_1
        m2 = env.LINK_MASS_2
        l1 = env.LINK_LENGTH_1
        lc1 = env.LINK_COM_POS_1
        lc2 = env.LINK_COM_POS_2
        I1 = env.LINK_MOI
        I2 = env.LINK_MOI
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2) \
                - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2) + phi2
        if env.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / \
                (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2) \
                / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return torch.Tensor((dtheta1, dtheta2, ddtheta1, ddtheta2, 0.))
    
    return env_model



env.reset()
state = torch.Tensor(env._get_state())
states = torch.zeros((4, horizon))
for i, a in enumerate(actions):
    s_augmented = torch.cat((state, a.unsqueeze(0).float()))
    ns = odeint(model(env), s_augmented, torch.Tensor([0, env.dt]))[1, :4]

    states[:, i] = ns
    state = ns

plt.figure()
plt.title("State 0 over 10 steps, random actions")
plt.plot(t, c_t - 1, alpha=0.3, label="action")
plt.plot(states[0, :], label="theta_0")
plt.legend()
plt.savefig("plots/model_test.png")


# Random NN pass
##################################################
class RandomModel(nn.Module):
    def __init__(self, controller=None):
        super(RandomModel, self).__init__()

        self.controller = controller

        self.net = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, t, y):
        return self.net(y) 

model_nn = RandomModel()

env.reset()
state = torch.Tensor(env._get_state())
states_nn = torch.zeros((4, horizon))
for i, a in enumerate(actions):
    s_augmented = torch.cat((state, a.unsqueeze(0).float()))
    ns = odeint(model_nn, s_augmented, torch.Tensor([0, env.dt]))[1, :4]

    states_nn[:, i] = ns
    state = ns

plt.figure()
plt.title("State 0 over 10 steps, random actions, random model")
plt.plot(t, c_t - 1, alpha=0.3, label="action")
plt.plot(states_nn[0, :].detach(), label="theta_0")
plt.legend()
plt.savefig("plots/nn_model_test.png")

plt.figure()
plt.title("State 0 over 10 steps, true vs nn model")
plt.plot(states[0, :], label="theta_true")
plt.plot(states_nn[0, :].detach(), label="theta_nn")
plt.legend()
plt.savefig("plots/nn_true_compare.png")
