import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import timeit

class LorenzODE(torch.nn.Module):

    def __init__(self):
        super(LorenzODE, self).__init__()

    def forward(self, t, u):
        x, y, z = u[0],u[1],u[2]
        du1 = 10.0 * (y - x)
        du2 = x * (28.0 - z) - y
        du3 = x * y - 2.66 * z
        return torch.stack([du1, du2, du3])

u0 = torch.tensor([1.0,0.0,0.0])
t = torch.linspace(0, 100, 1001)
sol = odeint(LorenzODE(), u0, t)

plt.plot(sol)
plt.show()
# def time_func():
#     odeint(LorenzODE(), u0, t, rtol = 1e-8, atol=1e-8)

# time_func()

# _t = timeit.Timer(time_func).timeit(number=2)
# print(_t) # 96.36945809999997 seconds