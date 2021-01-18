import torch
from torch import nn
from sillyode import sillyode
from torchdiffeq import odeint, odeint_adjoint
import time

def tictoc(s=None):
    global _t1
    if s is not None:
        print(f'{s} : {time.time() - _t1 : .2f} s')
    _t1 = time.time()


class F(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()

        self.a = nn.Parameter(torch.tensor(2.0, dtype=dtype))
        self.b = nn.Parameter(torch.tensor(3.0, dtype=dtype))

    def forward(self, t, y):
        r = torch.empty_like(y)
        r[0] = -(self.a * y[1])**3
        r[1] = self.b * y[0]
        return r


def run_code(backward, ode_method, jit=False):
    dtype = torch.double

    t = torch.linspace(0, 2.5, 50, dtype=dtype)
    y0 = torch.tensor([1.5, 0.5], dtype=dtype, requires_grad=True)
    func = F(dtype)

    if backward:
        if jit:
            yt = ode_method(func, y0, t, jit=True)
        else:
            yt = ode_method(func, y0, t)
        loss = torch.sum(yt ** 3)
        loss.backward()
    else:
        with torch.no_grad():
            yt = ode_method(func, y0, t)

    return yt

if __name__ == '__main__':
    n = 100

    for t, s in [(False, 'No .backward()'), (True, 'With .backward()')]:
        print(s)
        tictoc()
        for _ in range(n):
            run_code(t, odeint)
        tictoc('   torchdiffeq.odeint')
        for _ in range(n):
            run_code(t, odeint_adjoint)
        tictoc('   torchdiffeq.odeint_adjoint')
        for _ in range(n):
            run_code(t, sillyode)
        tictoc('   sillyode')
        print()
