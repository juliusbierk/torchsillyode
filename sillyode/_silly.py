import torch
import numpy as np
from scipy.integrate import solve_ivp

def rk4(func, y0, t, tt):
    sol = [y0]
    i = 1
    j = 1
    T = 1.0 * tt[0]
    y = y0
    b = torch.tensor([1/6, 2/6, 2/6, 1/6], dtype=t.dtype)

    while True:
        # Choose step size
        h = tt[j] - T
        save = False
        if h + T >= t[i]:
            h = t[i] - T
            save = True
            i += 1
        else:
            j += 1

        y = _take_step(T, b, func, h, y)
        T += h

        if save:
            sol.append(y)
            if i == len(t):
                break
    y = torch.stack(sol)
    return y


def _take_step(T, b, func, h, y):
    h2 = 0.5 * h
    k0 = func(T, y)
    k1 = func(T + h2, y + h2 * k0)
    k2 = func(T + h2, y + h2 * k1)
    k3 = func(T + h, y + k2 * h)
    return y + h * (b[0] * k0 + b[1] * k1 + b[2] * k2 + b[3] * k3)


def sillyode(func, y0, t, atol=1e-9, rtol=1e-7):
    assert not t.is_cuda, 'Only works on CPU for now.'

    requires_grad = func(t[0], y0).requires_grad

    with torch.no_grad():
        def np_f(t, y):
            return func(torch.from_numpy(np.asarray(t)), torch.from_numpy(y)).numpy()

        res = solve_ivp(np_f, (t.min(), t.max()), y0, t_eval=None if requires_grad else t.numpy(),
                        rtol=rtol, atol=atol)
        if not requires_grad:
            return torch.from_numpy(res.y).t()

    tt = torch.from_numpy(res.t)
    y = rk4(func, y0, t, tt)
    return y
