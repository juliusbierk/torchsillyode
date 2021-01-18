# Torch Silly ODE

An excellent package for torch
exists, which provide both a direct and an adjoint
method for backpropagation of errors through ODEs
[[torchdiffeq](https://github.com/rtqichen/torchdiffeq)].
The adjoint method uses basically no memory and so is suitable for
large problems.
The direct method is faster, however.
But not fast enough for my use-cases!

`torchsillyode` is a small script that makes the direct RK45 method about twice as fast.
The idea is simple: we run RK45 without thinking about the backward pass.
After this has been run, we rerun using RK4, which has far fewer FLOPs.

For now I've only implemented this for RK4 and only on the CPU.
Lots could be done to improve furhter.
We could for instance make a C++ extension, but I am struggling with [this issue](https://discuss.pytorch.org/t/pass-torchscript-from-python-to-c-without-serialization/72163).

----

### Simple Example
```python
import torch
from sillyode import sillyode
import matplotlib.pyplot as plt

def func(t, y):
    r = torch.empty_like(y)
    r[0] = -(y[1]) ** 3 * torch.exp(-t/5)
    r[1] = y[0]
    return r

if __name__ == '__main__':
    t = torch.linspace(0, 100, 1000)
    y0 = torch.tensor([1.0, 0.2], requires_grad=True)
    y = sillyode(func, y0, t)

    y[-1, 0].backward()
    print(y0.grad)

    plt.plot(t, y.detach())
    plt.show()
```

### Timing
See `examples/timings.py` for code which produces
```
No .backward()
   torchdiffeq.odeint :  13.38 s
   torchdiffeq.odeint_adjoint :  13.58 s
   sillyode :  5.93 s

With .backward()
   torchdiffeq.odeint :  39.54 s
   torchdiffeq.odeint_adjoint :  78.33 s
   sillyode :  19.94 s
```
