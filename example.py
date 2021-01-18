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




