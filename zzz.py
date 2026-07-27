import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

f = a * b
f.backward()
print(a.grad, b.grad)

g = a * b.detach()
g.backward()
print(a.grad, b.grad)