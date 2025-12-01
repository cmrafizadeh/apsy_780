import torch
import numpy as np

# a = 2
# a = np.array([2.0])
a = torch.tensor(2.0, requires_grad=True) # 1D tensor
b = torch.tensor(1.0, requires_grad=True)

c = a + b
d = b + 1
e = c * d

print("Tensor a: ", a)
print("Tensor b: ", b)
print("Tensor c: ", c)
print("Tensor d: ", d)
print("Tensor e: ", e)

# e.backward()
# print(f"Gradient of e wrt a = {a.grad}")
# print(f"Gradient of e wrt b = {b.grad}")


c.backward()
print(f"Gradient of c wrt a = {a.grad}")
print(f"Gradient of c wrt b = {b.grad}")