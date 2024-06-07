import torch
import numpy as np

# a = torch.rand(8, requires_grad=True)
# a.retain_grad()
# print(a)
#
# b = torch.reshape(a, (2, 2, 2))
# b.retain_grad()
# print(b)
#
# c = torch.permute(b, (2, 0, 1))
# c.retain_grad()
# print(c)
#
# d = torch.sum(c)
# d.retain_grad()
# print(d)
#
# d.backward()
#
# print(a.grad)
# print(b.grad)
# print(c.grad)
# print(d.grad)
arr = [i for i in range(8)]
n = np.array(arr)

a = torch.from_numpy(n)
print(a)

s = (2, 2, 2)
print("Reshaping to: " + str(s))
b = torch.reshape(a, shape=s)
print(b)

d = (2, 1, 0)
print("Permuting with dims" + str(d))
c = torch.permute(b, dims=d)
print(c)

e = torch.sum(c, dim=0)
print(e)
