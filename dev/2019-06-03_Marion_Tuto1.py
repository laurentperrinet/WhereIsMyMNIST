# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py

from __future__ import print_function
import torch


x = torch.empty(5, 3) # 5 lignes, 3 colonnes
print(x)

x = torch.rand(2, 5) # Xij nb aleatoire entre 0 et 1
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3]) # construction de tensor directement avec les donnees
print(x)

y = torch.empty(5, 3)
print("y1 :", y)
y = y.new_ones(5, 3, dtype=torch.double)
print("y2 :", y)
y = torch.randn_like(y, dtype=torch.float)
print("y3 :", y)


z = torch.empty(5, 3)
print("z1 :", z)
z = torch.randn_like(z)
print("z1 :", z)

print(z.size()) # la sortie est un tuple

print('y', y)
print('z', z)
print('y+z', torch.add(z, y))

result = torch.empty(5, 3)
torch.add(z, y, out=result)
print( "result", result)

y.add_(z)
print(y)

# Any operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.

print(result[:, 1]) # affichage colonne 1
print(result[:, 2])
print(result[0, :]) # affichage ligne 0


x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
print("x : ", x)
print("y : ", y)
print("z : ", z)

x = torch.randn(4, 3)
z = x.view(3, -1)
print(x.size(), z.size())
print("x : ", x)
print("z : ", z)

x = torch.randn(1)
print(x)
print(x.item())

a = torch.ones(5)
print("a ", a)

b = a.numpy()
print("b", b)

a.add_(1)
print("a2", a)
print("b2", b)

b = b+1
print("a3", a)
print("b3", b)

a.add_(1)
print("a4", a)
print("b4", b)

a.add_(1)
print("a5", a)
print("b5", b)


a = torch.ones(5)
print("a ", a)

b = a.numpy()
print("b", b)

a.add_(1)
print("a1bis", a)
print("b1bis", b)
a.add_(1)
print("a2bis", a)
print("b2bis", b)
a.add_(1)
print("a3bis", a)
print("b3bis", b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

if torch.cuda.is_available():
    print("Cuda available !")
    device = torch.device("cuda")
    y = torch.ones_like(x, device = device)
    x = x.to(device)
    z = x+y
    print(z)
    print(z.to("cpu", torch.double))
else:
    print("Cuda not available")



