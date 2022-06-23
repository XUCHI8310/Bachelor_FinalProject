#  对函数y = 2xTx求导
import torch

x = torch.arange(4.0)
x.requires_grad_(True)
x.grad
y = 2 * torch.dot(x, x)
y.backward()
print(x.grad)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

#  标量关于向量求导 得到的是一个向量；那向量关于向量求导 得到的就是一个矩阵；这里就是把向量通过sum的形式转化成了标量，进行求导
x.grad.zero_()
y = x * x
y.sum().backward()
print(x.grad)
