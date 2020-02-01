import torch

print('\n===== gradients =====');
x = torch.ones(2, 2, requires_grad = True);
print(x);
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean();
print(z, out);

print("\n .requires_grad_")
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
print(b)
print("calling backprop for out");
out.backward();
print("gradient of x is:");
print(x.grad)

print("randomized t3");
x = torch.randn(3, requires_grad = True)
print(x)
y = x * 2
print(y.data);
while y.data.norm() < 1000:
  print(y, y.data.norm())
  y = y * 2

v = torch.tensor([.1, 1.0, .0001], dtype=torch.float)
y.backward(v)
print(x.grad)
y = x.detach()
print(y)
print(x.eq(y).all())

