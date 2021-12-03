import julia
from julia import Main
import torch as th
from torch.autograd import grad

Main.eval("import Zygote")
#setup_code is julia code that does any setting up we need to define our function
setup_code = """
A = [1 2; 3 4]
f(x, y) = A * x + y
f(x) = A * x
"""
function_name = "f"#the name of the julia function we want to be able differentiate in pytorch
Main.eval(setup_code)
class JuliaFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, *x):
        Main.pytorch_arguments___ = tuple(map(lambda xi: xi.detach().numpy(), x))
        y, back = Main.eval(f"Zygote.pullback({function_name}, pytorch_arguments___...)")
        ctx.back = back
        return th.as_tensor(y)
    @staticmethod
    def backward(ctx, grad_output):
        return tuple(map(lambda x: th.as_tensor(x), ctx.back(grad_output.detach().numpy())))#it's a one-liner!

f = JuliaFunction.apply
x = th.zeros(2).requires_grad_(True)
y = th.zeros(2).requires_grad_(True)
f_eval = f(x)
g = grad(f_eval.sum(), [x])
print("gradient of f(x) is good?")
print((g[0].detach().numpy() == [4, 6]).all())
f_eval = f(x, y)
g = grad(f_eval.sum(), [x, y])
print("gradient of f(x, y) is good?")
print((g[0].detach().numpy() == [4, 6]).all() and (g[1].detach().numpy() == [1, 1]).all())
