import julia
from julia import Main
import torch as th
from torch.autograd import grad

Main.eval("import Zygote")
def julia2pytorch(julia_function_name, setup_code=""):
    Main.eval(setup_code)
    python_code = f"""class JuliaFunction_{julia_function_name}(th.autograd.Function):
    @staticmethod
    def forward(ctx, *x):
        Main.pytorch_arguments___ = tuple(map(lambda xi: xi.detach().numpy(), x))
        y, back = Main.eval(f"Zygote.pullback({julia_function_name}, pytorch_arguments___...)")
        ctx.back = back
        return th.as_tensor(y)
    @staticmethod
    def backward(ctx, grad_output):
        return tuple(map(lambda x: th.as_tensor(x), ctx.back(grad_output.detach().numpy())))#it's a one-liner!
    """
    exec(python_code)
    return eval(f"JuliaFunction_{julia_function_name}.apply")

#setup_code is julia code that does any setting up we need to define our function in julia
#you can use it to import packages in julia, define globals, etc.
setup_code = """
A = [1 2; 3 4]
f(x, y) = A * x + y
f(x) = A * x
"""
julia_function_name = "f"#the name of the julia function we want to be able differentiate in pytorch
f = julia2pytorch(julia_function_name, setup_code)
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
