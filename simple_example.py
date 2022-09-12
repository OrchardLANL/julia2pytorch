#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import grad

from src.Julia_coupling import julia2pytorch

"""
setup_code is julia code that does any setting up we need to define our 
function in julia you can use it to import packages in julia, define globals, etc.
"""

setup_code = """
                import Zygote
                A = [1 2; 3 4]
                f(x, y) = A * x + y
                f(x) = A * x
             """

func_name = "f" # the name of the julia
                # function we want to be
                # able to differentiate
                # in pytorch
                          
f = julia2pytorch(func_name, setup_code)
x = torch.zeros(2).requires_grad_(True)
y = torch.zeros(2).requires_grad_(True)

f_eval = f(x)
g = grad(f_eval.sum(), [x])
print("Is d/dx[f(x)] correct?")
g = g[0].detach().numpy()
t = (g == [4, 6]).all()
print(t) # prints "True"

f_eval = f(x, y)
g = grad(f_eval.sum(), [x, y])
print("Is d/d[x, y][f(x, y)] correct?")
g0 = g[0].detach().numpy()
g1 = g[1].detach().numpy()
t0 = (g0 == [4, 6]).all()
t1 = (g1 == [1, 1]).all()
print(t0 and t1) # prints "True"
