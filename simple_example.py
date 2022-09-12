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

julia_function_name = "f" # the name of the julia function we want to be able to 
                          # differentiate in pytorch
                          
f = julia2pytorch(julia_function_name, setup_code)
x,y = torch.zeros(2).requires_grad_(True), torch.zeros(2).requires_grad_(True)

f_eval = f(x)
g = grad(f_eval.sum(), [x])
print("Is the gradient of f(x) correct?")
print( ( g[0].detach().numpy() == [4, 6] ).all() )

f_eval = f(x, y)
g = grad(f_eval.sum(), [x, y])
print("Is the gradient of f(x, y) correct?")
print( ( g[0].detach().numpy() == [4, 6] ).all() and 
       ( g[1].detach().numpy() == [1, 1] ).all() )