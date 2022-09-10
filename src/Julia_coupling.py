#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd.function import once_differentiable


# julieries
from julia.api import Julia
jl = Julia(compiled_modules=False)
#import julia
from julia import Main





def julia2pytorch(fn_name,setup_code):
    Main.eval(setup_code)
    
    class _JuliaFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *tensor_args):   
            Main.pytorch_arguments___ = tuple(x.detach().numpy() for x in tensor_args)
            y, back = Main.eval(f"Zygote.pullback({fn_name}, pytorch_arguments___...)") 
            ctx.back = back
            return torch.as_tensor(y)    
    
        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            grad_inputs = ctx.back(grad_output.detach().numpy())
            grad_inputs = tuple(torch.from_numpy(gx) for gx in grad_inputs)
            return grad_inputs
    
    name = f"JuliaFunction_{fn_name}"
    _JuliaFunction.__name__= name
    _JuliaFunction.__qualname__= name
    return _JuliaFunction.apply












