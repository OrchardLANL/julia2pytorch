#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def singlephase_setup(domain_size=[260,256], dx=0.0001):
    
    
    domain_size_m = [i*dx for i in domain_size]

    code = f""" 
                import DPFEHM
                import Zygote
                
                ns   = {domain_size}
                mins = [0, 0]
                maxs = {domain_size_m} # meters
                
                coords, neighbors, areasoverlengths, _ = DPFEHM.regulargrid2d(mins, maxs, ns, 0.01)
                
                Qs = zeros(size(coords, 2))
                logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))
                boundaryhead(x, y) =  (x - maxs[1]) / (mins[1] - maxs[1]) + 1
                
                dirichletnodes = Int[]
                dirichleths = zeros(size(coords, 2))
                for i = 1:size(coords, 2)
                                if coords[1, i] == mins[1] || coords[1, i] == maxs[1]
                                                push!(dirichletnodes, i)
                                                dirichleths[i] = boundaryhead(coords[1:2, i]...)
                                end
                end
                
                function solveforp(logKs)
                                @assert length(logKs) == length(Qs)
                                Ks_neighbors = logKs2Ks_neighbors(logKs)
                                return DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
                end
    """
    
    
    return code

