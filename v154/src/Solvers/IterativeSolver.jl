// # FILE: .\src\Solvers\IterativeSolver.jl
module IterativeSolver 
 
using LinearAlgebra, Printf 
using ..CPUSolver 
using ..GPUSolver 
 
export solve_system_iterative 
 
function solve_system_iterative(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T, 
                                bc_indicator::Matrix{T}, f::Vector{T}; 
                                solver_type=:matrix_free, max_iter=1000, tol=1e-6, 
                                use_precond=true, density::Vector{T}=nothing, 
                                gpu_method=:native, krylov_solver=:cg, 
                                shift_factor::T=Float32(1.0e-6), 
                                min_stiffness_threshold::T=Float32(1.0e-3),
                                u_guess::AbstractVector=T[], 
                                config::Dict=Dict()) where T                                 
                                  
    if solver_type == :matrix_free 
        return CPUSolver.solve_system_cpu( 
            nodes, elements, E, nu, bc_indicator, f; 
            max_iter=max_iter, tol=tol, use_precond=use_precond,  
            density=density, shift_factor=shift_factor,  
            min_stiffness_threshold=min_stiffness_threshold 
        ) 
    elseif solver_type == :gpu 
        if density === nothing 
            error("You must provide a density array for GPU solver.") 
        end 
        
        # Explicit call to exported function
        return GPUSolver.solve_system_gpu( 
            nodes, elements, E, nu, bc_indicator, f, density; 
            max_iter=max_iter, tol=tol,  
            method=gpu_method, solver=krylov_solver, use_precond=use_precond, 
            shift_factor=shift_factor, 
            min_stiffness_threshold=min_stiffness_threshold,
            u_guess=u_guess, 
            config=config 
        ) 
    else 
        error("Unknown solver type: $solver_type. Use :matrix_free or :gpu.") 
    end 
end 
 
end