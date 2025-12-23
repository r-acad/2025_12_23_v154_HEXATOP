// # FILE: .\src\Solvers\Solver.jl
module Solver 

using CUDA 
using Printf
using ..Helpers  
using ..DirectSolver: solve_system as solve_system_direct 
using ..IterativeSolver: solve_system_iterative 
using ..MeshPruner 

export solve_system 

function choose_solver(nNodes, nElem, config) 
    solver_params = config["solver_parameters"] 
    configured_type = Symbol(lowercase(get(solver_params, "solver_type", "direct"))) 

    if configured_type == :direct 
        if nElem > 100_000 
            @warn "Direct solver requested for large mesh ($(nElem) elements). Switching to Matrix-Free iterative." 
            return :matrix_free 
        end 
        return :direct 
    elseif configured_type == :gpu 
        if CUDA.functional() 
            return :gpu 
        else 
            @warn "No CUDA GPU detected. Falling back to CPU."
            return :matrix_free 
        end 
    elseif configured_type == :matrix_free 
        return :matrix_free 
    else 
        @warn "Unknown solver_type: $(configured_type). Defaulting to matrix_free." 
        return :matrix_free 
    end 
end 

function solve_system(nodes::AbstractMatrix{<:AbstractFloat}, 
                      elements::Matrix{Int}, 
                      E::AbstractFloat, 
                      nu::AbstractFloat, 
                      bc_indicator::AbstractMatrix{<:AbstractFloat}, 
                      F::AbstractVector{<:AbstractFloat}; 
                      density::AbstractVector{<:AbstractFloat}=Float32[], 
                      config::Dict, 
                      min_stiffness_threshold::AbstractFloat=1.0e-3f0,
                      prune_voids::Bool=true,
                      u_prev::AbstractVector{<:AbstractFloat}=Float32[]) 
      
    nNodes_solve = size(nodes, 1) 
    nElem_solve = size(elements, 1) 
      
    solver_params = config["solver_parameters"] 
    solver_type = choose_solver(nNodes_solve, nElem_solve, config) 
      
    target_tol = Float32(get(solver_params, "tolerance", 1.0e-6)) 
    max_iter = Int(get(solver_params, "max_iterations", 1000)) 
    shift_factor = Float32(get(solver_params, "diagonal_shift_factor", 1.0e-6)) 
    precond_type = get(solver_params, "preconditioner", "jacobi")
      
    iter_current = get(config, "current_outer_iter", 1) 
    
    # --- PRE-SOLVE MEMORY & PRECONDITIONER CHECK ---
    if solver_type == :gpu
        tol_str = @sprintf("%.1e", target_tol)
        println("   [Solver] Strict Tol: $tol_str")
        
        is_huge_model = (nElem_solve > 5_000_000)
        use_double = get(config, "force_float64", false)

        if precond_type == "multigrid"
            # 1. Check if GMG fits in VRAM (using updated realistic estimator)
            # Note: We pass nElem_solve (Total) because GMG requires the full grid.
            feasible, req_gb, free_gb = Helpers.is_gmg_feasible_on_gpu(nElem_solve, use_double)
            
            if feasible
                # If fits: We MUST disable pruning to keep grid structured for GMG
                if prune_voids
                    println("   [Solver] Pruning DISABLED (Required for Geometric Multigrid).")
                    prune_voids = false
                end
            else
                # 2. FAILBACK STRATEGY
                # If GMG is too big, we switch to Jacobi.
                # CRITICAL: We also ENABLE PRUNING. This allows the solver to discard 
                # void elements, significantly reducing memory usage (Active vs Total).
                println("\n\u001b[33m" * ">>> [AUTO-FALLBACK] VRAM insufficient for GMG (Req: $(round(req_gb,digits=2)) GB)." * "\u001b[0m")
                println("\u001b[33m" * ">>> [AUTO-FALLBACK] Switching to JACOBI and enabling PRUNING to save memory." * "\u001b[0m\n")
                
                precond_type = "jacobi"
                config["solver_parameters"]["preconditioner"] = "jacobi" 
                prune_voids = true # <--- Explicitly enabling pruning to save VRAM
            end
        else
            # Default Jacobi behavior
            if prune_voids
                 if iter_current < 3 && !is_huge_model
                     prune_voids = false
                     println("   [Solver] Pruning skipped (Early Iteration protection).")
                 elseif iter_current < 3 && is_huge_model
                     println("   [Solver] Pruning FORCED (Huge Model Protection).")
                 end
            end
        end
    end

    # --- MESH PRUNING STEP ---
    active_system = nothing 
    solve_nodes = nodes
    solve_elements = elements
    solve_bc = bc_indicator
    solve_F = F
    solve_density = density
    solve_u_guess = u_prev

    if prune_voids && !isempty(density)
        prune_threshold = min_stiffness_threshold * 1.01f0 
        nElem_total = size(elements, 1)
        nActive = count(d -> d > prune_threshold, density)
          
        if nActive < (nElem_total * 0.99)
            active_system = MeshPruner.prune_system(nodes, elements, density, prune_threshold, bc_indicator, F)
              
            solve_nodes = active_system.nodes
            solve_elements = active_system.elements
            solve_bc = active_system.bc_indicator
            solve_F = active_system.F
            solve_density = active_system.density
            
            if !isempty(u_prev) && length(u_prev) == length(F)
                nActiveNodes = length(active_system.new_to_old_node_map)
                solve_u_guess = zeros(eltype(u_prev), nActiveNodes * 3)
                
                for (new_idx, old_idx) in enumerate(active_system.new_to_old_node_map)
                    base_old = 3 * (old_idx - 1)
                    base_new = 3 * (new_idx - 1)
                    solve_u_guess[base_new+1] = u_prev[base_old+1]
                    solve_u_guess[base_new+2] = u_prev[base_old+2]
                    solve_u_guess[base_new+3] = u_prev[base_old+3]
                end
            end
        else
            solve_u_guess = u_prev
        end
    else
        solve_u_guess = u_prev
    end

    use_precond = true 
      
    U_solved_tuple = if solver_type == :direct 
        solve_system_direct(solve_nodes, solve_elements, Float32(E), Float32(nu), solve_bc, solve_F; 
                            density=solve_density, 
                            shift_factor=shift_factor, 
                            min_stiffness_threshold=min_stiffness_threshold) 
                             
    elseif solver_type == :gpu 
        gpu_method = Symbol(lowercase(get(solver_params, "gpu_method", "krylov"))) 
        krylov_solver = Symbol(lowercase(get(solver_params, "krylov_solver", "cg"))) 
 
        solve_system_iterative(solve_nodes, solve_elements, E, nu, solve_bc, solve_F; 
                             solver_type=:gpu, max_iter=max_iter, tol=target_tol, 
                             density=solve_density, 
                             use_precond=use_precond,  
                             gpu_method=gpu_method, krylov_solver=krylov_solver, 
                             shift_factor=shift_factor, 
                             min_stiffness_threshold=min_stiffness_threshold,
                             u_guess=solve_u_guess, 
                             config=config) 
                             
    else    
        solve_system_iterative(solve_nodes, solve_elements, E, nu, solve_bc, solve_F; 
                             solver_type=:matrix_free, max_iter=max_iter, tol=target_tol, 
                             use_precond=use_precond, 
                             density=solve_density, 
                             shift_factor=shift_factor, 
                             min_stiffness_threshold=min_stiffness_threshold, 
                             config=config) 
    end 
 
    U_solved_vec = U_solved_tuple[1]
    res_val = U_solved_tuple[2]
    prec_str = U_solved_tuple[3]

    if active_system !== nothing
        U_full = MeshPruner.reconstruct_full_solution(U_solved_vec, active_system.new_to_old_node_map, size(nodes, 1))
        return (U_full, res_val, prec_str)
    else
        return U_solved_tuple
    end
end 
 
end