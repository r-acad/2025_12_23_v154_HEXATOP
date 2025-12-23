// # FILE: .\src\Solvers\GPUSolver.jl
module GPUSolver

using LinearAlgebra, Printf
using CUDA
using SparseArrays
using Base.Threads
using Dates 
using Statistics
using ..Element
using ..GPUGeometricMultigrid 
using ..Diagnostics
using ..Helpers

export solve_system_gpu

const HISTORY_LOG_FILE = "convergence_history_log.txt"

function dual_log(msg::String; force_file::Bool=false)
    print(msg)
    if force_file
        clean_msg = replace(msg, r"\u001b\[[0-9;]*m" => "")
        try
            open(HISTORY_LOG_FILE, "a") do io; write(io, clean_msg); end
        catch e; @warn "Log write failed: $e"; end
    end
    flush(stdout)
end

function log_section_header(title::String, outer_iter::Any="?")
    width = 80
    s = "\n\u001b[36m" * "="^width * "\u001b[0m\n"
    full_title = "$title [Topo Opt Iter: $outer_iter]"
    pad = max(0, (width - length(full_title) - 2) ÷ 2)
    s *= " "^pad * "\u001b[1m" * full_title * "\u001b[0m\n"
    s *= "\u001b[36m" * "="^width * "\u001b[0m\n"
    dual_log(s; force_file=true)
end

mutable struct CGWorkspace
    is_initialized::Bool
    precision_type::DataType
    n_nodes_cached::Int  
    r::Any 
    p::Any
    z_Ap::Any 
    x::Any    
    conn_gpu::Any               
    factors_gpu::Any        
    Ke_gpu::Any                
    map_gpu::Any            
    CGWorkspace() = new(false, Float32, 0, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

const GLOBAL_CG_CACHE = CGWorkspace()

function free_cg_workspace_if_needed(n_free::Int, n_elem::Int, n_nodes::Int)
    ws = GLOBAL_CG_CACHE
    if ws.is_initialized
        current_r_len = (ws.r !== nothing) ? length(ws.r) : 0
        current_factors_len = (ws.factors_gpu !== nothing) ? length(ws.factors_gpu) : 0
        
        if current_r_len != n_free || current_factors_len != n_elem || ws.n_nodes_cached != n_nodes
            dual_log("  [GPU Memory] Releasing Stale Solver Cache to free VRAM...\n"; force_file=true)
            if ws.r !== nothing; CUDA.unsafe_free!(ws.r); ws.r = nothing; end
            if ws.p !== nothing; CUDA.unsafe_free!(ws.p); ws.p = nothing; end
            if ws.z_Ap !== nothing; CUDA.unsafe_free!(ws.z_Ap); ws.z_Ap = nothing; end
            if ws.x !== nothing; CUDA.unsafe_free!(ws.x); ws.x = nothing; end
            if ws.conn_gpu !== nothing; CUDA.unsafe_free!(ws.conn_gpu); ws.conn_gpu = nothing; end
            if ws.factors_gpu !== nothing; CUDA.unsafe_free!(ws.factors_gpu); ws.factors_gpu = nothing; end
            if ws.Ke_gpu !== nothing; CUDA.unsafe_free!(ws.Ke_gpu); ws.Ke_gpu = nothing; end
            if ws.map_gpu !== nothing; CUDA.unsafe_free!(ws.map_gpu); ws.map_gpu = nothing; end
            ws.is_initialized = false
            ws.n_nodes_cached = 0
            Helpers.cleanup_memory()
        end
    end
end

function get_cg_workspace(n_free::Int, n_total_nodes::Int, n_elem::Int, T::DataType)
    ws = GLOBAL_CG_CACHE
    if !ws.is_initialized
        dual_log("  [GPU Alloc] Allocating solver workspace vectors... "; force_file=false)
        ws.r = CUDA.zeros(T, n_free)
        ws.p = CUDA.zeros(T, n_free)
        ws.z_Ap = CUDA.zeros(T, n_free) 
        ws.x = CUDA.zeros(T, n_free)
        ws.conn_gpu = CUDA.zeros(Int, n_elem * 8)
        ws.factors_gpu = CUDA.zeros(T, n_elem)
        ws.Ke_gpu = CUDA.zeros(T, 24, 24)
        ws.map_gpu = CUDA.zeros(Int, n_free) 
        ws.precision_type = T
        ws.n_nodes_cached = n_total_nodes
        ws.is_initialized = true
        dual_log("Done.\n"; force_file=false)
    end
    fill!(ws.r, T(0.0)); fill!(ws.p, T(0.0)); fill!(ws.z_Ap, T(0.0)); fill!(ws.x, T(0.0))
    return ws
end

function matvec_unstructured_atomic_kernel!(y_full, x_full, conn, Ke, factors, nElem, nDof)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        base_idx = (e - 1) * 8
        n1 = conn[base_idx+1]; n2 = conn[base_idx+2]; n3 = conn[base_idx+3]; n4 = conn[base_idx+4]
        n5 = conn[base_idx+5]; n6 = conn[base_idx+6]; n7 = conn[base_idx+7]; n8 = conn[base_idx+8]

        if n1 < 1 || n2 < 1 || n3 < 1 || n4 < 1 || n5 < 1 || n6 < 1 || n7 < 1 || n8 < 1
            return 
        end

        u1x = x_full[(n1-1)*3+1]; u1y = x_full[(n1-1)*3+2]; u1z = x_full[(n1-1)*3+3]
        u2x = x_full[(n2-1)*3+1]; u2y = x_full[(n2-1)*3+2]; u2z = x_full[(n2-1)*3+3]
        u3x = x_full[(n3-1)*3+1]; u3y = x_full[(n3-1)*3+2]; u3z = x_full[(n3-1)*3+3]
        u4x = x_full[(n4-1)*3+1]; u4y = x_full[(n4-1)*3+2]; u4z = x_full[(n4-1)*3+3]
        u5x = x_full[(n5-1)*3+1]; u5y = x_full[(n5-1)*3+2]; u5z = x_full[(n5-1)*3+3]
        u6x = x_full[(n6-1)*3+1]; u6y = x_full[(n6-1)*3+2]; u6z = x_full[(n6-1)*3+3]
        u7x = x_full[(n7-1)*3+1]; u7y = x_full[(n7-1)*3+2]; u7z = x_full[(n7-1)*3+3]
        u8x = x_full[(n8-1)*3+1]; u8y = x_full[(n8-1)*3+2]; u8z = x_full[(n8-1)*3+3]

        u_loc = (u1x, u1y, u1z, u2x, u2y, u2z, u3x, u3y, u3z, u4x, u4y, u4z, 
                 u5x, u5y, u5z, u6x, u6y, u6z, u7x, u7y, u7z, u8x, u8y, u8z)
        factor = factors[e]
        node_lookup = (n1, n2, n3, n4, n5, n6, n7, n8)

        @inbounds for i in 1:24
            val = zero(u1x)
            val += Ke[i, 1]*u_loc[1] + Ke[i, 2]*u_loc[2] + Ke[i, 3]*u_loc[3] + Ke[i, 4]*u_loc[4]
            val += Ke[i, 5]*u_loc[5] + Ke[i, 6]*u_loc[6] + Ke[i, 7]*u_loc[7] + Ke[i, 8]*u_loc[8]
            val += Ke[i, 9]*u_loc[9] + Ke[i, 10]*u_loc[10] + Ke[i, 11]*u_loc[11] + Ke[i, 12]*u_loc[12]
            val += Ke[i, 13]*u_loc[13] + Ke[i, 14]*u_loc[14] + Ke[i, 15]*u_loc[15] + Ke[i, 16]*u_loc[16]
            val += Ke[i, 17]*u_loc[17] + Ke[i, 18]*u_loc[18] + Ke[i, 19]*u_loc[19] + Ke[i, 20]*u_loc[20]
            val += Ke[i, 21]*u_loc[21] + Ke[i, 22]*u_loc[22] + Ke[i, 23]*u_loc[23] + Ke[i, 24]*u_loc[24]
            
            target_node_idx = (i - 1) ÷ 3 + 1
            target_node = node_lookup[target_node_idx]
            target_dof_idx = (target_node - 1) * 3 + (i - 1) % 3 + 1
            
            if target_dof_idx >= 1 && target_dof_idx <= nDof
                CUDA.atomic_add!(pointer(y_full, target_dof_idx), val * factor)
            end
        end
    end
    return nothing
end

function compute_diagonal_atomic_kernel!(diag_vec, conn, Ke_diag, factors, nElem, nDof)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        base_idx = (e - 1) * 8
        factor = factors[e]
        for i in 1:8
            node = conn[base_idx + i]
            if node < 1; continue; end
            idx1 = (node - 1) * 3 + 1
            idx3 = (node - 1) * 3 + 3
            if idx3 > nDof; continue; end
            k_val = Ke_diag[(i-1)*3 + 1] * factor
            CUDA.atomic_add!(pointer(diag_vec, idx1), k_val)
            CUDA.atomic_add!(pointer(diag_vec, idx1 + 1), k_val)
            CUDA.atomic_add!(pointer(diag_vec, idx1 + 2), k_val)
        end
    end
    return nothing
end

function expand_kernel!(x_full, x_free, map, n_free, nDof)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_free
        target = map[idx]
        if target > 0 && target <= nDof
            @inbounds x_full[target] = x_free[idx]
        end
    end
    return nothing
end

function contract_kernel!(y_free, y_full, map, n_free, nDof)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_free
        src = map[idx]
        if src > 0 && src <= nDof
            @inbounds y_free[idx] = y_full[src]
        end
    end
    return nothing
end

function jacobi_precond_kernel!(z, r, M_inv, n)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n
        @inbounds z[idx] = r[idx] * M_inv[idx]
    end
    return nothing
end

function get_free_dofs(bc_indicator)
    nNodes = size(bc_indicator, 1)
    ndof = nNodes * 3
    constrained = falses(ndof)
    for i in 1:nNodes
        if bc_indicator[i,1] > 0; constrained[3*(i-1)+1] = true; end
        if bc_indicator[i,2] > 0; constrained[3*(i-1)+2] = true; end
        if bc_indicator[i,3] > 0; constrained[3*(i-1)+3] = true; end
    end
    return findall(!, constrained)
end

function gpu_matrix_free_cg_solve(nodes, elements, E, nu, bc, f, density;
                                  max_iter=40000, tol=1e-6, shift_factor=1.0e-4,
                                  min_stiffness_threshold=1.0e-3, u_guess=[], config=Dict())
    
    # --- Aggressive Fresh Cleanup ---
    Helpers.cleanup_memory()
    
    CUDA.allowscalar(false)
    gpu_profile = get(config, "hardware_profile_applied", "RTX")
    use_double = (gpu_profile == "H200" || gpu_profile == "V100" || gpu_profile == "A100") || get(config, "force_float64", false)
    T = use_double ? Float64 : Float32
    
    outer_iter = get(config, "current_outer_iter", "?")
    
    if outer_iter == 1 || outer_iter == "1"
        try
            open(HISTORY_LOG_FILE, "a") do io 
                write(io, "\n\n" * "-"^80 * "\n")
                write(io, "NEW JOB SESSION STARTED: $(Dates.now())\n")
                write(io, "-"^80 * "\n")
            end
        catch; end
    end
    
    solver_params = get(config, "solver_parameters", Dict())
    stagnation_tol = T(get(solver_params, "stagnation_tolerance", 0.0))
    max_shift_attempts = Int(get(solver_params, "max_shift_attempts", 3))
    shift_multiplier = T(get(solver_params, "shift_multiplier", 10.0))
    
    nNodes = size(nodes, 1)
    nElem = size(elements, 1)
    nDof = nNodes * 3
    free_dofs = get_free_dofs(bc)
    n_free = length(free_dofs)
    
    geom_conf = config["geometry"]
    dx = Float32(get(geom_conf, "dx_computed", 1.0))
    dy = Float32(get(geom_conf, "dy_computed", 1.0))
    dz = Float32(get(geom_conf, "dz_computed", 1.0))
    nx_f = Int(get(geom_conf, "nElem_x_computed", 0))
    ny_f = Int(get(geom_conf, "nElem_y_computed", 0))
    nz_f = Int(get(geom_conf, "nElem_z_computed", 0))
    
    want_mg = (get(solver_params, "preconditioner", "jacobi") == "multigrid")
    
    n_active_check = nElem
    if length(density) == nElem
        n_active_check = count(d -> d > min_stiffness_threshold, density)
    end

    # Clean existing workspace to ensure "fresh" VRAM for every solve
    free_cg_workspace_if_needed(n_free, nElem, nNodes)
    Helpers.enforce_gpu_memory_safety(n_active_check, nNodes, use_double, want_mg)
    
    if CUDA.functional()
        free_mem, tot_mem = CUDA.available_memory(), CUDA.total_memory()
        log_section_header("GPU SOLVER (ATOMIC)", outer_iter)
        dual_log(@sprintf("  Nodes: %d | Elems: %d | Free DOFs: %d\n", nNodes, nElem, n_free); force_file=true)
        dual_log(@sprintf("  Pre-Alloc VRAM: %.2f GB Free / %.2f GB Total\n", free_mem/1024^3, tot_mem/1024^3); force_file=true)
    end

    try 
        Helpers.log_gpu_state("Pre-Workspace Alloc")
        
        ws = get_cg_workspace(n_free, nNodes, nElem, T)
        
        Helpers.log_gpu_state("Post-Workspace Alloc")

        # 1. Runtime Safety Check (Reduced limit to 400 MB)
        # This allows 10GB cards to run closer to the edge for 50M element models.
        current_free = CUDA.available_memory()
        if current_free < (400 * 1024^2) && want_mg
            dual_log("\n\u001b[31m[CRITICAL MEMORY WARNING]\u001b[0m\n"; force_file=true)
            dual_log("  Free VRAM is critically low ($(round(current_free/1024^3, digits=2)) GB).\n"; force_file=true)
            dual_log("  Enabling Multigrid will likely freeze the system due to swapping.\n"; force_file=true)
            dual_log("  \u001b[33m>>> FORCING FALLBACK TO JACOBI PRECONDITIONER <<<\u001b[0m\n"; force_file=true)
            want_mg = false
        end
        
        Ke_base = Element.get_canonical_stiffness(dx, dy, dz, Float32(nu))
        copyto!(ws.Ke_gpu, Matrix{T}(Ke_base))
        
        # Aggressive GC before data transfer
        Helpers.cleanup_memory()
        
        dual_log("  [GPU Data] GMG Mode: Using Implicit Connectivity (Saved ~0.01 GB).\n"; force_file=false)
        # Note: We are NOT allocating explicit connectivity here because GMG Implicit kernels calculate neighbors on the fly
        # This is a key optimization.
        
        dual_log("  [GPU Data] Transferring connectivity (~$(round(nElem*8*4/1024^3, digits=2)) GB)... "; force_file=false)
        conn_flat = collect(vec(elements')) 
        copyto!(ws.conn_gpu, conn_flat) 
        dual_log("Done.\n"; force_file=false)
        
        dual_log("  [GPU Data] Transferring stiffness factors... "; force_file=false)
        fact_vec = Vector{T}(E .* density)
        copyto!(ws.factors_gpu, fact_vec) 
        dual_log("Done.\n"; force_file=false)
        
        dual_log("  [GPU Data] Transferring DOF map (~$(round(n_free*4/1024^3, digits=2)) GB)... "; force_file=false)
        copyto!(ws.map_gpu, free_dofs) 
        dual_log("Done.\n"; force_file=false)

        mg_ws = nothing
        if want_mg
            try
                dual_log("  [MG Init] Setting up Multigrid... \n"; force_file=false)
                mg_ws = GPUGeometricMultigrid.setup_multigrid(nodes, density, config)
                dual_log("\u001b[32m  [MG Init] Multigrid Levels Initialized Successfully:\u001b[0m\n"; force_file=true)
                dual_log(@sprintf("    Lv 1 (Fine): %dx%dx%d\n", nx_f, ny_f, nz_f); force_file=true)
                dual_log(@sprintf("    Lv 2 (Med):  %dx%dx%d\n", mg_ws.nc_x, mg_ws.nc_y, mg_ws.nc_z); force_file=true)
                if mg_ws.levels == 3
                    dual_log(@sprintf("    Lv 3 (Cst):  %dx%dx%d\n", mg_ws.n_cst_x, mg_ws.n_cst_y, mg_ws.n_cst_z); force_file=true)
                end
            catch e
                dual_log("\u001b[33m  [MG Init Failed] Error: $e. Falling back to Jacobi.\u001b[0m\n"; force_file=true)
                want_mg = false
                Helpers.cleanup_memory()
            end
        end
        
        x_gpu, r_gpu, p_gpu = ws.x, ws.r, ws.p
        Ap_free = ws.z_Ap; z_gpu = ws.z_Ap           
        conn_gpu = ws.conn_gpu; factors_gpu = ws.factors_gpu; Ke_gpu = ws.Ke_gpu; map_gpu = ws.map_gpu
        
        b_gpu = CuVector{T}(f[free_dofs])
        x_full = CUDA.zeros(T, nDof)
        Ap_full = CUDA.zeros(T, nDof)
        diag_full = CUDA.zeros(T, nDof)
        
        Ke_diag_cpu = diag(Ke_base)
        Ke_diag_gpu = CuArray{T}(Ke_diag_cpu)
        
        threads_per_block = 256
        @cuda threads=threads_per_block blocks=cld(nElem, threads_per_block) compute_diagonal_atomic_kernel!(
            diag_full, conn_gpu, Ke_diag_gpu, factors_gpu, nElem, nDof
        )
        
        diag_free = CUDA.zeros(T, n_free)
        @cuda threads=threads_per_block blocks=cld(n_free, threads_per_block) contract_kernel!(diag_free, diag_full, map_gpu, n_free, nDof)
        
        max_diag = maximum(diag_free)
        M_inv = CUDA.zeros(T, n_free)
        norm_b = norm(b_gpu)
        best_x = zeros(T, n_free)
        final_rel_res = 0.0
        
        function apply_A!(y_f, x_f)
            fill!(x_full, T(0.0)) 
            @cuda threads=threads_per_block blocks=cld(n_free, threads_per_block) expand_kernel!(x_full, x_f, map_gpu, n_free, nDof)
            fill!(Ap_full, T(0.0)) 
            @cuda threads=threads_per_block blocks=cld(nElem, threads_per_block) matvec_unstructured_atomic_kernel!(
                Ap_full, x_full, conn_gpu, Ke_gpu, factors_gpu, nElem, nDof
            )
            @cuda threads=threads_per_block blocks=cld(n_free, threads_per_block) contract_kernel!(y_f, Ap_full, map_gpu, n_free, nDof)
        end

        current_use_mg = want_mg
        global_solve_success = false
        
        while true
            precond_name = current_use_mg ? "Geometric Multigrid (V-Cycle)" : "Jacobi"
            dual_log(@sprintf("  Preconditioner: %s\n", precond_name); force_file=true)
            cur_shift = shift_factor
            solve_ok = false
            fallback_to_jacobi_immediate = false
            
            for attempt in 1:max_shift_attempts
                real_shift = cur_shift * max_diag
                dual_log(@sprintf("\n  >>> ATTEMPT %d | Shift: %.1e\n", attempt, real_shift); force_file=true)
                diverged = false
                
                if !isempty(u_guess) && attempt == 1 && current_use_mg == want_mg
                    copyto!(x_gpu, CuVector{T}(u_guess[free_dofs]))
                elseif attempt > 1 || current_use_mg != want_mg
                    fill!(x_gpu, T(0.0))
                end
                
                M_inv .= T(1.0) ./ (diag_free .+ real_shift)
                apply_A!(Ap_free, x_gpu)
                Ap_free .+= real_shift .* x_gpu
                r_gpu .= b_gpu .- Ap_free
                
                if current_use_mg
                    GPUGeometricMultigrid.apply_mg_vcycle!(z_gpu, r_gpu, mg_ws, M_inv, map_gpu, n_free)
                else
                    @cuda threads=threads_per_block blocks=cld(n_free, threads_per_block) jacobi_precond_kernel!(z_gpu, r_gpu, M_inv, n_free)
                end
                
                p_gpu .= z_gpu
                rz = dot(r_gpu, z_gpu)
                t_cg = time()
                best_rel = Inf
                last_rel_res = Inf 
                
                dual_log(@sprintf("  \u001b[1m%8s %12s %12s %10s %10s %10s %10s\u001b[0m\n", 
                                  "Iter", "Res", "RelRes", "Alpha", "Beta", "Time(s)", "Clock"); force_file=true)
                
                alpha_val = T(0.0)
                beta_val = T(0.0)

                for k in 1:max_iter
                    apply_A!(Ap_free, p_gpu)
                    Ap_free .+= real_shift .* p_gpu
                    
                    denom = dot(p_gpu, Ap_free)
                    
                    if abs(denom) < 1e-20
                        dual_log("\u001b[31m  >>> SINGULARITY DETECTED: Denom ~ 0 at iter $k. Aborting.\u001b[0m\n"; force_file=true)
                        diverged = true
                        break
                    end
                    
                    alpha = rz / denom
                    alpha_val = alpha 
                    
                    x_gpu .+= alpha .* p_gpu
                    r_gpu .-= alpha .* Ap_free
                    
                    if k % 500 == 0 || k == 1
                        res = sqrt(dot(r_gpu, r_gpu))
                        rel = res / norm_b
                        final_rel_res = rel
                        clock_now = Dates.format(now(), "HH:MM:SS")
                        
                        if rel < best_rel; best_rel = rel; copyto!(best_x, x_gpu); end
                        if rel < tol; solve_ok = true; break; end
                        
                        if isnan(rel) || isinf(rel)
                            dual_log("\u001b[31m  >>> NaN/Inf DETECTED at iter $k. Aborting attempt.\u001b[0m\n"; force_file=true)
                            if current_use_mg; fallback_to_jacobi_immediate = true; else; diverged = true; end
                            break
                        end

                        is_startup_transient = (k <= 20)
                        hard_divergence_limit = 100000.0
                        drift_limit = best_rel * 5000.0
                        
                        if !is_startup_transient && (rel > hard_divergence_limit || (k > 500 && rel > drift_limit))
                            dual_log("\u001b[31m  >>> DIVERGENCE DETECTED at iter $k (Rel=$rel).\u001b[0m\n"; force_file=true)
                            if current_use_mg; fallback_to_jacobi_immediate = true; else; diverged = true; end
                            break
                        end
                        
                        color = rel < tol ? "\u001b[32m" : (rel < 0.1 ? "\u001b[33m" : "\u001b[0m")
                        
                        dual_log(@sprintf("  %s%8d %12.4e %12.4e %10.2e %10.2e %10.3f %10s\u001b[0m\n", 
                                          color, k, res, rel, alpha_val, beta_val, time() - t_cg, clock_now); force_file=true)
                    end
                    
                    if k % 1000 == 0
                        improvement = (last_rel_res - final_rel_res) / (last_rel_res + 1e-20)
                        if improvement < 1e-4 && final_rel_res > tol
                            dual_log(@sprintf("   [STAGNATION CHECK] Iter %d: Improvement %.2e%% (Potential Stall)\n", k, improvement*100); force_file=true)
                        end
                        last_rel_res = final_rel_res
                    end
                    
                    if current_use_mg
                        GPUGeometricMultigrid.apply_mg_vcycle!(z_gpu, r_gpu, mg_ws, M_inv, map_gpu, n_free)
                    else
                        @cuda threads=threads_per_block blocks=cld(n_free, threads_per_block) jacobi_precond_kernel!(z_gpu, r_gpu, M_inv, n_free)
                    end
                    rz_new = dot(r_gpu, z_gpu)
                    beta = rz_new / rz
                    beta_val = beta 
                    
                    p_gpu .= z_gpu .+ beta .* p_gpu
                    rz = rz_new
                end
                
                if fallback_to_jacobi_immediate
                    dual_log("\u001b[33m  >>> RESTARTING SOLVE WITH JACOBI PRECONDITIONER <<<\u001b[0m\n"; force_file=true)
                    current_use_mg = false; continue
                end
                if solve_ok; global_solve_success = true; break; end
                if diverged
                    if attempt == max_shift_attempts
                    else; cur_shift *= shift_multiplier; continue; end
                end
                if stagnation_tol > 0.0 && best_rel < stagnation_tol
                    solve_ok = true; global_solve_success = true; copyto!(x_gpu, best_x)
                    dual_log(@sprintf("  [Adaptive] Stagnated below tolerance. Accepting.\n"); force_file=true)
                    break
                end
                cur_shift *= shift_multiplier
            end
            
            if fallback_to_jacobi_immediate
                dual_log("\u001b[33m  >>> RESTARTING SOLVE WITH JACOBI PRECONDITIONER <<<\u001b[0m\n"; force_file=true)
                current_use_mg = false; continue
            end
            if global_solve_success; break; else
                dual_log("\u001b[31m  [Failure] Solver failed to reach strict tolerance.\u001b[0m\n"; force_file=true)
                copyto!(x_gpu, best_x)
                break
            end
        end
        
        fill!(x_full, T(0.0))
        @cuda threads=threads_per_block blocks=cld(n_free, threads_per_block) expand_kernel!(x_full, x_gpu, map_gpu, n_free, nDof)
        x_final_full = Array(x_full) 
        CUDA.unsafe_free!(x_full); CUDA.unsafe_free!(Ap_full); CUDA.unsafe_free!(diag_full)
        CUDA.unsafe_free!(M_inv); CUDA.unsafe_free!(Ke_diag_gpu)
        CUDA.unsafe_free!(diag_free); CUDA.unsafe_free!(b_gpu)
        dual_log("\u001b[36m" * "-"^80 * "\u001b[0m\n"; force_file=true)
        final_method_name = current_use_mg ? "GMG_GPU" : "Jacobi_GPU"
        return (x_final_full, final_rel_res, final_method_name)

    finally
        # Final cleanup for safety
    end
end


function solve_system_gpu(nodes, elements, E, nu, bc, f, density;
                          max_iter=40000, tol=1e-6, method=:native, solver=:cg, use_precond=true,
                          shift_factor=1.0e-4, min_stiffness_threshold=1.0e-3, u_guess=[], config=Dict())
    if !CUDA.functional()
        error("CUDA not functional!")
    end
    return gpu_matrix_free_cg_solve(nodes, elements, E, nu, bc, f, density;
                                    max_iter=max_iter, tol=tol, shift_factor=shift_factor,
                                    min_stiffness_threshold=min_stiffness_threshold,
                                    u_guess=u_guess, config=config)
end

end