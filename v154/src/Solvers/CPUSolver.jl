#// # FILE: .\src\Solvers\CPUSolver.jl
module CPUSolver

using LinearAlgebra, SparseArrays, Base.Threads, Printf
using ..Element

export MatrixFreeSystem, solve_system_cpu

struct MatrixFreeSystem{T}
    nodes::Matrix{T}
    elements::Matrix{Int}
    E::T
    nu::T
    bc_indicator::Matrix{T}
    free_dofs::Vector{Int}
    constrained_dofs::Vector{Int}
    density::Vector{T}
    min_stiffness_threshold::T 
    canonical_ke::Matrix{T}
end

function MatrixFreeSystem(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                          bc_indicator::Matrix{T}, density::Vector{T}=nothing,
                          min_stiffness_threshold::T=Float32(1.0e-3)) where T
                                    
    nElem = size(elements, 1)
    if density === nothing; density = ones(T, nElem); end

    nNodes = size(nodes, 1)
    ndof   = nNodes * 3
    constrained = falses(ndof)
    @inbounds for i in 1:nNodes
        if bc_indicator[i,1]>0; constrained[3*(i-1)+1]=true; end
        if bc_indicator[i,2]>0; constrained[3*(i-1)+2]=true; end
        if bc_indicator[i,3]>0; constrained[3*(i-1)+3]=true; end
    end

    free_dofs        = findall(!, constrained)
    constrained_dofs = findall(x->x, constrained)

    n1, n2, n4, n5 = nodes[elements[1,1], :], nodes[elements[1,2], :], nodes[elements[1,4], :], nodes[elements[1,5], :]
    dx, dy, dz = norm(n2-n1), norm(n4-n1), norm(n5-n1)
    canonical_ke = Element.get_canonical_stiffness(dx, dy, dz, nu)

    return MatrixFreeSystem(nodes, elements, E, nu, bc_indicator,
                            free_dofs, constrained_dofs, density,
                            min_stiffness_threshold, canonical_ke) 
end

"""
    apply_stiffness(system, x)

Updated to use Task-Based Partitioning instead of threadid() indexing.
This prevents BoundsErrors when thread IDs exceed nthreads().
"""
function apply_stiffness(system::MatrixFreeSystem{T}, x::Vector{T}) where T
    nNodes = size(system.nodes, 1)
    ndof   = nNodes * 3
    nElem  = size(system.elements, 1)

    result = zeros(T, ndof)
    
    # Determine chunking for parallelism
    n_chunks = Threads.nthreads()
    chunk_size = cld(nElem, n_chunks)
    
    # Storage for partial results from each task
    partial_results = Vector{Vector{T}}(undef, n_chunks)

    Ke_base = system.canonical_ke

    @sync for (i, chunk_range) in enumerate(Iterators.partition(1:nElem, chunk_size))
        Threads.@spawn begin
            # Allocate local buffer for this specific task
            local_res = zeros(T, ndof)
            
            # Temporary arrays to avoid allocation inside the loop
            u_elem = zeros(T, 24)
            
            for e in chunk_range
                dens = system.density[e]
                if dens >= system.min_stiffness_threshold
                    conn = view(system.elements, e, :)
                    factor = system.E * dens

                    # Gather displacement
                    for k in 1:8
                        node_id = conn[k]
                        base = 3*(node_id-1)
                        u_elem[3*(k-1)+1] = x[base+1]
                        u_elem[3*(k-1)+2] = x[base+2]
                        u_elem[3*(k-1)+3] = x[base+3]
                    end

                    # Compute Element Force
                    # Note: Manual multiplication or BLAS can be used here. 
                    # Assuming small matrix, simple mult is fine.
                    f_elem = (Ke_base * u_elem) .* factor

                    # Scatter force to local buffer
                    for k in 1:8
                        node_id = conn[k]
                        base = 3*(node_id-1)
                        local_res[base+1] += f_elem[3*(k-1)+1]
                        local_res[base+2] += f_elem[3*(k-1)+2]
                        local_res[base+3] += f_elem[3*(k-1)+3]
                    end
                end
            end
            partial_results[i] = local_res
        end
    end

    # Reduce results
    for res in partial_results
        if isassigned(partial_results, 1) && res !== nothing
            result .+= res
        end
    end
    
    return result
end

function apply_system(system::MatrixFreeSystem{T}, x::Vector{T}) where T
    return apply_stiffness(system, x)
end

function apply_system_free_dofs(system::MatrixFreeSystem{T}, x_free::Vector{T}) where T
    nNodes = size(system.nodes, 1)
    ndof   = nNodes * 3
    x_full = zeros(T, ndof)
    x_full[system.free_dofs] = x_free
    result_full = apply_system(system, x_full)
    return result_full[system.free_dofs]
end

"""
    compute_diagonal_preconditioner(system)

Updated to use Task-Based Partitioning.
"""
function compute_diagonal_preconditioner(system::MatrixFreeSystem{T}) where T
    nNodes = size(system.nodes, 1)
    ndof   = nNodes*3
    nElem  = size(system.elements, 1)
    
    diag_vec = zeros(T, ndof)
    
    n_chunks = Threads.nthreads()
    chunk_size = cld(nElem, n_chunks)
    partial_diags = Vector{Vector{T}}(undef, n_chunks)
    
    Ke_base = system.canonical_ke

    @sync for (i, chunk_range) in enumerate(Iterators.partition(1:nElem, chunk_size))
        Threads.@spawn begin
            local_diag = zeros(T, ndof)
            
            for e in chunk_range
                dens = system.density[e]
                if dens >= system.min_stiffness_threshold
                    conn = view(system.elements, e, :)
                    factor = system.E * dens

                    for k in 1:8
                        node_id = conn[k]
                        base_dof = 3*(k-1) # Local DOF index (0-23)
                        
                        diag_val_x = Ke_base[base_dof+1, base_dof+1] * factor
                        diag_val_y = Ke_base[base_dof+2, base_dof+2] * factor
                        diag_val_z = Ke_base[base_dof+3, base_dof+3] * factor
                        
                        global_idx = 3*(node_id-1)
                        local_diag[global_idx+1] += diag_val_x
                        local_diag[global_idx+2] += diag_val_y
                        local_diag[global_idx+3] += diag_val_z
                    end
                end
            end
            partial_diags[i] = local_diag
        end
    end

    for d in partial_diags
        if isassigned(partial_diags, 1) && d !== nothing
            diag_vec .+= d
        end
    end
    return diag_vec
end

function matrix_free_cg_solve(system::MatrixFreeSystem{T}, f::Vector{T};
                              max_iter=1000, tol=1e-6, use_precond=true,
                              shift_factor::T=Float32(1.0e-6)) where T  
    f_free = f[system.free_dofs]
    n_free = length(system.free_dofs)
    x_free = zeros(T, n_free)

    diag_full = compute_diagonal_preconditioner(system)
    diag_free = diag_full[system.free_dofs]

    shift = T(0.0)
    try
        max_diag = maximum(diag_free)
        shift = shift_factor * max_diag
        println("CPUSolver: Applying diagonal shift: $shift (Factor: $shift_factor)")
    catch e
        @warn "Could not calculate diagonal shift: $e"
    end
    
    r = copy(f_free)
    diag_free[diag_free .<= shift] .= shift
    z = use_precond ? r ./ diag_free : copy(r)
    p = copy(z)
    rz_old = dot(r, z)

    println("Starting matrix-free CG solve with $(n_free) unknowns on $(Threads.nthreads()) threads...")
    total_time = 0.0
    norm_f = norm(f_free)
    final_res_norm = 0.0

    if norm_f == 0
        return (zeros(T, length(f)), 0.0, "None")
    end

    for iter in 1:max_iter
        iter_start = time()
        Ap = apply_system_free_dofs(system, p) .+ (shift .* p)
        alpha = rz_old / dot(p, Ap)
        x_free .+= alpha .* p
        r .-= alpha .* Ap
        
        final_res_norm = norm(r) / norm_f
        total_time += (time() - iter_start)

        if final_res_norm < tol
            println("CG converged in $iter iterations, residual = $final_res_norm, total time = $total_time sec")
            break
        end

        # Re-apply preconditioner
        diag_free[diag_free .<= shift] .= shift
        z = use_precond ? r ./ diag_free : copy(r)
        
        rz_new = dot(r, z)
        beta = rz_new / rz_old
        p .= z .+ beta .* p
        rz_old = rz_new
    end

    x_full = zeros(T, length(f))
    x_full[system.free_dofs] = x_free
    
    return (x_full, final_res_norm, "Jacobi_CPU")
end

function solve_system_cpu(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                          bc_indicator::Matrix{T}, f::Vector{T};
                          max_iter=1000, tol=1e-6, use_precond=true,
                          density::Vector{T}=nothing,
                          shift_factor::T=Float32(1.0e-6),
                          min_stiffness_threshold::T=Float32(1.0e-3)) where T    
                                    
    system = MatrixFreeSystem(nodes, elements, E, nu, bc_indicator, density, min_stiffness_threshold)
    
    solve_start = time()
    solution = matrix_free_cg_solve(system, f, max_iter=max_iter, tol=tol, 
                                    use_precond=use_precond, shift_factor=shift_factor)
    solve_end = time()
    @printf("Total solution time (matrix-free CPU): %.6f sec\n", solve_end - solve_start)
    return solution
end

end