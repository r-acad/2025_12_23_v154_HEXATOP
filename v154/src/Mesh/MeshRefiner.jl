# FILE: .\src\Mesh\MeshRefiner.jl
module MeshRefiner

using LinearAlgebra
using Printf
using Base.Threads
using ..Mesh
using ..Helpers

export refine_mesh_and_fields

function estimate_element_memory_cost_bytes(hard_element_limit::Int)
    if hard_element_limit > 500_000_000 
        return 180 
    else
        return 220 
    end
end

"""
    sample_nearest_neighbor(grid, nx, ny, nz, tx, ty, tz)

Robust Piecewise Constant (Nearest Neighbor) sampling.
Preserves sharp boundaries (Solid vs Void) and avoids creating artificial "gray" elements
that the optimizer would immediately cull.
"""
@inline function sample_nearest_neighbor(grid::Vector{Float32}, nx::Int, ny::Int, nz::Int, 
                                         tx::Float32, ty::Float32, tz::Float32)
    # Map normalized coords (0.0 - 1.0) to grid indices
    # We add 0.5 to sample the center of the voxel, ensuring strict nearest neighbor behavior
    ix = clamp(floor(Int, tx * nx), 0, nx - 1)
    iy = clamp(floor(Int, ty * ny), 0, ny - 1)
    iz = clamp(floor(Int, tz * nz), 0, nz - 1)

    # Linear index calculation (1-based for Julia)
    idx = (ix + 1) + (iy * nx) + (iz * nx * ny)
    
    return @inbounds grid[idx]
end

function refine_mesh_and_fields(nodes::Matrix{Float32}, 
                                elements::Matrix{Int}, 
                                density::Vector{Float32}, 
                                alpha_field::Vector{Float32}, 
                                current_dims::Tuple{Int, Int, Int},
                                target_active_count::Int, 
                                domain_bounds::NamedTuple; 
                                max_growth_rate::Float64=1.2, 
                                hard_element_limit::Int=800_000_000,
                                update_method::String="hard") 

    C_RESET = "\u001b[0m"
    C_BOLD = "\u001b[1m"
    C_CYAN = "\u001b[36m"
    C_GREEN = "\u001b[32m"
    C_YELLOW = "\u001b[33m"

    println("\n" * C_CYAN * "="^60 * C_RESET)
    println(C_CYAN * C_BOLD * ">>> [MESH REFINER] Evaluating Refinement (Conservative Voxel Split)" * C_RESET)

    n_total_old = length(density)
    n_active_old = count(d -> d > 0.001f0, density) 
    active_ratio = max(0.0001, n_active_old / n_total_old) 
    
    println("    Current Total Elements:  $(n_total_old)")
    println("    Current Active Elements: $(n_active_old) ($(round(active_ratio*100, digits=2))%)")
    println("    Target Active Limit:     $(target_active_count)")

    ideal_total_from_target = round(Int, (target_active_count / active_ratio) * 0.90)
    rate_limit_elements = round(Int, n_total_old * max_growth_rate)
    
    final_new_total = n_total_old

    if n_active_old >= target_active_count
        println(C_GREEN * "    [OK] Target active count reached. Maintaining mesh resolution." * C_RESET)
        final_new_total = n_total_old 
    else
        limits = [
            ("Target Active Limit", ideal_total_from_target),
            ("Growth Rate Limit", rate_limit_elements),
            ("Config Hard Limit", hard_element_limit)
        ]
        
        sort!(limits, by = x -> x[2])
        
        limiting_factor_name = limits[1][1]
        final_new_total = limits[1][2]
        
        println("    Constraint Analysis:")
        for (name, val) in limits
            col = (name == limiting_factor_name) ? C_YELLOW : C_RESET
            println("      - $col$name: $(Base.format_bytes(val * 200)) approx ($val elems)$C_RESET")
        end
        println("    LIMIT APPLIED: $C_YELLOW$limiting_factor_name$C_RESET")
    end

    if final_new_total > hard_element_limit
        final_new_total = hard_element_limit
    end

    # Enforce a minimum growth to justify the reallocation cost
    if final_new_total < (n_total_old * 1.05)
        println(C_YELLOW * "    [SKIP] Calculated growth too small (< 5%). Skipping." * C_RESET)
        println(C_CYAN * "="^60 * "\n" * C_RESET)
        return nodes, elements, density, alpha_field, current_dims
    end

    println(C_GREEN * C_BOLD * "    >>> EXECUTING REFINEMENT TO: $final_new_total elements" * C_RESET)

    len_x, len_y, len_z = domain_bounds.len_x, domain_bounds.len_y, domain_bounds.len_z
    new_nx, new_ny, new_nz, new_dx, new_dy, new_dz, actual_count = 
        Helpers.calculate_element_distribution(len_x, len_y, len_z, final_new_total)
        
    println("      > Grid: $(new_nx)x$(new_ny)x$(new_nz) = $actual_count")
    println("      > Res:  $(new_dx) x $(new_dy) x $(new_dz)")

    new_nodes, new_elements, new_dims = Mesh.generate_mesh(
        new_nx, new_ny, new_nz; 
        dx=new_dx, dy=new_dy, dz=new_dz
    )
    
    # Offset new nodes to match original domain origin
    min_pt = domain_bounds.min_pt
    new_nodes[:, 1] .+= min_pt[1]
    new_nodes[:, 2] .+= min_pt[2]
    new_nodes[:, 3] .+= min_pt[3]
    
    println("      > Mapping fields (Nearest Neighbor / Voxel Split)...")
    n_new_total = size(new_elements, 1)
    new_density = zeros(Float32, n_new_total)
    new_alpha   = zeros(Float32, n_new_total) 
    
    old_nx = Int(current_dims[1] - 1)
    old_ny = Int(current_dims[2] - 1)
    old_nz = Int(current_dims[3] - 1)
    
    # Pre-calculate stride constants for the NEW mesh
    new_nx_64 = Int64(new_nx)
    new_ny_64 = Int64(new_ny)
    
    Threads.@threads for e_new in 1:n_new_total
        # 1. Get (i, j, k) of the NEW element
        e_new_idx = Int64(e_new)
        # 0-based calculation for math
        tmp = e_new_idx - 1
        iz = div(tmp, new_nx_64 * new_ny_64)
        rem = tmp % (new_nx_64 * new_ny_64)
        iy = div(rem, new_nx_64)
        ix = rem % new_nx_64
        
        # 2. Compute normalized position (0.0 to 1.0)
        # Using element centers (i + 0.5) to sample ensures we pick the correct parent voxel
        tx = (Float32(ix) + 0.5f0) / Float32(new_nx)
        ty = (Float32(iy) + 0.5f0) / Float32(new_ny)
        tz = (Float32(iz) + 0.5f0) / Float32(new_nz)
        
        # 3. Nearest Neighbor Sample from OLD grid
        # This effectively splits parent voxels into children without blurring
        new_density[e_new] = sample_nearest_neighbor(density, old_nx, old_ny, old_nz, tx, ty, tz)
        new_alpha[e_new]   = sample_nearest_neighbor(alpha_field, old_nx, old_ny, old_nz, tx, ty, tz)
    end
    
    println(C_GREEN * "    [DONE] Refinement Complete (Boundary Preserved)." * C_RESET)
    println(C_CYAN * "="^60 * "\n" * C_RESET)

    return new_nodes, new_elements, new_density, new_alpha, new_dims
end

end