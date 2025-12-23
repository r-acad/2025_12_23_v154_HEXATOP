# FILE: .\src\Optimization\TopOpt.jl
module TopologyOptimization 

using LinearAlgebra
using SparseArrays
using Printf  
using Statistics 
using SuiteSparse 
using CUDA
using Base.Threads
using ..Element
using ..Mesh
using ..GPUExplicitFilter
using ..Helpers

export update_density!, reset_filter_cache!

mutable struct FilterCache
    is_initialized::Bool
    radius::Float32
    K_filter::SuiteSparse.CHOLMOD.Factor{Float64} 
    FilterCache() = new(false, 0.0f0)
end

const GLOBAL_FILTER_CACHE = FilterCache()

function reset_filter_cache!()
    GLOBAL_FILTER_CACHE.is_initialized = false
end

function apply_emergency_box_filter(density::Vector{Float32}, nx::Int, ny::Int, nz::Int)
    println("    [EMERGENCY FILTER] Applying 3x3x3 box filter (CPU)...")
    
    nElem = length(density)
    filtered = copy(density)
    
    Threads.@threads for k in 2:nz-1
        for j in 2:ny-1
            for i in 2:nx-1
                e = i + (j-1)*nx + (k-1)*nx*ny
                
                if e < 1 || e > nElem; continue; end
                
                sum_rho = 0.0f0
                count = 0
                
                for dk in -1:1, dj in -1:1, di in -1:1
                    neighbor_i = i + di
                    neighbor_j = j + dj
                    neighbor_k = k + dk
                    
                    if neighbor_i >= 1 && neighbor_i <= nx &&
                       neighbor_j >= 1 && neighbor_j <= ny &&
                       neighbor_k >= 1 && neighbor_k <= nz
                        
                        neighbor_idx = neighbor_i + (neighbor_j-1)*nx + (neighbor_k-1)*nx*ny
                        
                        if neighbor_idx >= 1 && neighbor_idx <= nElem
                            sum_rho += density[neighbor_idx]
                            count += 1
                        end
                    end
                end
                
                filtered[e] = (count > 0) ? (sum_rho / count) : density[e]
            end
        end
    end
    
    return filtered
end

function create_transition_zone(protected_mask::BitVector, nx::Int, ny::Int, nz::Int, depth::Int=3)
    nElem = length(protected_mask)
    transition_zone = falses(nElem)
    
    for k in 1:nz, j in 1:ny, i in 1:nx
        e = i + (j-1)*nx + (k-1)*nx*ny
        if e < 1 || e > nElem; continue; end
        
        if protected_mask[e]; continue; end
        
        # Check if within 'depth' layers of a protected element
        found_protected = false
        for dk in -depth:depth, dj in -depth:depth, di in -depth:depth
            ni = i + di
            nj = j + dj
            nk = k + dk
            
            if ni >= 1 && ni <= nx && nj >= 1 && nj <= ny && nk >= 1 && nk <= nz
                neighbor_idx = ni + (nj-1)*nx + (nk-1)*nx*ny
                if neighbor_idx >= 1 && neighbor_idx <= nElem && protected_mask[neighbor_idx]
                    found_protected = true
                    break
                end
            end
        end
        
        if found_protected
            transition_zone[e] = true
        end
    end
    
    return transition_zone
end

function blend_transition_zone!(density::Vector{Float32}, 
                                 filtered_density::Vector{Float32},
                                 protected_mask::BitVector,
                                 transition_zone::BitVector,
                                 original_density::Vector{Float32},
                                 nx::Int, ny::Int, nz::Int,
                                 blend_depth::Int=3)
    
    nElem = length(density)
    
    Threads.@threads for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                e = i + (j-1)*nx + (k-1)*nx*ny
                if e < 1 || e > nElem; continue; end
                
                if !transition_zone[e]; continue; end
                
                min_dist = blend_depth + 1.0
                
                for dk in -blend_depth:blend_depth, dj in -blend_depth:blend_depth, di in -blend_depth:blend_depth
                    ni = i + di
                    nj = j + dj
                    nk = k + dk
                    
                    if ni >= 1 && ni <= nx && nj >= 1 && nj <= ny && nk >= 1 && nk <= nz
                        neighbor_idx = ni + (nj-1)*nx + (nk-1)*nx*ny
                        if neighbor_idx >= 1 && neighbor_idx <= nElem && protected_mask[neighbor_idx]
                            dist = sqrt(Float32(di^2 + dj^2 + dk^2))
                            min_dist = min(min_dist, dist)
                        end
                    end
                end
                
                alpha = clamp(min_dist / blend_depth, 0.0f0, 1.0f0)
                smooth_alpha = alpha * alpha * (3.0f0 - 2.0f0 * alpha)
                density[e] = (1.0f0 - smooth_alpha) * original_density[e] + smooth_alpha * filtered_density[e]
            end
        end
    end
end

function verify_boundary_filtering_detailed(density::Vector{Float32}, filtered::Vector{Float32}, 
                                            nx::Int, ny::Int, nz::Int)
    
    interior_changed = 0; interior_total = 0
    faces_changed = 0; faces_total = 0
    edges_changed = 0; edges_total = 0
    corners_changed = 0; corners_total = 0
    
    for k in 1:nz, j in 1:ny, i in 1:nx
        e = i + (j-1)*nx + (k-1)*nx*ny
        
        changed = abs(density[e] - filtered[e]) > 1e-6
        
        on_boundary_count = 0
        if i == 1 || i == nx; on_boundary_count += 1; end
        if j == 1 || j == ny; on_boundary_count += 1; end
        if k == 1 || k == nz; on_boundary_count += 1; end
        
        if on_boundary_count == 0
            interior_total += 1
            if changed; interior_changed += 1; end
        elseif on_boundary_count == 1
            faces_total += 1
            if changed; faces_changed += 1; end
        elseif on_boundary_count == 2
            edges_total += 1
            if changed; edges_changed += 1; end
        else  
            corners_total += 1
            if changed; corners_changed += 1; end
        end
    end
    
    println("    [Filter Check] Detailed Boundary Analysis:")
    
    if interior_total > 0
        pct = 100.0 * interior_changed / interior_total
        status = pct > 90.0 ? "✓" : "✗"
        println(@sprintf("      %s Interior:  %6d / %6d (%.1f%%)", 
                         status, interior_changed, interior_total, pct))
    end
    
    if faces_total > 0
        pct = 100.0 * faces_changed / faces_total
        status = pct > 90.0 ? "✓" : "✗"
        println(@sprintf("      %s Faces:       %6d / %6d (%.1f%%)", 
                         status, faces_changed, faces_total, pct))
    end
    
    if edges_total > 0
        pct = 100.0 * edges_changed / edges_total
        status = pct > 80.0 ? "✓" : "✗"
        println(@sprintf("      %s Edges:       %6d / %6d (%.1f%%)", 
                         status, edges_changed, edges_total, pct))
    end
    
    if corners_total > 0
        pct = 100.0 * corners_changed / corners_total
        status = pct > 70.0 ? "✓" : "✗"
        println(@sprintf("      %s Corners:     %6d / %6d (%.1f%%)", 
                         status, corners_changed, corners_total, pct))
    end
    
    total_boundary = faces_total + edges_total + corners_total
    total_boundary_changed = faces_changed + edges_changed + corners_changed
    
    if total_boundary > 0
        overall_pct = 100.0 * total_boundary_changed / total_boundary
        if overall_pct < 80.0
            println("    \u001b[33m[WARNING] <80%% of boundaries filtered properly!\u001b[0m")
        else
            println("    \u001b[32m[SUCCESS] Boundary filtering working correctly.\u001b[0m")
        end
    end
end

function update_density!(density::Vector{Float32}, 
                         l1_stress_norm_field::Vector{Float32}, 
                         protected_elements_mask::BitVector, 
                         E::Float32, 
                         l1_stress_allowable::Float32, 
                         iter::Int, 
                         number_of_iterations::Int, 
                         original_density::Vector{Float32}, 
                         min_density::Float32,  
                         max_density::Float32, 
                         config::Dict, 
                         elements::Matrix{Int},
                         is_annealing::Bool=false;
                         force_no_cull::Bool=false)  # <--- NEW ARGUMENT

    nElem = length(density)
    
    if any(isnan, l1_stress_norm_field)
        println("\n" * "\u001b[31m" * "!!!"^20 * "\u001b[0m")
        println("\u001b[31m" * ">>> [SAFEGUARD] CRITICAL: NaNs detected in stress field (Solver Diverged)." * "\u001b[0m")
        println("\u001b[31m" * ">>> [SAFEGUARD] Skipping topology update to prevent mesh corruption." * "\u001b[0m")
        println("\u001b[31m" * "!!!"^20 * "\n" * "\u001b[0m")
        return 0.0f0, 0.0f0, 0.0f0, 0.0, 0, 0.0
    end

    opt_params = config["optimization_parameters"]
    geom_params = config["geometry"]
    
    nElem_x = Int(geom_params["nElem_x_computed"]) 
    nElem_y = Int(geom_params["nElem_y_computed"])
    nElem_z = Int(geom_params["nElem_z_computed"])
    dx = Float32(geom_params["dx_computed"])
    dy = Float32(geom_params["dy_computed"])
    dz = Float32(geom_params["dz_computed"])
    avg_element_size = (dx + dy + dz) / 3.0f0
    
    proposed_density_field = zeros(Float32, nElem)
    Threads.@threads for e in 1:nElem
        if !protected_elements_mask[e] 
            current_l1_stress = l1_stress_norm_field[e]
            val = (current_l1_stress / l1_stress_allowable) / E
            proposed_density_field[e] = val
        else
            proposed_density_field[e] = original_density[e]
        end
    end

    target_d_phys = Float32(get(opt_params, "minimum_feature_size_physical", 0.0))
    floor_d_elems = Float32(get(opt_params, "minimum_feature_size_elements", 3.0)) 
    
    floor_d_phys = floor_d_elems * avg_element_size
    
    d_min_phys = 0.0f0
    active_constraint = ""
    
    if target_d_phys > floor_d_phys
        d_min_phys = target_d_phys
        active_constraint = "PHYSICAL TARGET"
    else
        d_min_phys = floor_d_phys
        active_constraint = "ELEMENT FLOOR (Stability)"
    end
    
    d_min_elems = d_min_phys / avg_element_size

    t = Float32(iter) / Float32(number_of_iterations)
    t = clamp(t, 0.0f0, 1.0f0)

    gamma = Float32(get(opt_params, "radius_decay_exponent", 1.8))
    r_max_mult = Float32(get(opt_params, "radius_max_multiplier", 4.0))
    r_min_mult = Float32(get(opt_params, "radius_min_multiplier", 0.5))
    C_safe = Float32(get(opt_params, "constraint_constant", 0.25))
    final_threshold_val = Float32(get(opt_params, "final_density_threshold", 0.98))
    
    # NEW PARAMETER
    exponent_cutoff = Float32(get(opt_params, "exponent_for_cutoff_schedule", 1.0))
    
    current_threshold = 0.0f0
    if iter > number_of_iterations
        current_threshold = final_threshold_val
    else
        # Apply exponent to curve the schedule
        t_scheduled = t ^ exponent_cutoff
        current_threshold = final_threshold_val * t_scheduled
    end
    
    effective_threshold = max(current_threshold, 0.001f0)

    decay_factor = 1.0f0 - (t^gamma)
    r_baseline = (r_max_mult * d_min_phys) * decay_factor + (r_min_mult * d_min_phys)
    
    constraint_limit_radius = (C_safe * d_min_phys) / effective_threshold
    R_final = min(r_baseline, constraint_limit_radius)
    
    R_abs_min = 1.0f0 * avg_element_size 
    R_final = max(R_final, R_abs_min)

    product_val = effective_threshold * R_final
    limit_val   = C_safe * d_min_phys
    margin_pct  = ((limit_val - product_val) / limit_val) * 100.0
    
    status_str = "[SAFE]"
    if margin_pct < 0; status_str = "[VIOLATION]";
    elseif margin_pct < 5.0; status_str = "[CRITICAL]";
    elseif margin_pct < 15.0; status_str = "[TIGHT]"; end
    
    is_constrained = (R_final < r_baseline)
    constrained_flag = is_constrained ? "[CONSTRAINED]" : ""
    R_in_elements = R_final / avg_element_size

    println("\n╔═══════════════════════════════════════════════════════════════╗")
    println(@sprintf("║  COUPLED FILTER SCHEDULE (Iter %d) [%s]", iter, active_constraint))
    println("╟───────────────────────────────────────────────────────────────╢")
    println(@sprintf("║  Avg Element Size:         %.4f m", avg_element_size))
    println(@sprintf("║  Requested Physical:       %.4f m", target_d_phys))
    println(@sprintf("║  Stability Floor:          %.4f m (%.1f elems)", floor_d_phys, floor_d_elems))
    println("╟───────────────────────────────────────────────────────────────╢")
    println(@sprintf("║  Effective d_min:          %.4f m (%.1f elems)", d_min_phys, d_min_elems))
    println(@sprintf("║  Radius (Final):           %.4f m (%.1f elems) %s", R_final, R_in_elements, constrained_flag))
    println("╟───────────────────────────────────────────────────────────────╢")
    println(@sprintf("║  Cutoff Exponent:          %.2f", exponent_cutoff))
    println(@sprintf("║  Constraint Margin:        %.1f%%  %s", margin_pct, status_str))
    println("╚═══════════════════════════════════════════════════════════════╝\n")

    filtered_density_field = proposed_density_field
    filter_time = 0.0
    
    if R_final > 1e-4
        t_start = time()
        
        # Call the GPU Explicit Filter (diffusion-based)
        filtered_density_field = GPUExplicitFilter.apply_explicit_filter!(
            proposed_density_field, 
            nElem_x, nElem_y, nElem_z,
            dx, dy, dz, R_final,
            min_density 
        )
        
        filter_time = time() - t_start
        
        if iter % 10 == 1
            verify_boundary_filtering_detailed(proposed_density_field, filtered_density_field, 
                                               nElem_x, nElem_y, nElem_z)
        end
        
        if any(isnan, filtered_density_field) || any(isinf, filtered_density_field)
            println("\u001b[33m" * ">>> [SAFEGUARD] Filter produced NaNs/Infs. Triggering emergency box filter." * "\u001b[0m")
            
            filtered_density_field = apply_emergency_box_filter(
                proposed_density_field, nElem_x, nElem_y, nElem_z
            )
            
            if any(isnan, filtered_density_field)
                println("\u001b[31m" * ">>> [CRITICAL] Emergency filter also failed. Using unfiltered density." * "\u001b[0m")
                filtered_density_field = proposed_density_field
            end
        end
    end
    
    filtered_density_field = clamp.(filtered_density_field, min_density, max_density)
    
    blend_depth = max(3, round(Int, R_final / avg_element_size / 2))
    transition_zone = create_transition_zone(protected_elements_mask, nElem_x, nElem_y, nElem_z, blend_depth)
    
    n_chunks = Threads.nthreads()
    chunk_size = cld(nElem, n_chunks)
    partial_stats = Vector{Tuple{Float32, Int}}(undef, n_chunks)
    
    @sync for (i, chunk_range) in enumerate(Iterators.partition(1:nElem, chunk_size))
        Threads.@spawn begin
            local_change = 0.0f0
            local_active = 0
            for e in chunk_range
                if !protected_elements_mask[e] 
                    old_val = density[e]
                    raw_new_val = filtered_density_field[e]
                    density[e] = raw_new_val
                    if density[e] > min_density
                        local_active += 1
                    end
                    local_change += abs(raw_new_val - old_val)
                else
                    density[e] = original_density[e]
                    if density[e] > min_density
                        local_active += 1
                    end
                end
            end
            partial_stats[i] = (local_change, local_active)
        end
    end
    
    blend_transition_zone!(density, filtered_density_field, protected_elements_mask, 
                           transition_zone, original_density, nElem_x, nElem_y, nElem_z, blend_depth)
    
    total_change = 0.0f0
    total_active = 0
    for i in 1:length(partial_stats)
        if isassigned(partial_stats, i)
            (c, a) = partial_stats[i]
            total_change += c
            total_active += a
        end
    end
    
    mean_change = (total_active > 0) ? (total_change / Float32(total_active)) : 0.0f0
    
    # --- SAFEGUARDED CULLING RATIO ---
    # Default fallback is 0.2 (20%) if config missing
    max_culling_ratio = Float32(get(opt_params, "max_culling_ratio", 0.2)) 
    
    # If we are in annealing mode, restrict culling even further to preserve thin features
    if is_annealing
        max_culling_ratio = min(max_culling_ratio, 0.02f0)
    end
    
    # --- GRACE PERIOD LOGIC ---
    if force_no_cull
        max_culling_ratio = 0.0f0
        println("      [GRACE PERIOD] Culling disabled for structure recovery.")
    end

    cull_candidates = Int[]
    active_count = 0
    
    for e in 1:nElem
        if !protected_elements_mask[e] && !transition_zone[e]
            if density[e] > max_density
                density[e] = max_density
            end
            
            if density[e] > min_density
                active_count += 1
                if density[e] < current_threshold
                    push!(cull_candidates, e)
                end
            end
        else
             if density[e] > min_density
                 active_count += 1
             end
        end
    end
    
    max_allowed_culls = floor(Int, active_count * max_culling_ratio)
    
    if length(cull_candidates) > max_allowed_culls
        sort!(cull_candidates, by = idx -> density[idx])
        for i in 1:max_allowed_culls
            idx = cull_candidates[i]
            density[idx] = min_density
        end
    else
        for idx in cull_candidates
            density[idx] = min_density
        end
    end

    update_method = get(opt_params, "density_update_method", "soft")
    if update_method == "hard"
        Threads.@threads for e in 1:nElem
            if !protected_elements_mask[e] && !transition_zone[e]
                if density[e] > min_density
                    density[e] = 1.0f0
                end
            end
        end
    end

    Threads.@threads for e in 1:nElem
        if protected_elements_mask[e]
            density[e] = original_density[e]
        end
    end
    
    return mean_change, R_final, current_threshold, filter_time, 0, 0.0
end

end