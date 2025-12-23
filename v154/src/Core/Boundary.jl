// # FILE: .\src\Core\Boundary.jl
module Boundary 

using JSON 
using SparseArrays 
using LinearAlgebra 
using Base.Threads
using ..Element

export get_bc_indicator, reduce_system, apply_external_forces!, add_self_weight!, compute_global_thermal_forces!

""" 
    get_affected_nodes(spec, nodes) 

Returns an array of *final* node indices affected by this BC specification `spec`. 
""" 
function get_affected_nodes(spec::AbstractDict, nodes::Matrix{Float32}) 
      
    nNodes = size(nodes, 1) 

    # 1) If user gave "node" 
    if haskey(spec, "node") 
        raw = spec["node"] 
        if isa(raw, Integer) 
            idx = clamp(raw, 1, nNodes) 
            return [idx] 
        elseif isa(raw, AbstractVector) 
            node_list = Int[] 
            for r in raw 
                push!(node_list, clamp(r, 1, nNodes)) 
            end 
            return unique(node_list) 
        else 
            error("'node' must be an integer or an array of integers") 
        end 
    end 

    # 2) If user gave "location" 
    if haskey(spec, "location") 
        loc_array = spec["location"] 
        if length(loc_array) < 3 
            error("Location specification must have at least 3 components (x,y,z)") 
        end 
        return get_nodes_by_location(loc_array, nodes) 
    end 

    error("Specification must include either 'node' or 'location'") 
end 

""" 
    get_nodes_by_location(loc_array, nodes) 

Find nodes whose (x,y,z) coordinates match the "location" pattern. 
""" 
function get_nodes_by_location(loc_array::AbstractVector, nodes::Matrix{Float32}) 
    xvals = @view nodes[:, 1] 
    yvals = @view nodes[:, 2] 
    zvals = @view nodes[:, 3] 

    xmin, xmax = extrema(xvals) 
    ymin, ymax = extrema(yvals) 
    zmin, zmax = extrema(zvals) 

    xspec = loc_array[1] 
    yspec = loc_array[2] 
    zspec = loc_array[3] 

    xmask = interpret_location_component(xspec, xvals, xmin, xmax) 
    ymask = interpret_location_component(yspec, yvals, ymin, ymax) 
    zmask = interpret_location_component(zspec, zvals, zmin, zmax) 

    return findall(xmask .& ymask .& zmask) 
end 

""" 
    interpret_location_component(spec, coords, cmin, cmax) 
""" 
function interpret_location_component(spec, 
                                      coords::AbstractVector{Float32}, 
                                      cmin::Float32, cmax::Float32) 
    nNodes = length(coords) 
    mask = falses(nNodes) 
      
    tol = Float32(1e-4) * max(Float32(1.0), abs(cmax - cmin)) 

    if spec == ":" 
        return trues(nNodes) 
          
    elseif isa(spec, Number) 
        val = resolve_coordinate_value(spec, cmin, cmax) 
          
        @inbounds for i in 1:nNodes 
            if abs(coords[i] - val) <= tol 
                mask[i] = true 
            end 
        end 

    elseif isa(spec, String) && endswith(spec, "%") 
        frac = parse(Float32, replace(spec, "%"=>"")) / Float32(100.0) 
        frac = clamp(frac, Float32(0.0), Float32(1.0)) 
        val = Float32(cmin + frac*(cmax - cmin)) 
          
        @inbounds for i in 1:nNodes 
            if abs(coords[i] - val) <= tol 
                mask[i] = true 
            end 
        end 
          
    else 
        error("Invalid location component: $spec") 
    end 

    return mask 
end 

function resolve_coordinate_value(spec, cmin::Float32, cmax::Float32)
    if isa(spec, Number)
        if spec >= Float32(0.0) && spec <= Float32(1.0) 
            return Float32(cmin + spec*(cmax - cmin)) 
        else 
            return Float32(spec) 
        end
    elseif isa(spec, String) && endswith(spec, "%")
        frac = parse(Float32, replace(spec, "%"=>"")) / Float32(100.0) 
        frac = clamp(frac, Float32(0.0), Float32(1.0)) 
        return Float32(cmin + frac*(cmax - cmin))
    end
    return Float32(cmin) 
end

""" 
    get_bc_indicator(nNodes, nodes, bc_data; T=Float32) 
""" 
function get_bc_indicator(nNodes::Int, 
                          nodes::Matrix{Float32}, 
                          bc_data::Vector{Any};  
                          T::Type{<:AbstractFloat} = Float32) 

    bc_indicator = zeros(T, nNodes, 3) 
      
    for bc in bc_data 
        dofs = bc["DoFs"] 
          
        for dof in dofs 
            if dof < 1 || dof > 3 
                error("Invalid DoF index: $dof (must be 1..3).") 
            end 
        end 

        affected = get_affected_nodes(bc, nodes) 
        for nd in affected 
            for d in dofs 
                bc_indicator[nd, d] = one(T) 
            end 
        end 
    end 

    return bc_indicator 
end 

""" 
    reduce_system(K, F, bc_data, nodes, elements) 
""" 
function reduce_system(K::SparseMatrixCSC{Float32,Int}, 
                       F::Vector{Float32}, 
                       bc_data::Vector{Any},  
                       nodes::Matrix{Float32}, 
                       elements::Matrix{Int}) 

    nNodes = size(nodes, 1) 
    ndof    = 3*nNodes 
    constrained = falses(ndof) 

    for bc in bc_data 
        dofs = bc["DoFs"] 
        affected = get_affected_nodes(bc, nodes) 
          
        for nd in affected 
            for d in dofs 
                gdof = 3*(nd-1) + d 
                constrained[gdof] = true 
                F[gdof] = Float32(0.0)  
            end 
        end 
    end 

    free_indices = findall(!, constrained) 
    K_reduced = K[free_indices, free_indices] 
    F_reduced = F[free_indices] 
      
    return K_reduced, F_reduced, free_indices 
end 

function find_nearest_node(target_coords::Vector{Float32}, nodes::Matrix{Float32})
    nNodes = size(nodes, 1)
    best_idx = -1
    min_dist_sq = Inf32

    @inbounds for i in 1:nNodes
        dx = nodes[i, 1] - target_coords[1]
        dy = nodes[i, 2] - target_coords[2]
        dz = nodes[i, 3] - target_coords[3]
        dist_sq = dx*dx + dy*dy + dz*dz

        if dist_sq < (min_dist_sq - 1e-9)
            min_dist_sq = dist_sq
            best_idx = i
        elseif abs(dist_sq - min_dist_sq) <= 1e-9
            if nodes[i, 1] > nodes[best_idx, 1]
                best_idx = i
            elseif nodes[i, 1] == nodes[best_idx, 1]
                if nodes[i, 2] > nodes[best_idx, 2]
                    best_idx = i
                elseif nodes[i, 2] == nodes[best_idx, 2]
                    if nodes[i, 3] > nodes[best_idx, 3]
                        best_idx = i
                    end
                end
            end
        end
    end
    return best_idx
end

""" 
    apply_external_forces!(F, forces_data, nodes, elements) 

Processes external forces.
""" 
function apply_external_forces!(F::Vector{T}, 
                                 forces_data::Vector{Any},  
                                 nodes::Matrix{Float32}, 
                                 elements::Matrix{Int}) where T<:AbstractFloat 

    x_bounds = extrema(view(nodes, :, 1))
    y_bounds = extrema(view(nodes, :, 2))
    z_bounds = extrema(view(nodes, :, 3))

    println("Processing $(length(forces_data)) external forces...")

    for force in forces_data 
        
        force_name = get(force, "name", "Unnamed Force")
        
        affected_nodes = get_affected_nodes(force, nodes) 

        
        if isempty(affected_nodes) && haskey(force, "location")
            loc = force["location"]
            is_point_spec = all(x -> x != ":", loc)
            
            if is_point_spec
                tx = resolve_coordinate_value(loc[1], x_bounds[1], x_bounds[2])
                ty = resolve_coordinate_value(loc[2], y_bounds[1], y_bounds[2])
                tz = resolve_coordinate_value(loc[3], z_bounds[1], z_bounds[2])
                target = Float32[tx, ty, tz]

                nearest_idx = find_nearest_node(target, nodes)
                if nearest_idx != -1
                    affected_nodes = [nearest_idx]
                    println("   -> Force '$force_name': mapped to nearest node #$nearest_idx")
                end
            end
        else
            println("   -> Force '$force_name': mapped to $(length(affected_nodes)) nodes")
        end
        
        if isempty(affected_nodes) 
            continue 
        end 
          
        f_raw = force["F"] 
        f_arr = zeros(T, 3) 
        len_to_copy = min(length(f_raw), 3) 
        f_arr[1:len_to_copy] = T.(f_raw[1:len_to_copy])  

        # If user gave "location", we spread the total force among the matched nodes 
        scale_factor = haskey(force, "location") ? (one(T) / length(affected_nodes)) : one(T) 

        for nd in affected_nodes 
            for i in 1:3 
                global_dof = 3*(nd-1) + i 
                F[global_dof] += scale_factor * f_arr[i] 
            end 
        end 
    end 

    return F 
end 

"""
    add_self_weight!(F, density, material_density, gravity_scale, elements, dx, dy, dz, g_accel)

Calculates the gravitational body force for each element and distributes it 
to the element nodes. 
    Force_elem = Volume * rho_material * rho_element * g_accel
    Direction = Negative Y
"""
function add_self_weight!(F::Vector{T}, 
                          density::Vector{T}, 
                          material_density::T, 
                          gravity_scale::T,
                          elements::Matrix{Int},
                          dx::T, dy::T, dz::T,
                          g_accel::T) where T<:AbstractFloat

    elem_vol = dx * dy * dz
    
    # Force per element
    base_weight = elem_vol * material_density * g_accel * gravity_scale
    
    # Distributed to 8 nodes
    node_factor = base_weight / T(8.0)
    
    # [FIX] Thread-local storage safety buffer
    # Allocated extra slots to handle potential thread ID drift in VSCode/Interactive environments
    # (e.g. threadid() returning 33 when nthreads() is 32).
    n_threads_safe = Threads.nthreads() + 16 
    ndof = length(F)
    F_local = [zeros(T, ndof) for _ in 1:n_threads_safe]

    nElem = length(density)

    Threads.@threads for e in 1:nElem
        rho = density[e]
        if rho > 1e-6 
            tid = Threads.threadid()
            
            # [FIX] Safety check to prevent BoundsError
            if tid > n_threads_safe
                tid = 1 # Fallback safe index
            end
            
            # Force value for this node (Downwards Y)
            fy_node = -1.0f0 * node_factor * rho
            
            conn = view(elements, e, :)
            for i in 1:8
                node_idx = conn[i]
                
                gdof_y = 3*(node_idx-1) + 2
                @inbounds F_local[tid][gdof_y] += fy_node
            end
        end
    end

    # Reduce thread-local vectors into global F
    for t in 1:n_threads_safe
        F .+= F_local[t]
    end
end

"""
    compute_global_thermal_forces!(F_total, nodes, elements, alpha_field, delta_T, E, nu, density)

Calculates thermal forces for all elements and adds them to the total force vector.
"""
function compute_global_thermal_forces!(F_total::Vector{Float32}, 
                                        nodes::Matrix{Float32}, 
                                        elements::Matrix{Int},
                                        alpha_field::Vector{Float32},
                                        delta_T::Float32, 
                                        E::Float32, 
                                        nu::Float32, 
                                        density::Vector{Float32})

    if abs(delta_T) < 1e-6
        return
    end

    nElem = size(elements, 1)
    ndof = length(F_total)
    
    # [FIX] Thread safety buffer
    n_threads_safe = Threads.nthreads() + 16
    F_local = [zeros(Float32, ndof) for _ in 1:n_threads_safe]

    Threads.@threads for e in 1:nElem
        
        if density[e] > 1e-4 && abs(alpha_field[e]) > 1e-9
            tid = Threads.threadid()
            if tid > n_threads_safe; tid = 1; end
            
            conn = view(elements, e, :)
            el_nodes = nodes[conn, :]
            
            # Note: We scale Young's modulus by density (SIMP) for stiffness, 
            # but for thermal force f = B' D e_th, D scales with E.
            # e_th depends on alpha.
            
            f_elem = Element.compute_element_thermal_force(el_nodes, E * density[e], nu, alpha_field[e], delta_T)
            
            # Assemble into local thread vector
            for i in 1:8
                node_idx = conn[i]
                base_dof = 3 * (node_idx - 1)
                @inbounds F_local[tid][base_dof + 1] += f_elem[3*(i-1) + 1]
                @inbounds F_local[tid][base_dof + 2] += f_elem[3*(i-1) + 2]
                @inbounds F_local[tid][base_dof + 3] += f_elem[3*(i-1) + 3]
            end
        end
    end

    # Reduction
    for t in 1:n_threads_safe
        F_total .+= F_local[t]
    end
end

end