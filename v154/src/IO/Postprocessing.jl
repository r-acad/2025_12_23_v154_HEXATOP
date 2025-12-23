// # FILE: .\src\IO\Postprocessing.jl
module Postprocessing

using JSON, Printf
using Base.Threads
using CUDA
using LinearAlgebra
using Logging 
using ..Mesh
using ..MeshUtilities 
using ..ExportVTK
using ..Diagnostics 
import MarchingCubes: MC, march

export export_iteration_results, export_smooth_watertight_stl

function suppress_specific_warnings(f::Function, module_to_suppress::Module)
    current_logger = global_logger()
    filtered_logger = EarlyFilteredLogger(current_logger) do args
        if args.level == Logging.Warn && args._module == module_to_suppress
            return false 
        end
        return true 
    end
    with_logger(f, filtered_logger)
end

function safe_parse_int(val, default::Int)
    if val === nothing; return default; end
    if isa(val, Number); return Int(val); end
    if isa(val, String)
        clean_val = replace(val, "_" => "")
        return try parse(Int, clean_val) catch; default end
    end
    return default
end

function get_smooth_nodal_densities(density::Vector{Float32}, elements::Matrix{Int}, nNodes::Int)
    node_sums = zeros(Float32, nNodes)
    node_counts = zeros(Int, nNodes)
    nElem = length(density)
    
    @inbounds for e in 1:nElem
        rho = density[e]
        for i in 1:8
            node_idx = elements[e, i]
            if node_idx > 0 && node_idx <= nNodes
                node_sums[node_idx] += rho
                node_counts[node_idx] += 1
            end
        end
    end
    nodal_density = zeros(Float32, nNodes)
    @inbounds for i in 1:nNodes
        if node_counts[i] > 0
            nodal_density[i] = node_sums[i] / Float32(node_counts[i])
        end
    end
    return nodal_density
end

function smooth_grid!(grid::Array{Float32, 3}, passes::Int)
    if passes <= 0; return; end
    nx, ny, nz = size(grid)
    temp_grid = copy(grid)
    
    for _ in 1:passes
        Threads.@threads for k in 2:(nz-1)
            for j in 2:(ny-1)
                for i in 2:(nx-1)
                    sum_neighbors = grid[i-1,j,k] + grid[i+1,j,k] +
                                    grid[i,j-1,k] + grid[i,j+1,k] +
                                    grid[i,j,k-1] + grid[i,j,k+1]
                    temp_grid[i,j,k] = (grid[i,j,k] * 4.0f0 + sum_neighbors) * 0.1f0
                end
            end
        end
        grid[2:end-1, 2:end-1, 2:end-1] .= temp_grid[2:end-1, 2:end-1, 2:end-1]
    end
end

function trilinear_interpolate(vals, xd::Float32, yd::Float32, zd::Float32)
    c00 = vals[1]*(1f0-xd) + vals[2]*xd
    c01 = vals[4]*(1f0-xd) + vals[3]*xd
    c10 = vals[5]*(1f0-xd) + vals[6]*xd
    c11 = vals[8]*(1f0-xd) + vals[7]*xd
    c0 = c00*(1f0-yd) + c01*yd
    c1 = c10*(1f0-yd) + c11*yd
    return c0*(1f0-zd) + c1*zd
end

function decimate_mesh!(vertices::Vector{Tuple{Float64, Float64, Float64}}, 
                        triangles::AbstractVector, 
                        target_triangle_count::Int)
    current_count = length(triangles)
    
    if current_count > 2_000_000 
        Diagnostics.print_warn("Mesh too large for decimation ($current_count tris). Skipping to preserve performance.")
        return triangles
    end

    if target_triangle_count <= 0 || current_count <= target_triangle_count
        return triangles
    end

    Diagnostics.print_info("Decimating mesh: $current_count -> $target_triangle_count triangles...")
    
    mutable_tris = Vector{Vector{Int}}(undef, current_count)
    for i in 1:current_count
        t = triangles[i]
        mutable_tris[i] = [Int(t[1]), Int(t[2]), Int(t[3])]
    end

    max_passes = 15
    for pass in 1:max_passes
        if length(mutable_tris) <= target_triangle_count; break; end

        edges = Vector{Tuple{Float64, Int, Int}}()
        sizehint!(edges, length(mutable_tris) * 3)

        for t in mutable_tris
            v1, v2, v3 = t[1], t[2], t[3]
            d12 = (vertices[v1][1]-vertices[v2][1])^2 + (vertices[v1][2]-vertices[v2][2])^2 + (vertices[v1][3]-vertices[v2][3])^2
            d23 = (vertices[v2][1]-vertices[v3][1])^2 + (vertices[v2][2]-vertices[v3][2])^2 + (vertices[v2][3]-vertices[v3][3])^2
            d31 = (vertices[v3][1]-vertices[v1][1])^2 + (vertices[v3][2]-vertices[v1][2])^2 + (vertices[v3][3]-vertices[v1][3])^2
            push!(edges, (d12, min(v1,v2), max(v1,v2)))
            push!(edges, (d23, min(v2,v3), max(v2,v3)))
            push!(edges, (d31, min(v3,v1), max(v3,v1)))
        end

        sort!(edges, by = x -> x[1])
        
        replacements = collect(1:length(vertices))
        collapsed_nodes = falses(length(vertices))
        n_collapsed = 0
        
        tris_to_remove = length(mutable_tris) - target_triangle_count
        limit_collapses = max(100, tris_to_remove) 

        for (dist, u, v) in edges
            if n_collapsed >= limit_collapses; break; end
            if !collapsed_nodes[u] && !collapsed_nodes[v]
                replacements[v] = u
                mx = (vertices[u][1] + vertices[v][1]) * 0.5
                my = (vertices[u][2] + vertices[v][2]) * 0.5
                mz = (vertices[u][3] + vertices[v][3]) * 0.5
                vertices[u] = (mx, my, mz)
                collapsed_nodes[u] = true 
                collapsed_nodes[v] = true
                n_collapsed += 1
            end
        end

        if n_collapsed == 0; break; end

        new_triangles = Vector{Vector{Int}}()
        sizehint!(new_triangles, length(mutable_tris))

        for t in mutable_tris
            v1 = replacements[t[1]]
            v2 = replacements[t[2]]
            v3 = replacements[t[3]]
            if v1 != v2 && v1 != v3 && v2 != v3
                push!(new_triangles, [v1, v2, v3])
            end
        end
        mutable_tris = new_triangles
    end
    return mutable_tris
end

function laplacian_smooth_mesh!(vertices::Vector{Tuple{Float64, Float64, Float64}}, 
                                triangles::AbstractVector, 
                                iterations::Int=3, lambda::Float64=0.5)
    if iterations <= 0; return; end
    nv = length(vertices)
    new_pos = Vector{Tuple{Float64, Float64, Float64}}(undef, nv)
    neighbor_counts = zeros(Int, nv)
    neighbor_sums_x = zeros(Float64, nv)
    neighbor_sums_y = zeros(Float64, nv)
    neighbor_sums_z = zeros(Float64, nv)

    for _ in 1:iterations
        fill!(neighbor_counts, 0)
        fill!(neighbor_sums_x, 0.0); fill!(neighbor_sums_y, 0.0); fill!(neighbor_sums_z, 0.0)

        for tri in triangles
            i1, i2, i3 = tri[1], tri[2], tri[3]
            if i1 < 1 || i1 > nv || i2 < 1 || i2 > nv || i3 < 1 || i3 > nv; continue; end
            v1 = vertices[i1]; v2 = vertices[i2]; v3 = vertices[i3]
            neighbor_sums_x[i1] += v2[1]; neighbor_sums_y[i1] += v2[2]; neighbor_sums_z[i1] += v2[3]; neighbor_counts[i1] += 1
            neighbor_sums_x[i1] += v3[1]; neighbor_sums_y[i1] += v3[2]; neighbor_sums_z[i1] += v3[3]; neighbor_counts[i1] += 1
            neighbor_sums_x[i2] += v1[1]; neighbor_sums_y[i2] += v1[2]; neighbor_sums_z[i2] += v1[3]; neighbor_counts[i2] += 1
            neighbor_sums_x[i2] += v3[1]; neighbor_sums_y[i2] += v3[2]; neighbor_sums_z[i2] += v3[3]; neighbor_counts[i2] += 1
            neighbor_sums_x[i3] += v1[1]; neighbor_sums_y[i3] += v1[2]; neighbor_sums_z[i3] += v1[3]; neighbor_counts[i3] += 1
            neighbor_sums_x[i3] += v2[1]; neighbor_sums_y[i3] += v2[2]; neighbor_sums_z[i3] += v2[3]; neighbor_counts[i3] += 1
        end
        
        Threads.@threads for i in 1:nv
            cnt = neighbor_counts[i]
            if cnt > 0
                old_x, old_y, old_z = vertices[i]
                avg_x, avg_y, avg_z = neighbor_sums_x[i]/cnt, neighbor_sums_y[i]/cnt, neighbor_sums_z[i]/cnt
                nx = old_x + lambda * (avg_x - old_x)
                ny = old_y + lambda * (avg_y - old_y)
                nz = old_z + lambda * (avg_z - old_z)
                new_pos[i] = (nx, ny, nz)
            else
                new_pos[i] = vertices[i]
            end
        end
        copyto!(vertices, new_pos)
    end
end

"""
    write_stl_chunked(filename, triangles, vertices)

OPTIMIZED: Uses batched multi-threading to eliminate false sharing and 
chunks output to minimize RAM spikes.
"""
function write_stl_chunked(filename::String, 
                           triangles::AbstractVector, 
                           vertices::Vector{Tuple{Float64, Float64, Float64}})
    
    n_tri = length(triangles)
    
    open(filename, "w") do io
        
        header_str = "HEXA TopOpt Optimized Binary STL"
        header = zeros(UInt8, 80)
        copyto!(header, 1, codeunits(header_str), 1, min(length(header_str), 80))
        write(io, header)
        write(io, UInt32(n_tri))

        
        
        CHUNK_SIZE = 1_000_000 
        buffer = Vector{UInt8}(undef, CHUNK_SIZE * 50)
        
        n_chunks = cld(n_tri, CHUNK_SIZE)
        
        for c in 1:n_chunks
            start_idx = (c - 1) * CHUNK_SIZE + 1
            end_idx = min(c * CHUNK_SIZE, n_tri)
            n_in_chunk = end_idx - start_idx + 1
            
            
            
            
            
            
            n_threads = Threads.nthreads()
            batch_size = cld(n_in_chunk, n_threads)
            
            Threads.@threads for t in 1:n_threads
                t_start = start_idx + (t - 1) * batch_size
                t_end = min(start_idx + t * batch_size - 1, end_idx)
                
                if t_start <= t_end
                    
                    for i in t_start:t_end
                        
                        local_i = i - start_idx
                        offset = local_i * 50
                        
                        tri = triangles[i]
                        
                        
                        if tri[1] < 1 || tri[1] > length(vertices) ||
                           tri[2] < 1 || tri[2] > length(vertices) ||
                           tri[3] < 1 || tri[3] > length(vertices)
                            continue
                        end

                        v1 = vertices[tri[1]]
                        v2 = vertices[tri[2]]
                        v3 = vertices[tri[3]]
                        
                        
                        e1x, e1y, e1z = v2[1]-v1[1], v2[2]-v1[2], v2[3]-v1[3]
                        e2x, e2y, e2z = v3[1]-v1[1], v3[2]-v1[2], v3[3]-v1[3]
                        nx, ny, nz = e1y*e2z - e1z*e2y, e1z*e2x - e1x*e2z, e1x*e2y - e1y*e2x
                        mag = sqrt(nx*nx + ny*ny + nz*nz)
                        if mag > 1e-12; nx/=mag; ny/=mag; nz/=mag; else; nx=0.0; ny=0.0; nz=0.0; end
                        
                        
                        ptr = pointer(buffer, offset + 1)
                        p_f32 = reinterpret(Ptr{Float32}, ptr)
                        
                        unsafe_store!(p_f32, Float32(nx), 1)
                        unsafe_store!(p_f32, Float32(ny), 2)
                        unsafe_store!(p_f32, Float32(nz), 3)
                        
                        unsafe_store!(p_f32, Float32(v1[1]), 4)
                        unsafe_store!(p_f32, Float32(v1[2]), 5)
                        unsafe_store!(p_f32, Float32(v1[3]), 6)
                        
                        unsafe_store!(p_f32, Float32(v2[1]), 7)
                        unsafe_store!(p_f32, Float32(v2[2]), 8)
                        unsafe_store!(p_f32, Float32(v2[3]), 9)
                        
                        unsafe_store!(p_f32, Float32(v3[1]), 10)
                        unsafe_store!(p_f32, Float32(v3[2]), 11)
                        unsafe_store!(p_f32, Float32(v3[3]), 12)
                        
                        p_u16 = reinterpret(Ptr{UInt16}, ptr + 48)
                        unsafe_store!(p_u16, UInt16(0), 1)
                    end
                end
            end
            
            
            
            bytes_to_write = n_in_chunk * 50
            write(io, view(buffer, 1:bytes_to_write))
        end
    end
end

function export_smooth_watertight_stl(density::Vector{Float32}, geom, threshold::Float32, filename::String; 
                                      subdivision_level::Int=2, smoothing_passes::Int=2, 
                                      mesh_smoothing_iters::Int=3, target_triangle_count::Int=0) 
    
    min_d, max_d = extrema(density)
    if max_d < threshold
        Diagnostics.print_info("Skipping STL: Max density ($max_d) < threshold ($threshold). No surface exists.")
        return
    end

    try
        dir_path = dirname(filename)
        if !isempty(dir_path) && !isdir(dir_path); mkpath(dir_path); end

        NX, NY, NZ = geom.nElem_x, geom.nElem_y, geom.nElem_z
        dx, dy, dz = geom.dx, geom.dy, geom.dz
        
        
        
        actual_subdivision = subdivision_level
        if length(density) > 5_000_000
             actual_subdivision = 1
        end

        nodes_coarse, elements_coarse, _ = Mesh.generate_mesh(NX, NY, NZ; dx=dx, dy=dy, dz=dz)
        nNodes_coarse = size(nodes_coarse, 1)
        if length(density) != size(elements_coarse, 1); return; end
        
        nodal_density_coarse = get_smooth_nodal_densities(density, elements_coarse, nNodes_coarse)
        grid_coarse = reshape(nodal_density_coarse, (NX+1, NY+1, NZ+1))
        smooth_grid!(grid_coarse, smoothing_passes)

        sub_NX, sub_NY, sub_NZ = NX * actual_subdivision, NY * actual_subdivision, NZ * actual_subdivision
        pad = 1 
        fine_dim_x, fine_dim_y, fine_dim_z = sub_NX+1+2*pad, sub_NY+1+2*pad, sub_NZ+1+2*pad
        sub_dx, sub_dy, sub_dz = dx/Float32(actual_subdivision), dy/Float32(actual_subdivision), dz/Float32(actual_subdivision)

        fine_grid = zeros(Float32, fine_dim_x, fine_dim_y, fine_dim_z)
        x_coords = collect(Float32, range(-pad*sub_dx, step=sub_dx, length=fine_dim_x))
        y_coords = collect(Float32, range(-pad*sub_dy, step=sub_dy, length=fine_dim_y))
        z_coords = collect(Float32, range(-pad*sub_dz, step=sub_dz, length=fine_dim_z))

        Threads.@threads for k_f in (1+pad):(fine_dim_z-pad)
            for j_f in (1+pad):(fine_dim_y-pad)
                for i_f in (1+pad):(fine_dim_x-pad)
                    ix, iy, iz = i_f-(1+pad), j_f-(1+pad), k_f-(1+pad)
                    idx_x, idx_y, idx_z = div(ix, actual_subdivision), div(iy, actual_subdivision), div(iz, actual_subdivision)
                    if idx_x >= NX; idx_x = NX - 1; end
                    if idx_y >= NY; idx_y = NY - 1; end
                    if idx_z >= NZ; idx_z = NZ - 1; end
                    c_i, c_j, c_k = idx_x + 1, idx_y + 1, idx_z + 1
                    rem_x, rem_y, rem_z = ix - idx_x*actual_subdivision, iy - idx_y*actual_subdivision, iz - idx_z*actual_subdivision
                    xd, yd, zd = Float32(rem_x)/actual_subdivision, Float32(rem_y)/actual_subdivision, Float32(rem_z)/actual_subdivision
                    vals = (grid_coarse[c_i,c_j,c_k], grid_coarse[c_i+1,c_j,c_k], grid_coarse[c_i+1,c_j+1,c_k], grid_coarse[c_i,c_j+1,c_k],
                            grid_coarse[c_i,c_j,c_k+1], grid_coarse[c_i+1,c_j,c_k+1], grid_coarse[c_i+1,c_j+1,c_k+1], grid_coarse[c_i,c_j+1,c_k+1])
                    fine_grid[i_f, j_f, k_f] = trilinear_interpolate(vals, xd, yd, zd)
                end
            end
        end

        mc_struct = MC(fine_grid, Int; normal_sign=1, x=x_coords, y=y_coords, z=z_coords)
        march(mc_struct, threshold)
        
        if length(mc_struct.vertices) == 0
            Diagnostics.print_warn("STL generation produced 0 vertices (Empty).")
            return
        end

        final_triangles = collect(mc_struct.triangles) 
        final_vertices = [(Float64(v[1]), Float64(v[2]), Float64(v[3])) for v in mc_struct.vertices]

        if mesh_smoothing_iters > 0
            try
                verts_tuple = copy(final_vertices)
                laplacian_smooth_mesh!(verts_tuple, final_triangles, mesh_smoothing_iters, 0.5)
                final_vertices = verts_tuple
            catch e
                Diagnostics.print_warn("Mesh smoothing failed: $e")
            end
        end

        if target_triangle_count > 0 && length(final_triangles) > target_triangle_count
             try
                 final_triangles = decimate_mesh!(final_vertices, final_triangles, target_triangle_count)
             catch e
                 Diagnostics.print_error("Mesh decimation failed ($e). Exporting un-decimated mesh.")
             end
        end

        
        write_stl_chunked(filename, final_triangles, final_vertices)

    catch e
        Diagnostics.print_error("STL Export crashed: $e")
        Diagnostics.write_crash_log("crash_log.txt", "STL_EXPORT", e, stacktrace(catch_backtrace()), 0, Dict(), Float32[])
    end
end

function export_binary_for_web(filename::String, 
                               nodes::Matrix{Float32}, 
                               elements::Matrix{Int}, 
                               density::Vector{Float32}, 
                               l1_stress::Vector{Float32}, 
                               principal_field::Matrix{Float32}, 
                               geom, 
                               threshold::Float32, 
                               iter::Int, 
                               current_radius::Float32, 
                               config::Dict; 
                               max_export_cells::Int=0) 
    
    all_active_indices = findall(x -> x >= threshold, density)
    n_active = length(all_active_indices)
    if n_active == 0; return; end

    if max_export_cells > 0 && n_active > max_export_cells
        step_val = n_active / max_export_cells
        indices_to_export = Int[]
        sizehint!(indices_to_export, max_export_cells)
        curr_float_idx = 1.0
        while curr_float_idx <= n_active
            idx_int = floor(Int, curr_float_idx)
            if idx_int <= n_active
                push!(indices_to_export, all_active_indices[idx_int])
            end
            curr_float_idx += step_val
        end
        valid_indices = indices_to_export
    else
        valid_indices = all_active_indices
    end

    count = length(valid_indices)

    meta = deepcopy(config)
    meta["iteration"] = iter
    meta["radius"] = current_radius
    meta["threshold"] = threshold
    if haskey(meta, "geometry") && isa(meta["geometry"], Dict)
        for (key, shape) in meta["geometry"]
            if isa(shape, Dict) && haskey(shape, "type")
                if !haskey(shape, "action") && haskey(shape, "stiffness_ratio")
                    ratio = Float32(shape["stiffness_ratio"])
                    shape["action"] = ratio > 0 ? "add" : "remove"
                end
            end
        end
    end
    meta["loads"] = get(config, "external_forces", [])
    meta["bcs"] = get(config, "boundary_conditions", [])
    meta["settings"] = get(config, "optimization_parameters", Dict())
    
    json_str = JSON.json(meta)
    json_bytes = Vector{UInt8}(json_str)
    json_len = UInt32(length(json_bytes))

    try
        open(filename, "w") do io
            write(io, 0x48455841) # Magic "HEXA"
            write(io, UInt32(2))  
            write(io, Int32(iter))
            write(io, Float32(current_radius))
            write(io, Float32(threshold))
            write(io, UInt32(count))
            write(io, Float32(geom.dx))
            write(io, Float32(geom.dy))
            write(io, Float32(geom.dz))

            centroids = zeros(Float32, count * 3)
            densities = zeros(Float32, count)
            signed_l1 = zeros(Float32, count)

            Threads.@threads for i in 1:count
                idx = valid_indices[i]
                c = MeshUtilities.element_centroid(idx, nodes, elements)
                centroids[3*(i-1)+1] = c[1]
                centroids[3*(i-1)+2] = c[2]
                centroids[3*(i-1)+3] = c[3]

                densities[i] = density[idx]

                s1 = principal_field[1, idx]
                s2 = principal_field[2, idx]
                s3 = principal_field[3, idx]
                abs_max = abs(s1); sign_val = sign(s1)
                if abs(s2) > abs_max; abs_max = abs(s2); sign_val = sign(s2); end
                if abs(s3) > abs_max; abs_max = abs(s3); sign_val = sign(s3); end
                signed_l1[i] = l1_stress[idx] * sign_val
            end

            write(io, centroids)
            write(io, densities)
            write(io, signed_l1)
            write(io, json_len)
            write(io, json_bytes)
        end
    catch e
        Diagnostics.print_error("[Binary Export] Failed to write file: $e")
    end
end

function export_iteration_results(iter::Int, base_name::String, RESULTS_DIR::String, 
                                  nodes::Matrix{Float32}, elements::Matrix{Int}, 
                                  U_full::AbstractVector, F::AbstractVector, 
                                  bc_indicator::Matrix{Float32}, principal_field::Matrix{Float32}, 
                                  vonmises_field::Vector{Float32}, full_stress_voigt::Matrix{Float32}, 
                                  l1_stress_norm_field::Vector{Float32}, principal_max_dir_field::Matrix{Float32}, principal_min_dir_field::Matrix{Float32}, 
                                  density::Vector{Float32}, E::Float32, geom; 
                                  iso_threshold::Float32=0.8f0, 
                                  current_radius::Float32=0.0f0, 
                                  config::Dict=Dict(), 
                                  save_bin::Bool=true, 
                                  save_stl::Bool=true, 
                                  save_vtk::Bool=true)
      
    U_f32 = (eltype(U_full) == Float32) ? U_full : Float32.(U_full)
    F_f32 = (eltype(F) == Float32) ? F : Float32.(F)

    iter_prefix = "iter_$(iter)_"

    out_settings = get(config, "output_settings", Dict())
    
    raw_max_cells = get(out_settings, "maximum_cells_in_binary_output", 25_000_000)
    max_export_cells = safe_parse_int(raw_max_cells, 25_000_000)

    raw_stl_target = get(out_settings, "stl_target_triangle_count", 0)
    target_triangles = safe_parse_int(raw_stl_target, 0)

    if save_bin
        try
            print("      > Writing Checkpoint/Web Binary...")
            t_web = time()
            # CHANGED: Extension from .bin to .bintop
            bin_filename = joinpath(RESULTS_DIR, "$(iter_prefix)$(base_name)_webdata.bintop")
            export_binary_for_web(
                bin_filename, nodes, elements, density, l1_stress_norm_field, 
                principal_field, geom, iso_threshold, iter, current_radius, 
                config; max_export_cells=max_export_cells
            )
            @printf(" done (%.3fs)\n", time() - t_web)
        catch e
            Diagnostics.print_warn("Web binary export failed. Continuing.")
            Diagnostics.write_crash_log("crash_log.txt", "WEB_EXPORT", e, stacktrace(catch_backtrace()), iter, config, density)
        end
    end

    if save_vtk
        try
            print("      > Writing VTK (Paraview)...")
            t_vtk = time()
            solution_filename = joinpath(RESULTS_DIR, "$(iter_prefix)$(base_name)_solution") 
            
            
            ExportVTK.export_solution(nodes, elements, U_f32, F_f32, bc_indicator, 
                                    principal_field, vonmises_field, full_stress_voigt, 
                                    l1_stress_norm_field, principal_max_dir_field, principal_min_dir_field; 
                                    density=density, threshold=iso_threshold, scale=Float32(1.0), 
                                    filename=solution_filename,
                                    config=config, 
                                    max_cells=0)

            @printf(" done (%.3fs)\n", time() - t_vtk)
        catch e
            Diagnostics.print_warn("VTK export failed. Continuing.")
            Diagnostics.write_crash_log("crash_log.txt", "VTK_EXPORT", e, stacktrace(catch_backtrace()), iter, config, density)
        end
    end

    if save_stl && iter > 0 
        print("      > Writing Isosurface STL...")
        t_stl = time()
        stl_filename = joinpath(RESULTS_DIR, "$(iter_prefix)$(base_name)_isosurface.stl")
        
        subdiv = get(out_settings, "stl_subdivision_level", 2)
        smooth = get(out_settings, "stl_smoothing_passes", 2)
        mesh_smooth = get(out_settings, "stl_mesh_smoothing_iters", 3)
        
        export_smooth_watertight_stl(density, geom, iso_threshold, stl_filename; 
                                     subdivision_level=subdiv, 
                                     smoothing_passes=smooth,
                                     mesh_smoothing_iters=mesh_smooth,
                                     target_triangle_count=target_triangles)
        @printf(" done (%.3fs)\n", time() - t_stl)
    end
end 

end