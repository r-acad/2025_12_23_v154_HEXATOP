# // # FILE: .\src\Mesh\MeshShapeProcessing.jl

module MeshShapeProcessing 
 
export apply_geometric_modifiers!
 
using LinearAlgebra 
using Base.Threads
using ..MeshUtilities     
 
"""
    precompute_shape_bboxes(shapes)

Precompute bounding boxes for each shape to enable early rejection.
"""
function precompute_shape_bboxes(shapes::Vector{Any})
    bboxes = Vector{NamedTuple{(:xmin, :xmax, :ymin, :ymax, :zmin, :zmax), NTuple{6, Float32}}}(undef, length(shapes))
    
    for (i, shape) in enumerate(shapes)
        shape_type = lowercase(get(shape, "type", ""))
        
        if shape_type == "sphere"
            center = tuple(Float32.(shape["center"])...)
            radius = Float32(shape["diameter"]) / 2.0f0
            bboxes[i] = (
                xmin = center[1] - radius, xmax = center[1] + radius,
                ymin = center[2] - radius, ymax = center[2] + radius,
                zmin = center[3] - radius, zmax = center[3] + radius
            )
        elseif shape_type == "box"
            center = tuple(Float32.(shape["center"])...)
            
            if haskey(shape, "size")
                sz_raw = shape["size"]
                if isa(sz_raw, AbstractVector) && length(sz_raw) >= 3
                    hx = Float32(sz_raw[1]) / 2.0f0
                    hy = Float32(sz_raw[2]) / 2.0f0
                    hz = Float32(sz_raw[3]) / 2.0f0
                else
                    hx = hy = hz = 0.5f0
                end
            elseif haskey(shape, "side")
                half_side = Float32(shape["side"]) / 2.0f0
                hx = hy = hz = half_side
            else
                hx = hy = hz = 0.5f0
            end
            
            bboxes[i] = (
                xmin = center[1] - hx, xmax = center[1] + hx,
                ymin = center[2] - hy, ymax = center[2] + hy,
                zmin = center[3] - hz, zmax = center[3] + hz
            )
        else
            # Unknown shape type - make infinite bbox
            bboxes[i] = (
                xmin = -Inf32, xmax = Inf32,
                ymin = -Inf32, ymax = Inf32,
                zmin = -Inf32, zmax = Inf32
            )
        end
    end
    
    return bboxes
end

"""
    bbox_contains(bbox, pt)

Fast bounding box check.
"""
@inline function bbox_contains(bbox, pt)
    return (pt[1] >= bbox.xmin && pt[1] <= bbox.xmax &&
            pt[2] >= bbox.ymin && pt[2] <= bbox.ymax &&
            pt[3] >= bbox.zmin && pt[3] <= bbox.zmax)
end

"""
    apply_geometric_modifiers!(density, alpha_field, nodes, elements, shapes, min_density)

Iterates over elements and modifies the `density` and `alpha_field` based on the 
`stiffness_ratio` of the geometric shapes defined in the configuration.

OPTIMIZED VERSION:
- Multi-threaded element processing
- Bounding box early rejection
- Vectorized centroid calculation
- Reduced allocations
"""
function apply_geometric_modifiers!(density::Vector{Float32}, 
                                    alpha_field::Vector{Float32},
                                    nodes::Matrix{Float32}, 
                                    elements::Matrix{Int}, 
                                    shapes::Vector{Any},
                                    min_density::Float32)
    
    if isempty(shapes)
        return
    end

    nElem = size(elements, 1)
    
    println("Processing geometric density and thermal modifiers...")
    t_start = time()
    
    # Precompute bounding boxes for all shapes
    bboxes = precompute_shape_bboxes(shapes)
    
    # Extract shape data into faster access structures
    n_shapes = length(shapes)
    shape_types = Vector{String}(undef, n_shapes)
    shape_ratios = Vector{Float32}(undef, n_shapes)
    shape_centers = Vector{Tuple{Float32, Float32, Float32}}(undef, n_shapes)
    shape_params = Vector{Any}(undef, n_shapes)
    
    for (i, shape) in enumerate(shapes)
        shape_types[i] = lowercase(get(shape, "type", ""))
        shape_ratios[i] = Float32(get(shape, "stiffness_ratio", 0.0))
        
        if haskey(shape, "center")
            c = shape["center"]
            shape_centers[i] = (Float32(c[1]), Float32(c[2]), Float32(c[3]))
        else
            shape_centers[i] = (0.0f0, 0.0f0, 0.0f0)
        end
        
        # Store type-specific parameters
        if shape_types[i] == "sphere"
            shape_params[i] = Float32(shape["diameter"]) / 2.0f0  # radius
        elseif shape_types[i] == "box"
            if haskey(shape, "size")
                sz_raw = shape["size"]
                if isa(sz_raw, AbstractVector) && length(sz_raw) >= 3
                    shape_params[i] = (Float32(sz_raw[1])/2.0f0, Float32(sz_raw[2])/2.0f0, Float32(sz_raw[3])/2.0f0)
                else
                    shape_params[i] = (0.5f0, 0.5f0, 0.5f0)
                end
            elseif haskey(shape, "side")
                half_side = Float32(shape["side"]) / 2.0f0
                shape_params[i] = (half_side, half_side, half_side)
            else
                shape_params[i] = (0.5f0, 0.5f0, 0.5f0)
            end
        else
            shape_params[i] = nothing
        end
    end
    
    # Precompute element centroids (parallelized)
    centroids = Vector{Tuple{Float32, Float32, Float32}}(undef, nElem)
    
    Threads.@threads for e in 1:nElem
        cx = 0.0f0; cy = 0.0f0; cz = 0.0f0
        
        @inbounds for i in 1:8
            node_idx = elements[e, i]
            cx += nodes[node_idx, 1]
            cy += nodes[node_idx, 2]
            cz += nodes[node_idx, 3]
        end
        
        centroids[e] = (cx * 0.125f0, cy * 0.125f0, cz * 0.125f0)
    end
    
    println("  [Optimization] Computed $(nElem) element centroids ($(round(time()-t_start, digits=2))s)")
    
    # Process elements in parallel
    Threads.@threads for e in 1:nElem
        centroid = centroids[e]
        
        # Check each shape
        for i in 1:n_shapes
            # Early rejection via bounding box
            if !bbox_contains(bboxes[i], centroid)
                continue
            end
            
            # Exact containment test
            is_inside = false
            
            if shape_types[i] == "sphere"
                center = shape_centers[i]
                radius = shape_params[i]
                
                dx = centroid[1] - center[1]
                dy = centroid[2] - center[2]
                dz = centroid[3] - center[3]
                dist_sq = dx*dx + dy*dy + dz*dz
                
                is_inside = (dist_sq <= radius*radius)
                
            elseif shape_types[i] == "box"
                center = shape_centers[i]
                half_sizes = shape_params[i]
                
                is_inside = (abs(centroid[1] - center[1]) <= half_sizes[1] &&
                            abs(centroid[2] - center[2]) <= half_sizes[2] &&
                            abs(centroid[3] - center[3]) <= half_sizes[3])
            end
            
            if is_inside
                ratio = shape_ratios[i]
                
                if ratio == 0.0f0
                    # Void
                    density[e] = min_density
                    alpha_field[e] = 0.0f0
                elseif ratio < 0.0f0
                    # Non-designable solid
                    density[e] = abs(ratio)
                    alpha_field[e] = 1.0f0
                else
                    # Stiff region
                    density[e] = ratio
                    alpha_field[e] = 0.0f0
                end
                
                # Stop checking other shapes for this element
                break
            end
        end
    end
    
    total_time = time() - t_start
    println("  [Optimization] Element density and thermal processing complete ($(round(total_time, digits=2))s)")
    println("                 Processed $(nElem) elements with $(n_shapes) shapes")
    println("                 Throughput: $(round(nElem/total_time/1e6, digits=2))M elements/sec")
end 
 
end