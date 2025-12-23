# // # FILE: .\src\Mesh\MeshPruner.jl

module MeshPruner

using LinearAlgebra
using SparseArrays

export prune_system, reconstruct_full_solution

function prune_system(nodes::Matrix{Float32}, 
                      elements::Matrix{Int}, 
                      density::Vector{Float32}, 
                      threshold::Float32, 
                      bc_indicator::Matrix{Float32}, 
                      F::Vector{Float32})

    nElem = size(elements, 1)
    nNodes = size(nodes, 1)

    active_mask = density .> threshold
    active_element_indices = findall(active_mask)
    nActiveElem = length(active_element_indices)

    if nActiveElem == 0
        error("MeshPruner: No active elements found (Threshold: $threshold). System is empty.")
    end

    active_nodes_mask = falses(nNodes)
    
    for e in active_element_indices
        for i in 1:8
            node_idx = elements[e, i]
            active_nodes_mask[node_idx] = true
        end
    end

    old_to_new_node_map = zeros(Int, nNodes)
    new_to_old_node_map = Int[]
    
    current_new_id = 1
    for i in 1:nNodes
        if active_nodes_mask[i]
            old_to_new_node_map[i] = current_new_id
            push!(new_to_old_node_map, i)
            current_new_id += 1
        end
    end
    
    nActiveNodes = length(new_to_old_node_map)
    reduced_nodes = nodes[new_to_old_node_map, :]

    reduced_elements = Matrix{Int}(undef, nActiveElem, 8)
    for (i, old_e_idx) in enumerate(active_element_indices)
        for j in 1:8
            old_node = elements[old_e_idx, j]
            new_node = old_to_new_node_map[old_node]
            reduced_elements[i, j] = new_node
        end
    end

    reduced_bc = bc_indicator[new_to_old_node_map, :]
    reduced_ndof = nActiveNodes * 3
    reduced_F = zeros(Float32, reduced_ndof)
    
    for (new_idx, old_idx) in enumerate(new_to_old_node_map)
        base_old = 3 * (old_idx - 1)
        base_new = 3 * (new_idx - 1)
        reduced_F[base_new+1] = F[base_old+1]
        reduced_F[base_new+2] = F[base_old+2]
        reduced_F[base_new+3] = F[base_old+3]
    end

    reduced_density = density[active_element_indices]

    return (
        nodes = reduced_nodes,
        elements = reduced_elements,
        bc_indicator = reduced_bc,
        F = reduced_F,
        density = reduced_density,
        old_to_new_node_map = old_to_new_node_map,
        new_to_old_node_map = new_to_old_node_map,
        active_element_indices = active_element_indices,
        n_original_nodes = nNodes,
        n_original_elems = nElem
    )
end

function reconstruct_full_solution(u_reduced::AbstractVector, 
                                   new_to_old_node_map::Vector{Int}, 
                                   n_full_nodes::Int)
    
    T = eltype(u_reduced)
    ndof_full = n_full_nodes * 3
    u_full = zeros(T, ndof_full)

    for (new_node_idx, old_node_idx) in enumerate(new_to_old_node_map)
        base_new = 3 * (new_node_idx - 1)
        base_old = 3 * (old_node_idx - 1)

        u_full[base_old+1] = u_reduced[base_new+1]
        u_full[base_old+2] = u_reduced[base_new+2]
        u_full[base_old+3] = u_reduced[base_new+3]
    end

    return u_full
end

end