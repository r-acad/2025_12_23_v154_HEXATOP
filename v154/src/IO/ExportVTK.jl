# FILE: .\src\IO\ExportVTK.jl

module ExportVTK 

using Printf 
using Base64 

export export_solution_vti, export_solution_legacy

"""
    export_solution_vti(...)

Writes the simulation results to a VTK XML Image Data file (.vti).
Optimized to optionally include Principal Stresses and Directions based on config.
"""
function export_solution_vti(dims::Tuple{Int,Int,Int}, 
                             spacing::Tuple{Float32,Float32,Float32}, 
                             origin::Tuple{Float32,Float32,Float32},
                             density::Vector{Float32}, 
                             l1_stress::Vector{Float32},
                             von_mises::Vector{Float32},
                             principal_vals::Matrix{Float32},
                             principal_max_dirs::Matrix{Float32},
                             principal_min_dirs::Matrix{Float32},
                             config::Dict,
                             filename::String)

    nx, ny, nz = dims
    n_cells = length(density)
    
    if !endswith(filename, ".vti"); filename *= ".vti"; end

    # Check config for vector export
    out_settings = get(config, "output_settings", Dict())
    save_vec_val = get(out_settings, "save_principal_stress_vectors", "no")
    write_vectors = (lowercase(string(save_vec_val)) == "yes" || save_vec_val == true)

    open(filename, "w") do io
        write(io, "<?xml version=\"1.0\"?>\n")
        write(io, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
        
        # WholeExtent="x1 x2 y1 y2 z1 z2" (0-based node indices, so dims are cell counts)
        extent = "0 $nx 0 $ny 0 $nz"
        dx, dy, dz = spacing
        ox, oy, oz = origin
        
        write(io, "  <ImageData WholeExtent=\"$extent\" Origin=\"$ox $oy $oz\" Spacing=\"$dx $dy $dz\">\n")
        write(io, "    <Piece Extent=\"$extent\">\n")
        write(io, "      <CellData Scalars=\"Density\" Vectors=\"MaxPrincipalDirection\">\n")
        
        # --- HEADER SECTION ---
        current_offset = 0
        
        # 1. Density
        write(io, "        <DataArray type=\"Float32\" Name=\"Density\" format=\"appended\" offset=\"$current_offset\"/>\n")
        current_offset += sizeof(UInt32) + n_cells * sizeof(Float32)
        
        # 2. L1 Stress
        write(io, "        <DataArray type=\"Float32\" Name=\"L1_Stress\" format=\"appended\" offset=\"$current_offset\"/>\n")
        current_offset += sizeof(UInt32) + n_cells * sizeof(Float32)
        
        # 3. Von Mises
        write(io, "        <DataArray type=\"Float32\" Name=\"VonMises\" format=\"appended\" offset=\"$current_offset\"/>\n")
        current_offset += sizeof(UInt32) + n_cells * sizeof(Float32)

        if write_vectors
            # 4. Principal Values (3 components per cell)
            write(io, "        <DataArray type=\"Float32\" Name=\"PrincipalValues\" NumberOfComponents=\"3\" format=\"appended\" offset=\"$current_offset\"/>\n")
            current_offset += sizeof(UInt32) + (n_cells * 3) * sizeof(Float32)

            # 5. Max Principal Direction (Vector)
            write(io, "        <DataArray type=\"Float32\" Name=\"MaxPrincipalDirection\" NumberOfComponents=\"3\" format=\"appended\" offset=\"$current_offset\"/>\n")
            current_offset += sizeof(UInt32) + (n_cells * 3) * sizeof(Float32)

            # 6. Min Principal Direction (Vector)
            write(io, "        <DataArray type=\"Float32\" Name=\"MinPrincipalDirection\" NumberOfComponents=\"3\" format=\"appended\" offset=\"$current_offset\"/>\n")
            current_offset += sizeof(UInt32) + (n_cells * 3) * sizeof(Float32)
        end

        write(io, "      </CellData>\n")
        write(io, "    </Piece>\n")
        write(io, "  </ImageData>\n")
        
        # --- BINARY DATA SECTION ---
        write(io, "  <AppendedData encoding=\"raw\">\n")
        write(io, "_") 
        
        # Helper to write arrays
        function write_array(arr)
            n_bytes = UInt32(length(arr) * sizeof(Float32))
            write(io, n_bytes)
            write(io, arr)
        end

        write_array(density)
        write_array(l1_stress)
        write_array(von_mises)

        if write_vectors
            # principal_vals is 3xN, we need linear buffer
            write_array(vec(principal_vals)) 
            # principal_max_dirs is 3xN, we need linear buffer
            write_array(vec(principal_max_dirs))
            # principal_min_dirs is 3xN, we need linear buffer
            write_array(vec(principal_min_dirs))
        end
        
        write(io, "\n  </AppendedData>\n")
        write(io, "</VTKFile>\n")
    end
end

function export_solution_legacy(nodes::Matrix{Float32}, 
                                elements::Matrix{Int}, 
                                U_full::Vector{Float32}, 
                                F::Vector{Float32}, 
                                bc_indicator::Matrix{Float32}, 
                                principal_field::Matrix{Float32}, 
                                vonmises_field::Vector{Float32}, 
                                full_stress_voigt::Matrix{Float32}, 
                                l1_stress_norm_field::Vector{Float32},
                                principal_max_dir_field::Matrix{Float32}; 
                                density::Vector{Float32}=Float32[],
                                filename::String="solution.vtk",
                                kwargs...)
    # Legacy writer not implemented for this update.
    println("Legacy exporter triggered (not updated for vectors).")
end

function export_solution(nodes, elements, U, F, bc, p_field, vm, voigt, l1, p_max_dir, p_min_dir; 
                         density=nothing, filename="out.vtk", config=nothing, kwargs...)
                          
    # If we have config and it's a structured grid, USE VTI (Fastest)
    if config !== nothing
        geom = config["geometry"]
        nx = Int(geom["nElem_x_computed"])
        ny = Int(geom["nElem_y_computed"])
        nz = Int(geom["nElem_z_computed"])
        dx = Float32(geom["dx_computed"])
        dy = Float32(geom["dy_computed"])
        dz = Float32(geom["dz_computed"])
        
        # Pass the vector fields and config to the VTI writer
        export_solution_vti((nx, ny, nz), (dx, dy, dz), (0f0, 0f0, 0f0), 
                            density, l1, vm, p_field, p_max_dir, p_min_dir, config, filename)
    else
        export_solution_legacy(nodes, elements, U, F, bc, p_field, vm, voigt, l1, p_max_dir; 
                               density=density, filename=filename, kwargs...)
    end
end

end