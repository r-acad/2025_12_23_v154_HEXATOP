module Configuration 
 
using YAML  
using JSON
using ..Mesh  
using ..Helpers 
using ..MeshShapeProcessing 
 
export load_configuration, setup_geometry, initialize_density_field, load_checkpoint
 
""" 
    load_configuration(filename::String) 
 
Load and parse a JSON/YAML configuration file. 
""" 
function load_configuration(filename::String) 
    if !isfile(filename) 
        error("Configuration file '$(filename)' not found") 
    end 
      
    return YAML.load_file(filename) 
end 

"""
    load_checkpoint(filename::String)

Reads a .bintop checkpoint file. 
Returns: (config, density, restart_iter, restart_radius, restart_threshold)
"""
function load_checkpoint(filename::String)
    println(">>> [Checkpoint] Reading restart data from: $filename")
    
    if !isfile(filename); error("Checkpoint file not found."); end

    data = open(filename, "r") do io
        
        magic = read(io, UInt32) # 0x48455841 "HEXA"
        version = read(io, UInt32)
        
        if magic != 0x48455841
            error("Invalid file format. Not a HEXA checkpoint.")
        end

        iter = Int(read(io, Int32))
        radius = Float32(read(io, Float32))
        threshold = Float32(read(io, Float32))
        
        count = Int(read(io, UInt32))
        dx = read(io, Float32)
        dy = read(io, Float32)
        dz = read(io, Float32)

        seek(io, position(io) + (count * 3 * 4))

        density = Vector{Float32}(undef, count)
        read!(io, density)

        seek(io, position(io) + (count * 4))

        json_len = Int(read(io, UInt32))
        json_bytes = Vector{UInt8}(undef, json_len)
        read!(io, json_bytes)
        
        config_str = String(json_bytes)
        config = JSON.parse(config_str)

        println("    Restarting at Iteration: $iter")
        println("    Filter Radius: $radius")
        println("    Threshold: $threshold")
        println("    Elements: $count")

        return (config, density, iter, radius, threshold)
    end

    return data
end
 
""" 
    setup_geometry(config) 
 
Process the geometry configuration and return parameters for mesh generation. 
""" 
function setup_geometry(config) 
      
    length_x = config["geometry"]["length_x"] 
    length_y = config["geometry"]["length_y"] 
    length_z = config["geometry"]["length_z"] 
     
    raw_count = config["geometry"]["target_elem_count"]
    target_elem_count = if isa(raw_count, String)
        parse(Int, replace(raw_count, "_" => ""))
    else
        Int(raw_count)
    end
      
    println("Domain dimensions:") 
    println("  X: 0 to $(length_x)") 
    println("  Y: 0 to $(length_y)") 
    println("  Z: 0 to $(length_z)") 
      
    shapes = Any[] 
     
    for (key, shape) in config["geometry"] 
        if key in ["length_x", "length_y", "length_z", "target_elem_count", "shape_notes", 
                   "nElem_x_computed", "nElem_y_computed", "nElem_z_computed", 
                   "dx_computed", "dy_computed", "dz_computed", "max_domain_dim"] 
            continue 
        end 
          
        if haskey(shape, "type") 
            push!(shapes, shape)
        end 
    end 
 
    println("Found $(length(shapes)) geometric modification shapes.") 
      
    nElem_x, nElem_y, nElem_z, dx, dy, dz, actual_elem_count = 
        Helpers.calculate_element_distribution(length_x, length_y, length_z, target_elem_count) 
      
    println("Mesh parameters:") 
    println("  Domain: $(length_x) x $(length_y) x $(length_z) meters") 
    println("  Elements: $(nElem_x) x $(nElem_y) x $(nElem_z) = $(actual_elem_count)") 
    println("  Element sizes: $(dx) x $(dy) x $(dz)") 
      
    max_domain_dim = max(length_x, length_y, length_z) 
 
    return ( 
        nElem_x = nElem_x,  
        nElem_y = nElem_y,  
        nElem_z = nElem_z, 
        dx = dx, 
        dy = dy, 
        dz = dz, 
        shapes = shapes, 
        actual_elem_count = actual_elem_count, 
        max_domain_dim = Float32(max_domain_dim)  
    ) 
end 
 
""" 
    initialize_density_field(nodes, elements, shapes, config)
 
Processes geometric shapes to set the initial density array AND the alpha (thermal expansion) field.
Returns `density`, `original_density`, `protected_elements_mask`, and `alpha_field`.

OPTIMIZED VERSION: Now includes timing and progress reporting.
""" 
function initialize_density_field(nodes::Matrix{Float32}, 
                                  elements::Matrix{Int}, 
                                  shapes::Vector{Any}, 
                                  config::Dict) 
      
    min_density = Float32(get(config["optimization_parameters"], "min_density", 1e-3)) 
 
    nElem = size(elements, 1)
    
    println("\n" * "="^80)
    println(">>> [GEOMETRY] Initializing Density Field")
    println("="^80)
    println("  Total Elements: $(nElem)")
    println("  Number of Shapes: $(length(shapes))")
    println("  Min Density Floor: $(min_density)")
    
    
    t_init = time()
    density = ones(Float32, nElem) 
    alpha_field = zeros(Float32, nElem)
    println("  [Timing] Field allocation: $(round((time()-t_init)*1000, digits=2))ms")

    
    t_geom = time()
    MeshShapeProcessing.apply_geometric_modifiers!(density, alpha_field, nodes, elements, shapes, min_density)
    geom_time = time() - t_geom
    
    println("  [Timing] Geometric processing: $(round(geom_time, digits=2))s")
    println("           Throughput: $(round(nElem/geom_time/1e6, digits=2))M elements/sec")
      
    
    t_stats = time()
    original_density = copy(density) 
    protected_elements_mask = (original_density .!= 1.0f0) 
    num_protected = sum(protected_elements_mask)
    
    
    num_voids = count(d -> d < 0.01f0, density)
    num_stiff = count(d -> d > 1.01f0, density)
    num_passive = count(i -> protected_elements_mask[i] && density[i] >= 0.01f0 && density[i] <= 1.01f0, 1:nElem)
    num_designable = nElem - num_protected
    
    println("  [Timing] Statistics computation: $(round((time()-t_stats)*1000, digits=2))ms")
    println("\n  Element Classification:")
    println("    Designable:        $(num_designable) ($(round(100*num_designable/nElem, digits=2))%)")
    println("    Protected Total:   $(num_protected) ($(round(100*num_protected/nElem, digits=2))%)")
    println("      ├─ Voids:         $(num_voids)")
    println("      ├─ Stiff:         $(num_stiff)")
    println("      └─ Passive:       $(num_passive)")
    
    println("="^80 * "\n")
 
    return density, original_density, protected_elements_mask, alpha_field
end 

"""
    validate_configuration(config)

Validates configuration parameters and warns about potential issues.
"""
function validate_configuration(config::Dict)
    warnings = String[]
    
    
    opt_params = get(config, "optimization_parameters", Dict())
    
    max_culling = Float32(get(opt_params, "max_culling_ratio", 0.15))
    if max_culling > 0.3
        push!(warnings, "max_culling_ratio is high ($(max_culling)). Values >0.3 may cause instability.")
    end
    
    final_threshold = Float32(get(opt_params, "final_density_threshold", 0.85))
    if final_threshold > 0.95
        push!(warnings, "final_density_threshold is high ($(final_threshold)). Values >0.95 may trigger collapse.")
    end
    
    filter_final = Float32(get(opt_params, "filter_R_final_perc", 0.5))
    if filter_final < 0.3
        push!(warnings, "filter_R_final_perc is low ($(filter_final)). Values <0.3 may cause checkerboarding.")
    end
    
    
    solver_params = get(config, "solver_parameters", Dict())
    
    max_iter = Int(get(solver_params, "max_iterations", 40000))
    if max_iter < 1000
        push!(warnings, "max_iterations is low ($max_iter). May not converge for large problems.")
    end
    
    
    geom = get(config, "geometry", Dict())
    
    lx = get(geom, "length_x", 1.0)
    ly = get(geom, "length_y", 1.0)
    lz = get(geom, "length_z", 1.0)
    
    aspect_ratio = max(lx, ly, lz) / min(lx, ly, lz)
    if aspect_ratio > 10.0
        push!(warnings, "Domain aspect ratio is extreme ($(round(aspect_ratio, digits=1))). May cause solver issues.")
    end
    
    
    if !isempty(warnings)
        println("\n" * "⚠"^80)
        println(">>> [CONFIG VALIDATION] Warnings:")
        for (i, w) in enumerate(warnings)
            println("  $i. $w")
        end
        println("⚠"^80 * "\n")
    else
        println(">>> [CONFIG VALIDATION] No issues detected.")
    end
    
    return length(warnings) == 0
end

"""
    print_configuration_summary(config)

Prints a formatted summary of the configuration.
"""
function print_configuration_summary(config::Dict)
    println("\n" * "="^80)
    println(">>> [CONFIGURATION SUMMARY]")
    println("="^80)
    
    
    hw_profile = get(config, "hardware_profile_applied", get(config, "gpu_profile", "Unknown"))
    println("  Hardware Profile: $hw_profile")
    
    
    geom = get(config, "geometry", Dict())
    lx = get(geom, "length_x", 0)
    ly = get(geom, "length_y", 0)
    lz = get(geom, "length_z", 0)
    println("  Domain: $(lx) × $(ly) × $(lz)")
    
    
    n_iter = get(config, "number_of_iterations", 30)
    println("  Iterations: $n_iter")
    
    
    solver = get(config, "solver_parameters", Dict())
    solver_type = get(solver, "solver_type", "unknown")
    precond = get(solver, "preconditioner", "unknown")
    tol = get(solver, "tolerance", 0.0)
    println("  Solver: $solver_type")
    println("  Preconditioner: $precond")
    println("  Tolerance: $tol")
    
    
    opt = get(config, "optimization_parameters", Dict())
    max_cull = get(opt, "max_culling_ratio", 0.0)
    final_thresh = get(opt, "final_density_threshold", 0.0)
    println("  Max Culling Ratio: $max_cull")
    println("  Final Density Threshold: $final_thresh")
    
    
    growth = get(config, "growth_settings", Dict())
    target_active = get(growth, "target_active_elements", 0)
    max_growth = get(growth, "max_growth_rate", 0.0)
    println("  Target Active Elements: $target_active")
    println("  Max Growth Rate: $max_growth")
    
    println("="^80 * "\n")
end

"""
    export_configuration(config, filename)

Exports the current configuration to a YAML file.
Useful for reproducibility and debugging.
"""
function export_configuration(config::Dict, filename::String)
    try
        YAML.write_file(filename, config)
        println(">>> [CONFIG] Exported configuration to: $filename")
        return true
    catch e
        @warn "Failed to export configuration: $e"
        return false
    end
end

"""
    compare_configurations(config1, config2)

Compares two configurations and highlights differences.
"""
function compare_configurations(config1::Dict, config2::Dict)
    println("\n" * "="^80)
    println(">>> [CONFIG COMPARISON]")
    println("="^80)
    
    all_keys = union(keys(config1), keys(config2))
    differences = 0
    
    for key in all_keys
        if !haskey(config1, key)
            println("  ADDED: $key = $(config2[key])")
            differences += 1
        elseif !haskey(config2, key)
            println("  REMOVED: $key = $(config1[key])")
            differences += 1
        elseif config1[key] != config2[key]
            println("  CHANGED: $key")
            println("    From: $(config1[key])")
            println("    To:   $(config2[key])")
            differences += 1
        end
    end
    
    if differences == 0
        println("  No differences found.")
    else
        println("\n  Total differences: $differences")
    end
    
    println("="^80 * "\n")
end

"""
    get_config_value(config, path, default)

Safely retrieves nested configuration values.
Example: get_config_value(config, "solver_parameters.tolerance", 1e-6)
"""
function get_config_value(config::Dict, path::String, default)
    keys_list = split(path, '.')
    current = config
    
    for key in keys_list
        if !haskey(current, key)
            return default
        end
        current = current[key]
    end
    
    return current
end

"""
    set_config_value!(config, path, value)

Safely sets nested configuration values.
Example: set_config_value!(config, "solver_parameters.tolerance", 1e-8)
"""
function set_config_value!(config::Dict, path::String, value)
    keys_list = split(path, '.')
    current = config
    
    for (i, key) in enumerate(keys_list)
        if i == length(keys_list)
            current[key] = value
        else
            if !haskey(current, key) || !isa(current[key], Dict)
                current[key] = Dict()
            end
            current = current[key]
        end
    end
end

"""
    apply_preset!(config, preset_name)

Applies predefined configuration presets.
"""
function apply_preset!(config::Dict, preset_name::String)
    presets = Dict(
        "conservative" => Dict(
            "optimization_parameters" => Dict(
                "max_culling_ratio" => 0.05,
                "final_density_threshold" => 0.75,
                "filter_R_final_perc" => 0.7
            ),
            "solver_parameters" => Dict(
                "tolerance" => 1.0e-7
            )
        ),
        "aggressive" => Dict(
            "optimization_parameters" => Dict(
                "max_culling_ratio" => 0.25,
                "final_density_threshold" => 0.90,
                "filter_R_final_perc" => 0.3
            ),
            "solver_parameters" => Dict(
                "tolerance" => 1.0e-5
            )
        ),
        "balanced" => Dict(
            "optimization_parameters" => Dict(
                "max_culling_ratio" => 0.15,
                "final_density_threshold" => 0.85,
                "filter_R_final_perc" => 0.5
            ),
            "solver_parameters" => Dict(
                "tolerance" => 1.0e-6
            )
        )
    )
    
    if !haskey(presets, preset_name)
        @warn "Unknown preset: $preset_name. Available: $(keys(presets))"
        return false
    end
    
    preset = presets[preset_name]
    
    for (section, values) in preset
        if !haskey(config, section)
            config[section] = Dict()
        end
        
        for (key, value) in values
            config[section][key] = value
        end
    end
    
    println(">>> [CONFIG] Applied preset: '$preset_name'")
    return true
end

end