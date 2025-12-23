# FILE: .\src\Main.jl

using Pkg
using LinearAlgebra
using SparseArrays
using Printf
using Base.Threads
using JSON
using Dates
using Statistics
using CUDA
using YAML

println("\n>>> SCRIPT START: Loading Modules...")
flush(stdout)

module HEXA
    using LinearAlgebra
    using SparseArrays
    using Printf
    using Base.Threads
    using JSON
    using Dates
    using Statistics
    using CUDA
    using YAML

    # Determine Project Root relative to this file (src/Main.jl -> ..)
    const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
    
    include("Utils/Diagnostics.jl")
    include("Utils/Helpers.jl")
    using .Diagnostics
    using .Helpers

    include("Core/Element.jl")
    include("Core/Boundary.jl")
    include("Core/Stress.jl")
    using .Element
    using .Boundary
    using .Stress

    include("Mesh/Mesh.jl")
    include("Mesh/MeshUtilities.jl")
    include("Mesh/MeshPruner.jl")
    include("Mesh/MeshRefiner.jl")
    include("Mesh/MeshShapeProcessing.jl")
    using .Mesh
    using .MeshUtilities
    using .MeshPruner
    using .MeshRefiner
    using .MeshShapeProcessing

    include("Solvers/CPUSolver.jl")
    include("Solvers/GPUGeometricMultigrid.jl")
    include("Solvers/GPUSolver.jl")
    include("Solvers/DirectSolver.jl")
    include("Solvers/IterativeSolver.jl")
    include("Solvers/Solver.jl")
    
    using .CPUSolver
    using .GPUGeometricMultigrid
    using .GPUSolver
    using .DirectSolver
    using .IterativeSolver
    using .Solver

    include("IO/Configuration.jl")
    include("IO/ExportVTK.jl")
    include("IO/Postprocessing.jl")
    
    include("Optimization/GPUExplicitFilter.jl")
    include("Optimization/TopOpt.jl")
    
    using .Configuration
    using .ExportVTK
    using .Postprocessing
    using .GPUExplicitFilter
    using .TopologyOptimization

    function __init__()
        Diagnostics.print_success("HEXA Finite Element Solver initialized")
        Helpers.clear_gpu_memory()
        flush(stdout)
    end
    
    function apply_hardware_profile!(config::Dict)
        gpu_type = get(config, "gpu_profile", "RTX")
        if gpu_type == "AUTO" && CUDA.functional()
            dev_name = CUDA.name(CUDA.device())
            if occursin("V100", dev_name); gpu_type = "V100"; end
            if occursin("A100", dev_name) || occursin("H100", dev_name); gpu_type = "H200"; end
            Diagnostics.print_info("Auto-Detected GPU: $dev_name -> Profile: $gpu_type")
        else
            Diagnostics.print_info("Using Configured Profile: $gpu_type")
        end
        growth = get(config, "growth_settings", Dict())
        solver = get(config, "solver_parameters", Dict())
        
        config["force_float64"] = false
        if uppercase(gpu_type) in ["H", "H200", "H100", "A100"]
            Diagnostics.print_substep("High-Performance Data Center GPU (H/A-Series). Precision: Float64.")
            solver["tolerance"] = get(solver, "tolerance", 1.0e-12)
            solver["diagonal_shift_factor"] = 1.0e-10
            solver["solver_type"] = "gpu"
            config["force_float64"] = true
        elseif uppercase(gpu_type) == "V100"
            Diagnostics.print_substep("Legacy Data Center GPU (Tesla V100). Precision: Float64.")
            solver["tolerance"] = get(solver, "tolerance", 1.0e-10)
            solver["diagonal_shift_factor"] = 1.0e-9
            solver["solver_type"] = "gpu"
            config["force_float64"] = true
        else 
            Diagnostics.print_substep("Consumer/Workstation GPU (RTX-Series). Precision: Float32.")
            solver["tolerance"] = get(solver, "tolerance", 1.0e-6)
            solver["solver_type"] = "gpu"
            config["force_float64"] = false
        end
        config["growth_settings"] = growth
        config["solver_parameters"] = solver
        config["hardware_profile_applied"] = gpu_type
    end

    function run_main(input_file=nothing)
        try
            _run_safe(input_file)
        catch e
            if isa(e, InterruptException)
                Diagnostics.print_banner("USER INTERRUPT", color="\u001b[33m")
                println(">>> Simulation stopped by user.")
                println(">>> Check RESULTS directory for the last successful iteration.")
            else
                Diagnostics.print_banner("FATAL ERROR DETECTED", color="\u001b[31m")
                showerror(stderr, e, catch_backtrace())
            end
            flush(stdout)
        finally
            if CUDA.functional()
                Diagnostics.print_info("Finalizing: Cleaning up GPU Memory...")
                Helpers.clear_gpu_memory()
                Diagnostics.print_success("GPU Memory Released.")
                flush(stdout)
            end
        end
    end

    function _run_safe(input_file)
        Diagnostics.print_banner("HEXA TOPOLOGY OPTIMIZER INITIALIZATION")
        Diagnostics.print_info("Clearing GPU Memory from previous runs...")
        
        if CUDA.functional()
            Helpers.clear_gpu_memory()
            CUDA.device!(0) 
            dev = CUDA.device()
            name = CUDA.name(dev)
            total_mem = CUDA.total_memory()
            
            if CUDA.runtime_version() >= v"11.2"
                threshold = min(total_mem * 0.10, 5 * 1024^3) 
                try
                    pool = CUDA.memory_pool(dev)
                    CUDA.attribute!(pool, CUDA.MEMPOOL_ATTR_RELEASE_THRESHOLD, UInt64(threshold))
                    Diagnostics.print_success("CUDA memory pool configured successfully")
                catch e
                    Diagnostics.print_warn("Memory pool configuration failed (non-critical): $e")
                end
            else
                Diagnostics.print_info("Skipping CUDA memory pool config (CUDA $(CUDA.runtime_version()) < 11.2)")
            end
            
            mem_gb = total_mem / 1024^3
            Diagnostics.print_success("GPU Detected: $name ($(round(mem_gb, digits=2)) GB)")
        else
            Diagnostics.print_warn("No CUDA GPU detected. Running in CPU Mode (Slow).")
        end
        GC.gc()
        
        if input_file === nothing
            input_file = joinpath(PROJECT_ROOT, "configs", "default.yaml")
        end
        
        if !isfile(input_file)
            error("Input file not found: $input_file")
        end

        Diagnostics.print_info("Loading configuration from: $input_file")
        current_config = Configuration.load_configuration(input_file)
        apply_hardware_profile!(current_config)
        
        restart_conf = get(current_config, "restart_configuration", Dict())
        enable_restart = get(restart_conf, "enable_restart", false)
        restart_path = get(restart_conf, "file_path", "")
        
        config = Dict{Any,Any}()
        density = Float32[]
        start_iter = 1
        restart_radius = 0.0f0
        restart_threshold = 0.0f0
        is_restart_active = false

        if enable_restart
             if isfile(restart_path)
                Diagnostics.print_banner("RESTART MODE ACTIVE", color="\u001b[35m")
                Diagnostics.print_info("Loading checkpoint: $restart_path")
                saved_config, density, saved_iter, restart_radius, restart_threshold = Configuration.load_checkpoint(restart_path)
                config = merge(saved_config, current_config)
                apply_hardware_profile!(config)
                start_iter = saved_iter + 1
                is_restart_active = true
            else
                Diagnostics.print_warn("Restart requested but file not found: '$restart_path'")
                Diagnostics.print_warn("Falling back to FRESH START.")
                config = current_config
                is_restart_active = false
            end
        else
            Diagnostics.print_info("Starting Fresh Simulation")
            config = current_config
            is_restart_active = false
        end
        
        hard_stop_iter = get(config, "hard_stop_after_iteration", -1)
        if hard_stop_iter > -1
            Diagnostics.print_info("HARD STOP ENABLED: Execution will stop after iteration $hard_stop_iter.")
        end

        out_settings = get(config, "output_settings", Dict())
        default_freq = get(out_settings, "export_frequency", 5)
        save_bin_freq = get(out_settings, "save_bin_frequency", default_freq)
        save_stl_freq = get(out_settings, "save_STL_frequency", default_freq)
        save_vtk_freq = get(out_settings, "save_VTK_frequency", default_freq)

        save_vec_val = get(out_settings, "save_principal_stress_vectors", "no")
        save_vectors_bool = (lowercase(save_vec_val) == "yes" || save_vec_val == true)
        
        RESULTS_DIR = joinpath(PROJECT_ROOT, "RESULTS")
        if !isdir(RESULTS_DIR); mkpath(RESULTS_DIR); end

        raw_log_name = get(out_settings, "log_filename", "simulation_log.txt")
        log_filename = joinpath(PROJECT_ROOT, basename(raw_log_name))
        iso_threshold_val = get(out_settings, "iso_surface_threshold", 0.8)
        iso_threshold = Float32(iso_threshold_val)
        
        if !is_restart_active
            Diagnostics.init_log_file(log_filename, config)
        else
            Diagnostics.log_status("--- RESTARTING SIMULATION (Iter $start_iter) ---")
        end
        
        geom = Configuration.setup_geometry(config)
        nodes, elements, dims = generate_mesh(geom.nElem_x, geom.nElem_y, geom.nElem_z; dx = geom.dx, dy = geom.dy, dz = geom.dz)
        initial_target_count = size(elements, 1)
        
        if is_restart_active && length(density) != initial_target_count
            error("Restart Mismatch: Checkpoint density size ($(length(density))) != Generated Mesh size ($initial_target_count).")
        end
        
        domain_bounds = (min_pt=[0.0f0,0.0f0,0.0f0], len_x=geom.dx*geom.nElem_x, len_y=geom.dy*geom.nElem_y, len_z=geom.dz*geom.nElem_z)
        config["geometry"]["nElem_x_computed"] = geom.nElem_x
        config["geometry"]["nElem_y_computed"] = geom.nElem_y
        config["geometry"]["nElem_z_computed"] = geom.nElem_z
        config["geometry"]["dx_computed"] = geom.dx
        config["geometry"]["dy_computed"] = geom.dy
        config["geometry"]["dz_computed"] = geom.dz
        config["geometry"]["max_domain_dim"] = geom.max_domain_dim
        
        nNodes = size(nodes, 1)
        ndof = nNodes * 3
        bc_data = config["boundary_conditions"]
        forces_data = config["external_forces"]
        
        bc_indicator = Boundary.get_bc_indicator(nNodes, nodes, Vector{Any}(bc_data))
        F_external = zeros(Float32, ndof)
        Boundary.apply_external_forces!(F_external, Vector{Any}(forces_data), nodes, elements)
        Diagnostics.print_success("Boundary Conditions & External Forces Mapped.")

        E = Float32(config["material"]["E"])
        nu = Float32(config["material"]["nu"])
        material_density = Float32(get(config["material"], "material_density", 0.0))
        gravity_accel = Float32(get(config["material"], "gravity_acceleration", 9.81))
        delta_T = Float32(get(config["material"], "delta_temperature", 0.0))
        if abs(delta_T) > 1e-6; Diagnostics.print_info("THERMOELASTICITY ENABLED: Delta T = $delta_T"); end

        original_density = ones(Float32, size(elements, 1)) 
        protected_elements_mask = falses(size(elements, 1)) 
        alpha_field = zeros(Float32, size(elements, 1))

        if !is_restart_active
            density, original_density, protected_elements_mask, alpha_field = Configuration.initialize_density_field(nodes, elements, geom.shapes, config)
        else
            _, original_density, protected_elements_mask, alpha_field = Configuration.initialize_density_field(nodes, elements, geom.shapes, config)
        end
        
        opt_params = config["optimization_parameters"]
        min_density = Float32(get(opt_params, "min_density", 1.0e-3))
        max_density_clamp = Float32(get(opt_params, "density_clamp_max", 1.0))
        base_name = splitext(basename(input_file))[1]
        growth_conf = get(config, "growth_settings", Dict())
        nominal_iterations = get(config, "number_of_iterations", 30)
        annealing_iterations = round(Int, nominal_iterations * 0.30)
        total_iterations = nominal_iterations + annealing_iterations

        Diagnostics.print_banner("OPTIMIZATION SCHEDULE")
        println("    Nominal Phase:    Iterations 1 to $nominal_iterations")
        println("    Annealing Phase: Iterations $(nominal_iterations + 1) to $total_iterations")

        raw_active_target = get(growth_conf, "target_active_elements", initial_target_count)
        final_target_active = isa(raw_active_target, String) ? parse(Int, replace(raw_active_target, "_" => "")) : Int(raw_active_target)
        max_growth_rate = Float64(get(growth_conf, "max_growth_rate", 1.2))
        raw_bg_limit = get(growth_conf, "max_background_elements", 800_000_000)
        hard_elem_limit = isa(raw_bg_limit, String) ? parse(Int, replace(raw_bg_limit, "_" => "")) : Int(raw_bg_limit)
        Diagnostics.print_info("Hard Element Limit: $(Base.format_bytes(hard_elem_limit * 100)) approx ($hard_elem_limit elems)")

        l1_stress_allowable = Float32(get(config, "l1_stress_allowable", 1.0))
        if l1_stress_allowable == 0.0f0; l1_stress_allowable = 1.0f0; end
        
        internal_l1_allowable = l1_stress_allowable 
        U_full = zeros(Float32, ndof)
        density_change_metric = 1.0f0 
        filter_R = is_restart_active ? restart_radius : 0.0f0
        curr_threshold = is_restart_active ? restart_threshold : 0.0f0
        
        iter = start_iter
        keep_running = true
        is_annealing = false
        prev_compliance = 0.0f0
        convergence_streak = 0
        CONVERGENCE_TOL = 0.005 
        CONVERGENCE_DENSITY_TOL = 0.001 
        CONVERGENCE_REQUIRED_STREAK = 5 
        
        Diagnostics.print_banner("STARTING MAIN LOOP")
        Diagnostics.print_info("Log File: $log_filename")
        
        flush(stdout) # Force print before loop

        while keep_running
            iter_start_time = time()
            status_msg = "Nominal"
            current_target_active = final_target_active
            phase_refinement_needed = false
            gravity_scale = 0.0f0
            
            if iter <= nominal_iterations
                status_msg = "Nominal"
                is_annealing = false
                progress = (iter - 1) / Float64(nominal_iterations)
                
                # --- Configurable Refinement Schedule Exponent ---
                growth_exponent = Float64(get(growth_conf, "exponent_for_refinement_schedule", 2.0))
                growth_factor = progress ^ growth_exponent 
                # ------------------------------------------------------

                target_interpolated = initial_target_count + (final_target_active - initial_target_count) * growth_factor
                current_target_active = round(Int, target_interpolated)
                current_active = count(d -> d > 0.01, density)
                nominal_ref_thresh = Float64(get(growth_conf, "nominal_refinement_threshold", 0.8))
                if current_active < (current_target_active * nominal_ref_thresh)
                    phase_refinement_needed = true
                end
                gravity_scale = 0.0f0
            else
                status_msg = "Annealing"
                is_annealing = true 
                current_target_active = final_target_active
                current_active = count(d -> d > 0.01, density)
                if current_active < (final_target_active * 0.95)
                    phase_refinement_needed = true
                end
                gravity_scale = 1.0f0 
            end
            
            if phase_refinement_needed
                 prev_elem_count = size(elements, 1)
                 nodes, elements, density, alpha_field, dims = MeshRefiner.refine_mesh_and_fields(
                    nodes, elements, density, alpha_field, dims, current_target_active, domain_bounds;
                    max_growth_rate = max_growth_rate, hard_element_limit = hard_elem_limit
                )
                GC.gc()
                
                if size(elements, 1) > prev_elem_count
                    status_msg = "Refined"
                    
                    convergence_streak = 0
                    nElem_x_new, nElem_y_new, nElem_z_new = dims[1]-1, dims[2]-1, dims[3]-1
                    current_dx = domain_bounds.len_x / nElem_x_new
                    current_dy = domain_bounds.len_y / nElem_y_new
                    current_dz = domain_bounds.len_z / nElem_z_new
                    
                    config["geometry"]["nElem_x_computed"] = nElem_x_new
                    config["geometry"]["nElem_y_computed"] = nElem_y_new
                    config["geometry"]["nElem_z_computed"] = nElem_z_new
                    config["geometry"]["dx_computed"] = current_dx
                    config["geometry"]["dy_computed"] = current_dy
                    config["geometry"]["dz_computed"] = current_dz
                    
                    geom = (nElem_x=nElem_x_new, nElem_y=nElem_y_new, nElem_z=nElem_z_new, dx=current_dx, dy=current_dy, dz=current_dz, shapes=geom.shapes, actual_elem_count=size(elements, 1), max_domain_dim=geom.max_domain_dim)

                    Diagnostics.print_substep("[Refinement] Re-mapping Boundary Conditions & Forces...")
                    nNodes = size(nodes, 1)
                    ndof = nNodes * 3
                    bc_indicator = Boundary.get_bc_indicator(nNodes, nodes, Vector{Any}(bc_data))
                    F_external = zeros(Float32, ndof)
                    Boundary.apply_external_forces!(F_external, Vector{Any}(forces_data), nodes, elements)
                    _, original_density, protected_elements_mask, _ = Configuration.initialize_density_field(nodes, elements, geom.shapes, config)
                    Diagnostics.print_substep("[Refinement] Resetting solution guess.")
                    U_full = zeros(Float32, ndof)
                    TopologyOptimization.reset_filter_cache!()
                else
                    status_msg = "Skip"
                end
            end

            if iter > 1
                Threads.@threads for e in 1:size(elements, 1)
                    if protected_elements_mask[e]; density[e] = original_density[e]; end
                end
            end
            
            config["current_outer_iter"] = iter
            F_total = copy(F_external)
            
            if gravity_scale > 1e-4 && material_density > 1e-9
                 dx_curr = Float32(config["geometry"]["dx_computed"]); dy_curr = Float32(config["geometry"]["dy_computed"]); dz_curr = Float32(config["geometry"]["dz_computed"])
                 Boundary.add_self_weight!(F_total, density, material_density, gravity_scale, elements, dx_curr, dy_curr, dz_curr, gravity_accel)
            end
            if abs(delta_T) > 1e-6
                 Boundary.compute_global_thermal_forces!(F_total, nodes, elements, alpha_field, delta_T, E, nu, density)
            end
            
            Diagnostics.print_substep("FEA Solve (Iter $iter)")
            sol_tuple = Solver.solve_system(
                nodes, elements, E, nu, bc_indicator, F_total;
                density=density, config=config, min_stiffness_threshold=min_density, 
                prune_voids=true, u_prev=U_full 
            )
            U_new = sol_tuple[1]
            last_residual = sol_tuple[2]
            prec_used = sol_tuple[3]
            U_full = U_new
            
            if CUDA.functional(); GC.gc(); CUDA.reclaim(); end
            
            compliance = dot(F_total, U_full)
            strain_energy = 0.5 * compliance
            
            Diagnostics.print_substep("Calculating Stress Field...")
            t_stress = time()
            
            principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, principal_max_dir_field, principal_min_dir_field = Stress.compute_stress_field(nodes, elements, U_full, E, nu, density; return_voigt=false)
            
            try
                if iter == 1
                    Diagnostics.print_info("Exporting INITIAL REFERENCE STATE (Iter 0)...")
                    do_bin_init = (save_bin_freq > 0); do_stl_init = (save_stl_freq > 0); do_vtk_init = (save_vtk_freq > 0)
                    if do_bin_init || do_stl_init || do_vtk_init
                        Postprocessing.export_iteration_results(0, base_name, RESULTS_DIR, nodes, elements, U_full, F_total, bc_indicator, 
                                                                principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, 
                                                                principal_max_dir_field, principal_min_dir_field, 
                                                                density, E, geom; iso_threshold=Float32(iso_threshold), current_radius=Float32(filter_R), config=config, save_bin=do_bin_init, save_stl=do_stl_init, save_vtk=do_vtk_init)
                    end
                    if hard_stop_iter == 0; println(">>> HARD STOP: Stopping after background analysis (Iter 0)."); keep_running = false; break; end
                end
            catch e_export
                Diagnostics.print_warn("Initial export failed ($e_export). Proceeding with optimization.")
                Diagnostics.write_crash_log("crash_log.txt", "INITIAL_EXPORT", e_export, stacktrace(catch_backtrace()), iter, config, density)
            end
            
            active_stress_indices = findall(d -> d > 0.1f0, density)
            avg_l1_stress = isempty(active_stress_indices) ? 0.0f0 : Float32(mean(view(l1_stress_norm_field, active_stress_indices)))
            vol_total = length(density); active_non_soft = count(d -> d > min_density, density); vol_frac = sum(density) / vol_total
            
            target_avg_stress = l1_stress_allowable * 1.10f0
            
            if is_annealing && status_msg != "Refined"
                stress_ratio = avg_l1_stress / (target_avg_stress + Float32(1.0e-9))
                damping = (stress_ratio > 1.0f0) ? 0.5f0 : 0.1f0
                correction_factor = (1.0f0 / stress_ratio) ^ damping
                internal_l1_allowable *= correction_factor
                internal_l1_allowable = max(internal_l1_allowable, l1_stress_allowable * 0.05f0)
                internal_l1_allowable = min(internal_l1_allowable, l1_stress_allowable * 5.0f0)
                internal_l1_allowable = Float32(internal_l1_allowable)
                Diagnostics.print_substep("Adaptive Stress Control: Avg=$(round(avg_l1_stress, digits=3)) (Target $(round(target_avg_stress, digits=3))) -> Ratio $(round(stress_ratio, digits=2)) -> Adj. Allowable: $(round(internal_l1_allowable, digits=3))")
            end
            
            Diagnostics.print_substep("Topology Update & Filtering...")
            t_filter = time()
            
            res_tuple = TopologyOptimization.update_density!(
                density, l1_stress_norm_field, protected_elements_mask, E, internal_l1_allowable, 
                iter, nominal_iterations, 
                original_density, min_density, max_density_clamp, config, elements, is_annealing;
                force_no_cull = false 
            )
            
            density_change_metric, filter_R, curr_threshold = res_tuple
            
            if iter > 5 && status_msg != "Refined"
                rel_comp_change = abs(compliance - prev_compliance) / (prev_compliance + 1e-9)
                if rel_comp_change < CONVERGENCE_TOL && density_change_metric < CONVERGENCE_DENSITY_TOL
                    convergence_streak += 1
                    Diagnostics.print_info("Convergence Streak: $convergence_streak/$CONVERGENCE_REQUIRED_STREAK (Comp: $(round(rel_comp_change*100, digits=3))%, Rho: $(round(density_change_metric*100, digits=4))%)")
                else
                    convergence_streak = 0
                end
                
                if convergence_streak >= CONVERGENCE_REQUIRED_STREAK
                    if iter <= nominal_iterations
                        Diagnostics.print_info("Nominal Phase converged. Continuing...")
                    elseif iter < total_iterations
                        Diagnostics.print_info("Annealing Phase converged. Continuing...")
                    else
                          Diagnostics.print_success("Fully Converged. Stopping.")
                          keep_running = false
                    end
                end
            else
                convergence_streak = 0
            end
            prev_compliance = compliance
            
            iter_time = time() - iter_start_time
            cur_dims_str = "$(config["geometry"]["nElem_x_computed"])x$(config["geometry"]["nElem_y_computed"])x$(config["geometry"]["nElem_z_computed"])"
            
            Diagnostics.write_iteration_log(
                log_filename, iter, cur_dims_str, vol_total, active_non_soft, 
                filter_R, curr_threshold, compliance, strain_energy, avg_l1_stress, vol_frac, density_change_metric, 
                status_msg, iter_time, last_residual, prec_used
            )

            is_last_iter = (!keep_running) || (hard_stop_iter > 0 && iter >= hard_stop_iter) || (iter >= total_iterations)
            
            do_bin = (save_bin_freq > 0) && (iter % save_bin_freq == 0)
            do_stl = ((save_stl_freq > 0) && (iter % save_stl_freq == 0)) || is_last_iter
            do_vtk = (save_vtk_freq > 0) && (iter % save_vtk_freq == 0)
            
            should_export = do_bin || do_stl || do_vtk || is_last_iter 

            if should_export
                Diagnostics.print_substep("Exporting results...")
                try
                    Postprocessing.export_iteration_results(iter, base_name, RESULTS_DIR, nodes, elements, U_full, F_total, bc_indicator, 
                                                            principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, 
                                                            principal_max_dir_field, principal_min_dir_field, 
                                                            density, E, geom; iso_threshold=Float32(iso_threshold), current_radius=Float32(filter_R), config=config, save_bin=do_bin, save_stl=do_stl, save_vtk=do_vtk)
                catch e_export
                    Diagnostics.print_error("Post-processing failed at Iter $iter. Logged to crash_log.txt. Continuing simulation.")
                    Diagnostics.write_crash_log("crash_log.txt", "ITER_EXPORT", e_export, stacktrace(catch_backtrace()), iter, config, density)
                end
            end
            
            if hard_stop_iter > 0 && iter >= hard_stop_iter; println(">>> HARD STOP: Reached target iteration $hard_stop_iter."); keep_running = false; break; end
            
            if iter >= total_iterations
                Diagnostics.print_success("Reached total iteration limit ($total_iterations).")
                keep_running = false
            end
            
            if CUDA.functional(); Helpers.clear_gpu_memory(); end
            iter += 1
            GC.gc()
            flush(stdout) 
        end
        Diagnostics.log_status("Finished.")
    end

end

function bootstrap()
    println(">>> [BOOTSTRAP] Parsing arguments and launching module...")
    flush(stdout)
    # CORRECTED: Access PROJECT_ROOT via the module HEXA
    config_file = joinpath(HEXA.PROJECT_ROOT, "configs", "default.yaml")
    if length(ARGS) >= 1
        if endswith(lowercase(ARGS[1]), ".yaml") || endswith(lowercase(ARGS[1]), ".json")
            config_file = ARGS[1]
        end
    end

    if isfile(config_file)
        HEXA.run_main(config_file)
    else
        println("\n!!! ERROR: Input file not found at: $config_file")
        println("Usage: julia src/Main.jl [config.yaml]")
    end
end

bootstrap()