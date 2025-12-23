# FILE: .\src\Utils\Diagnostics.jl

module Diagnostics

using CUDA
using Printf
using Dates
using JSON

export log_status, check_memory, init_log_file, write_iteration_log, write_crash_log
export print_banner, print_info, print_warn, print_error, print_success, print_substep
export log_memory_snapshot, get_hardware_info, log_full_config, config_to_string

const C_RESET   = "\u001b[0m"
const C_BOLD    = "\u001b[1m"
const C_RED     = "\u001b[31m"
const C_GREEN   = "\u001b[32m"
const C_YELLOW  = "\u001b[33m"
const C_BLUE    = "\u001b[34m"
const C_MAGENTA = "\u001b[35m"
const C_CYAN    = "\u001b[36m"
const C_WHITE   = "\u001b[37m"

const LOG_HEADER = """
| Iter |  Mesh Size  | Total El | Active El |  Radius  |  Cutoff  | Compliance | Strain Energy | Avg L1 Stress | Vol Frac | Delta Rho | Refine? | Residual | Preconditioner | Iter Time | Wall Time | VRAM |"""

function get_timestamp()
    return Dates.format(now(), "HH:MM:SS")
end

function print_banner(title::String; char="=", color=C_CYAN)
    width = 90
    println("\n" * color * char^width * C_RESET)
    println(color * C_BOLD * ">>> " * title * C_RESET)
    println(color * char^width * C_RESET)
    flush(stdout)
end

function print_info(msg::String)
    println(" " * C_CYAN * "[INFO] " * C_RESET * msg)
    flush(stdout)
end

function print_warn(msg::String)
    println(" " * C_YELLOW * C_BOLD * "[WARN] " * C_RESET * C_YELLOW * msg * C_RESET)
    flush(stdout)
end

function print_error(msg::String)
    println(" " * C_RED * C_BOLD * "[ERROR] " * C_RESET * C_RED * msg * C_RESET)
    flush(stdout)
end

function print_success(msg::String)
    println(" " * C_GREEN * C_BOLD * "[DONE] " * C_RESET * msg)
    flush(stdout)
end

function print_substep(msg::String)
    println("    " * C_MAGENTA * "-> " * C_RESET * msg)
    flush(stdout)
end

function log_status(msg::String)
    println(C_BOLD * "[$(get_timestamp())] " * C_RESET * msg)
    flush(stdout) 
end

function check_memory()
    if CUDA.functional()
        free_gpu, total_gpu = CUDA.available_memory(), CUDA.total_memory()
        return free_gpu
    end
    return 0
end

function get_hardware_info()
    cpu_threads = Threads.nthreads()
    sys_mem_free = Sys.free_memory() / 1024^3
    sys_mem_total = Sys.total_memory() / 1024^3
    
    gpu_info = "None"
    if CUDA.functional()
        dev = CUDA.device()
        name = CUDA.name(dev)
        free_gpu, total_gpu = CUDA.available_memory(), CUDA.total_memory()
        gpu_info = "$name | VRAM: $(round(free_gpu/1024^3, digits=2)) GB Free / $(round(total_gpu/1024^3, digits=2)) GB Total"
    end

    return """
    Hardware Profile:
      CPU Threads: $cpu_threads
      System RAM:  $(round(sys_mem_free, digits=2)) GB Free / $(round(sys_mem_total, digits=2)) GB Total
      GPU Device:  $gpu_info
      Julia Ver:   $VERSION
    """
end

function format_memory_str()
    if CUDA.functional()
        free_gpu, total_gpu = CUDA.available_memory(), CUDA.total_memory()
        used_gb = (total_gpu - free_gpu) / 1024^3
        total_gb = total_gpu / 1024^3
        pct = (used_gb / total_gb) * 100
        
        col = C_GREEN
        if pct > 80; col = C_YELLOW; end
        if pct > 95; col = C_RED; end
        
        return "$col" * @sprintf("%4.1fG", used_gb) * C_RESET
    end
    return " CPU"
end

function log_memory_snapshot(label::String)
    if CUDA.functional()
        free, total = CUDA.available_memory(), CUDA.total_memory()
        used = total - free
        return @sprintf("[%s] VRAM: %.2f GB Used / %.2f GB Total (%.1f%%)", 
            label, used/1024^3, total/1024^3, (used/total)*100)
    end
    return "[$label] VRAM: N/A"
end

function format_seconds_to_hms(seconds::Float64)
    total_seconds = round(Int, seconds)
    h = div(total_seconds, 3600)
    m = div(total_seconds % 3600, 60)
    s = total_seconds % 60
    return @sprintf("%02d:%02d:%02d", h, m, s)
end

function log_full_config(io::IO, config::Dict, indent::Int=0)
    prefix = " " ^ indent
    for (k, v) in config
        if isa(v, Dict)
            write(io, "$prefix$k:\n")
            log_full_config(io, v, indent + 2)
        elseif isa(v, Vector)
            write(io, "$prefix$k: [")
            join(io, v, ", ")
            write(io, "]\n")
        else
            write(io, "$prefix$k: $v\n")
        end
    end
end

function config_to_string(config::Dict)
    io = IOBuffer()
    log_full_config(io, config)
    return String(take!(io))
end

function init_log_file(filename::String, config::Dict)
    open(filename, "w") do io
        write(io, "================================================================================\n")
        write(io, "HEXA FEM TOPOLOGY OPTIMIZATION LOG\n")
        write(io, "Start Date: $(now())\n")
        write(io, "================================================================================\n\n")
        
        write(io, "--- HARDWARE INFO ---\n")
        write(io, get_hardware_info())
        write(io, "\n")

        write(io, "--- CONFIGURATION ECHO ---\n")
        log_full_config(io, config)
        write(io, "\n")
        
        write(io, "="^220 * "\n") 
        write(io, LOG_HEADER * "\n")
    end
    print_success("Log file initialized at: $filename")
end

function write_iteration_log(filename::String, iter, mesh_dims_str, nTotal, nActive, 
                             filter_R, threshold, compliance, strain_energy, avg_l1, 
                             vol_frac, delta_rho, refine_status, time_sec, 
                             lin_residual=0.0, precond_type="-")
    
    vram_str_clean = replace(format_memory_str(), r"\u001b\[[0-9;]*m" => "") 
    vram_str_colored = format_memory_str() 
    
    wall_time = Dates.format(now(), "HH:MM:SS")
    time_hms = format_seconds_to_hms(Float64(time_sec))
    
    # Use 4 digits for radius precision per debugging request
    f_R = Float64(filter_R)
    f_th = Float64(threshold)
    f_comp = Float64(compliance)
    f_se = Float64(strain_energy)
    f_l1 = Float64(avg_l1)
    f_vf = Float64(vol_frac)
    f_dr = Float64(delta_rho)
    f_res = Float64(lin_residual)
    f_total = Float64(nTotal)
    f_active = Float64(nActive)

    stat_col = C_RESET
    if refine_status == "Refined"; stat_col = C_CYAN; end
    if refine_status == "Annealing"; stat_col = C_MAGENTA; end
    if refine_status == "Skip"; stat_col = C_YELLOW; end

    line_console = @sprintf("| %s%4d%s | %11s | %8.2e | %9.2e | %8.4f | %8.3f | %10.2e | %13.2e | %13.2e | %8.3f | %7.2f%% | %s%7s%s | %8.2e | %-14s | %9s | %9s | %s |",
                    C_BOLD, iter, C_RESET, mesh_dims_str, f_total, f_active, f_R, f_th,
                    f_comp, f_se, f_l1, f_vf, 
                    f_dr*100, stat_col, refine_status, C_RESET, f_res, precond_type, time_hms, wall_time, vram_str_colored)

    line_file = @sprintf("| %4d | %11s | %8.2e | %9.2e | %8.4f | %8.3f | %10.2e | %13.2e | %13.2e | %8.3f | %7.2f%% | %7s | %8.2e | %-14s | %9s | %9s | %4s |",
                    iter, mesh_dims_str, f_total, f_active, f_R, f_th,
                    f_comp, f_se, f_l1, f_vf, 
                    f_dr*100, refine_status, f_res, precond_type, time_hms, wall_time, vram_str_clean)
    
    try
        open(filename, "a") do io
            println(io, line_file)
        end
    catch e
        println(C_RED * "[LOG ERROR] Could not write to log file." * C_RESET)
    end
    
    if iter == 1 || iter % 10 == 0 || refine_status != "Nominal"
        println("\n" * C_BOLD * LOG_HEADER * C_RESET)
    end
    println(line_console)
    flush(stdout)
end

function write_crash_log(filename, stage, err, stack, iter, config, density_sample)
    try
        open(filename, "a") do io
            write(io, "\n" * "="^80 * "\n")
            write(io, "CRASH REPORT [$(Dates.now())]\n")
            write(io, "="^80 * "\n")
            write(io, "Stage: $stage\n")
            write(io, "Iteration: $iter\n")
            write(io, "Error: $err\n")
            write(io, "\nStacktrace:\n")
            showerror(io, err, stack)
            write(io, "\n\nSystem State:\n")
            write(io, get_hardware_info())
            write(io, "="^80 * "\n")
        end
        print_warn("Detailed crash log written to: $filename")
    catch e
        print_error("Could not write crash log: $e")
    end
end

end