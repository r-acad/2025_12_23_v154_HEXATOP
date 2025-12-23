// # FILE: .\Run.jl
using Pkg
using Dates

const C_RESET   = "\u001b[0m"
const C_BOLD    = "\u001b[1m"
const C_CYAN    = "\u001b[36m"
const C_GREEN   = "\u001b[32m"
const C_YELLOW  = "\u001b[33m"
const C_RED     = "\u001b[31m"

# ==============================================================================
# DEPENDENCY DEFINITION
# ==============================================================================
# Restored MarchingCubes for STL generation (Stability Reversion)
const REQUIRED_PACKAGES = [
    "CUDA",             # GPU acceleration
    "JSON",             # Input/Output data formatting
    "YAML",             # Configuration file parsing
    "MarchingCubes"     # Original STL generation package
]

function setup_robust_environment()
    println("\n" * C_CYAN * "="^60 * C_RESET)
    println(C_CYAN * C_BOLD * ">>> [LAUNCHER] HEXA TopOpt: Robust Environment Setup (Restored MarchingCubes)" * C_RESET)
    println(C_CYAN * ">>> [INFO] Time: $(Dates.now()) | Julia Version: $VERSION" * C_RESET)
    println(C_CYAN * "="^60 * C_RESET)

    project_dir = @__DIR__
    println(">>> [ENV] Activating project at: " * C_BOLD * "$project_dir" * C_RESET)
    Pkg.activate(project_dir)

    manifest_path = joinpath(project_dir, "Manifest.toml")
    project_path = joinpath(project_dir, "Project.toml")
    
    if !isfile(project_path)
        println(C_YELLOW * ">>> [ENV] No Project.toml found. Initializing new project context..." * C_RESET)
    end
    
    # 1. Attempt Instantiate
    try
        print(">>> [ENV] Attempting to instantiate environment... ")
        Pkg.instantiate()
        println(C_GREEN * "[OK]" * C_RESET)
    catch e
        println(C_RED * "\n!!! [ENV] Instantiation failed." * C_RESET)
        println(C_YELLOW * "!!! [ENV] performing SELF-HEALING: Deleting Manifest.toml and Re-resolving..." * C_RESET)
        
        if isfile(manifest_path)
            rm(manifest_path, force=true)
            println(">>> [ENV] Deleted incompatible Manifest.toml.")
        end

        try
            Pkg.resolve()
            Pkg.instantiate()
            println(C_GREEN * ">>> [ENV] Self-healing successful. Environment instantiated." * C_RESET)
        catch e_heal
            println(C_RED * "!!! [FATAL] Self-healing failed. Critical dependency error." * C_RESET)
            println(C_RED * "!!! [FATAL] Error: $e_heal" * C_RESET)
            exit(1)
        end
    end

    # 2. Check explicitly for the required packages
    print(">>> [ENV] Verifying core package list... ")
    try
        deps = Pkg.dependencies()
        missing_pkgs = String[]
        for pkg in REQUIRED_PACKAGES
            is_present = false
            for (uuid, info) in deps
                if info.name == pkg
                    is_present = true
                    break
                end
            end
            if !is_present
                push!(missing_pkgs, pkg)
            end
        end

        if !isempty(missing_pkgs)
            error("Missing packages: $missing_pkgs")
        end

        println(C_GREEN * "[OK]" * C_RESET)
    catch
        println(C_YELLOW * "\n>>> [ENV] Core dependencies missing or broken. Re-adding packages..." * C_RESET)
        try
            Pkg.add(REQUIRED_PACKAGES)
            println(C_GREEN * ">>> [ENV] Packages added successfully." * C_RESET)
        catch e_add
            println(C_RED * "!!! [FATAL] Could not install core packages: $e_add" * C_RESET)
            exit(1)
        end
    end
    
    println(">>> [ENV] Precompiling project...")
    try
        Pkg.precompile()
    catch
        println(C_YELLOW * "!!! [WARN] Precompilation had warnings (safe to ignore if code runs)." * C_RESET)
    end

    println(C_GREEN * ">>> [ENV] Environment Ready." * C_RESET)
    println(C_CYAN * "-"^60 * C_RESET)
end

setup_robust_environment()

const MAIN_SCRIPT = joinpath(@__DIR__, "src", "Main.jl")

if !isfile(MAIN_SCRIPT)
    println(C_RED * "!!! [ERROR] Could not find Main.jl at: $MAIN_SCRIPT" * C_RESET)
    exit(1)
end

println(">>> [LAUNCHER] Handing off to Main.jl...\n")
include(MAIN_SCRIPT)