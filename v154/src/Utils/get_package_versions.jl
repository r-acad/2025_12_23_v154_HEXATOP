# // # FILE: .\src\Utils\get_package_versions.jl
import Pkg

const TARGET_FILE = joinpath(@__DIR__, "..", "..", "Project.toml") # Adjusted path to root

println(">>> Generating Project.toml from current environment...")

deps = Pkg.dependencies()
sorted_deps = sort(collect(deps), by=x->x[2].name)

buffer = IOBuffer()

println(buffer, "name = \"HEXA_TopOpt\"")
println(buffer, "uuid = \"a1b2c3d4-e5f6-4a5b-9c8d-7e6f5g4h3i2j\"")
println(buffer, "authors = [\"User\"]")
println(buffer, "version = \"1.0.0\"")
println(buffer, "")

println(buffer, "[deps]")
for (uuid, pkg) in sorted_deps
    if pkg.is_direct_dep
        println(buffer, "$(pkg.name) = \"$uuid\"")
    end
end
println(buffer, "")

println(buffer, "[compat]")
# Allow any Julia version from 1.6 upwards (LTS)
println(buffer, "julia = \"1.6\"") 

for (uuid, pkg) in sorted_deps
    if pkg.is_direct_dep
        if pkg.version !== nothing
            # FIX: Removed the "=" sign. 
            # "$(pkg.version)" implies semver compatibility (e.g., "1.2.3" allows "1.2.4" and "1.9.0")
            println(buffer, "$(pkg.name) = \"$(pkg.version)\"")
        else
            println(buffer, "# Warning: Could not detect version for $(pkg.name)")
        end
    end
end

new_content = String(take!(buffer))

if isfile(TARGET_FILE)
    mv(TARGET_FILE, TARGET_FILE * ".bak", force=true)
    println(">>> Existing Project.toml backed up to Project.toml.bak")
end

open(TARGET_FILE, "w") do io
    write(io, new_content)
end

println("-"^60)
println(">>> SUCCESS: Project.toml has been written successfully.")
println(">>> Location: $TARGET_FILE")
println("-"^60)