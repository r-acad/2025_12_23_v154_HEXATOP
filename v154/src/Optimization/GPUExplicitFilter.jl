# // # FILE: .\src\Optimization\GPUExplicitFilter.jl
# 
module GPUExplicitFilter

using CUDA
using LinearAlgebra
using Printf

export apply_explicit_filter!

# ------------------------------------------------------------------------------
# 1. KERNELS (Fully Expanded & Verbose)
# ------------------------------------------------------------------------------

"""
    diffusion_kernel_interior!(rho_new, rho_old, dt_over_dx2, nx, ny, nz)
    Standard 7-point Laplacian stencil for interior elements.
    Processes elements that are at least 1 element away from all boundaries.
"""
function diffusion_kernel_interior!(rho_new, rho_old, dt_over_dx2, nx, ny, nz)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    # Process only strictly interior elements
    if i > 1 && i < nx && j > 1 && j < ny && k > 1 && k < nz
        @inbounds begin
            idx = i + (j-1)*nx + (k-1)*nx*ny
            center = rho_old[idx]
            
            # Neighbors
            left   = rho_old[idx - 1]
            right  = rho_old[idx + 1]
            front  = rho_old[idx - nx]
            back   = rho_old[idx + nx]
            bottom = rho_old[idx - nx*ny]
            top    = rho_old[idx + nx*ny]
            
            laplacian = (left + right + front + back + bottom + top - 6.0f0 * center)
            
            # Update
            rho_new[idx] = center + dt_over_dx2 * laplacian
        end
    end
    return nothing
end

"""
    diffusion_kernel_faces!(rho_new, rho_old, dt_over_dx2, nx, ny, nz)
    Neumann BCs (ghost values) for the 6 planar faces.
"""
function diffusion_kernel_faces!(rho_new, rho_old, dt_over_dx2, nx, ny, nz)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    if i < 1 || i > nx || j < 1 || j > ny || k < 1 || k > nz; return nothing; end
    
    # Identify Face elements (On exactly 1 boundary surface)
    on_x = (i == 1 || i == nx) && (j > 1 && j < ny) && (k > 1 && k < nz)
    on_y = (j == 1 || j == ny) && (i > 1 && i < nx) && (k > 1 && k < nz)
    on_z = (k == 1 || k == nz) && (i > 1 && i < nx) && (j > 1 && j < ny)
    
    if !(on_x || on_y || on_z); return nothing; end
    
    @inbounds begin
        idx = i + (j-1)*nx + (k-1)*nx*ny
        center = rho_old[idx]
        sum_n = 0.0f0
        w_tot = 0.0f0
        
        # --- X-Axis Neighbors ---
        if i > 1
            sum_n += rho_old[idx-1]
            w_tot += 1.0f0
        elseif i == 1
            # Ghost Value: 2*center - next
            sum_n += (2.0f0*center - rho_old[idx+1])
            w_tot += 1.0f0
        end
        
        if i < nx
            sum_n += rho_old[idx+1]
            w_tot += 1.0f0
        elseif i == nx
            sum_n += (2.0f0*center - rho_old[idx-1])
            w_tot += 1.0f0
        end
        
        # --- Y-Axis Neighbors ---
        if j > 1
            sum_n += rho_old[idx-nx]
            w_tot += 1.0f0
        elseif j == 1
            sum_n += (2.0f0*center - rho_old[idx+nx])
            w_tot += 1.0f0
        end
        
        if j < ny
            sum_n += rho_old[idx+nx]
            w_tot += 1.0f0
        elseif j == ny
            sum_n += (2.0f0*center - rho_old[idx-nx])
            w_tot += 1.0f0
        end
        
        # --- Z-Axis Neighbors ---
        if k > 1
            sum_n += rho_old[idx-nx*ny]
            w_tot += 1.0f0
        elseif k == 1
            sum_n += (2.0f0*center - rho_old[idx+nx*ny])
            w_tot += 1.0f0
        end
        
        if k < nz
            sum_n += rho_old[idx+nx*ny]
            w_tot += 1.0f0
        elseif k == nz
            sum_n += (2.0f0*center - rho_old[idx-nx*ny])
            w_tot += 1.0f0
        end
        
        rho_new[idx] = center + dt_over_dx2 * (sum_n - w_tot * center)
    end
    return nothing
end

"""
    diffusion_kernel_edges!(rho_new, rho_old, dt_over_dx2, nx, ny, nz)
    Neumann BCs for the 12 edges (lines where 2 faces meet).
"""
function diffusion_kernel_edges!(rho_new, rho_old, dt_over_dx2, nx, ny, nz)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    if i < 1 || i > nx || j < 1 || j > ny || k < 1 || k > nz; return nothing; end
    
    # Identify Edge elements (On exactly 2 boundary surfaces)
    b_cnt = 0
    if i==1||i==nx; b_cnt+=1; end
    if j==1||j==ny; b_cnt+=1; end
    if k==1||k==nz; b_cnt+=1; end
    
    if b_cnt != 2; return nothing; end
    
    @inbounds begin
        idx = i + (j-1)*nx + (k-1)*nx*ny
        center = rho_old[idx]
        sum_n = 0.0f0
        w_tot = 0.0f0
        
        # --- X-Axis ---
        if i > 1
            sum_n += rho_old[idx-1]
            w_tot += 1.0f0
        elseif i == 1 && nx > 1
            sum_n += (2.0f0*center - rho_old[idx+1])
            w_tot += 1.0f0
        end
        
        if i < nx
            sum_n += rho_old[idx+1]
            w_tot += 1.0f0
        elseif i == nx && nx > 1
            sum_n += (2.0f0*center - rho_old[idx-1])
            w_tot += 1.0f0
        end
        
        # --- Y-Axis ---
        if j > 1
            sum_n += rho_old[idx-nx]
            w_tot += 1.0f0
        elseif j == 1 && ny > 1
            sum_n += (2.0f0*center - rho_old[idx+nx])
            w_tot += 1.0f0
        end
        
        if j < ny
            sum_n += rho_old[idx+nx]
            w_tot += 1.0f0
        elseif j == ny && ny > 1
            sum_n += (2.0f0*center - rho_old[idx-nx])
            w_tot += 1.0f0
        end
        
        # --- Z-Axis ---
        if k > 1
            sum_n += rho_old[idx-nx*ny]
            w_tot += 1.0f0
        elseif k == 1 && nz > 1
            sum_n += (2.0f0*center - rho_old[idx+nx*ny])
            w_tot += 1.0f0
        end
        
        if k < nz
            sum_n += rho_old[idx+nx*ny]
            w_tot += 1.0f0
        elseif k == nz && nz > 1
            sum_n += (2.0f0*center - rho_old[idx-nx*ny])
            w_tot += 1.0f0
        end
        
        if w_tot > 0.0f0
            rho_new[idx] = center + dt_over_dx2 * (sum_n - w_tot * center)
        else
            rho_new[idx] = center
        end
    end
    return nothing
end

"""
    diffusion_kernel_corners!(rho_new, rho_old, dt_over_dx2, nx, ny, nz)
    Neumann BCs for the 8 corners (points where 3 faces meet).
"""
function diffusion_kernel_corners!(rho_new, rho_old, dt_over_dx2, nx, ny, nz)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    if i < 1 || i > nx || j < 1 || j > ny || k < 1 || k > nz; return nothing; end
    
    # Identify Corner elements (On exactly 3 boundary surfaces)
    is_corner = (i==1||i==nx) && (j==1||j==ny) && (k==1||k==nz)
    if !is_corner; return nothing; end
    
    @inbounds begin
        idx = i + (j-1)*nx + (k-1)*nx*ny
        center = rho_old[idx]
        sum_n = 0.0f0
        w_tot = 0.0f0
        
        # --- X-Axis ---
        if i > 1
            sum_n += rho_old[idx-1]
            w_tot += 1.0f0
        elseif i == 1 && nx > 1
            sum_n += (2.0f0*center - rho_old[idx+1])
            w_tot += 1.0f0
        end
        
        if i < nx
            sum_n += rho_old[idx+1]
            w_tot += 1.0f0
        elseif i == nx && nx > 1
            sum_n += (2.0f0*center - rho_old[idx-1])
            w_tot += 1.0f0
        end
        
        # --- Y-Axis ---
        if j > 1
            sum_n += rho_old[idx-nx]
            w_tot += 1.0f0
        elseif j == 1 && ny > 1
            sum_n += (2.0f0*center - rho_old[idx+nx])
            w_tot += 1.0f0
        end
        
        if j < ny
            sum_n += rho_old[idx+nx]
            w_tot += 1.0f0
        elseif j == ny && ny > 1
            sum_n += (2.0f0*center - rho_old[idx-nx])
            w_tot += 1.0f0
        end
        
        # --- Z-Axis ---
        if k > 1
            sum_n += rho_old[idx-nx*ny]
            w_tot += 1.0f0
        elseif k == 1 && nz > 1
            sum_n += (2.0f0*center - rho_old[idx+nx*ny])
            w_tot += 1.0f0
        end
        
        if k < nz
            sum_n += rho_old[idx+nx*ny]
            w_tot += 1.0f0
        elseif k == nz && nz > 1
            sum_n += (2.0f0*center - rho_old[idx-nx*ny])
            w_tot += 1.0f0
        end
        
        if w_tot > 0.0f0
            rho_new[idx] = center + dt_over_dx2 * (sum_n - w_tot * center)
        else
            rho_new[idx] = center
        end
    end
    return nothing
end

# ------------------------------------------------------------------------------
# 2. MAIN FUNCTION (Optimized)
# ------------------------------------------------------------------------------

"""
    apply_explicit_filter!(density, nx, ny, nz, dx, dy, dz, radius, min_density)

OPTIMIZED VERSION:
1. Dynamic Slab Sizing: Utilizes all available VRAM instead of hardcoded 500MB.
2. Optimized Thread Blocks: (32, 4, 4) for memory coalescing on X-axis.
3. Execution Timer: Logs performance.
4. Sparsity Culling: Skips processing for empty slabs.
"""
function apply_explicit_filter!(density::Vector{Float32}, 
                                nx::Int, ny::Int, nz::Int,
                                dx::Float32, dy::Float32, dz::Float32,
                                radius::Float32,
                                min_density::Float32=0.0001f0) 
    
    if !CUDA.functional(); return density; end
    
    t_start = time()
    
    # Physics parameters
    D = (radius^2) / 6.0f0
    avg_dx = (dx + dy + dz) / 3.0f0
    dt_stable = 0.15f0 * (avg_dx^2) / D
    n_steps = max(10, round(Int, radius / avg_dx))
    dt_over_dx2 = dt_stable * D / (avg_dx^2)
    
    halo = 2
    
    # --- DYNAMIC MEMORY SIZING ---
    free_mem = CUDA.available_memory()
    bytes_per_node = 8 # 2x Float32 (old + new)
    
    # Reserve 500MB safety for display/OS, use rest
    usable_mem = max(500 * 1024^2, free_mem - (500 * 1024^2))
    max_nodes = div(usable_mem, bytes_per_node)
    
    slice_size = nx * ny
    max_slab_nz = div(max_nodes, slice_size)
    if max_slab_nz < (halo*2 + 1); max_slab_nz = halo*2 + 1; end # Emergency fit
    
    n_slabs = cld(nz, max_slab_nz - 2*halo)
    
    # --- LAUNCH CONFIG OPTIMIZATION ---
    # (32, 4, 4) ensures X-axis reads are coalesced into full warps
    threads = (32, 4, 4)
    
    println(@sprintf("    [ExplicitFilter] Radius=%.3f, Steps=%d, dt=%.2e, Slabs=%d (Unclamped Diffusion)", 
                     radius, n_steps, dt_stable, n_slabs))
    
    filtered_density = copy(density)
    
    skipped_slabs = 0
    
    for slab_idx in 1:n_slabs
        z_start = max(1, (slab_idx - 1) * (max_slab_nz - 2*halo) + 1 - halo)
        z_end   = min(nz, z_start + max_slab_nz - 1)
        slab_nz = z_end - z_start + 1
        slab_size = nx * ny * slab_nz
        
        # Gather (Host) - Dynamic Slab construction
        slab_data = zeros(Float32, slab_size)
        Threads.@threads for k in 1:slab_nz
            global_k = z_start + k - 1
            start_g = 1 + (global_k - 1) * nx * ny
            end_g   = start_g + nx * ny - 1
            start_l = 1 + (k - 1) * nx * ny
            end_l   = start_l + nx * ny - 1
            slab_data[start_l:end_l] = density[start_g:end_g]
        end
        
        # === OPTIMIZATION: SKIP EMPTY SLABS ===
        # If the max density in this slab (including halos) is "Void",
        # diffusion will not change anything. We can skip GPU processing.
        
        max_val = maximum(slab_data)
        
        # FIXED TYPO HERE: 1.0e-5f0 -> 1.0f-5
        if max_val <= (min_density * 1.01f0 + 1.0f-5)
            skipped_slabs += 1
            continue
        end
        
        rho_gpu = CuArray(slab_data)
        rho_new_gpu = CUDA.zeros(Float32, slab_size)
        
        blocks = (cld(nx, threads[1]), cld(ny, threads[2]), cld(slab_nz, threads[3]))
        
        for step in 1:n_steps
            @cuda threads=threads blocks=blocks diffusion_kernel_interior!(
                rho_new_gpu, rho_gpu, dt_over_dx2, nx, ny, slab_nz
            )
            @cuda threads=threads blocks=blocks diffusion_kernel_faces!(
                rho_new_gpu, rho_gpu, dt_over_dx2, nx, ny, slab_nz
            )
            @cuda threads=threads blocks=blocks diffusion_kernel_edges!(
                rho_new_gpu, rho_gpu, dt_over_dx2, nx, ny, slab_nz
            )
            @cuda threads=threads blocks=blocks diffusion_kernel_corners!(
                rho_new_gpu, rho_gpu, dt_over_dx2, nx, ny, slab_nz
            )
            rho_gpu, rho_new_gpu = rho_new_gpu, rho_gpu
        end
        
        CUDA.synchronize()
        copyto!(slab_data, rho_gpu)
        
        # Scatter (Host) - Discard Halo
        valid_z_start = (slab_idx == 1) ? 1 : halo + 1
        valid_z_end   = (slab_idx == n_slabs) ? slab_nz : slab_nz - halo
        
        Threads.@threads for k in valid_z_start:valid_z_end
            global_k = z_start + k - 1
            start_g = 1 + (global_k - 1) * nx * ny
            end_g   = start_g + nx * ny - 1
            start_l = 1 + (k - 1) * nx * ny
            end_l   = start_l + nx * ny - 1
            filtered_density[start_g:end_g] = slab_data[start_l:end_l]
        end
        
        CUDA.unsafe_free!(rho_gpu); CUDA.unsafe_free!(rho_new_gpu)
    end
    
    GC.gc(); CUDA.reclaim()
    
    elapsed = time() - t_start
    println(@sprintf("    [ExplicitFilter] R=%.3f, Slabs=%d/%d (Skipped %d Empty), Time=%.2fs", 
                     radius, n_slabs - skipped_slabs, n_slabs, skipped_slabs, elapsed))
    
    return filtered_density
end

end