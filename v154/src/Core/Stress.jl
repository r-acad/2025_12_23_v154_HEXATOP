module Stress 

using LinearAlgebra 
using Base.Threads
using CUDA
using Printf
using ..Element 

export compute_stress_field 

const GAUSS_VAL = 0.577350269f0 

"""
    estimate_stress_vram_required(nNodes, nElem, use_voigt)

Estimates total VRAM needed for stress calculation in bytes.
"""
function estimate_stress_vram_required(nNodes::Int, nElem::Int, use_voigt::Bool)
    bytes_per_float32 = 4
    
    # Input arrays
    nodes_mem = nNodes * 3 * bytes_per_float32
    elements_mem = nElem * 8 * 4  # Int32
    U_mem = nNodes * 3 * bytes_per_float32
    density_mem = nElem * bytes_per_float32
    
    # Output arrays
    principal_mem = nElem * 3 * bytes_per_float32
    vonmises_mem = nElem * bytes_per_float32
    l1_mem = nElem * bytes_per_float32
    dir_max_mem = nElem * 3 * bytes_per_float32
    dir_min_mem = nElem * 3 * bytes_per_float32
    voigt_mem = use_voigt ? (nElem * 6 * bytes_per_float32) : 0
    
    # Workspace (temporary full-DOF arrays in kernel)
    workspace_mem = nNodes * 3 * bytes_per_float32
    
    total = nodes_mem + elements_mem + U_mem + density_mem +
            principal_mem + vonmises_mem + l1_mem + 
            dir_max_mem + dir_min_mem + voigt_mem + workspace_mem
    
    return total
end

"""
    compute_element_stress(element_nodes, element_disp, E, nu)

CPU implementation - computes stress tensor for a single element.
"""
function compute_element_stress(element_nodes::Array{Float32,2}, 
                                element_disp::Array{Float32,1}, 
                                E::Float32, nu::Float32) 
    D = Element.material_matrix(E, nu) 
    
    xi, eta, zeta = 0.0f0, 0.0f0, 0.0f0 
    inv8 = 0.125f0 
    
    j11 = (-element_nodes[1,1] + element_nodes[2,1] + element_nodes[3,1] - element_nodes[4,1] - element_nodes[5,1] + element_nodes[6,1] + element_nodes[7,1] - element_nodes[8,1]) * inv8
    j12 = (-element_nodes[1,2] + element_nodes[2,2] + element_nodes[3,2] - element_nodes[4,2] - element_nodes[5,2] + element_nodes[6,2] + element_nodes[7,2] - element_nodes[8,2]) * inv8
    j13 = (-element_nodes[1,3] + element_nodes[2,3] + element_nodes[3,3] - element_nodes[4,3] - element_nodes[5,3] + element_nodes[6,3] + element_nodes[7,3] - element_nodes[8,3]) * inv8

    j21 = (-element_nodes[1,1] - element_nodes[2,1] + element_nodes[3,1] + element_nodes[4,1] - element_nodes[5,1] - element_nodes[6,1] + element_nodes[7,1] + element_nodes[8,1]) * inv8
    j22 = (-element_nodes[1,2] - element_nodes[2,2] + element_nodes[3,2] + element_nodes[4,2] - element_nodes[5,2] - element_nodes[6,2] + element_nodes[7,2] + element_nodes[8,2]) * inv8
    j23 = (-element_nodes[1,3] - element_nodes[2,3] + element_nodes[3,3] + element_nodes[4,3] - element_nodes[5,3] - element_nodes[6,3] + element_nodes[7,3] + element_nodes[8,3]) * inv8
    
    j31 = (-element_nodes[1,1] - element_nodes[2,1] - element_nodes[3,1] - element_nodes[4,1] + element_nodes[5,1] + element_nodes[6,1] + element_nodes[7,1] + element_nodes[8,1]) * inv8
    j32 = (-element_nodes[1,2] - element_nodes[2,2] - element_nodes[3,2] - element_nodes[4,2] + element_nodes[5,2] + element_nodes[6,2] + element_nodes[7,2] + element_nodes[8,2]) * inv8
    j33 = (-element_nodes[1,3] - element_nodes[2,3] - element_nodes[3,3] - element_nodes[4,3] + element_nodes[5,3] + element_nodes[6,3] + element_nodes[7,3] + element_nodes[8,3]) * inv8

    detJ = j11*(j22*j33 - j23*j32) - j12*(j21*j33 - j23*j31) + j13*(j21*j32 - j22*j31)
    if detJ <= 1.0e-9; detJ = 1.0f0; end
    invDet = 1.0f0 / detJ

    Jinv11 =  (j22*j33 - j23*j32) * invDet
    Jinv12 = -(j12*j33 - j13*j32) * invDet
    Jinv13 =  (j12*j23 - j13*j22) * invDet
    
    Jinv21 = -(j21*j33 - j23*j31) * invDet
    Jinv22 =  (j11*j33 - j13*j31) * invDet
    Jinv23 = -(j11*j23 - j13*j21) * invDet
    
    Jinv31 =  (j21*j32 - j22*j31) * invDet
    Jinv32 = -(j11*j32 - j12*j31) * invDet
    Jinv33 =  (j11*j22 - j12*j21) * invDet

    B = zeros(Float32, 6, 24)
    
    dNi_dxi  = Float32[-0.125,  0.125,  0.125, -0.125, -0.125,  0.125,  0.125, -0.125]
    dNi_deta = Float32[-0.125, -0.125,  0.125,  0.125, -0.125, -0.125,  0.125,  0.125]
    dNi_dzet = Float32[-0.125, -0.125, -0.125, -0.125,  0.125,  0.125,  0.125,  0.125]
    
    for i in 1:8
        dN_dx = dNi_dxi[i]*Jinv11 + dNi_deta[i]*Jinv21 + dNi_dzet[i]*Jinv31
        dN_dy = dNi_dxi[i]*Jinv12 + dNi_deta[i]*Jinv22 + dNi_dzet[i]*Jinv32
        dN_dz = dNi_dxi[i]*Jinv13 + dNi_deta[i]*Jinv23 + dNi_dzet[i]*Jinv33
        
        idx = 3*(i-1) + 1
        B[1, idx]   = dN_dx; B[2, idx+1] = dN_dy; B[3, idx+2] = dN_dz
        B[4, idx]   = dN_dy; B[4, idx+1] = dN_dx
        B[5, idx+1] = dN_dz; B[5, idx+2] = dN_dy
        B[6, idx]   = dN_dz; B[6, idx+2] = dN_dx
    end

    strain = B * element_disp 
    stress_voigt = D * strain 

    σ = zeros(Float32, 3, 3) 
    σ[1,1] = stress_voigt[1]; σ[2,2] = stress_voigt[2]; σ[3,3] = stress_voigt[3]       
    σ[1,2] = stress_voigt[4]; σ[2,1] = stress_voigt[4]       
    σ[2,3] = stress_voigt[5]; σ[3,2] = stress_voigt[5]       
    σ[1,3] = stress_voigt[6]; σ[3,1] = stress_voigt[6]       
    return σ 
end 

"""
    compute_principal_data(σ)

CPU implementation - eigenvalue decomposition of stress tensor.
"""
function compute_principal_data(σ::Matrix{Float32}) 
    F = eigen(σ)
    perm = sortperm(F.values, rev=true)
    principal_stresses = F.values[perm]
    principal_vectors  = F.vectors[:, perm]
    max_dir = principal_vectors[:, 1]
    min_dir = principal_vectors[:, 3] 

    σxx, σyy, σzz = σ[1,1], σ[2,2], σ[3,3]
    σxy, σyz, σxz = σ[1,2], σ[2,3], σ[1,3]
    vm = sqrt(0.5f0 * ((σxx-σyy)^2 + (σyy-σzz)^2 + (σzz-σxx)^2) + 3.0f0*(σxy^2 + σyz^2 + σxz^2))  

    return principal_stresses, vm, max_dir, min_dir
end 

# GPU Kernels (unchanged from original)
@inline function eigen3x3_device(s11, s22, s33, s12, s23, s13)
    v11, v12, v13 = 1.0f0, 0.0f0, 0.0f0
    v21, v22, v23 = 0.0f0, 1.0f0, 0.0f0
    v31, v32, v33 = 0.0f0, 0.0f0, 1.0f0
    
    @fastmath for iter in 1:12
        a12 = abs(s12); a23 = abs(s23); a13 = abs(s13)
        if max(a12, a23, a13) < 1.0e-9; break; end
        
        h = s12; g = 100.0f0 * abs(h)
        if abs(s11) + g != abs(s11) || abs(s22) + g != abs(s22)
            theta = 0.5f0 * (s22 - s11) / h
            t = 1.0f0 / (abs(theta) + sqrt(theta^2 + 1.0f0))
            if theta < 0.0f0; t = -t; end
            c = 1.0f0 / sqrt(t^2 + 1.0f0); s = t * c; tau = s / (1.0f0 + c)
            s11 -= t * s12; s22 += t * s12; s12 = 0.0f0
            tmp = s13; s13 -= s * (s23 + tau * s13); s23 += s * (tmp - tau * s23)
            tmp = v11; v11 -= s * (v12 + tau * v11); v12 += s * (tmp - tau * v12)
            tmp = v21; v21 -= s * (v22 + tau * v21); v22 += s * (tmp - tau * v22)
            tmp = v31; v31 -= s * (v32 + tau * v31); v32 += s * (tmp - tau * v32)
        end

        h = s13; g = 100.0f0 * abs(h)
        if abs(s11) + g != abs(s11) || abs(s33) + g != abs(s33)
            theta = 0.5f0 * (s33 - s11) / h
            t = 1.0f0 / (abs(theta) + sqrt(theta^2 + 1.0f0))
            if theta < 0.0f0; t = -t; end
            c = 1.0f0 / sqrt(t^2 + 1.0f0); s = t * c; tau = s / (1.0f0 + c)
            s11 -= t * s13; s33 += t * s13; s13 = 0.0f0
            tmp = s12; s12 -= s * (s23 + tau * s12); s23 += s * (tmp - tau * s23)
            tmp = v11; v11 -= s * (v13 + tau * v11); v13 += s * (tmp - tau * v13)
            tmp = v21; v21 -= s * (v23 + tau * v21); v23 += s * (tmp - tau * v23)
            tmp = v31; v31 -= s * (v33 + tau * v31); v33 += s * (tmp - tau * v33)
        end

        h = s23; g = 100.0f0 * abs(h)
        if abs(s22) + g != abs(s22) || abs(s33) + g != abs(s33)
            theta = 0.5f0 * (s33 - s22) / h
            t = 1.0f0 / (abs(theta) + sqrt(theta^2 + 1.0f0))
            if theta < 0.0f0; t = -t; end
            c = 1.0f0 / sqrt(t^2 + 1.0f0); s = t * c; tau = s / (1.0f0 + c)
            s22 -= t * s23; s33 += t * s23; s23 = 0.0f0
            tmp = s12; s12 -= s * (s13 + tau * s12); s13 += s * (tmp - tau * s13)
            tmp = v12; v12 -= s * (v13 + tau * v12); v13 += s * (tmp - tau * v13)
            tmp = v22; v22 -= s * (v23 + tau * v22); v23 += s * (tmp - tau * v23)
            tmp = v32; v32 -= s * (v33 + tau * v32); v33 += s * (tmp - tau * v33)
        end
    end
    
    if s11 < s22; t=s11; s11=s22; s22=t; t=v11; v11=v12; v12=t; t=v21; v21=v22; v22=t; t=v31; v31=v32; v32=t; end
    if s22 < s33; t=s22; s22=s33; s33=t; t=v12; v12=v13; v13=t; t=v22; v22=v23; v23=t; t=v32; v32=v33; v33=t; end
    if s11 < s22; t=s11; s11=s22; s22=t; t=v11; v11=v12; v12=t; t=v21; v21=v22; v22=t; t=v31; v31=v32; v32=t; end
    
    return s11, s22, s33, v11, v21, v31, v13, v23, v33
end

function stress_kernel!(principal_field, vonmises_field, voigt_field, l1_norm_field, dir_max_field, dir_min_field,
                        nodes, elements, U, density, E, nu, nElem, save_voigt)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e > nElem; return; end

    dens = density[e]
    
    @fastmath if dens < 1.0e-6
        @inbounds begin
            principal_field[1, e] = 0.0f0
            principal_field[2, e] = 0.0f0
            principal_field[3, e] = 0.0f0
            vonmises_field[e] = 0.0f0
            l1_norm_field[e] = 0.0f0
            dir_max_field[1, e] = 0.0f0; dir_max_field[2, e] = 0.0f0; dir_max_field[3, e] = 0.0f0
            dir_min_field[1, e] = 0.0f0; dir_min_field[2, e] = 0.0f0; dir_min_field[3, e] = 0.0f0
            if save_voigt
                voigt_field[1,e]=0; voigt_field[2,e]=0; voigt_field[3,e]=0
                voigt_field[4,e]=0; voigt_field[5,e]=0; voigt_field[6,e]=0
            end
        end
        return
    end
    
    E_loc = @fastmath E * dens
    
    j11, j12, j13 = 0.0f0, 0.0f0, 0.0f0
    j21, j22, j23 = 0.0f0, 0.0f0, 0.0f0
    j31, j32, j33 = 0.0f0, 0.0f0, 0.0f0
    
    @inbounds for i in 1:8
        nid = elements[e, i]
        nx = nodes[nid, 1]; ny = nodes[nid, 2]; nz = nodes[nid, 3]
        
        xi_val = (i==2 || i==3 || i==6 || i==7) ? 0.125f0 : -0.125f0
        eta_val = (i==3 || i==4 || i==7 || i==8) ? 0.125f0 : -0.125f0
        zet_val = (i>=5) ? 0.125f0 : -0.125f0
        
        j11 += xi_val * nx; j12 += xi_val * ny; j13 += xi_val * nz
        j21 += eta_val * nx; j22 += eta_val * ny; j23 += eta_val * nz
        j31 += zet_val * nx; j32 += zet_val * ny; j33 += zet_val * nz
    end
    
    @fastmath begin
        detJ = j11*(j22*j33 - j23*j32) - j12*(j21*j33 - j23*j31) + j13*(j21*j32 - j22*j31)
        if detJ <= 1.0e-12; detJ = 1.0f0; end
        invDet = 1.0f0 / detJ
        
        Jinv11 =  (j22*j33 - j23*j32) * invDet; Jinv12 = -(j12*j33 - j13*j32) * invDet; Jinv13 =  (j12*j23 - j13*j22) * invDet
        Jinv21 = -(j21*j33 - j23*j31) * invDet; Jinv22 =  (j11*j33 - j13*j31) * invDet; Jinv23 = -(j11*j23 - j13*j21) * invDet
        Jinv31 =  (j21*j32 - j22*j31) * invDet; Jinv32 = -(j11*j32 - j12*j31) * invDet; Jinv33 =  (j11*j22 - j12*j21) * invDet
    end

    eps_xx, eps_yy, eps_zz = 0.0f0, 0.0f0, 0.0f0
    gam_xy, gam_yz, gam_xz = 0.0f0, 0.0f0, 0.0f0
    
    @inbounds for i in 1:8
        nid = elements[e, i]
        base = 3*(nid-1)
        ux = U[base+1]; uy = U[base+2]; uz = U[base+3]
        
        dN_dxi = (i==2 || i==3 || i==6 || i==7) ? 0.125f0 : -0.125f0
        dN_deta = (i==3 || i==4 || i==7 || i==8) ? 0.125f0 : -0.125f0
        dN_dzet = (i>=5) ? 0.125f0 : -0.125f0
        
        dN_dx = dN_dxi*Jinv11 + dN_deta*Jinv21 + dN_dzet*Jinv31
        dN_dy = dN_dxi*Jinv12 + dN_deta*Jinv22 + dN_dzet*Jinv32
        dN_dz = dN_dxi*Jinv13 + dN_deta*Jinv23 + dN_dzet*Jinv33
        
        eps_xx += dN_dx * ux
        eps_yy += dN_dy * uy
        eps_zz += dN_dz * uz
        gam_xy += dN_dy * ux + dN_dx * uy
        gam_yz += dN_dz * uy + dN_dy * uz
        gam_xz += dN_dz * ux + dN_dx * uz
    end
    
    @fastmath begin
        fact = E_loc / ((1.0f0 + nu) * (1.0f0 - 2.0f0 * nu))
        c1 = (1.0f0 - nu) * fact
        c2 = nu * fact
        c3 = (0.5f0 - nu) * fact 
        
        sig_xx = c1*eps_xx + c2*eps_yy + c2*eps_zz
        sig_yy = c2*eps_xx + c1*eps_yy + c2*eps_zz
        sig_zz = c2*eps_xx + c2*eps_yy + c1*eps_zz
        sig_xy = c3 * gam_xy 
        sig_yz = c3 * gam_yz
        sig_xz = c3 * gam_xz
    end
    
    if save_voigt
        @inbounds begin
            voigt_field[1, e] = sig_xx
            voigt_field[2, e] = sig_yy
            voigt_field[3, e] = sig_zz
            voigt_field[4, e] = sig_xy
            voigt_field[5, e] = sig_yz
            voigt_field[6, e] = sig_xz
        end
    end
    
    vm_sq = 0.5f0 * ((sig_xx-sig_yy)^2 + (sig_yy-sig_zz)^2 + (sig_zz-sig_xx)^2) + 3.0f0 * (sig_xy^2 + sig_yz^2 + sig_xz^2)
    vm = sqrt(max(0.0f0, vm_sq))
    @inbounds vonmises_field[e] = vm
    
    s11, s22, s33, v1x, v1y, v1z, v3x, v3y, v3z = eigen3x3_device(sig_xx, sig_yy, sig_zz, sig_xy, sig_yz, sig_xz)
    
    @inbounds begin
        principal_field[1, e] = s11
        principal_field[2, e] = s22
        principal_field[3, e] = s33
        
        dir_max_field[1, e] = v1x
        dir_max_field[2, e] = v1y
        dir_max_field[3, e] = v1z

        dir_min_field[1, e] = v3x
        dir_min_field[2, e] = v3y
        dir_min_field[3, e] = v3z
        
        l1_norm_field[e] = abs(s11) + abs(s22) + abs(s33)
    end
    return nothing
end

"""
    compute_stress_field_cpu(nodes, elements, U, E, nu, density; return_voigt)

CPU fallback implementation using multi-threading.
"""
function compute_stress_field_cpu(nodes::Matrix{Float32}, elements::Matrix{Int}, 
                                   U::AbstractVector, E::Float32, nu::Float32, 
                                   density::Vector{Float32}; return_voigt::Bool=true)
    
    nElem = size(elements, 1) 
    U_f32 = (eltype(U) == Float32) ? U : Float32.(U)

    principal_field          = zeros(Float32, 3, nElem) 
    vonmises_field           = zeros(Float32, nElem) 
    l1_stress_norm_field     = zeros(Float32, nElem)  
    principal_max_dir_field  = zeros(Float32, 3, nElem)
    principal_min_dir_field  = zeros(Float32, 3, nElem)
    full_stress_voigt = return_voigt ? zeros(Float32, 6, nElem) : zeros(Float32, 0, 0)

    Threads.@threads for e in 1:nElem 
        if density[e] > 1e-6
            conn = elements[e, :] 
            element_nodes = nodes[conn, :] 
            
            element_disp = zeros(Float32, 24) 
            for i in 1:8 
                base_idx = 3*(conn[i]-1)
                element_disp[3*(i-1)+1] = U_f32[base_idx+1]
                element_disp[3*(i-1)+2] = U_f32[base_idx+2]
                element_disp[3*(i-1)+3] = U_f32[base_idx+3]
            end 

            E_local = E * density[e] 
            σ = compute_element_stress(element_nodes, element_disp, E_local, nu) 
            
            (principal, vm, max_dir, min_dir) = compute_principal_data(σ) 
        
            l1_norm = abs(principal[1]) + abs(principal[2]) + abs(principal[3]) 
                
            principal_field[:, e] = principal 
            vonmises_field[e]        = vm 
            l1_stress_norm_field[e] = l1_norm  
            principal_max_dir_field[:, e] = max_dir
            principal_min_dir_field[:, e] = min_dir

            if return_voigt
                full_stress_voigt[:, e] .= (σ[1,1], σ[2,2], σ[3,3], σ[1,2], σ[2,3], σ[1,3]) 
            end
        end
    end 
    
    return principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, principal_max_dir_field, principal_min_dir_field
end

"""
    compute_stress_field(nodes, elements, U, E, nu, density; return_voigt)

**HYBRID IMPLEMENTATION**
Automatically selects GPU or CPU based on available VRAM.
"""
function compute_stress_field(nodes::Matrix{Float32}, elements::Matrix{Int}, 
                              U::AbstractVector, E::Float32, nu::Float32, 
                              density::Vector{Float32}; return_voigt::Bool=true) 
    
    nElem = size(elements, 1) 
    nNodes = size(nodes, 1)
    
    # === STEP 1: Memory Check ===
    required_vram = estimate_stress_vram_required(nNodes, nElem, return_voigt)
    
    use_gpu = false
    if CUDA.functional()
        available_vram = CUDA.available_memory()
        vram_threshold = available_vram * 0.75  # 75% safety margin
        
        req_gb = required_vram / 1024^3
        avail_gb = available_vram / 1024^3
        
        if required_vram < vram_threshold
            use_gpu = true
            println(@sprintf("   [Stress] GPU Mode: %.2f GB required / %.2f GB available", req_gb, avail_gb))
        else
            println(@sprintf("   [Stress] CPU Mode: %.2f GB required > %.2f GB available (%.0f%% threshold)", 
                           req_gb, avail_gb, 75.0))
        end
    else
        println("   [Stress] CPU Mode: CUDA not functional")
    end
    
    # === STEP 2: The Fork ===
    if use_gpu
        # === PATH A: Fast GPU ===
        U_f32 = (eltype(U) == Float32) ? U : Float32.(U)
        
        principal_field          = zeros(Float32, 3, nElem) 
        vonmises_field           = zeros(Float32, nElem) 
        l1_stress_norm_field     = zeros(Float32, nElem)  
        principal_max_dir_field  = zeros(Float32, 3, nElem)
        principal_min_dir_field  = zeros(Float32, 3, nElem)
        full_stress_voigt = return_voigt ? zeros(Float32, 6, nElem) : zeros(Float32, 0, 0)
        
        # Transfer to GPU
        nodes_gpu    = CuArray(nodes)
        elements_gpu = CuArray(Int32.(elements)) 
        U_gpu        = CuArray(U_f32)
        density_gpu  = CuArray(density)
        
        principal_gpu = CUDA.zeros(Float32, 3, nElem)
        vonmises_gpu  = CUDA.zeros(Float32, nElem)
        l1_gpu        = CUDA.zeros(Float32, nElem)
        dir_max_gpu   = CUDA.zeros(Float32, 3, nElem)
        dir_min_gpu   = CUDA.zeros(Float32, 3, nElem)
        voigt_gpu     = return_voigt ? CUDA.zeros(Float32, 6, nElem) : CUDA.zeros(Float32, 1, 1)
        
        threads = 256
        blocks = cld(nElem, threads)
        
        @cuda threads=threads blocks=blocks stress_kernel!(
            principal_gpu, vonmises_gpu, voigt_gpu, l1_gpu, dir_max_gpu, dir_min_gpu,
            nodes_gpu, elements_gpu, U_gpu, density_gpu, E, nu, nElem, return_voigt
        )
        CUDA.synchronize()
        
        # Download results
        copyto!(principal_field, principal_gpu)
        copyto!(vonmises_field, vonmises_gpu)
        copyto!(l1_stress_norm_field, l1_gpu)
        copyto!(principal_max_dir_field, dir_max_gpu)
        copyto!(principal_min_dir_field, dir_min_gpu)
        
        if return_voigt
            copyto!(full_stress_voigt, voigt_gpu)
        end
        
        # Free GPU memory
        CUDA.unsafe_free!(nodes_gpu); CUDA.unsafe_free!(elements_gpu)
        CUDA.unsafe_free!(U_gpu); CUDA.unsafe_free!(density_gpu)
        CUDA.unsafe_free!(principal_gpu); CUDA.unsafe_free!(vonmises_gpu)
        CUDA.unsafe_free!(voigt_gpu); CUDA.unsafe_free!(l1_gpu)
        CUDA.unsafe_free!(dir_max_gpu); CUDA.unsafe_free!(dir_min_gpu)
        
        GC.gc(); CUDA.reclaim()
        
        return principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, principal_max_dir_field, principal_min_dir_field
        
    else
        # === PATH B: Robust CPU ===
        return compute_stress_field_cpu(nodes, elements, U, E, nu, density; return_voigt=return_voigt)
    end
end 
 
end