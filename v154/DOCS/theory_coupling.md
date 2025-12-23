# Mathematically Coupled Radius-Threshold Scheduling

## Overview

This document explains the mathematical framework governing the relationship between diffusion radius and culling threshold in HEXA TopOpt, ensuring that no structural features smaller than a specified minimum size are prematurely eliminated.

---

## The Fundamental Constraint

### Physical Basis

When a thin structural feature of width `d` is subjected to diffusion filtering with radius `r_d`, its peak density is reduced approximately as:
```
ρ_max ≈ d / (2·r_d)    [for box filter - 1D theory]
ρ_max ≈ d / (2√2·r_d)  [for Gaussian filter - 3D reality]
```

If this blurred density falls below the culling threshold `τ`, the feature is eliminated—even if it's mechanically critical.

### The Governing Equation

To preserve a feature of minimum width `d_min`, we require:
```
τ(k) · r_d(k) ≤ C · d_min
```

Where:
- `τ(k)` = culling threshold at iteration k
- `r_d(k)` = diffusion radius at iteration k  
- `d_min` = minimum feature size to preserve (in physical units)
- `C` = safety constant accounting for:
  - 3D geometry effects: factor of √2 ≈ 1.41
  - Gaussian kernel shape: factor of √π ≈ 1.77
  - Stress redistribution margin: factor of 1.2
  - **Combined: C ≈ 0.20-0.35**

---

## Implementation Strategy

### 1. Threshold Schedule (Primary Control)

The culling threshold follows a sigmoid evolution:
```julia
t = iter / total_iterations  # Progress [0, 1]

sigmoid = 1 / (1 + exp(-α·(t - β)))
τ(k) = τ_min + (τ_max - τ_min) · sigmoid
```

**Parameters:**
- `τ_min`: Starting threshold (default: 0.10)
- `τ_max`: Final threshold (default: 0.70)
- `α`: Steepness (default: 6.0, range: 4-8)
- `β`: Midpoint (default: 0.6, when τ reaches 50% of max)

### 2. Radius Schedule (Coupled Baseline)

The diffusion radius follows hyperbolic decay:
```julia
r_baseline(k) = r_max · (1 - t^γ) + r_min
```

Where:
- `r_max = 3.0 · d_min` (allows early exploration)
- `r_min = 0.5 · d_min` (preserves final features)
- `γ = 1.8` (decay exponent, range: 1.5-2.5)

### 3. Constraint Enforcement

The final radius is limited by the constraint:
```julia
r_final = min(r_baseline, C·d_min / τ_final)
```

This ensures the product `τ·r` never exceeds the limit.

---

## Configuration Parameters

### In `default.yaml`:
```yaml
optimization_parameters:
  # Feature preservation
  minimum_feature_size_elements: 4.0  # In mesh units
  
  # Threshold schedule
  threshold_min: 0.10
  threshold_max: 0.70
  sigmoid_steepness: 6.0
  sigmoid_midpoint: 0.6
  
  # Radius schedule
  radius_max_multiplier: 3.0
  radius_min_multiplier: 0.5
  radius_decay_exponent: 1.8
  
  # Constraint coupling
  constraint_constant: 0.25  # C value (0.20-0.35)
```

---

## Tuning Guide

### Problem: Thin features disappear prematurely

**Solution:** Decrease `constraint_constant`
- From 0.25 → 0.20 (more conservative)
- This tightens the constraint, reducing both τ and r

### Problem: Convergence too slow

**Solution:** Increase `sigmoid_steepness`
- From 6.0 → 8.0 (faster threshold growth)
- Or decrease `radius_decay_exponent`
- From 1.8 → 1.5 (faster radius decay)

### Problem: Final structure too coarse

**Solution:** Decrease `minimum_feature_size_elements`
- From 4.0 → 3.0 (allows finer features)
- **Warning:** Ensure mesh resolution supports this

### Problem: Early iterations too conservative

**Solution:** Increase `radius_max_multiplier`
- From 3.0 → 4.0 (more material redistribution)
- Or decrease `threshold_min`
- From 0.10 → 0.05 (less aggressive culling)

---

## Diagnostic Interpretation

During optimization, you'll see output like:
```
╔═══════════════════════════════════════════════════════════════╗
║  COUPLED FILTER-THRESHOLD SCHEDULE (Iter 25)
╟───────────────────────────────────────────────────────────────╢
║  Progress (t):              0.625
║  Min Feature Size:          0.3200 (4.0 elements)
╟───────────────────────────────────────────────────────────────╢
║  Radius (baseline):         0.8640
║  Radius (final):            0.8640
╟───────────────────────────────────────────────────────────────╢
║  Threshold (unconstrained): 0.4532
║  Threshold (final):         0.4532
╟───────────────────────────────────────────────────────────────╢
║  Constraint: τ·r            0.3916
║  Constraint Limit:          0.4000
║  Safety Margin:             2.1%
║  Status:                    [TIGHT]
╚═══════════════════════════════════════════════════════════════╝
```

**Key indicators:**
- **[SAFE]**: Margin > 15% - plenty of headroom
- **[TIGHT]**: Margin 5-15% - working as designed
- **[CRITICAL]**: Margin < 5% - constraint active, limiting schedule
- **[VIOLATION]**: Margin < 0% - ERROR, should never occur
- **[CONSTRAINED]**: Shows which parameter was limited

---

## Mathematical Validation

The constraint is verified at every iteration:
```julia
product = τ_final * r_final
if product > (C * d_min) * 1.01  # 1% tolerance
    @warn "Constraint violation!"
    # Force conservative fallback
end
```

You can also enable detailed feature tracking (see advanced section in main docs) to verify that features ≥ `d_min` survive through all iterations.

---

## References

- **Theoretical basis:** 1D convolution theory, extended to 3D via dimensional analysis
- **Gaussian filter scaling:** Error function approximation for thin features
- **Coupling strategy:** Dual-parameter schedule with constraint projection

---

**Last Updated:** 2025-01-20  
**Version:** 1.0.0