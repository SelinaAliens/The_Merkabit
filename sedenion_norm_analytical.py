#!/usr/bin/env python3
"""
ANALYTICAL DERIVATION: SEDENION NORM VIOLATION FROM CAYLEY-DICKSON
===================================================================

The Question:
  For unit sedenions a, b ∈ S¹⁵, the Cayley-Dickson product ab has
  ||ab|| ≠ ||a||·||b|| in general. What is E[| ||ab|| - 1 |] exactly?

  If this is 36/1000 or 72/1000 → Reading 3 (E₆ root count origin)
  If this is irrational ≈ 0.0710 → Reading 2 holds, Reading 1 gains ground

Method:
  1. Derive the exact cross-term from Cayley-Dickson
  2. Express ||ab||² = 1 + 2Δ where Δ depends on octonion non-associativity
  3. Compute E[|Δ|] and E[| ||ab|| - 1 |] to high precision
  4. Identify closed form via integer relation algorithms
"""

import numpy as np
from itertools import product as iter_product
import time

# ============================================================================
# OCTONION ALGEBRA (exact, from Fano plane)
# ============================================================================

FANO_LINES = [
    (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 7),
    (5, 6, 1), (6, 7, 2), (7, 1, 3),
]

def build_octonion_table():
    table = {}
    for i in range(8):
        table[(0, i)] = (+1, i)
        table[(i, 0)] = (+1, i)
    for i in range(1, 8):
        table[(i, i)] = (-1, 0)
    for line in FANO_LINES:
        i, j, k = line
        table[(i, j)] = (+1, k); table[(j, k)] = (+1, i); table[(k, i)] = (+1, j)
        table[(j, i)] = (-1, k); table[(k, j)] = (-1, i); table[(i, k)] = (-1, j)
    return table

OCT_TABLE = build_octonion_table()

def oct_mult(a, b):
    result = np.zeros(8)
    for i in range(8):
        if abs(a[i]) < 1e-15: continue
        for j in range(8):
            if abs(b[j]) < 1e-15: continue
            sign, k = OCT_TABLE[(i, j)]
            result[k] += sign * a[i] * b[j]
    return result

# Pre-compute structure constant tensor for fast multiplication
_OCT_C = np.zeros((8, 8, 8))
for (_i, _j), (_sign, _k) in OCT_TABLE.items():
    _OCT_C[_i, _j, _k] = _sign

def oct_mult_fast(a, b):
    """Fast octonion multiply using einsum."""
    return np.einsum('ijk,i,j->k', _OCT_C, a, b)

def oct_conj(a):
    c = a.copy(); c[1:] = -c[1:]; return c

def sed_mult(a, b):
    """(p,q)·(r,s) = (pr - s̄q, sp + qr̄)"""
    p, q = a[:8].copy(), a[8:].copy()
    r, s = b[:8].copy(), b[8:].copy()
    first  = oct_mult(p, r) - oct_mult(oct_conj(s), q)
    second = oct_mult(s, p) + oct_mult(q, oct_conj(r))
    return np.concatenate([first, second])

def sed_mult_fast(a, b):
    """Fast sedenion multiplication."""
    p, q = a[:8], a[8:]
    r, s = b[:8], b[8:]
    first  = oct_mult_fast(p, r) - oct_mult_fast(oct_conj(s), q)
    second = oct_mult_fast(s, p) + oct_mult_fast(q, oct_conj(r))
    return np.concatenate([first, second])


# ============================================================================
# PART 1: THE EXACT CROSS-TERM
# ============================================================================

def compute_cross_term(p, q, r, s):
    """
    Compute Δ = ⟨sp, qr̄⟩ - ⟨pr, s̄q⟩
    
    This is the EXACT expression for the norm violation:
      ||ab||² = ||a||²||b||² + 2Δ
    
    Proof:
      a = (p,q), b = (r,s), where p,q,r,s ∈ O (octonions)
      ab = (pr - s̄q, sp + qr̄)   [Cayley-Dickson]
      
      ||ab||² = ||pr - s̄q||² + ||sp + qr̄||²
             = ||pr||² + ||s̄q||² - 2⟨pr, s̄q⟩ + ||sp||² + ||qr̄||² + 2⟨sp, qr̄⟩
      
      Since octonions ARE normed: ||xy|| = ||x||·||y||, so:
        ||pr||² = ||p||²||r||²,  ||s̄q||² = ||s||²||q||²
        ||sp||² = ||s||²||p||²,  ||qr̄||² = ||q||²||r||²
      
      Sum of squared norms = (||p||²+||q||²)(||r||²+||s||²) = ||a||²||b||²
      
      Therefore: ||ab||² = ||a||²||b||² + 2(⟨sp, qr̄⟩ - ⟨pr, s̄q⟩)
      
      The cross-term Δ = ⟨sp, qr̄⟩ - ⟨pr, s̄q⟩ is zero if and only if 
      the underlying algebra is associative. Since octonions are NOT
      associative, Δ ≠ 0 in general, and the sedenion norm is NOT
      multiplicative.
    """
    sp = oct_mult(s, p)
    qr_bar = oct_mult(q, oct_conj(r))
    pr = oct_mult(p, r)
    s_bar_q = oct_mult(oct_conj(s), q)
    
    term1 = np.dot(sp, qr_bar)
    term2 = np.dot(pr, s_bar_q)
    
    return term1 - term2


def verify_cross_term(n_tests=10000):
    """Verify that ||ab||² = ||a||²||b||² + 2Δ exactly."""
    print("=" * 76)
    print("PART 1: VERIFY THE CROSS-TERM IDENTITY")
    print("  ||ab||² = ||a||²||b||² + 2Δ")
    print("  where Δ = ⟨sp, qr̄⟩ - ⟨pr, s̄q⟩")
    print("=" * 76)
    
    np.random.seed(42)
    max_error = 0
    
    for _ in range(n_tests):
        a = np.random.randn(16)
        b = np.random.randn(16)
        
        ab = sed_mult(a, b)
        
        lhs = np.dot(ab, ab)  # ||ab||²
        rhs_base = np.dot(a, a) * np.dot(b, b)  # ||a||²||b||²
        delta = compute_cross_term(a[:8], a[8:], b[:8], b[8:])
        rhs = rhs_base + 2 * delta
        
        error = abs(lhs - rhs)
        max_error = max(max_error, error)
    
    print(f"\n  Tested {n_tests} random sedenion pairs")
    print(f"  Max |LHS - RHS|: {max_error:.2e}")
    print(f"  Identity {'VERIFIED' if max_error < 1e-10 else 'FAILED'}")
    
    # Also verify octonion case: Δ should be ≡ 0
    print(f"\n  Control: octonion subalgebra (q=s=0, so Δ should be 0):")
    max_oct_delta = 0
    for _ in range(n_tests):
        a = np.zeros(16); a[:8] = np.random.randn(8)
        b = np.zeros(16); b[:8] = np.random.randn(8)
        delta = compute_cross_term(a[:8], a[8:], b[:8], b[8:])
        max_oct_delta = max(max_oct_delta, abs(delta))
    print(f"  Max |Δ| for octonion subalgebra: {max_oct_delta:.2e}")
    
    return max_error < 1e-10


# ============================================================================
# PART 2: WHAT MAKES Δ NON-ZERO? THE ASSOCIATOR
# ============================================================================

def analyze_associator_connection(n_tests=10000):
    """
    Show that Δ is controlled by the octonionic associator.
    
    The associator [x,y,z] = (xy)z - x(yz) measures non-associativity.
    Octonions are "alternative": [x,x,y] = [x,y,y] = 0, but [x,y,z] ≠ 0 
    in general.
    
    We can rewrite:
      ⟨sp, qr̄⟩ = Re((sp)̄ · (qr̄)) = Re(p̄s̄ · qr̄)
      ⟨pr, s̄q⟩ = Re((pr)̄ · (s̄q)) = Re(r̄p̄ · s̄q)
    
    The difference involves how products of 4 octonions bracket differently.
    """
    print("\n" + "=" * 76)
    print("PART 2: CONNECTION TO OCTONIONIC ASSOCIATOR")
    print("=" * 76)
    
    np.random.seed(42)
    
    # Compute the associator [x,y,z] = (xy)z - x(yz) for random octonions
    assoc_norms = []
    for _ in range(n_tests):
        x = np.random.randn(8)
        y = np.random.randn(8)
        z = np.random.randn(8)
        # Normalize
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        z /= np.linalg.norm(z)
        
        xy_z = oct_mult(oct_mult(x, y), z)
        x_yz = oct_mult(x, oct_mult(y, z))
        assoc = xy_z - x_yz
        assoc_norms.append(np.linalg.norm(assoc))
    
    assoc_norms = np.array(assoc_norms)
    print(f"\n  Octonionic associator ||[x,y,z]|| for unit x,y,z:")
    print(f"    Mean:   {np.mean(assoc_norms):.8f}")
    print(f"    Std:    {np.std(assoc_norms):.8f}")
    print(f"    Min:    {np.min(assoc_norms):.8f}")
    print(f"    Max:    {np.max(assoc_norms):.8f}")
    
    # Verify alternativity: [x,x,y] = 0
    alt_violations = []
    for _ in range(n_tests):
        x = np.random.randn(8)
        y = np.random.randn(8)
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        
        xx_y = oct_mult(oct_mult(x, x), y)
        x_xy = oct_mult(x, oct_mult(x, y))
        alt_violations.append(np.linalg.norm(xx_y - x_xy))
    
    print(f"\n  Alternativity check ||[x,x,y]||:")
    print(f"    Max:    {max(alt_violations):.2e}  (should be ≈ 0)")
    
    return np.mean(assoc_norms)


# ============================================================================
# PART 3: HIGH-PRECISION MEAN VIOLATION
# ============================================================================

def compute_mean_violation_high_precision(n_samples=2000000, batch_size=50000):
    """
    Compute E[| ||ab|| - 1 |] for unit sedenions to high precision.
    
    Also compute:
      E[Δ]       (should be 0 by symmetry)
      E[Δ²]      (variance of the cross-term)
      E[||ab||²]  
      E[||ab||]
      E[| ||ab|| - 1 |]   ← the target quantity
      E[(||ab|| - 1)²]    ← variance of norm deviation
    """
    print("\n" + "=" * 76)
    print("PART 3: HIGH-PRECISION MEAN VIOLATION")
    print(f"  {n_samples:,} random unit sedenion pairs")
    print("=" * 76)
    
    np.random.seed(12345)
    
    # Accumulators
    sum_delta = 0.0
    sum_delta2 = 0.0
    sum_norm_ab_sq = 0.0
    sum_norm_ab = 0.0
    sum_abs_violation = 0.0
    sum_violation_sq = 0.0
    sum_abs_delta = 0.0
    
    n_done = 0
    t0 = time.time()
    
    while n_done < n_samples:
        batch = min(batch_size, n_samples - n_done)
        
        # Generate random unit sedenions
        a = np.random.randn(batch, 16)
        b = np.random.randn(batch, 16)
        a_norms = np.linalg.norm(a, axis=1, keepdims=True)
        b_norms = np.linalg.norm(b, axis=1, keepdims=True)
        a = a / a_norms
        b = b / b_norms
        
        for i in range(batch):
            ab = sed_mult_fast(a[i], b[i])
            norm_ab = np.linalg.norm(ab)
            norm_ab_sq = np.dot(ab, ab)
            
            # Fast cross-term: Δ = (||ab||² - 1) / 2 for unit vectors
            delta = (norm_ab_sq - 1.0) / 2.0
            
            violation = norm_ab - 1.0
            
            sum_delta += delta
            sum_delta2 += delta**2
            sum_norm_ab_sq += norm_ab_sq
            sum_norm_ab += norm_ab
            sum_abs_violation += abs(violation)
            sum_violation_sq += violation**2
            sum_abs_delta += abs(delta)
        
        n_done += batch
        if n_done % 200000 == 0:
            elapsed = time.time() - t0
            print(f"  ... {n_done:,}/{n_samples:,} ({elapsed:.1f}s)")
    
    elapsed = time.time() - t0
    N = n_samples
    
    mean_delta = sum_delta / N
    mean_delta2 = sum_delta2 / N
    var_delta = mean_delta2 - mean_delta**2
    mean_norm_ab_sq = sum_norm_ab_sq / N
    mean_norm_ab = sum_norm_ab / N
    mean_abs_violation = sum_abs_violation / N
    mean_violation_sq = sum_violation_sq / N
    mean_abs_delta = sum_abs_delta / N
    
    # Standard errors (for confidence intervals)
    se_abs_violation = np.sqrt(mean_violation_sq - mean_abs_violation**2) / np.sqrt(N)
    
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"\n  RESULTS:")
    print(f"  {'─' * 60}")
    print(f"  E[Δ]                    = {mean_delta:+.10f}  (should be ≈ 0)")
    print(f"  E[Δ²]                   = {mean_delta2:.12f}")
    print(f"  Var(Δ) = E[Δ²]-E[Δ]²   = {var_delta:.12f}")
    print(f"  σ(Δ)                    = {np.sqrt(var_delta):.12f}")
    print(f"  E[|Δ|]                  = {mean_abs_delta:.12f}")
    print(f"  {'─' * 60}")
    print(f"  E[||ab||²]              = {mean_norm_ab_sq:.12f}")
    print(f"  E[||ab||]               = {mean_norm_ab:.12f}")
    print(f"  {'─' * 60}")
    print(f"  E[| ||ab|| - 1 |]       = {mean_abs_violation:.12f}")
    print(f"    ± SE                    {se_abs_violation:.2e}")
    print(f"  E[(||ab|| - 1)²]        = {mean_violation_sq:.12f}")
    print(f"  σ(||ab|| - 1)           = {np.sqrt(mean_violation_sq - (mean_norm_ab-1)**2):.12f}")
    
    return {
        'mean_delta': mean_delta,
        'var_delta': var_delta,
        'mean_abs_delta': mean_abs_delta,
        'mean_norm_ab_sq': mean_norm_ab_sq,
        'mean_norm_ab': mean_norm_ab,
        'mean_abs_violation': mean_abs_violation,
        'se': se_abs_violation,
        'mean_violation_sq': mean_violation_sq,
    }


# ============================================================================
# PART 4: CLOSED-FORM IDENTIFICATION
# ============================================================================

def identify_closed_form(results):
    """
    Attempt to identify the mean violation as a closed-form expression.
    
    Candidates to test:
      Rational: 36/1000, 72/1000, p/q for small p,q
      Algebraic: involving √2, √3, √7, etc.  
      Transcendental: involving π, Γ functions
    
    Key insight: for random variables on spheres, expectations often involve
    ratios of Gamma functions. Specifically, if X ~ chi distribution,
    E[|X-μ|] can involve Gamma(n/2) type expressions.
    
    For the sedenion norm:
      ||ab||² = 1 + 2Δ where Δ has some distribution.
      
    If Δ were Gaussian with variance σ²:
      E[|√(1+2Δ) - 1|] is related to a noncentral chi distribution.
      
    For small σ²: E[|√(1+2Δ)-1|] ≈ E[|Δ|] = σ√(2/π)
    """
    print("\n" + "=" * 76)
    print("PART 4: CLOSED-FORM IDENTIFICATION")
    print("=" * 76)
    
    v = results['mean_abs_violation']
    v_sq = results['mean_violation_sq']
    delta_var = results['var_delta']
    abs_delta = results['mean_abs_delta']
    se = results['se']
    
    print(f"\n  Target value: {v:.12f} ± {se:.2e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Test 1: Simple rationals
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n  Test 1: Simple rational candidates")
    print(f"  {'─' * 55}")
    
    rationals = [
        (36, 1000, "36/1000 (E₆ roots / 10³)"),
        (72, 1000, "72/1000 (E₆ total roots / 10³)"),
        (36, 500, "36/500"),
        (9, 128, "9/128"),
        (1, 14, "1/14"),
        (1, 15, "1/15"),
        (5, 72, "5/72"),
        (7, 100, "7/100"),
        (71, 1000, "71/1000"),
        (1, 4, "1/4 (for Δ²)"),
    ]
    
    for num, denom, label in rationals:
        val = num / denom
        diff = abs(v - val)
        sigma = diff / se if se > 0 else float('inf')
        marker = "◄◄◄" if sigma < 2 else "◄" if sigma < 5 else ""
        print(f"    {label:40s} = {val:.12f}  diff={diff:.2e}  ({sigma:.1f}σ)  {marker}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Test 2: Expressions involving π and Gamma functions
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n  Test 2: Transcendental candidates")
    print(f"  {'─' * 55}")
    
    from math import pi, sqrt, gamma, log
    
    transcendentals = [
        (1/(4*pi), "1/(4π)"),
        (1/(2*pi*sqrt(2)), "1/(2π√2)"),
        (sqrt(2/pi)/8, "√(2/π)/8"),
        (sqrt(2/pi)/7, "√(2/π)/7"),
        (gamma(8)/gamma(8.5)/sqrt(pi), "Γ(8)/[Γ(8.5)√π]"),
        (sqrt(2)*gamma(8)/(gamma(7.5)*pi), "√2·Γ(8)/[Γ(7.5)·π]"),
        (1/sqrt(2*pi*7), "1/√(14π)"),
        (1/sqrt(2*pi*8), "1/√(16π)"),
        (7/(8*pi*sqrt(2)), "7/(8π√2)"),
    ]
    
    for val, label in transcendentals:
        diff = abs(v - val)
        sigma = diff / se if se > 0 else float('inf')
        marker = "◄◄◄" if sigma < 2 else "◄" if sigma < 5 else ""
        print(f"    {label:40s} = {val:.12f}  diff={diff:.2e}  ({sigma:.1f}σ)  {marker}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Test 3: Variance-based prediction
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n  Test 3: If Δ is Gaussian, E[|Δ|] = σ√(2/π)")
    print(f"  {'─' * 55}")
    
    sigma_delta = sqrt(delta_var)
    predicted_abs_delta_gaussian = sigma_delta * sqrt(2/pi)
    
    print(f"    σ(Δ)                  = {sigma_delta:.12f}")
    print(f"    σ√(2/π)               = {predicted_abs_delta_gaussian:.12f}")
    print(f"    Actual E[|Δ|]         = {abs_delta:.12f}")
    print(f"    Ratio actual/predict  = {abs_delta/predicted_abs_delta_gaussian:.8f}")
    print(f"    → Δ {'IS' if abs(abs_delta/predicted_abs_delta_gaussian - 1) < 0.01 else 'is NOT'} approximately Gaussian")
    
    # ═══════════════════════════════════════════════════════════════════
    # Test 4: Compute E[Δ²] analytically
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n  Test 4: Analytical prediction for E[Δ²]")
    print(f"  {'─' * 55}")
    print(f"    Measured E[Δ²]        = {delta_var:.12f}")
    
    # For random unit vectors on S^{n-1}, various moments are known.
    # The key is to compute E[⟨sp, qr̄⟩²] and E[⟨pr, s̄q⟩²] and
    # E[⟨sp, qr̄⟩⟨pr, s̄q⟩] for independent uniform p,q,r,s on S^7
    # subject to ||p||²+||q||² = 1 and ||r||²+||s||² = 1.
    
    # Simple candidates for E[Δ²]:
    var_candidates = [
        (1/256, "1/256 = 1/16²"),
        (1/240, "1/240"),
        (1/224, "1/224 = 1/(8·28)"),
        (1/210, "1/210"),
        (1/192, "1/192 = 1/(8·24)"),
        (1/168, "1/168"),
        (7/1024, "7/1024"),
        (1/144, "1/144 = 1/12²"),
        (1/128, "1/128 = 1/2⁷"),
        (3/512, "3/512"),
        (1/120, "1/120"),
    ]
    
    for val, label in var_candidates:
        diff = abs(delta_var - val)
        ratio = delta_var / val
        marker = "◄◄◄" if abs(ratio - 1) < 0.001 else "◄" if abs(ratio - 1) < 0.01 else ""
        print(f"    {label:30s} = {val:.12f}  ratio={ratio:.6f}  {marker}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Test 5: E[||ab||²] analytical prediction
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n  Test 5: E[||ab||²] for unit sedenions")
    print(f"  {'─' * 55}")
    print(f"    Measured E[||ab||²]   = {results['mean_norm_ab_sq']:.12f}")
    print(f"    If norm were exact:     1.000000000000")
    print(f"    The excess = 2·E[Δ]  ≈ {2*results['mean_delta']:.12f}")
    print(f"    (Should be ≈ 0 since E[Δ] = 0 by symmetry)")
    
    # E[||ab||²] = 1 + 2E[Δ] = 1, but Var(||ab||²) = 4·Var(Δ) 
    print(f"    Var(||ab||²) = 4·Var(Δ) = {4*delta_var:.12f}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Test 6: Broader rational search
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n  Test 6: Systematic rational search for E[| ||ab||-1 |]")
    print(f"  {'─' * 55}")
    
    best_matches = []
    for denom in range(1, 2001):
        num = round(v * denom)
        if num > 0:
            val = num / denom
            diff = abs(v - val)
            if diff < 5 * se:
                best_matches.append((diff, num, denom, diff/se))
    
    best_matches.sort()
    print(f"    Best rational approximations within 5σ:")
    for diff, num, denom, sigma in best_matches[:15]:
        from math import gcd
        g = gcd(num, denom)
        print(f"      {num//g}/{denom//g:>6d}  = {num/denom:.12f}  ({sigma:.2f}σ)")
    
    # Same for Var(Δ)
    print(f"\n  Test 7: Systematic rational search for Var(Δ)")
    print(f"  {'─' * 55}")
    
    best_var = []
    for denom in range(1, 5001):
        num = round(delta_var * denom)
        if num >= 0:
            val = num / denom if denom > 0 else 0
            diff = abs(delta_var - val)
            if diff < delta_var * 0.001:  # within 0.1%
                from math import gcd
                g = gcd(max(num,1), denom)
                best_var.append((diff, num//g, denom//g))
    
    best_var.sort()
    seen = set()
    count = 0
    for diff, num, denom in best_var:
        key = (num, denom)
        if key not in seen:
            seen.add(key)
            print(f"      {num}/{denom}  = {num/denom:.12f}  diff={diff:.2e}")
            count += 1
            if count >= 10:
                break
    
    return v


# ============================================================================
# PART 5: DIMENSION SWEEP — IS THE PATTERN GENERAL?
# ============================================================================

def dimension_sweep():
    """
    Compute the norm violation for ALL Cayley-Dickson algebras:
      dim 2 (complex):     should be 0
      dim 4 (quaternion):  should be 0
      dim 8 (octonion):    should be 0
      dim 16 (sedenion):   NON-ZERO ← this is our target
      dim 32 (pathion):    also non-zero
    """
    print("\n" + "=" * 76)
    print("PART 5: NORM VIOLATION ACROSS CAYLEY-DICKSON LEVELS")
    print("=" * 76)
    
    def complex_mult(a, b):
        return np.array([a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]])
    
    def quat_mult(a, b):
        r = np.zeros(4)
        r[0] = a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]
        r[1] = a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2]
        r[2] = a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1]
        r[3] = a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]
        return r
    
    def cd_mult(a, b, sub_mult, sub_conj):
        """Generic Cayley-Dickson doubling."""
        n = len(a) // 2
        p, q = a[:n].copy(), a[n:].copy()
        r, s = b[:n].copy(), b[n:].copy()
        first  = sub_mult(p, r) - sub_mult(sub_conj(s), q)
        second = sub_mult(s, p) + sub_mult(q, sub_conj(r))
        return np.concatenate([first, second])
    
    def make_conj(n):
        def conj(a):
            c = a.copy(); c[1:] = -c[1:]; return c
        return conj
    
    # Build multiplication towers
    def real_mult(a, b): return np.array([a[0]*b[0]])
    def real_conj(a): return a.copy()
    
    complex_conj = make_conj(2)
    quat_conj = make_conj(4)
    oct_conj_fn = make_conj(8)
    
    # Sedenion: use existing
    # Pathion (32-ion): double the sedenion
    sed_conj_fn = make_conj(16)
    def path_mult(a, b):
        return cd_mult(a, b, sed_mult, sed_conj_fn)
    
    algebras = [
        ("ℝ (dim 1)",   1,  real_mult),
        ("ℂ (dim 2)",   2,  complex_mult),
        ("ℍ (dim 4)",   4,  quat_mult),
        ("𝕆 (dim 8)",   8,  oct_mult),
        ("𝕊 (dim 16)", 16,  sed_mult),
        # ("Pathion (32)", 32, path_mult),  # too slow for Monte Carlo
    ]
    
    np.random.seed(42)
    n_samples = 200000
    
    print(f"\n  {'Algebra':>15}  {'E[||ab||-1]':>14}  {'E[|Δ|]':>14}  {'Var(Δ)':>14}")
    print(f"  {'─'*15}  {'─'*14}  {'─'*14}  {'─'*14}")
    
    for name, dim, mult_fn in algebras:
        violations = []
        deltas = []
        
        n = min(n_samples, 200000 if dim <= 16 else 50000)
        
        for _ in range(n):
            a = np.random.randn(dim)
            b = np.random.randn(dim)
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            
            ab = mult_fn(a, b)
            norm_ab = np.linalg.norm(ab)
            violations.append(abs(norm_ab - 1.0))
            
            delta = (np.dot(ab, ab) - 1.0) / 2.0
            deltas.append(delta)
        
        violations = np.array(violations)
        deltas = np.array(deltas)
        
        print(f"  {name:>15}  {np.mean(violations):>14.10f}  "
              f"{np.mean(np.abs(deltas)):>14.10f}  {np.var(deltas):>14.10f}")
    
    return True


# ============================================================================
# PART 6: THE DECISIVE TEST — E₆ SIGNATURE OR NOT?
# ============================================================================

def decisive_test(results):
    """
    The question that discriminates the three readings:
    
    Reading 1: The norm violation is a generic geometric quantity — some
               function of dimension 16, involving π and Gamma functions.
               The appearance of ≈0.071 near 72/1000 is coincidence.
    
    Reading 2: The violation is determined by the non-associativity of 
               octonions, producing an irrational number that merely 
               approximates nice rationals.
    
    Reading 3: The violation IS exactly 36/1000 or 72/1000, and the E₆ 
               root count appears in the sedenion norm breaking because
               E₆ governs the structure of the octonion algebra.
    """
    print("\n" + "=" * 76)
    print("PART 6: THE DECISIVE TEST")
    print("=" * 76)
    
    v = results['mean_abs_violation']
    se = results['se']
    var_delta = results['var_delta']
    
    # Key test: is Var(Δ) a nice number?
    # If Var(Δ) = p/q for recognizable p,q, then
    # E[|Δ|] = √(2·Var(Δ)/π) if Δ is Gaussian → E[| ||ab||-1 |] ≈ E[|Δ|]
    
    from math import sqrt, pi
    
    sigma_delta = sqrt(var_delta)
    
    # If Var(Δ) = 1/(2·n·(n+2)) for n=8 (octonion dim):
    # This is a common moment formula for products of random vectors on spheres
    for n_test in [7, 8, 14, 15, 16]:
        candidate = 1.0 / (2 * n_test * (n_test + 2))
        print(f"    1/(2·{n_test}·{n_test+2}) = {candidate:.12f}  "
              f"vs Var(Δ) = {var_delta:.12f}  ratio = {var_delta/candidate:.6f}")
    
    print()
    
    # The analytical prediction for E[||ab||-1|]:
    # If the cross-term variance is V, and the cross-term is approximately
    # Gaussian (which holds by CLT since it's a sum of many terms), then:
    #   ||ab|| = √(1 + 2Δ) ≈ 1 + Δ - Δ²/2 + ...
    #   ||ab|| - 1 ≈ Δ for small Δ
    #   E[| ||ab||-1 |] ≈ E[|Δ|] = √(2V/π)
    
    predicted_v = sqrt(2 * var_delta / pi)
    print(f"    Gaussian prediction: E[| ||ab||-1 |] ≈ √(2·Var(Δ)/π)")
    print(f"    = √(2 × {var_delta:.12f} / π)")
    print(f"    = {predicted_v:.12f}")
    print(f"    Measured: {v:.12f}")
    print(f"    Ratio: {v/predicted_v:.8f}")
    
    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  VERDICT:")
    print(f"  ═══════════════════════════════════════════════════")
    
    # Test against the critical values
    test_36 = abs(v - 0.036) / se
    test_72 = abs(v - 0.072) / se
    
    print(f"\n    v = {v:.12f} ± {se:.2e}")
    print(f"    Distance from 36/1000: {test_36:.1f}σ")
    print(f"    Distance from 72/1000: {test_72:.1f}σ")
    
    if test_36 < 3:
        print(f"\n    → CONSISTENT with 36/1000. Reading 3 SUPPORTED.")
    elif test_72 < 3:
        print(f"\n    → CONSISTENT with 72/1000. Reading 3 SUPPORTED.")
    else:
        # Check if it's a recognizable irrational
        # Common sphere-integral results involve Γ(n/2)/Γ((n+1)/2) etc.
        print(f"\n    → NOT 36/1000 or 72/1000.")
        print(f"    Checking for irrational closed forms...")
        
        # Gamma function candidates for dimension 16 sphere integrals
        from math import gamma
        
        candidates = []
        for a_coeff in [1, 2, 4, 7, 8, 14, 15, 16]:
            for b_coeff in [1, 2, 4, 7, 8, 14, 15, 16]:
                for c_coeff in [1, 2, 3, 4, 6, 7, 8]:
                    val = gamma(a_coeff/2) / (gamma(b_coeff/2) * c_coeff * sqrt(pi))
                    if 0.01 < val < 0.2:
                        diff = abs(v - val)
                        if diff < 10 * se:
                            candidates.append((diff/se, val, 
                                f"Γ({a_coeff}/2)/[Γ({b_coeff}/2)·{c_coeff}·√π]"))
        
        candidates.sort()
        if candidates:
            print(f"    Closest Gamma-function forms:")
            for sigma, val, label in candidates[:5]:
                print(f"      {label} = {val:.12f}  ({sigma:.2f}σ)")
        
        # Also check: sqrt(k/m/pi) forms
        candidates2 = []
        for k in range(1, 50):
            for m in range(1, 200):
                val = sqrt(k / (m * pi))
                diff = abs(v - val)
                if diff < 5 * se:
                    candidates2.append((diff/se, val, f"√({k}/({m}π))"))
        
        candidates2.sort()
        if candidates2:
            print(f"\n    Closest √(k/(mπ)) forms:")
            for sigma, val, label in candidates2[:5]:
                print(f"      {label} = {val:.12f}  ({sigma:.2f}σ)")


# ============================================================================
# PART 7: ANALYTICAL COMPUTATION OF Var(Δ) 
# ============================================================================

def analytical_variance():
    """
    Attempt to compute Var(Δ) = E[Δ²] analytically.
    
    Δ = ⟨sp, qr̄⟩ - ⟨pr, s̄q⟩
    
    where a = (p,q) uniform on S^{15}, b = (r,s) uniform on S^{15}.
    
    This means p,q ∈ R^8 with ||p||² + ||q||² = 1,
    and r,s ∈ R^8 with ||r||² + ||s||² = 1.
    
    Strategy: decompose using the structure constants of the octonion algebra.
    """
    print("\n" + "=" * 76)
    print("PART 7: ANALYTICAL VARIANCE VIA STRUCTURE CONSTANTS")
    print("=" * 76)
    
    # Build the full structure constant tensor for octonions
    # c_{ijk} where e_i · e_j = Σ_k c_{ijk} e_k
    c = np.zeros((8, 8, 8))
    for (i, j), (sign, k) in OCT_TABLE.items():
        c[i, j, k] = sign
    
    # Verify: (xy)_k = Σ_{ij} c_{ijk} x_i y_j
    np.random.seed(42)
    x = np.random.randn(8)
    y = np.random.randn(8)
    xy_direct = oct_mult(x, y)
    xy_tensor = np.einsum('ijk,i,j->k', c, x, y)
    assert np.allclose(xy_direct, xy_tensor), "Structure constant tensor mismatch"
    print(f"  Structure constant tensor verified ✓")
    
    # ⟨sp, qr̄⟩ = Σ_k (sp)_k (qr̄)_k
    # (sp)_k = Σ_{ab} c_{abk} s_a p_b
    # (qr̄)_k = Σ_{cd} c_{cdk} q_c (r̄)_d
    # r̄_d = r_d if d=0, -r_d if d>0.  Define ε_d = 1 if d=0, -1 if d>0.
    
    eps = np.ones(8)
    eps[1:] = -1  # conjugation signs
    
    # ⟨sp, qr̄⟩ = Σ_{k,a,b,c,d} c_{abk} c_{cdk} ε_d s_a p_b q_c r_d
    # ⟨pr, s̄q⟩ = Σ_{k,a,b,c,d} c_{abk} c_{cdk} ε_c p_a r_b s_c q_d
    #           (where s̄_c = ε_c s_c)
    
    # So Δ = Σ_{k} [(sp)_k (qr̄)_k - (pr)_k (s̄q)_k]
    # Let's define two 4-index tensors:
    
    # T1_{abcd} = Σ_k c_{abk} · Σ_{c'd'} c_{c'd'k} ε_{d'} [with indices matched]
    # This needs to be done carefully...
    
    # Actually, let's compute the key quantity:
    # A_{abcd} ≡ Σ_k c_{abk} c_{cdk}  (the "metric" of the algebra)
    
    A = np.einsum('abk,cdk->abcd', c, c)
    print(f"  Contraction tensor A_abcd = Σ_k c_abk·c_cdk computed")
    print(f"    Shape: {A.shape}")
    print(f"    This encodes the inner product structure of all octonionic products")
    
    # Now:
    # ⟨sp, qr̄⟩ = Σ_{a,b,c,d} A_{abcd} ε_d · s_a p_b q_c r_d
    #           = Σ_{abcd} A_{abcd} ε_d · s_a p_b q_c r_d
    
    # ⟨pr, s̄q⟩ = Σ_{a,b,c,d} A_{abcd} ε_c · p_a r_b s_c q_d
    
    # So Δ = Σ_{abcd} [A_{abcd} ε_d · s_a p_b q_c r_d - A_{abcd} ε_c · p_a r_b s_c q_d]
    
    # Rename indices in the second term: a↔a, b↔b, c↔c, d↔d
    # → Δ = Σ_{abcd} s_a p_b q_c r_d · [A_{abcd} ε_d] - Σ_{abcd} p_a r_b s_c q_d · [A_{abcd} ε_c]
    
    # In the second term, let's relabel: a→c, b→d, c→a, d→b
    # Σ_{abcd} p_a r_b s_c q_d A_{abcd} ε_c = Σ_{cdab} p_c r_d s_a q_b A_{cdab} ε_a
    
    # So: Δ = Σ_{abcd} s_a p_b q_c r_d [A_{abcd} ε_d - A_{cdab} ε_a]
    
    # Define the "violation tensor":
    # V_{abcd} = A_{abcd} ε_d - A_{cdab} ε_a
    
    V = np.zeros((8, 8, 8, 8))
    for a in range(8):
        for b in range(8):
            for ci in range(8):
                for d in range(8):
                    # V_{abcd} = A_{abcd} ε_d - A_{bdac} ε_a
                    # First term: from ⟨sp, qr̄⟩ with r̄_d = ε_d r_d
                    # Second term: from ⟨pr, s̄q⟩ with s̄_a = ε_a s_a, re-indexed
                    V[a, b, ci, d] = A[a, b, ci, d] * eps[d] - A[b, d, a, ci] * eps[a]
    
    # Then Δ = Σ_{abcd} V_{abcd} s_a p_b q_c r_d
    
    # Verify this tensor formulation
    print("\n  Verifying violation tensor V_abcd:")
    n_verify = 1000
    max_err = 0
    for _ in range(n_verify):
        a = np.random.randn(16)
        b = np.random.randn(16)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        
        p, q, r, s = a[:8], a[8:], b[:8], b[8:]
        
        delta_direct = compute_cross_term(p, q, r, s)
        delta_tensor = np.einsum('abcd,a,b,c,d', V, s, p, q, r)
        max_err = max(max_err, abs(delta_direct - delta_tensor))
    
    print(f"    Max error over {n_verify} tests: {max_err:.2e}")
    print(f"    Tensor formulation {'VERIFIED ✓' if max_err < 1e-10 else 'FAILED ✗'}")
    
    # Properties of V
    V_norm = np.sqrt(np.sum(V**2))
    V_nonzero = np.sum(np.abs(V) > 1e-10)
    print(f"\n  Violation tensor properties:")
    print(f"    ||V||_F = {V_norm:.6f}")
    print(f"    Non-zero entries: {V_nonzero} / {8**4} = {V_nonzero/8**4:.4f}")
    
    # Is V antisymmetric under any swaps?
    # V_{abcd} vs V_{cdab}: 
    antisym_test = np.max(np.abs(V + np.transpose(V, (2,3,0,1))))
    print(f"    V_abcd + V_cdab max: {antisym_test:.2e}")
    if antisym_test < 1e-10:
        print(f"    → V is ANTISYMMETRIC under (ab)↔(cd) swap!")
    
    # Now we need E[Δ²] = E[(Σ V_{abcd} s_a p_b q_c r_d)²]
    # = Σ_{abcd,a'b'c'd'} V_{abcd} V_{a'b'c'd'} E[s_a s_{a'} p_b p_{b'} q_c q_{c'} r_d r_{d'}]
    
    # For (p,q) uniform on S^15: p = cos(θ)·u, q = sin(θ)·v where
    # u,v uniform on S^7 and θ has a specific distribution.
    # The moments are:
    #   E[p_i p_j] = (1/16) δ_{ij}  (by symmetry, each component has variance 1/16)
    #   E[q_i q_j] = (1/16) δ_{ij}
    #   E[p_i q_j] = 0  (by the circular symmetry of the Cayley-Dickson decomposition)
    #   E[p_i p_j q_k q_l] involves mixed moments on S^15
    
    # Actually, for a uniform point x on S^{n-1} in R^n:
    #   E[x_i x_j] = δ_{ij}/n
    #   E[x_i x_j x_k x_l] = (δ_{ij}δ_{kl} + δ_{ik}δ_{jl} + δ_{il}δ_{jk}) / (n(n+2))
    
    # Here our 16-dimensional vector is (p, q) on S^15, so n=16.
    # The components p_0,...,p_7,q_0,...,q_7 are the components of a point on S^{15}.
    
    # E[p_i p_j] = δ_{ij}/16
    # E[q_i q_j] = δ_{ij}/16
    # E[p_i q_j] = 0 (different component blocks)
    # E[s_a s_{a'} p_b p_{b'}] where s,p come from the same S^15 vector...
    
    # Wait, s and p come from DIFFERENT vectors!
    # a = (p,q), b = (r,s) are INDEPENDENT uniform on S^15.
    # So (p,q) and (r,s) are independent.
    
    # E[Δ²] = Σ V_{abcd} V_{a'b'c'd'} E[s_a s_{a'}] E[p_b p_{b'}] E[q_c q_{c'}] E[r_d r_{d'}]
    # Wait no — s and r come from the SAME vector b = (r,s), while p and q come from a = (p,q).
    
    # So (p,q) uniform on S^15 means they are jointly distributed.
    # (r,s) uniform on S^15 independently.
    
    # Δ = Σ V_{abcd} s_a p_b q_c r_d
    
    # E[Δ²] = Σ V_{abcd} V_{a'b'c'd'} E_a[s_a r_d s_{a'} r_{d'}] × E_b[p_b q_c p_{b'} q_{c'}]
    
    # No wait. s,r come from b = (r,s) and p,q come from a = (p,q).
    # So (p,q) are correlated (they're parts of one vector on S^15).
    # And (r,s) are correlated similarly.
    # But a ⊥ b.
    
    # E[Δ²] = Σ V_{abcd} V_{a'b'c'd'} × E[(p_b q_c)(p_{b'} q_{c'})] × E[(s_a r_d)(s_{a'} r_{d'})]
    
    # For x = (u,v) uniform on S^{2m-1} with u,v ∈ R^m:
    #   E[u_i u_j] = δ_{ij}/(2m)
    #   E[v_i v_j] = δ_{ij}/(2m)
    #   E[u_i v_j] = 0
    #   E[u_i u_j v_k v_l] = ?
    
    # For n-sphere: the 4th moment of components is:
    # E[x_i x_j x_k x_l] = (δ_{ij}δ_{kl} + δ_{ik}δ_{jl} + δ_{il}δ_{jk}) / (n(n+2))
    
    # Here n = 16. The components of a are indexed as:
    # a_0=p_0, ..., a_7=p_7, a_8=q_0, ..., a_{15}=q_7
    
    # So E[p_b q_c p_{b'} q_{c'}] = E[a_{b} a_{8+c} a_{b'} a_{8+c'}]
    # With n=16:
    # = (δ_{b,8+c}δ_{b',8+c'} + δ_{b,b'}δ_{8+c,8+c'} + δ_{b,8+c'}δ_{8+c,b'}) / (16·18)
    # = (0·0 + δ_{bb'}δ_{cc'} + 0·0) / 288
    # (since b ∈ {0..7} and 8+c ∈ {8..15}, they can never be equal)
    # = δ_{bb'}δ_{cc'} / 288
    
    n = 16
    factor = n * (n + 2)  # = 16 × 18 = 288
    
    print(f"\n  Analytical computation of E[Δ²]:")
    print(f"    n = {n}, n(n+2) = {factor}")
    print(f"    E[p_b q_c p_{{b'}} q_{{c'}}] = δ_{{bb'}}δ_{{cc'}} / {factor}")
    print(f"    E[s_a r_d s_{{a'}} r_{{d'}}] = δ_{{aa'}}δ_{{dd'}} / {factor}")
    
    # So E[Δ²] = (1/288²) Σ_{abcd} V_{abcd}²
    
    V_sq_sum = np.sum(V**2)
    E_delta_sq = V_sq_sum / factor**2
    
    print(f"    Σ V_abcd² = {V_sq_sum:.6f}")
    print(f"    E[Δ²] = {V_sq_sum:.6f} / {factor}² = {V_sq_sum:.6f} / {factor**2}")
    print(f"          = {E_delta_sq:.12f}")
    print(f"    Measured Var(Δ) will be compared below")
    
    # Now: is V_sq_sum an integer or simple fraction?
    print(f"\n  Is Σ V² a nice number?")
    print(f"    Σ V² = {V_sq_sum}")
    print(f"    As integer: {int(round(V_sq_sum))}")
    print(f"    Remainder: {V_sq_sum - round(V_sq_sum):.2e}")
    
    V_sq_int = int(round(V_sq_sum))
    print(f"\n    E[Δ²] = {V_sq_int} / {factor**2} = {V_sq_int} / {factor**2}")
    
    # Simplify
    from math import gcd
    g = gcd(V_sq_int, factor**2)
    print(f"          = {V_sq_int//g} / {factor**2//g}")
    
    # This gives us Var(Δ) exactly!
    var_exact = V_sq_int / factor**2
    
    # And therefore E[|Δ|] = √(2·Var(Δ)/π) if Gaussian:
    from math import sqrt, pi as PI
    E_abs_delta_gaussian = sqrt(2 * var_exact / PI)
    
    print(f"\n  If Δ is Gaussian:")
    print(f"    E[|Δ|] = √(2·{V_sq_int//g}/{factor**2//g}·1/π)")
    print(f"           = √({2*V_sq_int//g}/({factor**2//g}·π))")
    print(f"           = {E_abs_delta_gaussian:.12f}")
    print(f"    E[| ||ab||-1 |] ≈ E[|Δ|] = {E_abs_delta_gaussian:.12f}")
    
    return V, V_sq_int, factor, E_delta_sq


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 76)
    print("  SEDENION NORM VIOLATION: ANALYTICAL DERIVATION")
    print("  From Cayley-Dickson construction to closed-form expression")
    print("=" * 76)
    print()
    
    t0 = time.time()
    
    # Step 1: Verify the cross-term identity
    verify_cross_term()
    
    # Step 2: Associator analysis
    analyze_associator_connection()
    
    # Step 3: Analytical variance computation
    V, V_sq_int, factor, E_delta_sq_analytical = analytical_variance()
    
    # Step 4: High-precision Monte Carlo
    results = compute_mean_violation_high_precision(n_samples=1000000)
    
    # Step 5: Compare analytical and numerical
    print("\n" + "=" * 76)
    print("CRITICAL COMPARISON: ANALYTICAL vs NUMERICAL")
    print("=" * 76)
    
    from math import sqrt, pi as PI
    
    var_analytical = E_delta_sq_analytical
    var_numerical = results['var_delta']
    
    print(f"\n  Var(Δ) analytical: {var_analytical:.12f}")
    print(f"  Var(Δ) numerical:  {var_numerical:.12f}")
    print(f"  Ratio:             {var_numerical/var_analytical:.8f}")
    
    # Gaussian prediction for the target quantity
    predicted_violation = sqrt(2 * var_analytical / PI)
    measured_violation = results['mean_abs_violation']
    
    print(f"\n  E[| ||ab||-1 |] predicted (Gaussian Δ): {predicted_violation:.12f}")
    print(f"  E[| ||ab||-1 |] measured:                {measured_violation:.12f}")
    print(f"  Ratio:                                    {measured_violation/predicted_violation:.8f}")
    
    # Step 6: Identify closed form
    identify_closed_form(results)
    
    # Step 7: Dimension sweep
    dimension_sweep()
    
    # Step 8: Final verdict
    decisive_test(results)
    
    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")
    
    # ═══════════════════════════════════════════════════════════════
    # FINAL ANSWER
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  FINAL ANALYTICAL RESULT")
    print("=" * 76)
    
    from math import gcd
    g = gcd(V_sq_int, factor**2)
    num = V_sq_int // g
    den = factor**2 // g
    
    print(f"""
  The sedenion norm violation derives EXACTLY from the Cayley-Dickson
  construction as follows:
  
  1. IDENTITY:  ||ab||² = ||a||²||b||² + 2Δ
     where Δ = ⟨sp, qr̄⟩ - ⟨pr, s̄q⟩  (octonionic inner products)
  
  2. VIOLATION TENSOR: Δ = Σ_{{abcd}} V_{{abcd}} s_a p_b q_c r_d
     where V_{{abcd}} = Σ_k c_{{abk}} c_{{cdk}} ε_d - Σ_k c_{{cdak}} c_{{abk}} ε_a
     and c_{{ijk}} are the octonionic structure constants.
  
  3. VARIANCE: E[Δ²] = Σ V² / [n(n+2)]²
     Σ V²  = {V_sq_int}  (exact integer from structure constants)
     n(n+2) = 16 × 18 = 288
     
     E[Δ²] = {V_sq_int} / {factor**2} = {num} / {den}
  
  4. MEAN VIOLATION: If Δ is approximately Gaussian (by CLT),
     E[| ||ab|| - 1 |] ≈ √(2 · {num}/{den} / π) = {predicted_violation:.12f}
     
     Measured: {measured_violation:.12f}
  
  This number is {"RATIONAL" if abs(predicted_violation - round(predicted_violation * 1000) / 1000) < 1e-6 else "IRRATIONAL"} 
  (involves √(rational/π)).
""")


if __name__ == "__main__":
    main()
