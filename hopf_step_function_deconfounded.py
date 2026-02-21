#!/usr/bin/env python3
"""
HOPF FIBRATION STEP-FUNCTION TEST v2 â€” DECONFOUNDED
=====================================================

Fixes the critical flaw in v1: block-diagonal SU(2) gates created an
odd/even artifact (orphaned component at odd dimensions) that confused
the Hopf signal with a parity effect.

FIX: Use proper SU(n) generators (generalized Gell-Mann matrices) for
each dimension. Every component participates equally in every gate
regardless of whether n is odd or even. The ONLY structural difference
between dimensions is the topology of S^{2n-1}, which is exactly what
we want to test.

Gate construction:
  Rx-type: exp(i*theta*H_sym) where H_sym is the "democratic" Hermitian
           matrix coupling ALL pairs of components equally.
           H_sym[j,k] = 1 for all j != k.
           
  Rz-type: exp(i*theta*H_diag) where H_diag is the angular-momentum-like
           diagonal: diag(n-1, n-3, ..., -(n-1)) / (n-1).
           Every component gets a different phase â€” no orphans.
           
  P gate:  Asymmetric Rz â€” opposite diagonal for u and v.
  
  Cross:   exp(i*theta*H_cross) where H_cross has 1s ONLY in the blocks
           connecting first half to second half.
           This is the Cayley-Dickson coupling at ANY dimension.

Key guarantee: At n=3, all 3 components rotate. At n=5, all 5 rotate.
At n=7, all 7 rotate. No orphan, no parity artifact.

Usage: python3 hopf_step_function_deconfounded.py
"""

import numpy as np
import time

np.random.seed(42)

COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H
OUROBOROS_GATES = ['S', 'R', 'T', 'F', 'P']
NUM_GATES = 5
DIMS_TO_TEST = [1, 2, 3, 4, 5, 6, 7, 8, 10]
HOPF_DIMS = {1, 2, 4, 8}


# ============================================================================
# MATRIX EXPONENTIAL FOR HERMITIAN MATRICES
# ============================================================================

def expm_hermitian(H, coeff=1.0):
    """
    Compute exp(i * coeff * H) for Hermitian H via eigendecomposition.
    Exact â€” no Taylor truncation.
    """
    eigenvalues, U = np.linalg.eigh(H)
    exp_diag = np.diag(np.exp(1j * coeff * eigenvalues))
    return U @ exp_diag @ U.conj().T


# ============================================================================
# SU(n) GENERATOR CONSTRUCTION
# ============================================================================

def make_symmetric_generator(n):
    """
    'Democratic' Hermitian matrix: H_sym[j,k] = 1 for all j != k, 0 on diagonal.
    This couples ALL pairs of components equally.
    Normalized so ||H|| ~ 1 for any n.
    
    Eigenvalues: (n-1) once, and -1 with multiplicity (n-1).
    So spectral norm = (n-1). We normalize by 1/(n-1).
    """
    H = np.ones((n, n), dtype=complex) - np.eye(n, dtype=complex)
    return H / (n - 1) if n > 1 else H


def make_antisymmetric_generator(n):
    """
    Antisymmetric Hermitian: H_asym[j,k] = -i for j<k, +i for j>k, 0 on diag.
    All pairs participate. Normalized by spectral norm.
    """
    H = np.zeros((n, n), dtype=complex)
    for j in range(n):
        for k in range(j+1, n):
            H[j, k] = -1j
            H[k, j] = 1j
    norm = np.max(np.abs(np.linalg.eigvalsh(H)))
    if norm > 1e-10:
        H /= norm
    return H


def make_diagonal_generator(n):
    """
    Angular-momentum-like diagonal: diag(n-1, n-3, n-5, ..., -(n-1)) / (n-1).
    Every component gets a DISTINCT phase. No degeneracies for any n.
    Normalized to [-1, +1] range.
    """
    if n == 1:
        return np.zeros((1, 1), dtype=complex)
    diag_vals = np.array([n - 1 - 2*k for k in range(n)], dtype=complex)
    return np.diag(diag_vals / (n - 1))


def make_cross_generator(n):
    """
    Cross-coupling: 1s only in the off-diagonal blocks connecting
    first half [0..n//2) to second half [n//2..n).
    
    For even n: clean half-half coupling.
    For odd n: floor(n/2) x ceil(n/2) rectangular coupling â€” ALL components
    still participate (the middle component couples to the second half).
    
    This is the Cayley-Dickson analogue: at n=4, it's H <-> Hj coupling;
    at n=8, it's O = H + Hl coupling. At non-Hopf dims, the coupling
    exists but the resulting structure lacks division algebra properties.
    """
    H = np.zeros((n, n), dtype=complex)
    half = n // 2
    for i in range(half):
        for j in range(half, n):
            H[i, j] = 1.0
            H[j, i] = 1.0
    # Normalize
    norm = np.max(np.abs(np.linalg.eigvalsh(H)))
    if norm > 1e-10:
        H /= norm
    return H


def make_cross_antisym_generator(n):
    """
    Asymmetric cross-coupling: -i in upper-right block, +i in lower-left.
    For the P-like asymmetric action on the cross-coupling channel.
    """
    H = np.zeros((n, n), dtype=complex)
    half = n // 2
    for i in range(half):
        for j in range(half, n):
            H[i, j] = -1j
            H[j, i] = 1j
    norm = np.max(np.abs(np.linalg.eigvalsh(H)))
    if norm > 1e-10:
        H /= norm
    return H


# ============================================================================
# PRECOMPUTED GENERATORS (cached per dimension)
# ============================================================================

_generator_cache = {}

def get_generators(n):
    """Get (or create and cache) all generators for dimension n."""
    if n in _generator_cache:
        return _generator_cache[n]
    
    gens = {
        'sym': make_symmetric_generator(n),
        'asym': make_antisymmetric_generator(n),
        'diag': make_diagonal_generator(n),
        'cross': make_cross_generator(n),
        'cross_asym': make_cross_antisym_generator(n),
    }
    _generator_cache[n] = gens
    return gens


# ============================================================================
# GENERIC N-SPINOR STATE (same as v1)
# ============================================================================

class MerkabitNState:
    def __init__(self, u, v, omega=1.0):
        self.dim = len(u)
        self.u = np.array(u, dtype=complex).flatten()
        self.v = np.array(v, dtype=complex).flatten()
        self.omega = omega
        norm_u = np.linalg.norm(self.u)
        norm_v = np.linalg.norm(self.v)
        if norm_u > 1e-15: self.u /= norm_u
        if norm_v > 1e-15: self.v /= norm_v
    
    @property
    def overlap(self):
        return np.vdot(self.u, self.v)
    
    @property
    def coherence(self):
        return np.real(self.overlap)
    
    @property
    def overlap_magnitude(self):
        return abs(self.overlap)
    
    @property
    def relative_phase(self):
        return np.angle(self.overlap)
    
    @property
    def trit_value(self):
        c = self.coherence; r = self.overlap_magnitude
        if r < 0.1: return 0
        if c > r * 0.5: return +1
        elif c < -r * 0.5: return -1
        return 0
    
    def copy(self):
        return MerkabitNState(self.u.copy(), self.v.copy(), self.omega)


# ============================================================================
# BASIS STATES (same as v1)
# ============================================================================

def make_trit_plus(n):
    u = np.zeros(n, dtype=complex); u[0] = 1.0
    return MerkabitNState(u, u.copy())

def make_trit_zero(n):
    u = np.zeros(n, dtype=complex); u[0] = 1.0
    v = np.zeros(n, dtype=complex); v[n-1] = 1.0
    return MerkabitNState(u, v)

def make_trit_minus(n):
    u = np.zeros(n, dtype=complex); u[0] = 1.0
    return MerkabitNState(u, -u.copy())

def make_near_zero(n, eps=0.05):
    u = np.zeros(n, dtype=complex); u[0] = 1.0
    v = np.zeros(n, dtype=complex)
    v[n-1] = np.sqrt(1 - eps**2); v[0] = eps
    return MerkabitNState(u, v)

def make_random_state(n):
    u = np.random.randn(n) + 1j * np.random.randn(n)
    v = np.random.randn(n) + 1j * np.random.randn(n)
    return MerkabitNState(u, v)


# ============================================================================
# NEW GATE IMPLEMENTATIONS â€” SU(n) GENERATORS, NO ORPHANS
# ============================================================================

def gate_Rx_n(state, theta):
    """
    Symmetric rotation: exp(i*theta*H_sym) applied identically to u and v.
    H_sym is the democratic generator â€” ALL components couple equally.
    """
    n = state.dim
    if n == 1:
        return state
    gens = get_generators(n)
    R = expm_hermitian(gens['sym'], theta)
    return MerkabitNState(R @ state.u, R @ state.v, state.omega)


def gate_Ry_n(state, theta):
    """
    Antisymmetric rotation: exp(i*theta*H_asym) applied identically to u and v.
    Orthogonal to Rx â€” provides a second rotation axis.
    """
    n = state.dim
    if n == 1:
        return state
    gens = get_generators(n)
    R = expm_hermitian(gens['asym'], theta)
    return MerkabitNState(R @ state.u, R @ state.v, state.omega)


def gate_Rz_n(state, theta):
    """
    Phase rotation: exp(i*theta*H_diag) applied identically to u and v.
    H_diag gives each component a DISTINCT phase â€” no degeneracies.
    """
    n = state.dim
    gens = get_generators(n)
    R = expm_hermitian(gens['diag'], theta)
    return MerkabitNState(R @ state.u, R @ state.v, state.omega)


def gate_P_n(state, phi):
    """
    Asymmetric phase gate: exp(+i*phi*H_diag) on u, exp(-i*phi*H_diag) on v.
    This advances relative phase using ALL components â€” no orphans.
    """
    n = state.dim
    gens = get_generators(n)
    Pf = expm_hermitian(gens['diag'], phi)
    Pi = expm_hermitian(gens['diag'], -phi)
    return MerkabitNState(Pf @ state.u, Pi @ state.v, state.omega)


def gate_cross_n(state, theta):
    """
    Cross-coupling: exp(i*theta*H_cross) with opposite sign for u vs v.
    H_cross connects first half to second half.
    At ANY dimension, all components participate (no orphan).
    
    The ASYMMETRIC action (opposite sign for u vs v) is what creates
    the torsion / counter-rotation between Cayley-Dickson halves.
    """
    n = state.dim
    if n < 3:  # need at least 3 dims for meaningful cross-coupling
        return state
    gens = get_generators(n)
    Cf = expm_hermitian(gens['cross'], theta)
    Ci = expm_hermitian(gens['cross'], -theta)
    return MerkabitNState(Cf @ state.u, Ci @ state.v, state.omega)


def gate_cross_asym_n(state, theta):
    """
    Asymmetric cross: uses the antisymmetric cross generator.
    This provides a second cross-coupling channel orthogonal to the first.
    """
    n = state.dim
    if n < 3:
        return state
    gens = get_generators(n)
    Cf = expm_hermitian(gens['cross_asym'], theta)
    Ci = expm_hermitian(gens['cross_asym'], -theta)
    return MerkabitNState(Cf @ state.u, Ci @ state.v, state.omega)


# ============================================================================
# N-DIMENSIONAL OUROBOROS STEP â€” DECONFOUNDED
# ============================================================================

def ouroboros_step_n(state, step_index, theta=STEP_PHASE, cross_strength=0.3):
    """
    Ouroboros step using SU(n) generators.
    
    IDENTICAL gate logic for ALL dimensions. The generators automatically
    adapt to the dimension â€” no special-casing, no orphans.
    
    Gate sequence:
      1. P (asymmetric diagonal) â€” advances relative phase
      2. Cross (asymmetric half-coupling) â€” Cayley-Dickson torsion
      3. Rz (symmetric diagonal) â€” modulated phase rotation
      4. Rx (symmetric democratic) â€” modulated full rotation
    """
    k = step_index
    absent = k % NUM_GATES
    
    p_angle = theta
    sym_base = theta / 3
    omega_k = 2 * np.pi * k / COXETER_H
    
    rx_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    rz_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
    cross_angle = cross_strength * theta * (1.0 + 0.5 * np.cos(omega_k + 4*np.pi/3))
    
    gate_label = OUROBOROS_GATES[absent]
    if gate_label == 'S':
        rz_angle *= 0.4; rx_angle *= 1.3; cross_angle *= 1.2
    elif gate_label == 'R':
        rx_angle *= 0.4; rz_angle *= 1.3; cross_angle *= 0.8
    elif gate_label == 'T':
        rx_angle *= 0.7; rz_angle *= 0.7; cross_angle *= 1.5
    elif gate_label == 'P':
        p_angle *= 0.6; rx_angle *= 1.8; rz_angle *= 1.5; cross_angle *= 0.5
    
    s = gate_P_n(state, p_angle)
    s = gate_cross_n(s, cross_angle)
    s = gate_Rz_n(s, rz_angle)
    s = gate_Rx_n(s, rx_angle)
    return s


# ============================================================================
# BERRY PHASE COMPUTATION (same as v1)
# ============================================================================

def compute_berry_phase_n(states):
    n_states = len(states)
    gamma = 0.0; gamma_u = 0.0; gamma_v = 0.0
    min_fidelity = 1.0
    
    for k in range(n_states):
        k_next = (k + 1) % n_states
        ou = np.vdot(states[k].u, states[k_next].u)
        ov = np.vdot(states[k].v, states[k_next].v)
        gamma += np.angle(ou * ov)
        gamma_u += np.angle(ou)
        gamma_v += np.angle(ov)
        fid = abs(ou * ov)
        if fid < min_fidelity:
            min_fidelity = fid
    
    return -gamma, -gamma_u, -gamma_v, min_fidelity


# ============================================================================
# METRICS (adapted from v1)
# ============================================================================

def measure_berry_separation(dim):
    trit_states = {
        '+1': make_trit_plus(dim),
        ' 0': make_trit_zero(dim),
        '-1': make_trit_minus(dim),
    }
    
    berry_phases = {}
    for label, s0 in trit_states.items():
        states_cycle = [s0]
        s = s0.copy()
        for step in range(COXETER_H):
            s = ouroboros_step_n(s, step)
            states_cycle.append(s.copy())
        
        gamma, gamma_u, gamma_v, min_fid = compute_berry_phase_n(states_cycle[:-1])
        berry_phases[label] = {
            'gamma': gamma, 'gamma_u': gamma_u,
            'gamma_v': gamma_v, 'min_fidelity': min_fid,
        }
    
    g_plus = berry_phases['+1']['gamma']
    g_zero = berry_phases[' 0']['gamma']
    g_minus = berry_phases['-1']['gamma']
    
    sep_0_plus = abs(g_zero - g_plus)
    sep_0_minus = abs(g_zero - g_minus)
    sep_plus_minus = abs(g_plus - g_minus)
    
    return {
        'berry_phases': berry_phases,
        'sep_0_pm': (sep_0_plus + sep_0_minus) / 2,
        'sep_pm': sep_plus_minus,
        'total_separation': sep_0_plus + sep_0_minus + sep_plus_minus,
    }


def measure_cycle_fidelity(dim, n_trials=20):
    fidelities = []
    for _ in range(n_trials):
        s0 = make_random_state(dim)
        s = s0.copy()
        for step in range(COXETER_H):
            s = ouroboros_step_n(s, step)
        fid_u = abs(np.vdot(s0.u, s.u))**2
        fid_v = abs(np.vdot(s0.v, s.v))**2
        fidelities.append(fid_u * fid_v)
    
    return {
        'mean_fidelity': np.mean(fidelities),
        'std_fidelity': np.std(fidelities),
        'min_fidelity': np.min(fidelities),
    }


def measure_attractor_strength(dim, n_trials=30, n_multi_cycles=5):
    """
    Attractor strength: run near-|0> states through multiple ouroboros
    cycles, measure contraction of overlap magnitude.
    
    Reports BOTH the mean contraction AND the eps-dependence profile,
    since v1 showed the basin topology (eps-dependence) may be a
    stronger Hopf signature than the mean.
    """
    eps_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    contraction_ratios = []
    per_eps = []
    
    for eps in eps_values:
        contractions = []
        for _ in range(n_trials):
            s0 = make_near_zero(dim, eps)
            initial_overlap = s0.overlap_magnitude
            
            s = s0.copy()
            for cycle in range(n_multi_cycles):
                for step in range(COXETER_H):
                    s = ouroboros_step_n(s, step)
            
            final_overlap = s.overlap_magnitude
            if initial_overlap > 1e-10:
                contractions.append(final_overlap / initial_overlap)
        
        mean_c = np.mean(contractions) if contractions else 1.0
        per_eps.append((eps, mean_c))
        contraction_ratios.append(mean_c)
    
    mean_contraction = np.mean(contraction_ratios)
    
    # Basin topology metric: standard deviation of contraction across eps values.
    # Flat profile (same contraction at all eps) = no basin structure.
    # Varying profile (different contraction at different eps) = basin structure.
    basin_variation = np.std(contraction_ratios)
    
    # Monotonicity: does contraction decrease with eps? (attractor basin shape)
    # Compute Spearman-like correlation: negative = stronger contraction at larger eps
    if len(contraction_ratios) > 2:
        diffs = [contraction_ratios[i+1] - contraction_ratios[i] for i in range(len(contraction_ratios)-1)]
        monotone_decreasing = sum(1 for d in diffs if d < -0.01) / len(diffs)
    else:
        monotone_decreasing = 0.0
    
    return {
        'mean_contraction': mean_contraction,
        'contraction_by_eps': per_eps,
        'is_attractor': mean_contraction < 0.95,
        'basin_variation': basin_variation,
        'monotone_decreasing': monotone_decreasing,
    }


def measure_fiber_dimension(dim):
    if dim < 2:
        return {'effective_fiber_dim': 0, 'participation_ratio': 0, 'sv_entropy': 0}
    
    trajectories_u = []
    trajectories_v = []
    
    s = make_trit_zero(dim)
    for step in range(COXETER_H):
        s = ouroboros_step_n(s, step)
        trajectories_u.append(s.u.copy())
        trajectories_v.append(s.v.copy())
    
    combined = np.hstack([np.array(trajectories_u), np.array(trajectories_v)])
    sv = np.linalg.svd(combined, compute_uv=False)
    sv_normalized = sv / sv[0] if sv[0] > 1e-15 else sv
    
    effective_dim = int(np.sum(sv_normalized > 0.1))
    sv2 = sv_normalized**2
    participation = float(np.sum(sv2)**2 / (np.sum(sv2**2) + 1e-15))
    
    return {
        'effective_fiber_dim': effective_dim,
        'participation_ratio': participation,
        'sv_entropy': float(-np.sum(sv2 * np.log(sv2 + 1e-15))),
    }


def measure_hopf_proxy(dim, n_samples=50):
    """
    Hopf invariant proxy: correlation between base-space and fiber-space
    evolution under the ouroboros step.
    """
    if dim < 3:
        return {'hopf_proxy': 0.0}
    
    half = dim // 2
    correlations = []
    
    for _ in range(n_samples):
        s = make_random_state(dim)
        s_next = ouroboros_step_n(s, 0)
        
        base_u = np.vdot(s.u[:half], s_next.u[:half])
        base_v = np.vdot(s.v[:half], s_next.v[:half])
        fiber_u = np.vdot(s.u[half:], s_next.u[half:])
        fiber_v = np.vdot(s.v[half:], s_next.v[half:])
        
        if all(abs(x) > 1e-10 for x in [base_u, base_v, fiber_u, fiber_v]):
            pd_u = abs(np.angle(base_u) - np.angle(fiber_u))
            pd_v = abs(np.angle(base_v) - np.angle(fiber_v))
            correlations.append(abs(np.cos(pd_u) * np.cos(pd_v)))
    
    return {'hopf_proxy': np.mean(correlations) if correlations else 0.0}


# ============================================================================
# ADDITIONAL METRIC: SPECTRAL GAP OF OUROBOROS TRANSFER MATRIX
# ============================================================================

def measure_spectral_gap(dim, n_samples=10):
    """
    Build the effective transfer matrix of one full ouroboros cycle
    (12 steps) by tracking how it acts on random states in C^n x C^n.
    
    The spectral gap (ratio of second to first singular value) measures
    how strongly the cycle contracts toward a fixed structure.
    At Hopf dimensions, we expect a LARGER gap (stronger contraction
    into a lower-dimensional submanifold).
    """
    if dim == 1:
        return {'spectral_gap': 1.0, 'leading_sv': 1.0}
    
    # Probe the cycle's action on many initial states
    initial_vecs = []
    final_vecs = []
    
    for _ in range(max(n_samples, 4 * dim)):
        s0 = make_random_state(dim)
        # Pack into a single vector: [Re(u), Im(u), Re(v), Im(v)]
        init_vec = np.concatenate([s0.u.real, s0.u.imag, s0.v.real, s0.v.imag])
        
        s = s0.copy()
        for step in range(COXETER_H):
            s = ouroboros_step_n(s, step)
        
        final_vec = np.concatenate([s.u.real, s.u.imag, s.v.real, s.v.imag])
        
        initial_vecs.append(init_vec)
        final_vecs.append(final_vec)
    
    # Stack into matrices
    X = np.array(initial_vecs).T  # (4n, n_samples)
    Y = np.array(final_vecs).T    # (4n, n_samples)
    
    # SVD of the output to measure effective dimensionality
    sv_Y = np.linalg.svd(Y, compute_uv=False)
    sv_Y_norm = sv_Y / sv_Y[0] if sv_Y[0] > 1e-15 else sv_Y
    
    # Spectral gap: ratio of 2nd to 1st SV
    if len(sv_Y_norm) > 1:
        gap = sv_Y_norm[1] / sv_Y_norm[0] if sv_Y_norm[0] > 1e-15 else 0.0
    else:
        gap = 0.0
    
    return {
        'spectral_gap': float(gap),
        'leading_sv': float(sv_Y[0]),
        'sv_profile': sv_Y_norm[:min(8, len(sv_Y_norm))].tolist(),
    }


# ============================================================================
# RUN ALL
# ============================================================================

def run_dimension_sweep():
    print("=" * 80)
    print("  HOPF FIBRATION STEP-FUNCTION TEST v2 â€” DECONFOUNDED")
    print("  SU(n) generators: no orphaned components, no odd/even artifact")
    print("=" * 80)
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dimensions tested: {DIMS_TO_TEST}")
    print(f"  Hopf dimensions: {sorted(HOPF_DIMS)}")
    print(f"  Gate construction: generalized Gell-Mann matrices for each n")
    print(f"  Coxeter number: {COXETER_H}")
    print()
    
    results = {}
    for dim in DIMS_TO_TEST:
        is_hopf = dim in HOPF_DIMS
        label = f"n={dim}" + (" [HOPF]" if is_hopf else " [non-Hopf]")
        print(f"  Testing {label}...", end="", flush=True)
        t0 = time.time()
        
        berry = measure_berry_separation(dim)
        fidelity = measure_cycle_fidelity(dim)
        attractor = measure_attractor_strength(dim)
        fiber = measure_fiber_dimension(dim)
        hopf = measure_hopf_proxy(dim)
        spectral = measure_spectral_gap(dim)
        
        elapsed = time.time() - t0
        results[dim] = {
            'is_hopf': is_hopf, 'berry': berry, 'fidelity': fidelity,
            'attractor': attractor, 'fiber': fiber, 'hopf': hopf,
            'spectral': spectral, 'time': elapsed,
        }
        print(f" done ({elapsed:.1f}s)")
    
    return results


def print_results(results):
    print("\n" + "=" * 80)
    print("  RESULTS: DECONFOUNDED METRICS")
    print("=" * 80)
    
    # Table 1: Berry phase separation
    print("\n  TABLE 1: BERRY PHASE SEPARATION")
    print(f"  {'dim':>4}  {'Hopf?':>6}  {'Î³(+1)':>10}  {'Î³(0)':>10}  {'Î³(-1)':>10}  "
          f"{'|0-Â±| sep':>10}  {'total':>10}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    
    for dim in DIMS_TO_TEST:
        r = results[dim]; b = r['berry']; bp = b['berry_phases']
        hopf_str = "YES" if r['is_hopf'] else "no"
        print(f"  {dim:4d}  {hopf_str:>6}  "
              f"{bp['+1']['gamma']:10.4f}  {bp[' 0']['gamma']:10.4f}  {bp['-1']['gamma']:10.4f}  "
              f"{b['sep_0_pm']:10.4f}  {b['total_separation']:10.4f}")
    
    # Table 2: Attractor + Basin
    print("\n  TABLE 2: ATTRACTOR STRENGTH & BASIN TOPOLOGY")
    print(f"  {'dim':>4}  {'Hopf?':>6}  {'mean ctr':>10}  {'basin var':>10}  "
          f"{'monotoneâ†“':>10}  {'attractor':>10}  {'spec gap':>10}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    
    for dim in DIMS_TO_TEST:
        r = results[dim]; hopf_str = "YES" if r['is_hopf'] else "no"
        a = r['attractor']; sp = r['spectral']
        att_str = "YES" if a['is_attractor'] else "no"
        print(f"  {dim:4d}  {hopf_str:>6}  "
              f"{a['mean_contraction']:10.4f}  {a['basin_variation']:10.4f}  "
              f"{a['monotone_decreasing']:10.2f}  {att_str:>10}  {sp['spectral_gap']:10.4f}")
    
    # Table 3: Fiber + fidelity
    print("\n  TABLE 3: FIBER DIMENSION & CYCLE FIDELITY")
    print(f"  {'dim':>4}  {'Hopf?':>6}  {'eff fiber':>10}  {'particip':>10}  "
          f"{'SV entropy':>10}  {'mean fid':>10}  {'Hopf prx':>10}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    
    for dim in DIMS_TO_TEST:
        r = results[dim]; hopf_str = "YES" if r['is_hopf'] else "no"
        fi = r['fiber']; fd = r['fidelity']; hp = r['hopf']
        print(f"  {dim:4d}  {hopf_str:>6}  "
              f"{fi['effective_fiber_dim']:10d}  {fi['participation_ratio']:10.3f}  "
              f"{fi['sv_entropy']:10.4f}  {fd['mean_fidelity']:10.6f}  {hp['hopf_proxy']:10.4f}")


def analyze_step_function(results):
    print("\n" + "=" * 80)
    print("  CRITICAL ANALYSIS: STEP FUNCTION vs SMOOTH INTERPOLATION (v2)")
    print("  Gate construction: SU(n) generators â€” NO odd/even confound")
    print("=" * 80)
    
    verdict = {}
    
    # ---- TEST A: n=6 between n=4 and n=8 (Berry separation) ----
    if all(d in results for d in [4, 6, 8]):
        sep_4 = results[4]['berry']['total_separation']
        sep_6 = results[6]['berry']['total_separation']
        sep_8 = results[8]['berry']['total_separation']
        midpoint = (sep_4 + sep_8) / 2
        
        dist_to_4 = abs(sep_6 - sep_4)
        dist_to_mid = abs(sep_6 - midpoint)
        
        print(f"\n  TEST A: Berry separation â€” is n=6 step-function or smooth?")
        print(f"    n=4 (Hopf):     {sep_4:.6f}")
        print(f"    n=6 (non-Hopf): {sep_6:.6f}")
        print(f"    n=8 (Hopf):     {sep_8:.6f}")
        print(f"    Midpoint:       {midpoint:.6f}")
        print(f"    Dist to n=4:    {dist_to_4:.6f}")
        print(f"    Dist to mid:    {dist_to_mid:.6f}")
        
        verdict['A'] = dist_to_4 < dist_to_mid
        tag = "STEP (closer to n=4)" if verdict['A'] else "SMOOTH (closer to midpoint)"
        print(f"    â†’ {tag}")
    
    # ---- TEST B: n=3 between n=2 and n=4 (Berry separation) ----
    if all(d in results for d in [2, 3, 4]):
        sep_2 = results[2]['berry']['total_separation']
        sep_3 = results[3]['berry']['total_separation']
        sep_4 = results[4]['berry']['total_separation']
        midpoint = (sep_2 + sep_4) / 2
        
        dist_to_2 = abs(sep_3 - sep_2)
        dist_to_mid = abs(sep_3 - midpoint)
        
        print(f"\n  TEST B: Berry separation â€” is n=3 step-function or smooth?")
        print(f"    n=2 (Hopf):     {sep_2:.6f}")
        print(f"    n=3 (non-Hopf): {sep_3:.6f}")
        print(f"    n=4 (Hopf):     {sep_4:.6f}")
        print(f"    Midpoint:       {midpoint:.6f}")
        print(f"    Dist to n=2:    {dist_to_2:.6f}")
        print(f"    Dist to mid:    {dist_to_mid:.6f}")
        
        verdict['B'] = dist_to_2 < dist_to_mid
        tag = "STEP (closer to n=2)" if verdict['B'] else "SMOOTH (closer to midpoint)"
        print(f"    â†’ {tag}")
    
    # ---- TEST C: Attractor contraction â€” step function? ----
    print(f"\n  TEST C: Attractor contraction by dimension")
    print(f"  (With SU(n) generators, odd/even artifact should be eliminated)")
    
    for dim in DIMS_TO_TEST:
        r = results[dim]
        tag = " [HOPF]" if r['is_hopf'] else ""
        c = r['attractor']['mean_contraction']
        bv = r['attractor']['basin_variation']
        print(f"    n={dim:2d}{tag:>8}:  ctr={c:.4f}  basin_var={bv:.4f}  "
              f"{'ATTRACTOR' if c < 0.95 else ''}")
    
    # Test: do non-Hopf dims cluster with lower Hopf?
    n_cluster_ok = 0; n_cluster_tot = 0
    for dim in DIMS_TO_TEST:
        if dim not in HOPF_DIMS and dim > 1:
            lower_hopfs = [h for h in HOPF_DIMS if h < dim]
            upper_hopfs = [h for h in HOPF_DIMS if h > dim]
            if lower_hopfs and upper_hopfs:
                lh = max(lower_hopfs); uh = min(upper_hopfs)
                if lh in results and uh in results:
                    cl = results[lh]['attractor']['mean_contraction']
                    cu = results[uh]['attractor']['mean_contraction']
                    cd = results[dim]['attractor']['mean_contraction']
                    mid_c = (cl + cu) / 2
                    n_cluster_tot += 1
                    if abs(cd - cl) < abs(cd - mid_c):
                        n_cluster_ok += 1
                    tag = f"closer to n={lh}" if abs(cd - cl) < abs(cd - mid_c) else f"closer to mid"
                    print(f"      n={dim}: {cd:.4f}  (n={lh}={cl:.4f}, n={uh}={cu:.4f}, mid={mid_c:.4f}) â†’ {tag}")
    
    if n_cluster_tot > 0:
        verdict['C'] = n_cluster_ok / n_cluster_tot > 0.5
        print(f"    Clustering score: {n_cluster_ok}/{n_cluster_tot}")
    
    # ---- TEST D: Basin topology ----
    print(f"\n  TEST D: Basin topology (eps-dependent contraction)")
    print(f"  (Hopf dims should show VARYING contraction vs eps;")
    print(f"   non-Hopf dims should show FLAT profile)")
    
    for dim in DIMS_TO_TEST:
        r = results[dim]
        tag = " [HOPF]" if r['is_hopf'] else ""
        bv = r['attractor']['basin_variation']
        md = r['attractor']['monotone_decreasing']
        
        profile = "  ".join([f"{c:.3f}" for _, c in r['attractor']['contraction_by_eps']])
        print(f"    n={dim:2d}{tag:>8}: [{profile}]  var={bv:.4f}  monoâ†“={md:.2f}")
    
    # Test: Hopf dims have higher basin variation than non-Hopf dims
    hopf_vars = [results[d]['attractor']['basin_variation'] for d in DIMS_TO_TEST 
                  if d in HOPF_DIMS and d > 1 and d in results]
    nonhopf_vars = [results[d]['attractor']['basin_variation'] for d in DIMS_TO_TEST 
                     if d not in HOPF_DIMS and d > 1 and d in results]
    
    if hopf_vars and nonhopf_vars:
        mean_hopf_var = np.mean(hopf_vars)
        mean_nonhopf_var = np.mean(nonhopf_vars)
        verdict['D'] = mean_hopf_var > mean_nonhopf_var
        print(f"    Mean basin variation: Hopf={mean_hopf_var:.4f}, non-Hopf={mean_nonhopf_var:.4f}")
        tag = "Hopf has MORE structure" if verdict['D'] else "non-Hopf has more structure"
        print(f"    â†’ {tag}")
    
    # ---- TEST E: Hopf ceiling ----
    if all(d in results for d in [8, 10]):
        sep_8 = results[8]['berry']['total_separation']
        sep_10 = results[10]['berry']['total_separation']
        verdict['E'] = sep_10 <= sep_8 * 1.1
        
        print(f"\n  TEST E: Hopf ceiling (n=10 should not exceed n=8)")
        print(f"    n=8  (Hopf):     {sep_8:.6f}")
        print(f"    n=10 (non-Hopf): {sep_10:.6f}")
        print(f"    â†’ {'CEILING HOLDS' if verdict['E'] else 'CEILING BROKEN'}")
    
    # ---- TEST F: Spectral gap comparison ----
    print(f"\n  TEST F: Spectral gap of ouroboros transfer matrix")
    print(f"  (Larger gap = stronger dimensional reduction by cycle)")
    for dim in DIMS_TO_TEST:
        r = results[dim]
        tag = " [HOPF]" if r['is_hopf'] else ""
        sg = r['spectral']['spectral_gap']
        svp = r['spectral'].get('sv_profile', [])
        svp_str = "  ".join([f"{v:.3f}" for v in svp[:6]])
        print(f"    n={dim:2d}{tag:>8}: gap={sg:.4f}  SVs=[{svp_str}]")
    
    # ---- VISUALIZATIONS ----
    print(f"\n  VISUALIZATION: Berry separation vs dimension")
    max_sep = max(results[d]['berry']['total_separation'] for d in DIMS_TO_TEST)
    if max_sep < 1e-10: max_sep = 1.0
    
    for dim in DIMS_TO_TEST:
        sep = results[dim]['berry']['total_separation']
        bar_len = int(50 * sep / max_sep)
        hopf_mark = "*" if dim in HOPF_DIMS else " "
        print(f"  n={dim:2d} {hopf_mark} |{'#' * bar_len:<50}| {sep:.4f}")
    
    print(f"\n  VISUALIZATION: Attractor contraction vs dimension")
    for dim in DIMS_TO_TEST:
        c = results[dim]['attractor']['mean_contraction']
        strength = max(0, min(1.0 - c, 0.5))  # cap for display
        bar_len = int(50 * (strength + 0.5))   # shift so neutral is at 25
        hopf_mark = "*" if dim in HOPF_DIMS else " "
        print(f"  n={dim:2d} {hopf_mark} |{'#' * bar_len:<50}| c={c:.4f}")
    
    print(f"\n  VISUALIZATION: Basin variation vs dimension")
    max_bv = max(results[d]['attractor']['basin_variation'] for d in DIMS_TO_TEST)
    if max_bv < 1e-10: max_bv = 1.0
    
    for dim in DIMS_TO_TEST:
        bv = results[dim]['attractor']['basin_variation']
        bar_len = int(50 * bv / max_bv)
        hopf_mark = "*" if dim in HOPF_DIMS else " "
        print(f"  n={dim:2d} {hopf_mark} |{'#' * bar_len:<50}| bv={bv:.4f}")
    
    # ---- FINAL VERDICT ----
    print(f"\n  {'='*65}")
    print(f"  FINAL VERDICT (v2 â€” deconfounded)")
    print(f"  {'='*65}")
    
    for name, key in [("Test A (n=6 Berry step-function)", 'A'),
                       ("Test B (n=3 Berry step-function)", 'B'),
                       ("Test C (attractor clustering)", 'C'),
                       ("Test D (basin topology Hopf > non-Hopf)", 'D'),
                       ("Test E (Hopf ceiling at n=8)", 'E')]:
        if key in verdict:
            print(f"  {name}: {'PASS' if verdict[key] else 'FAIL'}")
    
    passed = sum(1 for v in verdict.values() if v)
    total = len(verdict)
    print(f"\n  Score: {passed}/{total}")
    
    if total > 0:
        frac = passed / total
        if frac >= 0.8:
            print(f"\n  CONCLUSION: STRONG SUPPORT for step-function at Hopf thresholds.")
            print(f"  With the odd/even confound removed, the division algebra structure")
            print(f"  is the dominant factor â€” not raw dimensionality.")
        elif frac >= 0.5:
            print(f"\n  CONCLUSION: PARTIAL SUPPORT. Some metrics show step-function,")
            print(f"  others are ambiguous. The Hopf structure matters but may not be")
            print(f"  the only factor.")
        else:
            print(f"\n  CONCLUSION: WEAK/NO SUPPORT. With proper SU(n) gates, the")
            print(f"  contraction appears to scale smoothly with dimension.")
            print(f"  The Hopf hypothesis is challenged by deconfounded data.")


def attractor_basin_detail(results):
    print("\n" + "=" * 80)
    print("  SUPPLEMENTARY: ATTRACTOR BASIN PROFILES (deconfounded)")
    print("=" * 80)
    
    for dim in DIMS_TO_TEST:
        r = results[dim]
        is_hopf = "HOPF" if r['is_hopf'] else "    "
        print(f"\n  n={dim} [{is_hopf}]:")
        for eps, ratio in r['attractor']['contraction_by_eps']:
            bar_len = int(40 * min(ratio, 2.0) / 2.0)
            bar = "=" * bar_len
            if ratio < 0.98:
                direction = "<-|0>"
            elif ratio > 1.02:
                direction = "->away"
            else:
                direction = " ~neutral"
            print(f"    eps={eps:.2f}: {ratio:8.4f} [{bar:<40}] {direction}")


# ============================================================================
# COMPARISON WITH V1
# ============================================================================

def print_v1_comparison():
    """Print the v1 results for direct comparison."""
    print("\n" + "=" * 80)
    print("  REFERENCE: v1 RESULTS (block-diagonal SU(2), with odd/even confound)")
    print("=" * 80)
    print("""
  v1 attractor contraction (block-diagonal SU(2)):
    n=1  [HOPF]: 1.000000  (neutral)
    n=2  [HOPF]: 17.531049 (repeller!)
    n=3        : 0.631660  (attractor â€” ORPHAN ARTIFACT)
    n=4  [HOPF]: 1.172942  (neutral/weak repeller)
    n=5        : 0.672325  (attractor â€” ORPHAN ARTIFACT)
    n=6        : 4.875261  (repeller)
    n=7        : 0.479126  (attractor â€” ORPHAN ARTIFACT)
    n=8  [HOPF]: 0.930834  (attractor)
    n=10       : 2.654999  (repeller)

  Key confound: ALL odd dims showed attractor behavior due to the
  orphaned component in block-diagonal SU(2). This masked the true
  Hopf signal. The v2 results above use SU(n) generators where
  ALL components participate, eliminating this artifact.
  """)


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    
    results = run_dimension_sweep()
    print_results(results)
    analyze_step_function(results)
    attractor_basin_detail(results)
    print_v1_comparison()
    
    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f} seconds")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()
