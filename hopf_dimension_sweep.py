#!/usr/bin/env python3
"""
HOPF FIBRATION STEP-FUNCTION TEST
==================================

Critical falsifiable prediction:
  The ouroboros contraction strength depends on Hopf fibration structure,
  NOT merely on dimensionality. Specifically:
  
  Division algebra dimensions (Hopf fibrations exist):
    n=1: R   (trivial)           S^1 -> S^1
    n=2: C   (complex Hopf)      S^3 -> S^2,  fiber S^1
    n=4: H   (quaternionic Hopf)  S^7 -> S^4,  fiber S^3
    n=8: O   (octonionic Hopf)    S^15 -> S^8, fiber S^7
    
  Non-division-algebra dimensions (NO Hopf fibration):
    n=3, 5, 6, 7: no normed division algebra, no Hopf bundle
    
  PREDICTION: contraction strength is a STEP FUNCTION at division algebra
  thresholds, not a smooth interpolation. In particular:
    - n=6 should behave more like n=4 than like a midpoint between n=4 and n=8
    - n=3 should behave more like n=2 than like a midpoint between n=2 and n=4
    - Jumps should occur AT and only at n=1, 2, 4, 8

  If contraction strength interpolates smoothly with dimension, the Hopf
  hypothesis is falsified. If it shows step-function jumps at 2, 4, 8,
  the framework's reliance on division algebra structure is confirmed.

Metrics measured:
  1. Berry phase separation (|gamma_0 - gamma_pm|)
  2. Cycle fidelity (|<psi_0|psi_T>|^2 after full ouroboros)
  3. Trit preservation (does the cycle return the trit?)
  4. Attractor strength (how nearby states converge toward |0>)
  5. Effective fiber dimension (rank of off-diagonal coupling)

Usage: python3 hopf_dimension_sweep.py
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
# GENERIC N-SPINOR STATE
# ============================================================================

class MerkabitNState:
    """N-spinor merkabit: (u, v) where u, v in C^n, |u| = |v| = 1."""
    
    def __init__(self, u, v, omega=1.0):
        self.dim = len(u)
        self.u = np.array(u, dtype=complex).flatten()
        self.v = np.array(v, dtype=complex).flatten()
        self.omega = omega
        norm_u = np.linalg.norm(self.u)
        norm_v = np.linalg.norm(self.v)
        if norm_u > 1e-15:
            self.u /= norm_u
        if norm_v > 1e-15:
            self.v /= norm_v
    
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
        c = self.coherence
        r = self.overlap_magnitude
        if r < 0.1:
            return 0
        if c > r * 0.5:
            return +1
        elif c < -r * 0.5:
            return -1
        return 0
    
    def copy(self):
        return MerkabitNState(self.u.copy(), self.v.copy(), self.omega)


# ============================================================================
# BASIS STATES FOR ARBITRARY DIMENSION
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
# GATE IMPLEMENTATIONS (DIMENSION-GENERIC)
# ============================================================================

def make_block_diagonal_su2(n, theta, gate_type='Rx'):
    """
    Build n x n unitary by embedding SU(2) blocks along the diagonal.
    
    For even n: n/2 blocks of 2x2
    For odd n: (n-1)/2 blocks + one 1x1 identity (orphan component)
    
    CRITICAL: At non-division-algebra dims, the orphaned component
    breaks the Cayley-Dickson pairing structure.
    """
    M = np.eye(n, dtype=complex)
    
    if gate_type == 'Rx':
        c, s = np.cos(theta/2), -1j * np.sin(theta/2)
        block = np.array([[c, s], [s, c]], dtype=complex)
    elif gate_type == 'Rz':
        block = np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2)])
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")
    
    for i in range(0, n - 1, 2):
        M[i:i+2, i:i+2] = block
    
    return M


def gate_Rx_n(state, theta):
    R = make_block_diagonal_su2(state.dim, theta, 'Rx')
    return MerkabitNState(R @ state.u, R @ state.v, state.omega)

def gate_Rz_n(state, theta):
    R = make_block_diagonal_su2(state.dim, theta, 'Rz')
    return MerkabitNState(R @ state.u, R @ state.v, state.omega)

def gate_P_n(state, phi):
    n = state.dim
    Pf = make_block_diagonal_su2(n, phi, 'Rz')
    Pi = make_block_diagonal_su2(n, -phi, 'Rz')
    return MerkabitNState(Pf @ state.u, Pi @ state.v, state.omega)


def gate_cross_n(state, theta):
    """
    Cross-coupling: connects first half to second half of spinor.
    
    At Hopf dims, this connects Cayley-Dickson halves.
    At non-Hopf dims, the halves are unequal (odd dim) or
    don't form a proper algebra pair.
    """
    n = state.dim
    if n < 4:
        return state
    
    half = n // 2
    Cf = np.eye(n, dtype=complex)
    Ci = np.eye(n, dtype=complex)
    
    c, s = np.cos(theta/2), np.sin(theta/2)
    
    for i in range(min(half, n - half)):
        j = half + i
        if j < n:
            Cf[i, i] = c; Cf[i, j] = -s; Cf[j, i] = s; Cf[j, j] = c
            Ci[i, i] = c; Ci[i, j] = s; Ci[j, i] = -s; Ci[j, j] = c
    
    return MerkabitNState(Cf @ state.u, Ci @ state.v, state.omega)


def gate_cross_recursive_n(state, theta):
    """
    Recursive cross-coupling across ALL Cayley-Dickson levels.
    
    n=8: octonionic (4<->4) + quaternionic (2<->2 within each 4)
    n=4: quaternionic (2<->2) only
    n=2: no cross
    
    At non-Hopf dims, the recursion BREAKS: halves aren't clean
    powers of 2, so sublevel coupling is malformed.
    """
    n = state.dim
    if n < 4:
        return state
    
    # Level 1: half-space coupling
    s = gate_cross_n(state, theta)
    
    # Level 2: quarter-space coupling
    if n >= 4:
        quarter = n // 4
        if quarter >= 1:
            Cf2 = np.eye(n, dtype=complex)
            Ci2 = np.eye(n, dtype=complex)
            c2, s2 = np.cos(theta/4), np.sin(theta/4)
            
            # First half: first quarter <-> second quarter
            for i in range(quarter):
                j = quarter + i
                if j < n // 2:
                    Cf2[i, i] = c2; Cf2[i, j] = -s2; Cf2[j, i] = s2; Cf2[j, j] = c2
                    Ci2[i, i] = c2; Ci2[i, j] = s2; Ci2[j, i] = -s2; Ci2[j, j] = c2
            
            # Second half: third quarter <-> fourth quarter
            offset = n // 2
            for i in range(quarter):
                j = quarter + i
                ii, jj = offset + i, offset + j
                if jj < n:
                    Cf2[ii, ii] = c2; Cf2[ii, jj] = -s2; Cf2[jj, ii] = s2; Cf2[jj, jj] = c2
                    Ci2[ii, ii] = c2; Ci2[ii, jj] = s2; Ci2[jj, ii] = -s2; Ci2[jj, jj] = c2
            
            s = MerkabitNState(Cf2 @ s.u, Ci2 @ s.v, s.omega)
    
    return s


# ============================================================================
# N-DIMENSIONAL OUROBOROS STEP
# ============================================================================

def ouroboros_step_n(state, step_index, theta=STEP_PHASE, cross_strength=0.3):
    """
    Ouroboros step for n-spinor merkabit.
    
    IDENTICAL gate logic across ALL dimensions â€” the ONLY thing that
    changes is the matrix dimension. This ensures we test the
    GEOMETRY, not the gate tuning.
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
    s = gate_cross_recursive_n(s, cross_angle)
    s = gate_Rz_n(s, rz_angle)
    s = gate_Rx_n(s, rx_angle)
    return s


# ============================================================================
# BERRY PHASE COMPUTATION
# ============================================================================

def compute_berry_phase_n(states):
    n_states = len(states)
    gamma = 0.0
    gamma_u = 0.0
    gamma_v = 0.0
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
# METRIC 1: BERRY PHASE SEPARATION
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


# ============================================================================
# METRIC 2: CYCLE FIDELITY
# ============================================================================

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


# ============================================================================
# METRIC 3: ATTRACTOR STRENGTH
# ============================================================================

def measure_attractor_strength(dim, n_trials=30, n_multi_cycles=5):
    eps_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    contraction_ratios = []
    
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
        
        if contractions:
            contraction_ratios.append(np.mean(contractions))
    
    mean_contraction = np.mean(contraction_ratios) if contraction_ratios else 1.0
    
    return {
        'mean_contraction': mean_contraction,
        'contraction_by_eps': list(zip(eps_values[:len(contraction_ratios)], contraction_ratios)),
        'is_attractor': mean_contraction < 0.95,
    }


# ============================================================================
# METRIC 4: EFFECTIVE FIBER DIMENSION
# ============================================================================

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


# ============================================================================
# METRIC 5: HOPF INVARIANT PROXY
# ============================================================================

def measure_hopf_invariant_proxy(dim, n_samples=50):
    if dim < 4:
        return {'hopf_proxy': 0.0, 'base_fiber_correlation': 0.0}
    
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
    
    mean_corr = np.mean(correlations) if correlations else 0.0
    return {'hopf_proxy': mean_corr, 'base_fiber_correlation': mean_corr}


# ============================================================================
# MAIN DIMENSION SWEEP
# ============================================================================

def run_dimension_sweep():
    print("=" * 80)
    print("  HOPF FIBRATION STEP-FUNCTION TEST")
    print("  Testing: contraction strength vs dimension")
    print("  Prediction: step function at division algebra thresholds (1, 2, 4, 8)")
    print("  NOT smooth interpolation with dimension")
    print("=" * 80)
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dimensions tested: {DIMS_TO_TEST}")
    print(f"  Hopf dimensions: {sorted(HOPF_DIMS)}")
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
        hopf = measure_hopf_invariant_proxy(dim)
        
        elapsed = time.time() - t0
        results[dim] = {
            'is_hopf': is_hopf, 'berry': berry, 'fidelity': fidelity,
            'attractor': attractor, 'fiber': fiber, 'hopf': hopf, 'time': elapsed,
        }
        print(f" done ({elapsed:.1f}s)")
    
    return results


def print_results_table(results):
    print("\n" + "=" * 80)
    print("  RESULTS: CONTRACTION STRENGTH vs DIMENSION")
    print("=" * 80)
    
    print("\n  TABLE 1: BERRY PHASE SEPARATION (primary metric)")
    print(f"  {'dim':>4}  {'Hopf?':>6}  {'gamma(+1)':>10}  {'gamma(0)':>10}  {'gamma(-1)':>10}  "
          f"{'|0-pm| sep':>10}  {'|+-| sep':>10}  {'total':>10}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    
    for dim in DIMS_TO_TEST:
        r = results[dim]; b = r['berry']; bp = b['berry_phases']
        hopf_str = "YES" if r['is_hopf'] else "no"
        print(f"  {dim:4d}  {hopf_str:>6}  "
              f"{bp['+1']['gamma']:10.4f}  {bp[' 0']['gamma']:10.4f}  {bp['-1']['gamma']:10.4f}  "
              f"{b['sep_0_pm']:10.4f}  {b['sep_pm']:10.4f}  {b['total_separation']:10.4f}")
    
    print("\n  TABLE 2: CYCLE FIDELITY & ATTRACTOR STRENGTH")
    print(f"  {'dim':>4}  {'Hopf?':>6}  {'mean fid':>10}  {'std fid':>10}  "
          f"{'contraction':>12}  {'attractor?':>10}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")
    
    for dim in DIMS_TO_TEST:
        r = results[dim]; hopf_str = "YES" if r['is_hopf'] else "no"
        f_data = r['fidelity']; a_data = r['attractor']
        att_str = "YES" if a_data['is_attractor'] else "no"
        print(f"  {dim:4d}  {hopf_str:>6}  "
              f"{f_data['mean_fidelity']:10.6f}  {f_data['std_fidelity']:10.6f}  "
              f"{a_data['mean_contraction']:12.6f}  {att_str:>10}")
    
    print("\n  TABLE 3: FIBER STRUCTURE & HOPF INVARIANT PROXY")
    print(f"  {'dim':>4}  {'Hopf?':>6}  {'eff fiber':>10}  {'particip':>10}  "
          f"{'SV entropy':>10}  {'Hopf proxy':>10}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    
    for dim in DIMS_TO_TEST:
        r = results[dim]; hopf_str = "YES" if r['is_hopf'] else "no"
        fi = r['fiber']; hp = r['hopf']
        print(f"  {dim:4d}  {hopf_str:>6}  "
              f"{fi['effective_fiber_dim']:10d}  {fi['participation_ratio']:10.3f}  "
              f"{fi['sv_entropy']:10.4f}  {hp['hopf_proxy']:10.4f}")


def analyze_step_function(results):
    print("\n" + "=" * 80)
    print("  CRITICAL ANALYSIS: STEP FUNCTION vs SMOOTH INTERPOLATION")
    print("=" * 80)
    
    # Test A: n=6 between n=4 and n=8
    step_test_a = None
    if all(d in results for d in [4, 6, 8]):
        sep_4 = results[4]['berry']['total_separation']
        sep_6 = results[6]['berry']['total_separation']
        sep_8 = results[8]['berry']['total_separation']
        midpoint = (sep_4 + sep_8) / 2
        
        dist_to_4 = abs(sep_6 - sep_4)
        dist_to_mid = abs(sep_6 - midpoint)
        dist_to_8 = abs(sep_6 - sep_8)
        
        print(f"\n  TEST A: Is n=6 a step function or smooth interpolation?")
        print(f"    Berry separation at n=4 (Hopf):      {sep_4:.6f}")
        print(f"    Berry separation at n=6 (non-Hopf):   {sep_6:.6f}")
        print(f"    Berry separation at n=8 (Hopf):      {sep_8:.6f}")
        print(f"    Smooth midpoint would be:             {midpoint:.6f}")
        print(f"    Distance n=6 to n=4:                  {dist_to_4:.6f}")
        print(f"    Distance n=6 to midpoint:             {dist_to_mid:.6f}")
        print(f"    Distance n=6 to n=8:                  {dist_to_8:.6f}")
        
        if dist_to_mid > 1e-10:
            ratio = dist_to_4 / dist_to_mid
        else:
            ratio = 0.0
        
        step_test_a = dist_to_4 < dist_to_mid
        result = "STEP FUNCTION" if step_test_a else "SMOOTH CURVE"
        print(f"    RESULT: n=6 is CLOSER to {'n=4' if step_test_a else 'midpoint'} -> {result}")
        print(f"    (Ratio: {ratio:.3f}, <1 supports step function)")
    
    # Test B: n=3 between n=2 and n=4
    step_test_b = None
    if all(d in results for d in [2, 3, 4]):
        sep_2 = results[2]['berry']['total_separation']
        sep_3 = results[3]['berry']['total_separation']
        sep_4 = results[4]['berry']['total_separation']
        midpoint = (sep_2 + sep_4) / 2
        
        dist_to_2 = abs(sep_3 - sep_2)
        dist_to_mid = abs(sep_3 - midpoint)
        
        print(f"\n  TEST B: Is n=3 a step function or smooth interpolation?")
        print(f"    Berry separation at n=2 (Hopf):      {sep_2:.6f}")
        print(f"    Berry separation at n=3 (non-Hopf):   {sep_3:.6f}")
        print(f"    Berry separation at n=4 (Hopf):      {sep_4:.6f}")
        print(f"    Smooth midpoint would be:             {midpoint:.6f}")
        print(f"    Distance n=3 to n=2:                  {dist_to_2:.6f}")
        print(f"    Distance n=3 to midpoint:             {dist_to_mid:.6f}")
        
        step_test_b = dist_to_2 < dist_to_mid
        result = "STEP FUNCTION" if step_test_b else "SMOOTH CURVE"
        print(f"    RESULT: n=3 is CLOSER to {'n=2' if step_test_b else 'midpoint'} -> {result}")
    
    # Test C: attractor contraction by dimension
    print(f"\n  TEST C: Attractor contraction ratios by dimension")
    for dim in DIMS_TO_TEST:
        r = results[dim]
        tag = " [HOPF]" if r['is_hopf'] else ""
        c = r['attractor']['mean_contraction']
        print(f"    n={dim:2d}{tag:>8}:  contraction = {c:.6f}  "
              f"{'<-- ATTRACTOR' if c < 0.95 else ''}")
    
    # Test D: jumps between consecutive Hopf dimensions
    print(f"\n  TEST D: Jumps at Hopf thresholds (Berry separation)")
    hopf_sorted = sorted(HOPF_DIMS)
    for i in range(len(hopf_sorted) - 1):
        d1, d2 = hopf_sorted[i], hopf_sorted[i+1]
        if d1 in results and d2 in results:
            s1 = results[d1]['berry']['total_separation']
            s2 = results[d2]['berry']['total_separation']
            intermediates = [d for d in DIMS_TO_TEST if d1 < d < d2 and d in results]
            print(f"    n={d1} -> n={d2}: separation {s1:.4f} -> {s2:.4f}")
            for d in intermediates:
                s = results[d]['berry']['total_separation']
                closer_to = f"n={d1}" if abs(s-s1) < abs(s-s2) else f"n={d2}"
                print(f"      n={d} (non-Hopf): {s:.4f}  (closer to {closer_to})")
    
    # Test E: beyond octonions
    if all(d in results for d in [8, 10]):
        sep_8 = results[8]['berry']['total_separation']
        sep_10 = results[10]['berry']['total_separation']
        print(f"\n  TEST E: Beyond octonions (n=10, no division algebra)")
        print(f"    n=8  (Hopf):     {sep_8:.6f}")
        print(f"    n=10 (non-Hopf): {sep_10:.6f}")
        if sep_10 <= sep_8:
            print(f"    RESULT: n=10 does NOT exceed n=8 -> consistent with Hopf ceiling")
        else:
            print(f"    RESULT: n=10 EXCEEDS n=8 -> challenges Hopf ceiling hypothesis")
    
    # Visualizations
    print(f"\n  VISUALIZATION: Berry separation vs dimension")
    print(f"  (each '#' = relative strength, '*' marks Hopf dimensions)")
    
    max_sep = max(results[d]['berry']['total_separation'] for d in DIMS_TO_TEST)
    if max_sep < 1e-10:
        max_sep = 1.0
    
    for dim in DIMS_TO_TEST:
        sep = results[dim]['berry']['total_separation']
        bar_len = int(50 * sep / max_sep)
        hopf_mark = "*" if dim in HOPF_DIMS else " "
        bar = "#" * bar_len
        print(f"  n={dim:2d} {hopf_mark} |{bar:<50}| {sep:.4f}")
    
    print(f"\n  VISUALIZATION: Attractor strength vs dimension")
    print(f"  (longer bar = stronger attractor, '*' marks Hopf dimensions)")
    
    for dim in DIMS_TO_TEST:
        c = results[dim]['attractor']['mean_contraction']
        strength = max(0, 1.0 - c)
        bar_len = int(50 * min(strength * 5, 1.0))  # scale up for visibility
        hopf_mark = "*" if dim in HOPF_DIMS else " "
        bar = "#" * bar_len
        print(f"  n={dim:2d} {hopf_mark} |{bar:<50}| c={c:.4f}")
    
    # Overall verdict
    print(f"\n  {'='*60}")
    print(f"  OVERALL VERDICT")
    print(f"  {'='*60}")
    
    tests_passed = 0; tests_total = 0
    
    for name, result in [("Test A (n=6 step function)", step_test_a),
                          ("Test B (n=3 step function)", step_test_b)]:
        if result is not None:
            tests_total += 1
            if result:
                tests_passed += 1
            print(f"  {name}:    {'PASS' if result else 'FAIL'}")
    
    # Clustering test
    n_cluster_ok = 0; n_cluster_tot = 0
    for dim in DIMS_TO_TEST:
        if dim not in HOPF_DIMS and dim > 1:
            lower_hopfs = [h for h in HOPF_DIMS if h < dim]
            upper_hopfs = [h for h in HOPF_DIMS if h > dim]
            if lower_hopfs and upper_hopfs:
                lh = max(lower_hopfs); uh = min(upper_hopfs)
                if lh in results and uh in results:
                    sl = results[lh]['berry']['total_separation']
                    su = results[uh]['berry']['total_separation']
                    sd = results[dim]['berry']['total_separation']
                    mid = (sl + su) / 2
                    n_cluster_tot += 1
                    if abs(sd - sl) < abs(sd - mid):
                        n_cluster_ok += 1
    
    if n_cluster_tot > 0:
        tests_total += 1
        frac = n_cluster_ok / n_cluster_tot
        passed = frac > 0.5
        if passed:
            tests_passed += 1
        print(f"  Test C (non-Hopf clusters with lower Hopf): {'PASS' if passed else 'FAIL'} ({n_cluster_ok}/{n_cluster_tot})")
    
    # Beyond-octonions test
    if all(d in results for d in [8, 10]):
        tests_total += 1
        ceiling_ok = results[10]['berry']['total_separation'] <= results[8]['berry']['total_separation'] * 1.1
        if ceiling_ok:
            tests_passed += 1
        print(f"  Test E (Hopf ceiling at n=8):    {'PASS' if ceiling_ok else 'FAIL'}")
    
    print(f"\n  Score: {tests_passed}/{tests_total} tests support step-function hypothesis")
    
    if tests_total > 0:
        if tests_passed == tests_total:
            print(f"\n  CONCLUSION: Contraction strength shows STEP-FUNCTION behavior")
            print(f"  at division algebra thresholds, consistent with Hopf fibration")
            print(f"  being the operative geometric structure.")
        elif tests_passed > tests_total / 2:
            print(f"\n  CONCLUSION: Mixed results -- partial step-function behavior.")
            print(f"  Some but not all metrics align with Hopf hypothesis.")
        else:
            print(f"\n  CONCLUSION: Contraction strength appears to interpolate SMOOTHLY.")
            print(f"  This CHALLENGES the Hopf fibration hypothesis.")


def attractor_basin_analysis(results):
    print("\n" + "=" * 80)
    print("  SUPPLEMENTARY: ATTRACTOR BASIN DETAIL")
    print("=" * 80)
    
    for dim in DIMS_TO_TEST:
        r = results[dim]
        is_hopf = "HOPF" if r['is_hopf'] else "    "
        print(f"\n  n={dim} [{is_hopf}]:")
        for eps, ratio in r['attractor']['contraction_by_eps']:
            bar_len = int(40 * min(ratio, 2.0) / 2.0)
            bar = "=" * bar_len
            marker = "|" if abs(ratio - 1.0) < 0.02 else " "
            direction = "<-|0>" if ratio < 0.98 else ("->away" if ratio > 1.02 else " ~neutral")
            print(f"    eps={eps:.2f}: {ratio:8.4f} [{bar:<40}]{marker} {direction}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    
    results = run_dimension_sweep()
    print_results_table(results)
    analyze_step_function(results)
    attractor_basin_analysis(results)
    
    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f} seconds")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()
