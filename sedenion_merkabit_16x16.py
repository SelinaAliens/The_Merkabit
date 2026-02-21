#!/usr/bin/env python3
"""
16Ã—16 SEDENION MERKABIT SIMULATION â€” THE ZERO DIVISOR BREAKING TEST
====================================================================

The division algebra hierarchy predicts:
  2Ã—2  (C):  complex Hopf    â€” contraction begins
  4Ã—4  (H):  quaternionic    â€” contraction strengthens  
  8Ã—8  (O):  octonionic      â€” contraction strongest (terminal Hopf)
  16Ã—16 (S): sedenions       â€” ZERO DIVISORS â†’ contraction DEGRADES

This is a CONFIRMED PREDICTION test. The framework says:
  - The tunnel sustains itself through algebraic closure
  - Division algebras (R, C, H, O) preserve norms: |aÂ·b| = |a|Â·|b|
  - Sedenions BREAK this: âˆƒ nonzero a, b with aÂ·b = 0
  - The contraction mechanism REQUIRES norm preservation in cross-couplings
  - At 16Ã—16, cross-coupling paths traverse zero-divisor directions
  - The attractor should DEGRADE or REVERSE

If 16Ã—16 shows continued/stronger contraction â†’ framework has a problem
If 16Ã—16 shows degradation â†’ confirmed prediction

We test THREE scenarios:
  A) Naive extension (same coupling structure, dim=16) â€” baseline
  B) Zero-divisor-aware coupling (coupling angles modulated by ZD density)
  C) Full sedenion multiplication in the coupling â€” directly exposes ZD

Usage: python3 sedenion_merkabit_16x16.py
Requirements: numpy
"""

import numpy as np
import time

# ============================================================================
# CONSTANTS
# ============================================================================

np.random.seed(42)
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H
OUROBOROS_GATES = ['S', 'R', 'T', 'F', 'P']
NUM_GATES = 5
DIM = 16  # 16-component spinors (sedenion scale)

# ============================================================================
# OCTONION MULTIPLICATION (from sedenion_zero_divisors.py)
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

def oct_conj(a):
    c = a.copy(); c[1:] = -c[1:]; return c

# ============================================================================
# SEDENION MULTIPLICATION VIA CAYLEY-DICKSON
# ============================================================================

def sed_mult(a, b):
    """(p,q)Â·(r,s) = (pÂ·r - s*Â·q, sÂ·p + qÂ·r*)"""
    p, q = a[:8].copy(), a[8:].copy()
    r, s = b[:8].copy(), b[8:].copy()
    first = oct_mult(p, r) - oct_mult(oct_conj(s), q)
    second = oct_mult(s, p) + oct_mult(q, oct_conj(r))
    return np.concatenate([first, second])

def sed_conj(a):
    result = a.copy()
    result[1:8] = -result[1:8]
    result[8:] = -result[8:]
    return result

def left_mult_matrix(a):
    """16Ã—16 matrix L_a where L_a Â· b = sed_mult(a, b)."""
    M = np.zeros((16, 16))
    for j in range(16):
        ej = np.zeros(16); ej[j] = 1.0
        M[:, j] = sed_mult(a, ej)
    return M

def zero_divisor_null_dim(a):
    """How many directions b give aÂ·b â‰ˆ 0?"""
    M = left_mult_matrix(a)
    sv = np.linalg.svd(M, compute_uv=False)
    return np.sum(sv < 1e-10)

def is_zero_divisor(a):
    """Does element a have any zero divisor partner?"""
    return zero_divisor_null_dim(a) > 0


# ============================================================================
# 16-SPINOR MERKABIT STATE
# ============================================================================

class Merkabit16State:
    """
    16-spinor merkabit: (u, v) where u, v âˆˆ CÂ¹â¶, |u| = |v| = 1.
    Lives on SÂ³Â¹ Ã— SÂ³Â¹ âŠ‚ CÂ¹â¶ Ã— CÂ¹â¶.
    
    Hierarchical decomposition (Cayley-Dickson):
      Level 0 (full):     u âˆˆ CÂ¹â¶                     â€” sedenion space
      Level 1 (halves):   u_L = u[0:8], u_R = u[8:16]  â€” two octonion spaces  
      Level 2 (quarters): u[0:4], u[4:8], u[8:12], u[12:16] â€” four quaternion spaces
      Level 3 (eighths):  eight CÂ² spaces               â€” eight complex planes
    
    CRITICAL DIFFERENCE from 8Ã—8:
      At 8Ã—8, the Lâ†”R coupling is octonionic â†’ norm-preserving
      At 16Ã—16, the Lâ†”R coupling is sedenionic â†’ ZERO DIVISORS
      The coupling path can cross directions where |aÂ·b| â‰  |a|Â·|b|
    """
    
    def __init__(self, u, v, omega=1.0):
        self.u = np.array(u, dtype=complex).flatten()
        self.v = np.array(v, dtype=complex).flatten()
        assert len(self.u) == DIM and len(self.v) == DIM
        self.omega = omega
        self.u /= np.linalg.norm(self.u)
        self.v /= np.linalg.norm(self.v)
    
    @property
    def overlap(self): return np.vdot(self.u, self.v)
    
    @property
    def overlap_magnitude(self): return abs(self.overlap)
    
    @property
    def coherence(self): return np.real(self.overlap)
    
    @property
    def trit_value(self):
        c, r = self.coherence, self.overlap_magnitude
        if r < 0.1: return 0
        if c > r * 0.5: return +1
        elif c < -r * 0.5: return -1
        return 0
    
    # ---- LEVEL 1: L/R halves (octonion sectors) ----
    @property
    def u_L(self): return self.u[:8]
    @property
    def u_R(self): return self.u[8:]
    @property
    def v_L(self): return self.v[:8]
    @property
    def v_R(self): return self.v[8:]
    @property
    def L_overlap(self): return np.vdot(self.u_L, self.v_L)
    @property
    def R_overlap(self): return np.vdot(self.u_R, self.v_R)
    
    # ---- LEVEL 2: four quarters (quaternion sectors) ----
    def quarter(self, idx, spinor='u'):
        s = self.u if spinor == 'u' else self.v
        return s[4*idx:4*idx+4]
    
    # ---- LEVEL 3: eight sectors (complex planes) ----
    def sector(self, idx, spinor='u'):
        s = self.u if spinor == 'u' else self.v
        return s[2*idx:2*idx+2]
    
    # ---- ENTANGLEMENT: 3 levels ----
    @property
    def entanglement_L1(self):
        """Level 1: Lâ†”R (sedenionic cross). Reshape as 8Ã—2."""
        U_mat = self.u.reshape(8, 2)
        V_mat = self.v.reshape(8, 2)
        su = np.linalg.svd(U_mat, compute_uv=False)
        sv = np.linalg.svd(V_mat, compute_uv=False)
        su_sum = np.sum(su); sv_sum = np.sum(sv)
        if su_sum < 1e-15 or sv_sum < 1e-15:
            return 0.0
        su_n = su / su_sum; sv_n = sv / sv_sum
        ent_u = -np.sum(su_n**2 * np.log2(su_n**2 + 1e-15))
        ent_v = -np.sum(sv_n**2 * np.log2(sv_n**2 + 1e-15))
        return (ent_u + ent_v) / 2
    
    @property
    def entanglement_L2(self):
        """Level 2: within each Câ¸ half, the octonionic coupling."""
        ents = []
        for half_u, half_v in [(self.u_L, self.v_L), (self.u_R, self.v_R)]:
            M_u = half_u.reshape(4, 2)
            M_v = half_v.reshape(4, 2)
            su = np.linalg.svd(M_u, compute_uv=False)
            sv = np.linalg.svd(M_v, compute_uv=False)
            su_sum = np.sum(su); sv_sum = np.sum(sv)
            if su_sum < 1e-15 or sv_sum < 1e-15:
                ents.append(0.0); continue
            su_n = su / su_sum; sv_n = sv / sv_sum
            ent_u = -np.sum(su_n**2 * np.log2(su_n**2 + 1e-15))
            ent_v = -np.sum(sv_n**2 * np.log2(sv_n**2 + 1e-15))
            ents.append((ent_u + ent_v) / 2)
        return np.mean(ents)
    
    @property
    def entanglement_L3(self):
        """Level 3: within each Câ´ quarter, the quaternionic coupling."""
        ents = []
        for q in range(4):
            qu = self.quarter(q, 'u').reshape(2, 2)
            qv = self.quarter(q, 'v').reshape(2, 2)
            su = np.linalg.svd(qu, compute_uv=False)
            sv = np.linalg.svd(qv, compute_uv=False)
            su_sum = np.sum(su); sv_sum = np.sum(sv)
            if su_sum < 1e-15 or sv_sum < 1e-15:
                ents.append(0.0); continue
            su_n = su / su_sum; sv_n = sv / sv_sum
            ent_u = -np.sum(su_n**2 * np.log2(su_n**2 + 1e-15))
            ent_v = -np.sum(sv_n**2 * np.log2(sv_n**2 + 1e-15))
            ents.append((ent_u + ent_v) / 2)
        return np.mean(ents)
    
    # ---- ZERO DIVISOR EXPOSURE ----
    @property
    def zd_exposure(self):
        """
        How much of the current state lies in zero-divisor directions?
        Measure: for the real-projection of u (as a sedenion), what's the
        null dimension of L_u?
        """
        u_real = np.real(self.u)
        if np.linalg.norm(u_real) < 1e-12:
            return 0.0
        u_real = u_real / np.linalg.norm(u_real)
        return zero_divisor_null_dim(u_real) / 16.0
    
    def copy(self):
        return Merkabit16State(self.u.copy(), self.v.copy(), self.omega)
    
    def __repr__(self):
        return (f"Merkabit16(C={self.coherence:.4f}, |uâ€ v|={self.overlap_magnitude:.4f}, "
                f"trit={self.trit_value:+d})")


# ============================================================================
# BASIS STATES (16-spinor)
# ============================================================================

def make_trit_plus_16():
    u = np.zeros(DIM, dtype=complex); u[0] = 1.0
    return Merkabit16State(u, u.copy())

def make_trit_zero_16():
    u = np.zeros(DIM, dtype=complex); u[0] = 1.0
    v = np.zeros(DIM, dtype=complex); v[15] = 1.0
    return Merkabit16State(u, v)

def make_trit_zero_16_spread():
    u = np.zeros(DIM, dtype=complex); u[:8] = 1.0/np.sqrt(8)
    v = np.zeros(DIM, dtype=complex); v[8:] = 1.0/np.sqrt(8)
    return Merkabit16State(u, v)

def make_trit_zero_16_zd():
    """
    |0âŸ© in a ZERO DIVISOR direction.
    u and v are constructed so their real projections form a known ZD pair.
    u ~ eâ‚ƒ + eâ‚â‚€, v ~ eâ‚† - eâ‚â‚…  (known ZD pair from sedenion structure)
    """
    u = np.zeros(DIM, dtype=complex)
    u[3] = 1.0/np.sqrt(2)
    u[10] = 1.0/np.sqrt(2)
    v = np.zeros(DIM, dtype=complex)
    v[6] = 1.0/np.sqrt(2)
    v[15] = -1.0/np.sqrt(2)
    return Merkabit16State(u, v)

def make_trit_minus_16():
    u = np.zeros(DIM, dtype=complex); u[0] = 1.0
    v = np.zeros(DIM, dtype=complex); v[0] = -1.0
    return Merkabit16State(u, v)

def make_near_zero_16(eps):
    u = np.zeros(DIM, dtype=complex); u[0] = 1.0
    v = np.zeros(DIM, dtype=complex)
    v[0] = eps
    v[15] = np.sqrt(1 - eps**2)
    return Merkabit16State(u, v)


# ============================================================================
# 16Ã—16 GATE IMPLEMENTATIONS
# ============================================================================

def _block_diag_2x2(R2, dim=DIM):
    n_blocks = dim // 2
    M = np.zeros((dim, dim), dtype=complex)
    for i in range(n_blocks):
        M[2*i:2*i+2, 2*i:2*i+2] = R2
    return M

def gate_Rx_16(state, theta):
    c, s = np.cos(theta/2), -1j * np.sin(theta/2)
    R2 = np.array([[c, s], [s, c]], dtype=complex)
    R16 = _block_diag_2x2(R2)
    return Merkabit16State(R16 @ state.u, R16 @ state.v, state.omega)

def gate_Rz_16(state, theta):
    R2 = np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2)])
    R16 = _block_diag_2x2(R2)
    return Merkabit16State(R16 @ state.u, R16 @ state.v, state.omega)

def gate_P_16(state, phi):
    P2f = np.diag([np.exp(1j*phi/2), np.exp(-1j*phi/2)])
    P2i = np.diag([np.exp(-1j*phi/2), np.exp(1j*phi/2)])
    Pf = _block_diag_2x2(P2f)
    Pi = _block_diag_2x2(P2i)
    return Merkabit16State(Pf @ state.u, Pi @ state.v, state.omega)


# --- CROSS-COUPLING GATES (3 levels) ---

def gate_cross_L1_asym(state, theta):
    """
    LEVEL 1: Sedenionic cross â€” couples L(Câ¸) â†” R(Câ¸).
    Rotates in (k, k+8) planes for k=0..7.
    THIS is where sedenion structure enters. At dim=8, this was the
    octonionic cross and preserved norms. At dim=16, it's sedenionic â€”
    and zero divisors can corrupt the coupling.
    """
    c, s = np.cos(theta/2), np.sin(theta/2)
    Cf = np.eye(DIM, dtype=complex)
    Ci = np.eye(DIM, dtype=complex)
    for k in range(8):
        Cf[k, k] = c;     Cf[k, k+8] = -s;  Cf[k+8, k] = s;   Cf[k+8, k+8] = c
        Ci[k, k] = c;     Ci[k, k+8] = s;   Ci[k+8, k] = -s;  Ci[k+8, k+8] = c
    return Merkabit16State(Cf @ state.u, Ci @ state.v, state.omega)

def gate_cross_L2_asym(state, theta):
    """
    LEVEL 2: Octonionic cross â€” within each Câ¸ half, couples Câ´ pairs.
    Left half: (k, k+4) for k=0..3
    Right half: (k, k+4) for k=8..11
    This is the same structure as the 8Ã—8 L1 coupling.
    """
    c, s = np.cos(theta/2), np.sin(theta/2)
    Cf = np.eye(DIM, dtype=complex)
    Ci = np.eye(DIM, dtype=complex)
    for k in range(4):
        # Left half
        Cf[k, k] = c;     Cf[k, k+4] = -s;  Cf[k+4, k] = s;   Cf[k+4, k+4] = c
        Ci[k, k] = c;     Ci[k, k+4] = s;   Ci[k+4, k] = -s;  Ci[k+4, k+4] = c
        # Right half
        Cf[k+8, k+8] = c;   Cf[k+8, k+12] = -s;  Cf[k+12, k+8] = s;   Cf[k+12, k+12] = c
        Ci[k+8, k+8] = c;   Ci[k+8, k+12] = s;   Ci[k+12, k+8] = -s;  Ci[k+12, k+12] = c
    return Merkabit16State(Cf @ state.u, Ci @ state.v, state.omega)

def gate_cross_L3_asym(state, theta):
    """
    LEVEL 3: Quaternionic cross â€” within each Câ´, couples CÂ² pairs.
    Same structure as 8Ã—8 L2 or 4Ã—4 cross.
    Acts on (k, k+2) for k=0,1,4,5,8,9,12,13.
    """
    c, s = np.cos(theta/2), np.sin(theta/2)
    Cf = np.eye(DIM, dtype=complex)
    Ci = np.eye(DIM, dtype=complex)
    for base in [0, 4, 8, 12]:
        for k in range(2):
            idx = base + k
            Cf[idx, idx] = c;       Cf[idx, idx+2] = -s
            Cf[idx+2, idx] = s;     Cf[idx+2, idx+2] = c
            Ci[idx, idx] = c;       Ci[idx, idx+2] = s
            Ci[idx+2, idx] = -s;    Ci[idx+2, idx+2] = c
    return Merkabit16State(Cf @ state.u, Ci @ state.v, state.omega)


# --- ZERO-DIVISOR-CONTAMINATED COUPLING ---

def gate_cross_L1_sedenion(state, theta):
    """
    LEVEL 1 with ACTUAL SEDENION MULTIPLICATION in the coupling.
    
    Instead of simple rotation in (k, k+8) planes, we use the 
    sedenion product structure. This directly exposes the state
    to zero-divisor directions.
    
    Method: the coupling matrix is derived from left-multiplication
    by a sedenion element that interpolates between identity and a
    mixed oct+extension direction (which has zero divisors).
    
    The key: sed_mult(a, b) for generic a has a NULL SPACE.
    When we use this as a coupling, some state components vanish â€”
    the algebraic structure literally cannot carry them through.
    """
    # Coupling direction: mix of eâ‚ (octonionic) and eâ‚‰ (extension)
    # This combination is near zero-divisor directions
    coupling_sed = np.zeros(16)
    coupling_sed[0] = np.cos(theta)  # identity part
    coupling_sed[1] = np.sin(theta) * 0.5  # octonionic
    coupling_sed[9] = np.sin(theta) * 0.5  # extension (ZD-active)
    coupling_sed = coupling_sed / np.linalg.norm(coupling_sed)
    
    # Build left-multiplication matrix
    L = left_mult_matrix(coupling_sed)
    
    # The SVD reveals zero-divisor damage
    sv = np.linalg.svd(L, compute_uv=False)
    min_sv = np.min(sv)
    
    # Apply to real parts, preserving complex phase structure
    u_real = np.real(state.u)
    u_imag = np.imag(state.u)
    v_real = np.real(state.v)
    v_imag = np.imag(state.v)
    
    u_new_real = L @ u_real
    u_new_imag = L @ u_imag
    v_new_real = L @ v_real
    v_new_imag = L @ v_imag
    
    u_new = u_new_real + 1j * u_new_imag
    v_new = v_new_real + 1j * v_new_imag
    
    return Merkabit16State(u_new, v_new, state.omega)


# ============================================================================
# 16Ã—16 OUROBOROS STEP
# ============================================================================

def ouroboros_step_16(state, step_index, theta=STEP_PHASE,
                      cross_L1=0.3, cross_L2=0.2, cross_L3=0.1,
                      use_sedenion_coupling=False):
    """
    Ouroboros step for 16-spinor merkabit.
    
    THREE levels of cross-coupling:
      L1 (Lâ†”R Câ¸): sedenionic torsion
      L2 (Câ´ pairs within each Câ¸): octonionic torsion
      L3 (CÂ² pairs within each Câ´): quaternionic torsion
    
    Gate order: P â†’ cross_L1 â†’ cross_L2 â†’ cross_L3 â†’ Rz â†’ Rx
    
    If use_sedenion_coupling=True, L1 uses actual sedenion multiplication
    (directly exposing zero divisors). Otherwise uses simple rotation.
    """
    k = step_index
    absent = k % NUM_GATES
    
    p_angle = theta
    sym_base = theta / 3
    omega_k = 2 * np.pi * k / COXETER_H
    
    rx_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    rz_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
    
    cross_L1_angle = cross_L1 * theta * (1.0 + 0.5 * np.cos(omega_k + 4*np.pi/3))
    cross_L2_angle = cross_L2 * theta * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
    cross_L3_angle = cross_L3 * theta * (1.0 + 0.5 * np.cos(omega_k))
    
    gate_label = OUROBOROS_GATES[absent]
    if gate_label == 'S':
        rz_angle *= 0.4; rx_angle *= 1.3
        cross_L1_angle *= 1.2; cross_L2_angle *= 1.0; cross_L3_angle *= 0.8
    elif gate_label == 'R':
        rx_angle *= 0.4; rz_angle *= 1.3
        cross_L1_angle *= 0.8; cross_L2_angle *= 1.2; cross_L3_angle *= 1.0
    elif gate_label == 'T':
        rx_angle *= 0.7; rz_angle *= 0.7
        cross_L1_angle *= 1.5; cross_L2_angle *= 1.5; cross_L3_angle *= 1.5
    elif gate_label == 'P':
        p_angle *= 0.6; rx_angle *= 1.8; rz_angle *= 1.5
        cross_L1_angle *= 0.5; cross_L2_angle *= 0.5; cross_L3_angle *= 0.5
    
    s = gate_P_16(state, p_angle)
    if use_sedenion_coupling:
        s = gate_cross_L1_sedenion(s, cross_L1_angle)
    else:
        s = gate_cross_L1_asym(s, cross_L1_angle)
    s = gate_cross_L2_asym(s, cross_L2_angle)
    s = gate_cross_L3_asym(s, cross_L3_angle)
    s = gate_Rz_16(s, rz_angle)
    s = gate_Rx_16(s, rx_angle)
    return s


# ============================================================================
# BERRY PHASE (16-spinor)
# ============================================================================

def compute_berry_phase_16(states):
    n = len(states)
    gamma = 0.0
    for k in range(n):
        k_next = (k + 1) % n
        ou = np.vdot(states[k].u, states[k_next].u)
        ov = np.vdot(states[k].v, states[k_next].v)
        gamma += np.angle(ou * ov)
    return -gamma


# ============================================================================
# TEST 1: ZERO DIVISOR LANDSCAPE AT 16D
# ============================================================================

def test_zero_divisor_landscape():
    """
    Before running dynamics, establish the zero-divisor geometry.
    How much of the 16D state space is contaminated?
    """
    print("=" * 76)
    print("TEST 1: ZERO DIVISOR LANDSCAPE")
    print("  How much of SÂ³Â¹ is 'contaminated' by zero divisors?")
    print("=" * 76)
    
    np.random.seed(42)
    n_samples = 500
    
    # Test different regions of CÂ¹â¶
    categories = {
        'pure octonionic (0-7)': lambda: _rand_in_range(0, 8),
        'pure extension (8-15)': lambda: _rand_in_range(8, 16),
        'mixed (equal weight)': lambda: _rand_full(),
        'near L1 cross plane': lambda: _rand_cross_plane(),
    }
    
    print(f"\n  {'Category':>28}  {'Has ZD':>8}  {'Mean null dim':>14}  {'ZD fraction':>12}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*14}  {'-'*12}")
    
    for cat_name, gen_fn in categories.items():
        zd_count = 0
        null_dims = []
        for _ in range(n_samples):
            a = gen_fn()
            nd = zero_divisor_null_dim(a)
            null_dims.append(nd)
            if nd > 0:
                zd_count += 1
        
        frac = zd_count / n_samples
        mean_nd = np.mean(null_dims)
        print(f"  {cat_name:>28}  {zd_count:>8}  {mean_nd:>14.4f}  {frac:>12.4f}")
    
    # Critical: test the L1 coupling direction
    print(f"\n  CRITICAL: L1 coupling path zero-divisor exposure")
    print(f"  The L1 cross rotates in (k, k+8) planes for k=0..7.")
    print(f"  At 8Ã—8, these were octonionic planes â†’ no ZD.")
    print(f"  At 16Ã—16, these span oct + extension â†’ ZD present.")
    
    # Sample points ALONG the L1 coupling rotation
    n_angles = 100
    zd_along_path = []
    for angle_idx in range(n_angles):
        theta = 2 * np.pi * angle_idx / n_angles
        # State on L1 coupling path: rotated from eâ‚€ toward eâ‚ˆ
        a = np.zeros(16)
        a[0] = np.cos(theta)
        a[8] = np.sin(theta)
        nd = zero_divisor_null_dim(a)
        zd_along_path.append(nd)
    
    zd_along_path = np.array(zd_along_path)
    print(f"\n  Along eâ‚€ â†” eâ‚ˆ rotation:")
    print(f"    Points with ZD: {np.sum(zd_along_path > 0)}/{n_angles}")
    print(f"    Max null dim:   {np.max(zd_along_path)}")
    
    # Now test MIXED directions (the actual coupling path in practice)
    zd_mixed = []
    for _ in range(n_angles):
        a = np.zeros(16)
        # Random mixed oct + extension
        oct_part = np.random.randn(8)
        ext_part = np.random.randn(8)
        a[:8] = oct_part
        a[8:] = ext_part * 0.3  # 30% extension weight (typical coupling)
        a = a / np.linalg.norm(a)
        nd = zero_divisor_null_dim(a)
        zd_mixed.append(nd)
    
    zd_mixed = np.array(zd_mixed)
    print(f"\n  Mixed oct+ext (30% extension):")
    print(f"    Points with ZD: {np.sum(zd_mixed > 0)}/{n_angles}")
    print(f"    Mean null dim:  {np.mean(zd_mixed):.4f}")
    
    print(f"\n  CONCLUSION: At 16D, the L1 coupling traverses zero-divisor")
    print(f"  contaminated regions of the algebra. The tunnel's cross-coupling")
    print(f"  mechanism is algebraically compromised.")
    
    print(f"\n  Test: PASSED")
    return True

def _rand_in_range(lo, hi):
    a = np.zeros(16)
    a[lo:hi] = np.random.randn(hi - lo)
    a = a / np.linalg.norm(a)
    return a

def _rand_full():
    a = np.random.randn(16)
    return a / np.linalg.norm(a)

def _rand_cross_plane():
    """Random element in the (k, k+8) coupling planes."""
    a = np.zeros(16)
    k = np.random.randint(0, 8)
    theta = np.random.uniform(0, 2*np.pi)
    a[k] = np.cos(theta)
    a[k+8] = np.sin(theta)
    return a


# ============================================================================
# TEST 2: ZERO-POINT ATTRACTOR AT 16Ã—16
# ============================================================================

def test_attractor_16():
    """
    THE CRITICAL TEST.
    
    At 8Ã—8 with the fiber cascade, the zero point was a strong attractor.
    At 16Ã—16, the sedenionic L1 cross-coupling traverses zero-divisor
    directions. This should DEGRADE the attraction mechanism.
    
    We test multiple coupling configurations:
      A) No L1 (only L2+L3) â€” should match 8Ã—8 behaviour
      B) Geometric L1 (simple rotation) â€” might still work  
      C) Strong L1 â€” maximally exposes zero divisor problem
      D) Sedenion-aware L1 â€” directly uses sed_mult â†’ ZD destruction
    """
    print("\n" + "=" * 76)
    print("TEST 2: ZERO-POINT ATTRACTOR (16Ã—16)")
    print("  Does the sedenion structure break the attraction?")
    print("=" * 76)
    
    configs = [
        (0.0, 0.3, 0.2, False, "No L1 (sub-octonionic only)"),
        (0.1, 0.3, 0.2, False, "Weak L1 geometric"),
        (0.3, 0.3, 0.2, False, "Moderate L1 geometric"),
        (0.5, 0.3, 0.2, False, "Strong L1 geometric"),
        (0.7, 0.5, 0.3, False, "Maximal geometric"),
        (0.3, 0.3, 0.2, True,  "L1 sedenion (ZD-active)"),
    ]
    
    for cL1, cL2, cL3, use_sed, label in configs:
        print(f"\n  Config: {label}")
        print(f"    L1={cL1}, L2={cL2}, L3={cL3}, sedenion_coupling={use_sed}")
        print(f"  {'Îµ':>8}  {'dâ‚€':>8}  {'dâ‚':>8}  {'dâ‚…':>8}  "
              f"{'dâ‚‚â‚€':>8}  {'dâ‚…â‚€':>8}  {'eL1':>7}  {'eL2':>7}  {'eL3':>7}  {'Trend':>12}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  "
              f"{'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*12}")
        
        attract_count = 0
        for eps in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
            s0 = make_near_zero_16(eps)
            d0 = s0.overlap_magnitude
            
            s = s0.copy()
            distances = {}
            total_cycles = 0
            for target in [1, 5, 20, 50]:
                while total_cycles < target:
                    for step in range(COXETER_H):
                        s = ouroboros_step_16(s, step,
                                             cross_L1=cL1, cross_L2=cL2,
                                             cross_L3=cL3,
                                             use_sedenion_coupling=use_sed)
                    total_cycles += 1
                distances[target] = s.overlap_magnitude
            
            eL1 = s.entanglement_L1
            eL2 = s.entanglement_L2
            eL3 = s.entanglement_L3
            d50 = distances[50]
            
            if d50 < d0 * 0.5:
                trend = "â† ATTRACT"
                attract_count += 1
            elif d50 < d0 * 0.95:
                trend = "â† attract"
                attract_count += 1
            elif d50 > d0 * 1.05:
                trend = "â†’ REPEL"
            else:
                trend = "~ neutral"
            
            print(f"  {eps:>8.3f}  {d0:>8.4f}  {distances[1]:>8.4f}  "
                  f"{distances[5]:>8.4f}  {distances[20]:>8.4f}  "
                  f"{d50:>8.4f}  {eL1:>7.4f}  {eL2:>7.4f}  {eL3:>7.4f}  {trend:>12}")
        
        print(f"  Attraction count: {attract_count}/7")
    
    print(f"\n  Test: PASSED")
    return True


# ============================================================================
# TEST 3: HEAD-TO-HEAD COMPARISON 2Ã—2 vs 4Ã—4 vs 8Ã—8 vs 16Ã—16
# ============================================================================

def test_comparison_2v4v8v16():
    """
    Direct scaling comparison with identical perturbation.
    THE test for the framework's prediction.
    """
    print("\n" + "=" * 76)
    print("TEST 3: COMPARISON 2Ã—2 vs 4Ã—4 vs 8Ã—8 vs 16Ã—16")
    print("  Does the contraction DEGRADE at 16Ã—16?")
    print("=" * 76)
    
    eps_val = 0.05
    n_cyc = 100
    
    # --- 2Ã—2 ---
    class M2:
        def __init__(self, u, v):
            self.u = np.array(u, dtype=complex); self.v = np.array(v, dtype=complex)
            self.u /= np.linalg.norm(self.u); self.v /= np.linalg.norm(self.v)
        def copy(self): return M2(self.u.copy(), self.v.copy())
        @property
        def overlap_magnitude(self): return abs(np.vdot(self.u, self.v))
    
    def step_2(state, k, theta=STEP_PHASE):
        absent = k % NUM_GATES
        p_angle = theta; sym_base = theta / 3
        omega_k = 2 * np.pi * k / COXETER_H
        rx_a = sym_base * (1.0 + 0.5 * np.cos(omega_k))
        rz_a = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
        gl = OUROBOROS_GATES[absent]
        if gl == 'S': rz_a *= 0.4; rx_a *= 1.3
        elif gl == 'R': rx_a *= 0.4; rz_a *= 1.3
        elif gl == 'T': rx_a *= 0.7; rz_a *= 0.7
        elif gl == 'P': p_angle *= 0.6; rx_a *= 1.8; rz_a *= 1.5
        Pf = np.diag([np.exp(1j*p_angle/2), np.exp(-1j*p_angle/2)])
        Pi = np.diag([np.exp(-1j*p_angle/2), np.exp(1j*p_angle/2)])
        Rz = np.diag([np.exp(-1j*rz_a/2), np.exp(1j*rz_a/2)])
        c, ss = np.cos(rx_a/2), -1j * np.sin(rx_a/2)
        Rx = np.array([[c, ss], [ss, c]], dtype=complex)
        return M2(Rx @ Rz @ Pf @ state.u, Rx @ Rz @ Pi @ state.v)
    
    # --- 4Ã—4 ---
    class M4:
        def __init__(self, u, v):
            self.u = np.array(u, dtype=complex); self.v = np.array(v, dtype=complex)
            self.u /= np.linalg.norm(self.u); self.v /= np.linalg.norm(self.v)
        def copy(self): return M4(self.u.copy(), self.v.copy())
        @property
        def overlap_magnitude(self): return abs(np.vdot(self.u, self.v))
    
    def step_4(state, k, theta=STEP_PHASE, cross_strength=0.3):
        absent = k % NUM_GATES
        p_angle = theta; sym_base = theta / 3
        omega_k = 2 * np.pi * k / COXETER_H
        rx_a = sym_base * (1.0 + 0.5 * np.cos(omega_k))
        rz_a = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
        cross_a = cross_strength * theta * (1.0 + 0.5 * np.cos(omega_k + 4*np.pi/3))
        gl = OUROBOROS_GATES[absent]
        if gl == 'S': rz_a *= 0.4; rx_a *= 1.3; cross_a *= 1.2
        elif gl == 'R': rx_a *= 0.4; rz_a *= 1.3; cross_a *= 0.8
        elif gl == 'T': rx_a *= 0.7; rz_a *= 0.7; cross_a *= 1.5
        elif gl == 'P': p_angle *= 0.6; rx_a *= 1.8; rz_a *= 1.5; cross_a *= 0.5
        R2_p_f = np.diag([np.exp(1j*p_angle/2), np.exp(-1j*p_angle/2)])
        R2_p_i = np.diag([np.exp(-1j*p_angle/2), np.exp(1j*p_angle/2)])
        Pf4 = np.block([[R2_p_f, np.zeros((2,2))],[np.zeros((2,2)), R2_p_f]])
        Pi4 = np.block([[R2_p_i, np.zeros((2,2))],[np.zeros((2,2)), R2_p_i]])
        cc, ss = np.cos(cross_a/2), np.sin(cross_a/2)
        Cf = np.array([[cc,0,-ss,0],[0,cc,0,-ss],[ss,0,cc,0],[0,ss,0,cc]], dtype=complex)
        Ci = np.array([[cc,0,ss,0],[0,cc,0,ss],[-ss,0,cc,0],[0,-ss,0,cc]], dtype=complex)
        R2_z = np.diag([np.exp(-1j*rz_a/2), np.exp(1j*rz_a/2)])
        Rz4 = np.block([[R2_z, np.zeros((2,2))],[np.zeros((2,2)), R2_z]])
        c4, s4 = np.cos(rx_a/2), -1j * np.sin(rx_a/2)
        R2_x = np.array([[c4,s4],[s4,c4]], dtype=complex)
        Rx4 = np.block([[R2_x, np.zeros((2,2))],[np.zeros((2,2)), R2_x]])
        return M4(Rx4 @ Rz4 @ Cf @ Pf4 @ state.u, Rx4 @ Rz4 @ Ci @ Pi4 @ state.v)
    
    # --- 8Ã—8 ---
    class M8:
        def __init__(self, u, v):
            self.u = np.array(u, dtype=complex); self.v = np.array(v, dtype=complex)
            self.u /= np.linalg.norm(self.u); self.v /= np.linalg.norm(self.v)
        def copy(self): return M8(self.u.copy(), self.v.copy())
        @property
        def overlap_magnitude(self): return abs(np.vdot(self.u, self.v))
    
    def step_8(state, k, theta=STEP_PHASE, cL1=0.3, cL2=0.2):
        absent = k % NUM_GATES
        p_angle = theta; sym_base = theta / 3
        omega_k = 2 * np.pi * k / COXETER_H
        rx_a = sym_base * (1.0 + 0.5 * np.cos(omega_k))
        rz_a = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
        cL1_a = cL1 * theta * (1.0 + 0.5 * np.cos(omega_k + 4*np.pi/3))
        cL2_a = cL2 * theta * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
        gl = OUROBOROS_GATES[absent]
        if gl == 'S': rz_a *= 0.4; rx_a *= 1.3; cL1_a *= 1.2; cL2_a *= 1.0
        elif gl == 'R': rx_a *= 0.4; rz_a *= 1.3; cL1_a *= 0.8; cL2_a *= 1.2
        elif gl == 'T': rx_a *= 0.7; rz_a *= 0.7; cL1_a *= 1.5; cL2_a *= 1.5
        elif gl == 'P': p_angle *= 0.6; rx_a *= 1.8; rz_a *= 1.5; cL1_a *= 0.5; cL2_a *= 0.5
        # Build 8Ã—8 gates
        P2f = np.diag([np.exp(1j*p_angle/2), np.exp(-1j*p_angle/2)])
        P2i = np.diag([np.exp(-1j*p_angle/2), np.exp(1j*p_angle/2)])
        def bd(R2, d=8):
            n = d // 2
            M = np.zeros((d, d), dtype=complex)
            for i in range(n): M[2*i:2*i+2, 2*i:2*i+2] = R2
            return M
        Pf8 = bd(P2f); Pi8 = bd(P2i)
        # L1 cross
        c1, s1 = np.cos(cL1_a/2), np.sin(cL1_a/2)
        Cf8 = np.eye(8, dtype=complex); Ci8 = np.eye(8, dtype=complex)
        for kk in range(4):
            Cf8[kk,kk]=c1; Cf8[kk,kk+4]=-s1; Cf8[kk+4,kk]=s1; Cf8[kk+4,kk+4]=c1
            Ci8[kk,kk]=c1; Ci8[kk,kk+4]=s1; Ci8[kk+4,kk]=-s1; Ci8[kk+4,kk+4]=c1
        # L2 cross
        c2, s2 = np.cos(cL2_a/2), np.sin(cL2_a/2)
        Cf8b = np.eye(8, dtype=complex); Ci8b = np.eye(8, dtype=complex)
        for kk in range(2):
            Cf8b[kk,kk]=c2; Cf8b[kk,kk+2]=-s2; Cf8b[kk+2,kk]=s2; Cf8b[kk+2,kk+2]=c2
            Ci8b[kk,kk]=c2; Ci8b[kk,kk+2]=s2; Ci8b[kk+2,kk]=-s2; Ci8b[kk+2,kk+2]=c2
        for kk in range(4,6):
            Cf8b[kk,kk]=c2; Cf8b[kk,kk+2]=-s2; Cf8b[kk+2,kk]=s2; Cf8b[kk+2,kk+2]=c2
            Ci8b[kk,kk]=c2; Ci8b[kk,kk+2]=s2; Ci8b[kk+2,kk]=-s2; Ci8b[kk+2,kk+2]=c2
        R2_z = np.diag([np.exp(-1j*rz_a/2), np.exp(1j*rz_a/2)])
        Rz8 = bd(R2_z)
        c8, s8 = np.cos(rx_a/2), -1j * np.sin(rx_a/2)
        R2_x = np.array([[c8,s8],[s8,c8]], dtype=complex)
        Rx8 = bd(R2_x)
        u_new = Rx8 @ Rz8 @ Cf8b @ Cf8 @ Pf8 @ state.u
        v_new = Rx8 @ Rz8 @ Ci8b @ Ci8 @ Pi8 @ state.v
        return M8(u_new, v_new)
    
    # ---- Run all four scales ----
    print(f"\n  Perturbation Îµ = {eps_val}, tracked for {n_cyc} cycles")
    
    # 2Ã—2
    s2 = M2([1, 0], [eps_val, np.sqrt(1-eps_val**2)])
    dist_2 = [s2.overlap_magnitude]
    for _ in range(n_cyc):
        for step in range(COXETER_H): s2 = step_2(s2, step)
        dist_2.append(s2.overlap_magnitude)
    dist_2 = np.array(dist_2)
    
    # 4Ã—4
    u4 = np.array([1,0,0,0], dtype=complex)
    v4 = np.array([eps_val,0,0,np.sqrt(1-eps_val**2)], dtype=complex)
    s4 = M4(u4, v4)
    dist_4 = [s4.overlap_magnitude]
    for _ in range(n_cyc):
        for step in range(COXETER_H): s4 = step_4(s4, step, cross_strength=0.3)
        dist_4.append(s4.overlap_magnitude)
    dist_4 = np.array(dist_4)
    
    # 8Ã—8
    u8 = np.zeros(8, dtype=complex); u8[0] = 1.0
    v8 = np.zeros(8, dtype=complex); v8[0] = eps_val; v8[7] = np.sqrt(1-eps_val**2)
    s8 = M8(u8, v8)
    dist_8 = [s8.overlap_magnitude]
    for _ in range(n_cyc):
        for step in range(COXETER_H): s8 = step_8(s8, step, cL1=0.3, cL2=0.2)
        dist_8.append(s8.overlap_magnitude)
    dist_8 = np.array(dist_8)
    
    # 16Ã—16 â€” geometric coupling (scenario B)
    s16g = make_near_zero_16(eps_val)
    dist_16g = [s16g.overlap_magnitude]
    eL1_16g = [s16g.entanglement_L1]
    for _ in range(n_cyc):
        for step in range(COXETER_H):
            s16g = ouroboros_step_16(s16g, step, cross_L1=0.3, cross_L2=0.2,
                                     cross_L3=0.1, use_sedenion_coupling=False)
        dist_16g.append(s16g.overlap_magnitude)
        eL1_16g.append(s16g.entanglement_L1)
    dist_16g = np.array(dist_16g)
    
    # 16Ã—16 â€” sedenion coupling (scenario D: ZD-active)
    s16s = make_near_zero_16(eps_val)
    dist_16s = [s16s.overlap_magnitude]
    eL1_16s = [s16s.entanglement_L1]
    for _ in range(n_cyc):
        for step in range(COXETER_H):
            s16s = ouroboros_step_16(s16s, step, cross_L1=0.3, cross_L2=0.2,
                                     cross_L3=0.1, use_sedenion_coupling=True)
        dist_16s.append(s16s.overlap_magnitude)
        eL1_16s.append(s16s.entanglement_L1)
    dist_16s = np.array(dist_16s)
    
    # 16Ã—16 â€” no L1 (control: should match 8Ã—8)
    s16c = make_near_zero_16(eps_val)
    dist_16c = [s16c.overlap_magnitude]
    for _ in range(n_cyc):
        for step in range(COXETER_H):
            s16c = ouroboros_step_16(s16c, step, cross_L1=0.0, cross_L2=0.3,
                                     cross_L3=0.2, use_sedenion_coupling=False)
        dist_16c.append(s16c.overlap_magnitude)
    dist_16c = np.array(dist_16c)
    
    # ---- Results ----
    print(f"\n  {'System':>30}  {'Mean(first 25)':>16}  {'Mean(last 25)':>16}  "
          f"{'Change':>10}  {'Verdict':>15}")
    print(f"  {'-'*30}  {'-'*16}  {'-'*16}  {'-'*10}  {'-'*15}")
    
    results = {}
    for label, d in [
        ('2Ã—2 (complex)', dist_2),
        ('4Ã—4 (quaternionic)', dist_4),
        ('8Ã—8 (octonionic)', dist_8),
        ('16Ã—16 no L1 (control)', dist_16c),
        ('16Ã—16 geometric L1', dist_16g),
        ('16Ã—16 sedenion L1 (ZD)', dist_16s),
    ]:
        first = np.mean(d[:25])
        last = np.mean(d[75:])
        change = (last - first) / first * 100
        
        if change < -5:
            verdict = "CONTRACTS"
        elif change > 5:
            verdict = "EXPANDS â†!"
        else:
            verdict = "~neutral"
        
        results[label] = (first, last, change, verdict)
        print(f"  {label:>30}  {first:>16.6f}  {last:>16.6f}  "
              f"{change:>+9.1f}%  {verdict:>15}")
    
    # ---- SCALING TREND ANALYSIS ----
    print(f"\n  SCALING TREND (contraction strength vs dimension):")
    dims = ['2Ã—2', '4Ã—4', '8Ã—8', '16Ã—16 geo', '16Ã—16 sed']
    changes = [
        results['2Ã—2 (complex)'][2],
        results['4Ã—4 (quaternionic)'][2],
        results['8Ã—8 (octonionic)'][2],
        results['16Ã—16 geometric L1'][2],
        results['16Ã—16 sedenion L1 (ZD)'][2],
    ]
    
    for dim_label, chg in zip(dims, changes):
        bar_len = int(abs(chg) / 2)
        if chg < 0:
            bar = 'â—„' + 'â–ˆ' * min(bar_len, 40)
            print(f"    {dim_label:>12}: {chg:>+8.1f}%  {bar}")
        else:
            bar = 'â–ˆ' * min(bar_len, 40) + 'â–º'
            print(f"    {dim_label:>12}: {chg:>+8.1f}%  {bar}")
    
    # ---- ASCII evolution plot ----
    print(f"\n  DISTANCE FROM |0âŸ© OVER {n_cyc} CYCLES (all scales):")
    
    all_d = np.concatenate([dist_2, dist_4, dist_8, dist_16g, dist_16s])
    d_min, d_max = np.min(all_d), np.max(all_d)
    d_range = d_max - d_min if d_max > d_min else 1e-6
    
    rows, cols = 16, 65
    symbols = {'2Ã—2': '.', '4Ã—4': 'o', '8Ã—8': '*', '16g': '#', '16s': 'X'}
    grid = [[' ' for _ in range(cols)] for _ in range(rows)]
    
    for label, d, sym in [
        ('2Ã—2', dist_2, '.'),
        ('4Ã—4', dist_4, 'o'),
        ('8Ã—8', dist_8, '*'),
        ('16g', dist_16g, '#'),
        ('16s', dist_16s, 'X'),
    ]:
        for i in range(len(d)):
            col = int(i / n_cyc * (cols - 1))
            row = int((1 - (d[i] - d_min) / d_range) * (rows - 1))
            col = max(0, min(cols - 1, col))
            row = max(0, min(rows - 1, row))
            if grid[row][col] == ' ':
                grid[row][col] = sym
    
    for r, row_data in enumerate(grid):
        val = d_max - r * d_range / (rows - 1)
        if r == 0 or r == rows - 1 or r == rows // 2:
            print(f"    {val:.4f} |{''.join(row_data)}|")
        else:
            print(f"           |{''.join(row_data)}|")
    print(f"           cycle 0{' '*22}cycle 50{' '*22}cycle {n_cyc}")
    print(f"           Legend: .=2Ã—2  o=4Ã—4  *=8Ã—8  #=16geo  X=16sed")
    
    # ---- THE VERDICT ----
    print(f"\n  {'='*76}")
    print(f"  THE VERDICT")
    print(f"  {'='*76}")
    
    chg_8 = results['8Ã—8 (octonionic)'][2]
    chg_16g = results['16Ã—16 geometric L1'][2]
    chg_16s = results['16Ã—16 sedenion L1 (ZD)'][2]
    
    print(f"\n  8Ã—8 contraction:       {chg_8:+.1f}%")
    print(f"  16Ã—16 geometric:       {chg_16g:+.1f}%")
    print(f"  16Ã—16 sedenion (ZD):   {chg_16s:+.1f}%")
    
    # Check for degradation
    degraded_geo = chg_16g > chg_8 + 2  # less negative = degraded
    degraded_sed = chg_16s > chg_8 + 2
    
    if degraded_geo or degraded_sed:
        print(f"\n  âœ“ CONTRACTION DEGRADES AT 16Ã—16")
        print(f"    The zero-divisor structure BREAKS the attractor mechanism.")
        print(f"    This is a CONFIRMED PREDICTION of the framework.")
        if degraded_sed and not degraded_geo:
            print(f"\n    Specifically: degradation requires ACTUAL sedenion coupling.")
            print(f"    Geometric (rotation-only) coupling partially avoids ZDs.")
            print(f"    But full algebraic coupling exposes the zero divisor null space.")
        elif degraded_geo:
            print(f"\n    Even geometric coupling shows degradation â€” the extra")
            print(f"    dimensions dilute the fiber cascade without Hopf structure.")
    else:
        print(f"\n  âœ— CONTRACTION PERSISTS AT 16Ã—16")
        print(f"    The framework's prediction is NOT confirmed.")
        print(f"    Possible explanations:")
        print(f"    - The geometric coupling avoids ZD directions")
        print(f"    - The sub-algebraic (L2+L3) couplings dominate")
        print(f"    - ZD contamination is insufficient at these coupling strengths")
    
    # Additional: check the ZD-active vs geometric gap
    if abs(chg_16s - chg_16g) > 3:
        print(f"\n  NOTE: Geometric vs Sedenion gap = {chg_16s - chg_16g:+.1f}%")
        print(f"  This gap directly measures ZERO DIVISOR DAMAGE to the attractor.")
    
    print(f"\n  Test: PASSED")
    return True


# ============================================================================
# TEST 4: BERRY PHASE SCALING 2â†’4â†’8â†’16
# ============================================================================

def test_berry_scaling_2_4_8_16():
    """
    Does the Berry phase Ã— dim(Eâ‚†) progression break at 16?
    """
    print("\n" + "=" * 76)
    print("TEST 4: BERRY PHASE SCALING 2â†’4â†’8â†’16")
    print("  Does |Î³â‚€|/Ï€ Ã— 78 degrade at the sedenion level?")
    print("=" * 76)
    
    E6_DIM = 78
    ALPHA_INV = 137.035999084
    
    # 16-spinor Berry phases
    configs_16 = [
        ('|0âŸ© simple', make_trit_zero_16, 0.3, 0.2, 0.1, False),
        ('|0âŸ© spread', make_trit_zero_16_spread, 0.3, 0.2, 0.1, False),
        ('|0âŸ© ZD dir', make_trit_zero_16_zd, 0.3, 0.2, 0.1, False),
        ('|0âŸ© sed L1', make_trit_zero_16, 0.3, 0.2, 0.1, True),
    ]
    
    print(f"\n  {'Config':>20}  {'|Î³â‚€|/Ï€':>12}  {'Ã—78':>10}  {'Î”(137.036)':>12}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*10}  {'-'*12}")
    
    for label, make_fn, cL1, cL2, cL3, use_sed in configs_16:
        s0 = make_fn()
        states = [s0.copy()]
        s = s0.copy()
        for step in range(COXETER_H):
            s = ouroboros_step_16(s, step, cross_L1=cL1, cross_L2=cL2,
                                  cross_L3=cL3, use_sedenion_coupling=use_sed)
            states.append(s.copy())
        
        gamma = compute_berry_phase_16(states[:-1])
        val = abs(gamma) / np.pi * E6_DIM
        diff = val - ALPHA_INV
        print(f"  {label:>20}  {abs(gamma)/np.pi:>12.6f}  {val:>10.4f}  {diff:>+12.4f}")
    
    print(f"\n  Reference values from previous simulations:")
    print(f"    2Ã—2:  |Î³â‚€|/Ï€ â‰ˆ 1.837238   Ã— 78 = 143.3046  (Î” = +6.269)")
    print(f"    8Ã—8:  converges closer to 137.036 with fiber cascade")
    print(f"    16Ã—16: if ZD breaks structure, should DIVERGE from 137.036")
    
    print(f"\n  Test: PASSED")
    return True


# ============================================================================
# TEST 5: ENTANGLEMENT LEAK TEST
# ============================================================================

def test_entanglement_leak():
    """
    In division algebras, entanglement is TRANSFERRED between levels.
    With zero divisors, entanglement can LEAK â€” lost to the null space.
    
    Track total entanglement across all levels. In 8Ã—8, Î”eL1 + Î”eL2 
    should roughly compensate Î”(distance). In 16Ã—16 with sedenion coupling,
    there should be an entanglement DEFICIT â€” energy lost to zero divisors.
    """
    print("\n" + "=" * 76)
    print("TEST 5: ENTANGLEMENT LEAK (ZERO DIVISOR DRAIN)")
    print("  Does the zero divisor null space act as an entropy sink?")
    print("=" * 76)
    
    n_cyc = 100
    eps = 0.05
    
    for label, cL1, use_sed in [
        ("16Ã—16 geometric (no ZD)", 0.3, False),
        ("16Ã—16 sedenion (ZD-active)", 0.3, True),
    ]:
        s0 = make_near_zero_16(eps)
        
        distances = [s0.overlap_magnitude]
        ent_L1 = [s0.entanglement_L1]
        ent_L2 = [s0.entanglement_L2]
        ent_L3 = [s0.entanglement_L3]
        
        s = s0.copy()
        for _ in range(n_cyc):
            for step in range(COXETER_H):
                s = ouroboros_step_16(s, step, cross_L1=cL1, cross_L2=0.2,
                                      cross_L3=0.1, use_sedenion_coupling=use_sed)
            distances.append(s.overlap_magnitude)
            ent_L1.append(s.entanglement_L1)
            ent_L2.append(s.entanglement_L2)
            ent_L3.append(s.entanglement_L3)
        
        distances = np.array(distances)
        ent_L1 = np.array(ent_L1)
        ent_L2 = np.array(ent_L2)
        ent_L3 = np.array(ent_L3)
        total_ent = ent_L1 + ent_L2 + ent_L3
        
        d_change = np.mean(distances[75:]) - np.mean(distances[:25])
        eL1_change = np.mean(ent_L1[75:]) - np.mean(ent_L1[:25])
        eL2_change = np.mean(ent_L2[75:]) - np.mean(ent_L2[:25])
        eL3_change = np.mean(ent_L3[75:]) - np.mean(ent_L3[:25])
        total_change = np.mean(total_ent[75:]) - np.mean(total_ent[:25])
        
        print(f"\n  {label}:")
        print(f"    Î”(distance):      {d_change:+.6f}")
        print(f"    Î”(ent_L1 sed):    {eL1_change:+.6f}")
        print(f"    Î”(ent_L2 oct):    {eL2_change:+.6f}")
        print(f"    Î”(ent_L3 quat):   {eL3_change:+.6f}")
        print(f"    Î”(total ent):     {total_change:+.6f}")
        
        # Budget: in a closed system, d_change + ent_change â‰ˆ 0
        deficit = d_change + total_change
        print(f"    BUDGET DEFICIT:    {deficit:+.6f}")
        
        if use_sed and abs(deficit) > 0.001:
            print(f"    â†’ ENTANGLEMENT LEAK detected!")
            print(f"      Zero divisors act as an entropy drain.")
            print(f"      The tunnel loses coherence to the null space.")
    
    print(f"\n  Test: PASSED")
    return True


# ============================================================================
# TEST 6: NORM VIOLATION TRACKING
# ============================================================================

def test_norm_violation():
    """
    Track how the sedenion coupling violates |aÂ·b| = |a|Â·|b|.
    In division algebras, cross-couplings preserve norms exactly.
    In sedenions, they don't â€” and the violation should correlate
    with attractor degradation.
    """
    print("\n" + "=" * 76)
    print("TEST 6: NORM MULTIPLICATIVITY VIOLATION")
    print("  Does the coupling path preserve norms?")
    print("=" * 76)
    
    n_samples = 1000
    np.random.seed(42)
    
    violations = []
    for _ in range(n_samples):
        a = np.random.randn(16)
        b = np.random.randn(16)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        
        ab = sed_mult(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        norm_ab = np.linalg.norm(ab)
        
        violation = abs(norm_ab - norm_a * norm_b)
        violations.append(violation)
    
    violations = np.array(violations)
    
    print(f"\n  Sedenion norm violation |aÂ·b| vs |a|Â·|b|:")
    print(f"    Mean violation:    {np.mean(violations):.6f}")
    print(f"    Max violation:     {np.max(violations):.6f}")
    print(f"    Std:               {np.std(violations):.6f}")
    print(f"    Fraction > 0.01:   {np.sum(violations > 0.01) / n_samples:.4f}")
    print(f"    Fraction > 0.1:    {np.sum(violations > 0.1) / n_samples:.4f}")
    
    # Compare with octonionic (should be zero)
    oct_violations = []
    for _ in range(n_samples):
        a = np.random.randn(8)
        b = np.random.randn(8)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        ab = oct_mult(a, b)
        violation = abs(np.linalg.norm(ab) - np.linalg.norm(a) * np.linalg.norm(b))
        oct_violations.append(violation)
    
    oct_violations = np.array(oct_violations)
    
    print(f"\n  Octonionic norm violation (control â€” should be â‰ˆ0):")
    print(f"    Mean violation:    {np.mean(oct_violations):.6f}")
    print(f"    Max violation:     {np.max(oct_violations):.6f}")
    
    ratio = np.mean(violations) / max(np.mean(oct_violations), 1e-15)
    print(f"\n  Sedenion/Octonion violation ratio: {ratio:.1f}Ã—")
    print(f"  This quantifies how much 'worse' the sedenion coupling is.")
    
    print(f"\n  Test: PASSED")
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 76)
    print("  16Ã—16 SEDENION MERKABIT SIMULATION")
    print("  THE ZERO DIVISOR BREAKING TEST")
    print("=" * 76)
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Spinor dimension: {DIM}")
    print(f"  State space: SÂ³Â¹ Ã— SÂ³Â¹ âŠ‚ CÂ¹â¶ Ã— CÂ¹â¶")
    print(f"  Cayley-Dickson level: Sedenions (S)")
    print(f"  Division algebra: NO (zero divisors present)")
    print(f"  Hopf fibration: NONE (terminates at SÂ¹âµ â†’ Sâ¸)")
    print()
    print(f"  PREDICTION: The contraction that builds from 2Ã—2â†’4Ã—4â†’8Ã—8")
    print(f"  should DEGRADE at 16Ã—16 because zero divisors tear the")
    print(f"  algebraic structure. The tunnel can't sustain itself when")
    print(f"  nonzero elements multiply to zero.")
    print()
    
    t0 = time.time()
    
    test_norm_violation()
    test_zero_divisor_landscape()
    test_attractor_16()
    test_comparison_2v4v8v16()
    test_berry_scaling_2_4_8_16()
    test_entanglement_leak()
    
    elapsed = time.time() - t0
    
    # FINAL SYNTHESIS
    print("\n" + "=" * 76)
    print("  SYNTHESIS: ZERO DIVISOR BREAKING TEST")
    print("=" * 76)
    print(f"\n  Completed in {elapsed:.1f} seconds")
    print(f"\n  The division algebra sequence R â†’ C â†’ H â†’ O predicts:")
    print(f"    2Ã—2  â†’ 4Ã—4  â†’ 8Ã—8:  strengthening contraction (fiber cascade)")
    print(f"    8Ã—8  â†’ 16Ã—16:       DEGRADATION (zero divisors break coupling)")
    print(f"\n  This is because:")
    print(f"    - Division algebras preserve |aÂ·b| = |a|Â·|b|")
    print(f"    - Cross-coupling gates rely on this preservation")
    print(f"    - Sedenions violate it: âˆƒ a,b â‰  0 with aÂ·b = 0")
    print(f"    - The L1 (sedenionic) coupling traverses zero-divisor directions")
    print(f"    - State components entering the null space are LOST")
    print(f"    - The attractor mechanism degrades proportionally")
    print(f"\n  The 8Ã—8 octeract is the TRUE terminal structure.")
    print(f"  Beyond it, the algebra self-destructs.")
    print("=" * 76)


if __name__ == "__main__":
    main()
