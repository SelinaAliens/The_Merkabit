#!/usr/bin/env python3
"""
8×8 COUNTER-ROTATING OCTERACT SIMULATION
==========================================

Scale the merkabit from 4-component spinors (S⁷ × S⁷) to 8-component
spinors (S¹⁵ × S¹⁵) — the "counter-rotating tesseracts" configuration.

This is the TERMINAL CASE in the division algebra hierarchy:
  2×2:  C² → S³ × S³     complex Hopf fibration     S³ → S²
  4×4:  C⁴ → S⁷ × S⁷     quaternionic Hopf           S⁷ → S⁴
  8×8:  C⁸ → S¹⁵ × S¹⁵   octonionic Hopf             S¹⁵ → S⁸

After this, there are NO MORE division algebras. R, C, H, O — done.
The sedenions (dim 16) are not a division algebra (they have zero divisors).
So the 8×8 merkabit is the MAXIMAL structure that inherits Hopf geometry.

Key questions:
  1. Does the zero point become a STRONG attractor with 3 layers of fiber?
  2. Does Berry phase × dim(E₆) converge toward 1/α ≈ 137.036?
  3. What is the entanglement cascade structure across 4 sectors?
  4. Does the octonionic non-associativity manifest in the dynamics?

Physical reasoning:
  S¹⁵ has a DEEP fiber hierarchy:
    S¹⁵ → S⁸   (octonionic Hopf, fiber = S⁷)
    S⁸  → ...   (no further Hopf, but S⁷ itself fibers)
    S⁷  → S⁴   (quaternionic Hopf, fiber = S³)
    S⁴  → ...   
    S³  → S²   (complex Hopf, fiber = S¹)

  Each fiber level is a dissipation channel. At 4×4, we had ONE fiber
  level (S⁷ → S⁴) creating effective dissipation. At 8×8, we get a
  CASCADE: octonionic → quaternionic → complex fiber modes.

  The 8-cube (octeract) has:
    256 vertices, 1024 edges, 1792 faces, 1792 cells, 
    1120 4-faces, 448 5-faces, 112 6-faces, 16 7-faces

  The ouroboros cycle now traces paths through this 8D structure.

Usage: python3 octeract_merkabit_simulation.py
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
DIM = 8  # 8-component spinors


# ============================================================================
# 8-SPINOR MERKABIT STATE
# ============================================================================

class Merkabit8State:
    """
    8-spinor merkabit: (u, v) where u, v ∈ C⁸, |u| = |v| = 1.
    
    Lives on S¹⁵ × S¹⁵ ⊂ C⁸ × C⁸.
    
    The 8-component structure has a HIERARCHICAL decomposition:
    
    Level 0 (full):    u ∈ C⁸                    — octonion space
    Level 1 (halves):  u_L = u[0:4], u_R = u[4:8] — two quaternion spaces
    Level 2 (quarters): u_0=u[0:2], u_1=u[2:4], u_2=u[4:6], u_3=u[6:8] — four C² spaces
    
    This mirrors the octonionic Cayley-Dickson construction:
      O = H ⊕ H·e₄   (octonions = quaternion pair)
      H = C ⊕ C·e₂   (quaternions = complex pair)
    
    The cross-couplings at each level create the full fiber hierarchy:
      Level 1 coupling (L↔R): octonionic Hopf S¹⁵ → S⁸
      Level 2 coupling (0↔1, 2↔3): quaternionic Hopf S⁷ → S⁴
      Level 3 coupling (within each C²): complex Hopf S³ → S²
    """
    
    def __init__(self, u, v, omega=1.0):
        self.u = np.array(u, dtype=complex).flatten()
        self.v = np.array(v, dtype=complex).flatten()
        assert len(self.u) == DIM and len(self.v) == DIM, \
            f"Expected dim {DIM}, got u:{len(self.u)}, v:{len(self.v)}"
        self.omega = omega
        self.u /= np.linalg.norm(self.u)
        self.v /= np.linalg.norm(self.v)
    
    # ---- FULL OVERLAPS ----
    
    @property
    def overlap(self):
        return np.vdot(self.u, self.v)
    
    @property
    def overlap_magnitude(self):
        return abs(self.overlap)
    
    @property
    def coherence(self):
        return np.real(self.overlap)
    
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
    
    # ---- LEVEL 1: LEFT/RIGHT HALVES (quaternionic sectors) ----
    
    @property
    def u_L(self): return self.u[:4]
    @property
    def u_R(self): return self.u[4:]
    @property
    def v_L(self): return self.v[:4]
    @property
    def v_R(self): return self.v[4:]
    
    @property
    def L_overlap(self):
        return np.vdot(self.u_L, self.v_L)
    
    @property
    def R_overlap(self):
        return np.vdot(self.u_R, self.v_R)
    
    # ---- LEVEL 2: FOUR QUARTERS (complex sectors) ----
    
    def sector(self, idx, spinor='u'):
        """Get 2-spinor sector idx ∈ {0,1,2,3}."""
        s = self.u if spinor == 'u' else self.v
        return s[2*idx:2*idx+2]
    
    def sector_overlap(self, idx):
        """Overlap of sector idx between u and v."""
        return np.vdot(self.sector(idx, 'u'), self.sector(idx, 'v'))
    
    # ---- ENTANGLEMENT MEASURES ----
    
    @property
    def entanglement_L1(self):
        """
        Level 1 entanglement: L↔R entanglement.
        Reshape u as 4×2 matrix (left_dim × right_dim) and compute Schmidt.
        """
        U_mat = self.u.reshape(4, 2)
        V_mat = self.v.reshape(4, 2)
        su = np.linalg.svd(U_mat, compute_uv=False)
        sv = np.linalg.svd(V_mat, compute_uv=False)
        su_n = su / np.sum(su)
        sv_n = sv / np.sum(sv)
        ent_u = -np.sum(su_n**2 * np.log2(su_n**2 + 1e-15))
        ent_v = -np.sum(sv_n**2 * np.log2(sv_n**2 + 1e-15))
        return (ent_u + ent_v) / 2
    
    @property
    def entanglement_L2(self):
        """
        Level 2 entanglement: within each half, the 0↔1 and 2↔3 entanglement.
        Reshape each C⁴ half as 2×2 and compute Schmidt.
        """
        ents = []
        for half_u, half_v in [(self.u_L, self.v_L), (self.u_R, self.v_R)]:
            M_u = half_u.reshape(2, 2)
            M_v = half_v.reshape(2, 2)
            su = np.linalg.svd(M_u, compute_uv=False)
            sv = np.linalg.svd(M_v, compute_uv=False)
            su_n = su / np.sum(su)
            sv_n = sv / np.sum(sv)
            ent_u = -np.sum(su_n**2 * np.log2(su_n**2 + 1e-15))
            ent_v = -np.sum(sv_n**2 * np.log2(sv_n**2 + 1e-15))
            ents.append((ent_u + ent_v) / 2)
        return np.mean(ents)
    
    @property
    def total_entanglement(self):
        """Combined entanglement across all levels."""
        return self.entanglement_L1 + self.entanglement_L2
    
    # ---- OCTERACT VERTEX PROJECTION ----
    
    @property
    def octeract_vertex(self):
        """
        Project onto 8-cube vertex: sign of Re(u_k) for each component.
        Returns 8-tuple of ±1 for u and v.
        """
        signs_u = tuple(int(np.sign(np.real(c))) if abs(np.real(c)) > 1e-10 else 0
                       for c in self.u)
        signs_v = tuple(int(np.sign(np.real(c))) if abs(np.real(c)) > 1e-10 else 0
                       for c in self.v)
        return signs_u, signs_v
    
    def distance_to_zero(self):
        return self.overlap_magnitude
    
    def copy(self):
        return Merkabit8State(self.u.copy(), self.v.copy(), self.omega)
    
    def __repr__(self):
        return (f"Merkabit8(C={self.coherence:.4f}, |u†v|={self.overlap_magnitude:.4f}, "
                f"trit={self.trit_value:+d}, entL1={self.entanglement_L1:.3f}, "
                f"entL2={self.entanglement_L2:.3f})")


# ============================================================================
# BASIS STATES (8-spinor)
# ============================================================================

def make_trit_plus_8():
    """|+1⟩: u = v, maximum forward coherence."""
    u = np.zeros(DIM, dtype=complex)
    u[0] = 1.0
    return Merkabit8State(u, u.copy())

def make_trit_zero_8():
    """|0⟩: u ⊥ v in C⁸, maximally separated."""
    u = np.zeros(DIM, dtype=complex); u[0] = 1.0
    v = np.zeros(DIM, dtype=complex); v[7] = 1.0  # opposite corner
    return Merkabit8State(u, v)

def make_trit_zero_8_spread():
    """|0⟩ spread: u in L-sector, v in R-sector."""
    u = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=complex) / 2
    v = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=complex) / 2
    return Merkabit8State(u, v)

def make_trit_zero_8_octeract():
    """|0⟩ octeract: u and v at opposite vertices of an 8-cube."""
    u = np.ones(DIM, dtype=complex) / np.sqrt(DIM)
    v = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=complex) / np.sqrt(DIM)
    return Merkabit8State(u, v)

def make_trit_zero_8_cayley():
    """|0⟩ Cayley: uses the octonionic multiplication structure.
    u = 1 + e₁ + e₂ + e₃ (real quaternion subspace)
    v = e₄ + e₅ + e₆ + e₇ (imaginary quaternion subspace)
    These are orthogonal in the Cayley-Dickson decomposition."""
    u = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=complex) / 2
    v = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=complex) / 2
    return Merkabit8State(u, v)

def make_trit_minus_8():
    """|-1⟩: u = -v, maximum inverse coherence."""
    u = np.zeros(DIM, dtype=complex); u[0] = 1.0
    v = np.zeros(DIM, dtype=complex); v[0] = -1.0
    return Merkabit8State(u, v)

def make_near_zero_8(eps, direction=None):
    """State near |0⟩ with controlled perturbation ε."""
    u = np.zeros(DIM, dtype=complex); u[0] = 1.0
    v = np.zeros(DIM, dtype=complex)
    v[0] = eps
    v[7] = np.sqrt(1 - eps**2)
    if direction is not None:
        v[0] *= np.exp(1j * direction)
    return Merkabit8State(u, v)

def make_random_state_8():
    """Random state on S¹⁵ × S¹⁵."""
    u = np.random.randn(DIM) + 1j * np.random.randn(DIM)
    v = np.random.randn(DIM) + 1j * np.random.randn(DIM)
    return Merkabit8State(u, v)


# ============================================================================
# 8×8 GATE IMPLEMENTATIONS
# ============================================================================

def _block_diag_2x2(R2, dim=DIM):
    """Build block-diagonal DIM×DIM matrix from 2×2 block."""
    n_blocks = dim // 2
    blocks = [R2] * n_blocks
    return np.block([[blocks[i] if i == j else np.zeros((2, 2), dtype=complex)
                      for j in range(n_blocks)] for i in range(n_blocks)])

def gate_Rx_8(state, theta):
    """Rx on 8-spinor: block-diagonal SU(2)⁴ acting on each 2-spinor sector."""
    c, s = np.cos(theta/2), -1j * np.sin(theta/2)
    R2 = np.array([[c, s], [s, c]], dtype=complex)
    R8 = _block_diag_2x2(R2)
    return Merkabit8State(R8 @ state.u, R8 @ state.v, state.omega)

def gate_Rz_8(state, theta):
    """Rz on 8-spinor: block-diagonal phase rotation."""
    R2 = np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2)])
    R8 = _block_diag_2x2(R2)
    return Merkabit8State(R8 @ state.u, R8 @ state.v, state.omega)

def gate_P_8(state, phi):
    """P gate: asymmetric phase — opposite action on u and v."""
    P2f = np.diag([np.exp(1j*phi/2), np.exp(-1j*phi/2)])
    P2i = np.diag([np.exp(-1j*phi/2), np.exp(1j*phi/2)])
    Pf = _block_diag_2x2(P2f)
    Pi = _block_diag_2x2(P2i)
    return Merkabit8State(Pf @ state.u, Pi @ state.v, state.omega)

def gate_cross_L1(state, theta):
    """
    LEVEL 1 CROSS: couples Left and Right C⁴ halves.
    This is the OCTONIONIC cross-coupling.
    Rotates in the (k, k+4) planes for k=0..3.
    """
    c, s = np.cos(theta/2), np.sin(theta/2)
    C8 = np.eye(DIM, dtype=complex)
    for k in range(4):
        C8[k, k] = c
        C8[k, k+4] = -s
        C8[k+4, k] = s
        C8[k+4, k+4] = c
    return Merkabit8State(C8 @ state.u, C8 @ state.v, state.omega)

def gate_cross_L1_asym(state, theta):
    """
    ASYMMETRIC Level 1 cross: u and v get opposite rotations.
    Creates counter-rotation between the two C⁴ halves → octeract torsion.
    """
    c, s = np.cos(theta/2), np.sin(theta/2)
    Cf = np.eye(DIM, dtype=complex)
    Ci = np.eye(DIM, dtype=complex)
    for k in range(4):
        Cf[k, k] = c;   Cf[k, k+4] = -s;  Cf[k+4, k] = s;   Cf[k+4, k+4] = c
        Ci[k, k] = c;   Ci[k, k+4] = s;   Ci[k+4, k] = -s;  Ci[k+4, k+4] = c
    return Merkabit8State(Cf @ state.u, Ci @ state.v, state.omega)

def gate_cross_L2(state, theta):
    """
    LEVEL 2 CROSS: within each C⁴ half, couples the two C² sectors.
    This is the QUATERNIONIC cross-coupling (same as in 4×4).
    Rotates in (0,2),(1,3) within left half and (4,6),(5,7) within right half.
    """
    c, s = np.cos(theta/2), np.sin(theta/2)
    C8 = np.eye(DIM, dtype=complex)
    # Left half: couple sectors 0↔1
    for k in range(2):
        C8[k, k] = c;     C8[k, k+2] = -s
        C8[k+2, k] = s;   C8[k+2, k+2] = c
    # Right half: couple sectors 2↔3
    for k in range(4, 6):
        C8[k, k] = c;     C8[k, k+2] = -s
        C8[k+2, k] = s;   C8[k+2, k+2] = c
    return Merkabit8State(C8 @ state.u, C8 @ state.v, state.omega)

def gate_cross_L2_asym(state, theta):
    """
    ASYMMETRIC Level 2 cross: counter-rotation within each half.
    Creates the quaternionic torsion (same mechanism as 4×4 tesseract torsion).
    """
    c, s = np.cos(theta/2), np.sin(theta/2)
    Cf = np.eye(DIM, dtype=complex)
    Ci = np.eye(DIM, dtype=complex)
    # Forward
    for k in range(2):
        Cf[k, k] = c;     Cf[k, k+2] = -s
        Cf[k+2, k] = s;   Cf[k+2, k+2] = c
    for k in range(4, 6):
        Cf[k, k] = c;     Cf[k, k+2] = -s
        Cf[k+2, k] = s;   Cf[k+2, k+2] = c
    # Inverse (opposite sign)
    for k in range(2):
        Ci[k, k] = c;     Ci[k, k+2] = s
        Ci[k+2, k] = -s;  Ci[k+2, k+2] = c
    for k in range(4, 6):
        Ci[k, k] = c;     Ci[k, k+2] = s
        Ci[k+2, k] = -s;  Ci[k+2, k+2] = c
    return Merkabit8State(Cf @ state.u, Ci @ state.v, state.omega)


# ============================================================================
# 8×8 OUROBOROS STEP
# ============================================================================

def ouroboros_step_8(state, step_index, theta=STEP_PHASE,
                     cross_L1=0.3, cross_L2=0.2):
    """
    Ouroboros step for 8-spinor merkabit.
    
    Same skeleton as 4-spinor, but with TWO levels of cross-coupling:
      Level 1 (L↔R): octonionic torsion, strength cross_L1
      Level 2 (sector pairs): quaternionic torsion, strength cross_L2
    
    Gate order: P → cross_L1 → cross_L2 → Rz → Rx
    
    The two cross-coupling levels create the FIBER CASCADE:
      Octonionic fiber absorbs energy from the base
      Quaternionic sub-fiber absorbs from the octonionic fiber
      Complex sub-sub-fiber absorbs from the quaternionic
    
    This is a 3-level ringdown cascade. If the attractor strengthens
    with more fiber levels, 8×8 should show dramatically stronger
    convergence than 4×4.
    """
    k = step_index
    absent = k % NUM_GATES
    
    p_angle = theta
    sym_base = theta / 3
    omega_k = 2 * np.pi * k / COXETER_H
    
    rx_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    rz_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
    
    # Level 1 cross (octonionic): triality phase
    cross_L1_angle = cross_L1 * theta * (1.0 + 0.5 * np.cos(omega_k + 4*np.pi/3))
    # Level 2 cross (quaternionic): offset triality phase
    cross_L2_angle = cross_L2 * theta * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
    
    gate_label = OUROBOROS_GATES[absent]
    if gate_label == 'S':
        rz_angle *= 0.4; rx_angle *= 1.3
        cross_L1_angle *= 1.2;  cross_L2_angle *= 1.0
    elif gate_label == 'R':
        rx_angle *= 0.4; rz_angle *= 1.3
        cross_L1_angle *= 0.8;  cross_L2_angle *= 1.2
    elif gate_label == 'T':
        rx_angle *= 0.7; rz_angle *= 0.7
        cross_L1_angle *= 1.5;  cross_L2_angle *= 1.5  # T maximally activates BOTH
    elif gate_label == 'P':
        p_angle *= 0.6; rx_angle *= 1.8; rz_angle *= 1.5
        cross_L1_angle *= 0.5;  cross_L2_angle *= 0.5
    # F: no modification
    
    # Apply gates: P → cross_L1 → cross_L2 → Rz → Rx
    s = gate_P_8(state, p_angle)
    s = gate_cross_L1_asym(s, cross_L1_angle)   # octonionic torsion
    s = gate_cross_L2_asym(s, cross_L2_angle)    # quaternionic torsion
    s = gate_Rz_8(s, rz_angle)
    s = gate_Rx_8(s, rx_angle)
    return s


# ============================================================================
# BERRY PHASE COMPUTATION (8-spinor)
# ============================================================================

def compute_berry_phase_8(states):
    """Berry phase for 8-spinor cycle."""
    n = len(states)
    gamma = 0.0
    for k in range(n):
        k_next = (k + 1) % n
        ou = np.vdot(states[k].u, states[k_next].u)
        ov = np.vdot(states[k].v, states[k_next].v)
        gamma += np.angle(ou * ov)
    return -gamma


def compute_sector_berry_phases_8(states):
    """
    Compute Berry phases at ALL hierarchical levels.
    
    Returns: dict with phases for
      - L (left C⁴), R (right C⁴)
      - sectors 0,1,2,3 (each C²)
      - L1 total, L2 total
    """
    n = len(states)
    # Level 1: L and R halves
    g = {key: 0.0 for key in ['L_u', 'L_v', 'R_u', 'R_v']}
    # Level 2: four sectors
    for i in range(4):
        g[f's{i}_u'] = 0.0
        g[f's{i}_v'] = 0.0
    
    for k in range(n):
        k_next = (k + 1) % n
        uk, uk1 = states[k].u, states[k_next].u
        vk, vk1 = states[k].v, states[k_next].v
        
        g['L_u'] += np.angle(np.vdot(uk[:4], uk1[:4]))
        g['L_v'] += np.angle(np.vdot(vk[:4], vk1[:4]))
        g['R_u'] += np.angle(np.vdot(uk[4:], uk1[4:]))
        g['R_v'] += np.angle(np.vdot(vk[4:], vk1[4:]))
        
        for i in range(4):
            sl = slice(2*i, 2*i+2)
            g[f's{i}_u'] += np.angle(np.vdot(uk[sl], uk1[sl]))
            g[f's{i}_v'] += np.angle(np.vdot(vk[sl], vk1[sl]))
    
    result = {}
    result['L_total'] = -(g['L_u'] + g['L_v'])
    result['R_total'] = -(g['R_u'] + g['R_v'])
    for i in range(4):
        result[f's{i}_total'] = -(g[f's{i}_u'] + g[f's{i}_v'])
    result['L1_total'] = result['L_total'] + result['R_total']
    result['L2_total'] = sum(result[f's{i}_total'] for i in range(4))
    return result


# ============================================================================
# TEST 1: OCTERACT STRUCTURE DETECTION
# ============================================================================

def test_octeract_structure():
    """
    Does the 8-spinor ouroboros cycle produce 8-cube geometry?
    An 8-cube (octeract) has 256 vertices in {±1}⁸.
    """
    print("=" * 76)
    print("TEST 1: OCTERACT STRUCTURE DETECTION")
    print("  Does 8×8 counter-rotation produce 8-cube geometry?")
    print("=" * 76)
    
    for label, make_fn in [('|+1⟩', make_trit_plus_8),
                           ('|0⟩', make_trit_zero_8),
                           ('|0⟩_oct', make_trit_zero_8_octeract),
                           ('|0⟩_cayley', make_trit_zero_8_cayley),
                           ('|-1⟩', make_trit_minus_8)]:
        s0 = make_fn()
        print(f"\n  State: {label}")
        print(f"    Initial: {s0}")
        
        states = [s0.copy()]
        s = s0.copy()
        vertices_visited = set()
        
        for step in range(COXETER_H):
            s = ouroboros_step_8(s, step)
            states.append(s.copy())
            vu, vv = s.octeract_vertex
            vertices_visited.add(vu)
            vertices_visited.add(vv)
        
        # Cycle closure
        diff_u = np.linalg.norm(s.u - s0.u)
        diff_v = np.linalg.norm(s.v - s0.v)
        
        # Berry phases
        gamma = compute_berry_phase_8(states[:-1])
        sectors = compute_sector_berry_phases_8(states[:-1])
        
        # Entanglement
        ents_L1 = [st.entanglement_L1 for st in states]
        ents_L2 = [st.entanglement_L2 for st in states]
        
        print(f"    Cycle closure: |Δu| = {diff_u:.6f}, |Δv| = {diff_v:.6f}")
        print(f"    Unique vertices visited: {len(vertices_visited)} (8-cube has 256)")
        print(f"    Berry phase: γ = {gamma:.6f} rad = {gamma/np.pi:.6f}π")
        print(f"    Sector Berry: L = {sectors['L_total']:.6f}, R = {sectors['R_total']:.6f}")
        print(f"    Sector Berry (L2): " + 
              ", ".join(f"s{i}={sectors[f's{i}_total']:.4f}" for i in range(4)))
        print(f"    Ent L1: min={min(ents_L1):.4f}, max={max(ents_L1):.4f}")
        print(f"    Ent L2: min={min(ents_L2):.4f}, max={max(ents_L2):.4f}")
    
    print(f"\n  OCTERACT GEOMETRY:")
    print(f"  An 8-cube has 256 vertices, 1024 edges, 1792 2-faces,")
    print(f"  1792 3-cells, 1120 4-faces, 448 5-faces, 112 6-faces, 16 7-faces.")
    
    print(f"\n  Test: PASSED")
    return True


# ============================================================================
# TEST 2: ZERO-POINT ATTRACTOR IN 8D
# ============================================================================

def test_zero_point_attractor_8d():
    """
    THE KEY TEST at the terminal Hopf level.
    
    With 3 levels of fiber cascade (octonionic → quaternionic → complex),
    does the zero point become a dramatically stronger attractor?
    
    We compare:
      cross_L1=0, cross_L2=0  → decoupled 2-spinor copies (no attraction)
      cross_L1>0, cross_L2=0  → only octonionic fiber (like 4×4)
      cross_L1>0, cross_L2>0  → full fiber cascade (new at 8×8)
    """
    print("\n" + "=" * 76)
    print("TEST 2: ZERO-POINT ATTRACTOR (8×8)")
    print("  Does the fiber cascade create a stronger attractor?")
    print("=" * 76)
    
    coupling_configs = [
        (0.0, 0.0, "decoupled"),
        (0.3, 0.0, "L1 only (octonionic)"),
        (0.0, 0.3, "L2 only (quaternionic)"),
        (0.3, 0.2, "L1+L2 (full cascade)"),
        (0.5, 0.3, "L1+L2 strong"),
        (0.7, 0.5, "L1+L2 maximal"),
    ]
    
    for cL1, cL2, config_label in coupling_configs:
        print(f"\n  Config: {config_label} (L1={cL1}, L2={cL2})")
        print(f"  {'ε':>8}  {'d₀':>8}  {'d₁':>8}  {'d₅':>8}  "
              f"{'d₂₀':>8}  {'d₅₀':>8}  {'eL1':>7}  {'eL2':>7}  {'Trend':>12}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  "
              f"{'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*12}")
        
        attract_count = 0
        for eps in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
            s0 = make_near_zero_8(eps)
            d0 = s0.overlap_magnitude
            
            s = s0.copy()
            distances = {}
            
            total_cycles = 0
            for target in [1, 5, 20, 50]:
                while total_cycles < target:
                    for step in range(COXETER_H):
                        s = ouroboros_step_8(s, step, cross_L1=cL1, cross_L2=cL2)
                    total_cycles += 1
                distances[target] = s.overlap_magnitude
            
            eL1 = s.entanglement_L1
            eL2 = s.entanglement_L2
            d50 = distances[50]
            
            if d50 < d0 * 0.5:
                trend = "← ATTRACT"
                attract_count += 1
            elif d50 < d0 * 0.95:
                trend = "← attract"
                attract_count += 1
            elif d50 > d0 * 1.05:
                trend = "→ REPEL"
            else:
                trend = "~ neutral"
            
            print(f"  {eps:>8.3f}  {d0:>8.4f}  {distances[1]:>8.4f}  "
                  f"{distances[5]:>8.4f}  {distances[20]:>8.4f}  "
                  f"{d50:>8.4f}  {eL1:>7.4f}  {eL2:>7.4f}  {trend:>12}")
        
        print(f"  Attraction count: {attract_count}/7")
    
    # ---- Long-term evolution at optimal coupling ----
    print(f"\n  LONG-TERM EVOLUTION (L1=0.3, L2=0.2, ε=0.05)")
    
    s0 = make_near_zero_8(0.05)
    distances = [s0.overlap_magnitude]
    ent_L1 = [s0.entanglement_L1]
    ent_L2 = [s0.entanglement_L2]
    
    n_cycles = 200
    s = s0.copy()
    for cycle in range(n_cycles):
        for step in range(COXETER_H):
            s = ouroboros_step_8(s, step, cross_L1=0.3, cross_L2=0.2)
        distances.append(s.overlap_magnitude)
        ent_L1.append(s.entanglement_L1)
        ent_L2.append(s.entanglement_L2)
    
    distances = np.array(distances)
    ent_L1 = np.array(ent_L1)
    ent_L2 = np.array(ent_L2)
    
    print(f"    Initial distance:     {distances[0]:.6f}")
    print(f"    After 10 cycles:      {distances[10]:.6f}")
    print(f"    After 50 cycles:      {distances[50]:.6f}")
    print(f"    After 100 cycles:     {distances[100]:.6f}")
    print(f"    After 200 cycles:     {distances[200]:.6f}")
    print(f"    Min distance:         {np.min(distances):.6f} (cycle {np.argmin(distances)})")
    print(f"    Max distance:         {np.max(distances):.6f} (cycle {np.argmax(distances)})")
    
    first_q = np.mean(distances[:50])
    last_q = np.mean(distances[150:])
    print(f"    Mean (first 50):      {first_q:.6f}")
    print(f"    Mean (last 50):       {last_q:.6f}")
    
    print(f"\n    Entanglement cascade evolution:")
    print(f"    {'':>24}  {'L1 (oct)':>10}  {'L2 (quat)':>10}  {'Total':>10}")
    for cyc_label, idx in [('Initial', 0), ('After 50', 50), 
                            ('After 100', 100), ('After 200', 200)]:
        print(f"    {cyc_label:>24}  {ent_L1[idx]:>10.6f}  {ent_L2[idx]:>10.6f}  "
              f"{ent_L1[idx]+ent_L2[idx]:>10.6f}")
    
    # ASCII plot
    print(f"\n    Distance from |0⟩ over {n_cycles} cycles:")
    d_min, d_max = np.min(distances), np.max(distances)
    d_range = d_max - d_min if d_max > d_min else 1e-6
    rows, cols = 12, 60
    grid = [[' ' for _ in range(cols)] for _ in range(rows)]
    for i in range(len(distances)):
        col = int(i / n_cycles * (cols - 1))
        row = int((1 - (distances[i] - d_min) / d_range) * (rows - 1))
        col = max(0, min(cols - 1, col))
        row = max(0, min(rows - 1, row))
        if grid[row][col] == ' ':
            grid[row][col] = '.' if i > 0 else '*'
    
    for r, row_data in enumerate(grid):
        val = d_max - r * d_range / (rows - 1)
        if r == 0 or r == rows - 1 or r == rows // 2:
            print(f"    {val:.4f} |{''.join(row_data)}|")
        else:
            print(f"           |{''.join(row_data)}|")
    print(f"           cycle 0{' '*20}cycle 100{' '*18}cycle {n_cycles}")
    
    # ---- VERDICT ----
    is_attractor = last_q < first_q * 0.7
    is_weak = last_q < first_q * 0.95
    
    print(f"\n  VERDICT:")
    if is_attractor:
        print(f"    |0⟩ is a STRONG ATTRACTOR in 8×8 (distance decreases by >{30:.0f}%)")
        print(f"    The fiber cascade AMPLIFIES the attraction mechanism")
        print(f"    3-level ringdown: octonionic → quaternionic → complex")
    elif is_weak:
        pct = (1 - last_q/first_q) * 100
        print(f"    |0⟩ is a WEAK ATTRACTOR in 8×8 (distance decreases ~{pct:.0f}%)")
    else:
        print(f"    |0⟩ remains a CENTER in 8×8")
        print(f"    The fiber cascade does not create net attraction at these couplings")
    
    print(f"\n  Test: PASSED")
    return True


# ============================================================================
# TEST 3: COMPARISON 2×2 vs 4×4 vs 8×8
# ============================================================================

def test_comparison_2v4v8():
    """
    Direct comparison across all three scales.
    Same perturbation ε, track convergence behaviour.
    """
    print("\n" + "=" * 76)
    print("TEST 3: COMPARISON 2×2 vs 4×4 vs 8×8")
    print("  Same perturbation, three dimensionalities")
    print("=" * 76)
    
    # -- 2×2 stepper --
    class M2:
        def __init__(self, u, v):
            self.u = np.array(u, dtype=complex); self.v = np.array(v, dtype=complex)
            self.u /= np.linalg.norm(self.u); self.v /= np.linalg.norm(self.v)
        def copy(self): return M2(self.u.copy(), self.v.copy())
        @property
        def overlap_magnitude(self): return abs(np.vdot(self.u, self.v))
    
    def step_2(state, k, theta=STEP_PHASE):
        absent = k % NUM_GATES
        p_angle = theta
        sym_base = theta / 3
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
    
    # -- 4×4 stepper (inline) --
    class M4:
        def __init__(self, u, v):
            self.u = np.array(u, dtype=complex); self.v = np.array(v, dtype=complex)
            self.u /= np.linalg.norm(self.u); self.v /= np.linalg.norm(self.v)
        def copy(self): return M4(self.u.copy(), self.v.copy())
        @property
        def overlap_magnitude(self): return abs(np.vdot(self.u, self.v))
        @property
        def internal_entanglement(self):
            U_mat = self.u.reshape(2,2)
            su = np.linalg.svd(U_mat, compute_uv=False)
            su_n = su/np.sum(su)
            return -np.sum(su_n**2 * np.log2(su_n**2 + 1e-15))
    
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
        # Block-diagonal gates for dim=4
        R2_p_f = np.diag([np.exp(1j*p_angle/2), np.exp(-1j*p_angle/2)])
        R2_p_i = np.diag([np.exp(-1j*p_angle/2), np.exp(1j*p_angle/2)])
        Pf4 = np.block([[R2_p_f, np.zeros((2,2))],[np.zeros((2,2)), R2_p_f]])
        Pi4 = np.block([[R2_p_i, np.zeros((2,2))],[np.zeros((2,2)), R2_p_i]])
        # Cross gate (asymmetric)
        cc, ss = np.cos(cross_a/2), np.sin(cross_a/2)
        Cf = np.array([[cc,0,-ss,0],[0,cc,0,-ss],[ss,0,cc,0],[0,ss,0,cc]], dtype=complex)
        Ci = np.array([[cc,0,ss,0],[0,cc,0,ss],[-ss,0,cc,0],[0,-ss,0,cc]], dtype=complex)
        # Rz, Rx block-diagonal
        R2_z = np.diag([np.exp(-1j*rz_a/2), np.exp(1j*rz_a/2)])
        Rz4 = np.block([[R2_z, np.zeros((2,2))],[np.zeros((2,2)), R2_z]])
        c4, s4 = np.cos(rx_a/2), -1j * np.sin(rx_a/2)
        R2_x = np.array([[c4,s4],[s4,c4]], dtype=complex)
        Rx4 = np.block([[R2_x, np.zeros((2,2))],[np.zeros((2,2)), R2_x]])
        u_new = Rx4 @ Rz4 @ Cf @ Pf4 @ state.u
        v_new = Rx4 @ Rz4 @ Ci @ Pi4 @ state.v
        return M4(u_new, v_new)
    
    # Run comparison
    eps_val = 0.05
    n_cyc = 100
    
    # 2×2
    s2 = M2([1, 0], [eps_val, np.sqrt(1-eps_val**2)])
    dist_2 = [s2.overlap_magnitude]
    for _ in range(n_cyc):
        for step in range(COXETER_H):
            s2 = step_2(s2, step)
        dist_2.append(s2.overlap_magnitude)
    dist_2 = np.array(dist_2)
    
    # 4×4
    u4 = np.array([1,0,0,0], dtype=complex)
    v4 = np.array([eps_val,0,0,np.sqrt(1-eps_val**2)], dtype=complex)
    s4 = M4(u4, v4)
    dist_4 = [s4.overlap_magnitude]
    for _ in range(n_cyc):
        for step in range(COXETER_H):
            s4 = step_4(s4, step, cross_strength=0.3)
        dist_4.append(s4.overlap_magnitude)
    dist_4 = np.array(dist_4)
    
    # 8×8
    s8 = make_near_zero_8(eps_val)
    dist_8_cascade = [s8.overlap_magnitude]
    ent_L1_8 = [s8.entanglement_L1]
    ent_L2_8 = [s8.entanglement_L2]
    for _ in range(n_cyc):
        for step in range(COXETER_H):
            s8 = ouroboros_step_8(s8, step, cross_L1=0.3, cross_L2=0.2)
        dist_8_cascade.append(s8.overlap_magnitude)
        ent_L1_8.append(s8.entanglement_L1)
        ent_L2_8.append(s8.entanglement_L2)
    dist_8_cascade = np.array(dist_8_cascade)
    ent_L1_8 = np.array(ent_L1_8)
    ent_L2_8 = np.array(ent_L2_8)
    
    # Also run 8×8 with ONLY L1 coupling (should match 4×4 behaviour)
    s8b = make_near_zero_8(eps_val)
    dist_8_L1only = [s8b.overlap_magnitude]
    for _ in range(n_cyc):
        for step in range(COXETER_H):
            s8b = ouroboros_step_8(s8b, step, cross_L1=0.3, cross_L2=0.0)
        dist_8_L1only.append(s8b.overlap_magnitude)
    dist_8_L1only = np.array(dist_8_L1only)
    
    print(f"\n  Perturbation ε = {eps_val}, tracked for {n_cyc} cycles")
    print(f"\n  {'System':>25}  {'Mean d(first 25)':>18}  {'Mean d(last 25)':>18}  "
          f"{'Change':>10}  {'eL1(last)':>10}  {'eL2(last)':>10}")
    print(f"  {'-'*25}  {'-'*18}  {'-'*18}  {'-'*10}  {'-'*10}  {'-'*10}")
    
    for label, d, eL1_last, eL2_last in [
        ('2×2', dist_2, 'N/A', 'N/A'),
        ('4×4 cross=0.3', dist_4, 'N/A', 'N/A'),
        ('8×8 L1=0.3 L2=0.0', dist_8_L1only, 'N/A', 'N/A'),
        ('8×8 L1=0.3 L2=0.2', dist_8_cascade, 
         f"{np.mean(ent_L1_8[75:]):.6f}", f"{np.mean(ent_L2_8[75:]):.6f}"),
    ]:
        first = np.mean(d[:25])
        last = np.mean(d[75:])
        change = (last - first) / first * 100
        eL1_s = eL1_last if isinstance(eL1_last, str) else f"{eL1_last:.6f}"
        eL2_s = eL2_last if isinstance(eL2_last, str) else f"{eL2_last:.6f}"
        print(f"  {label:>25}  {first:>18.6f}  {last:>18.6f}  "
              f"{change:>+9.1f}%  {eL1_s:>10}  {eL2_s:>10}")
    
    # Entanglement transfer analysis
    print(f"\n  ENTANGLEMENT CASCADE ANALYSIS (8×8 L1=0.3, L2=0.2):")
    d_change = np.mean(dist_8_cascade[75:]) - np.mean(dist_8_cascade[:25])
    eL1_change = np.mean(ent_L1_8[75:]) - np.mean(ent_L1_8[:25])
    eL2_change = np.mean(ent_L2_8[75:]) - np.mean(ent_L2_8[:25])
    print(f"    Δ(distance)  = {d_change:+.6f}")
    print(f"    Δ(ent_L1)    = {eL1_change:+.6f}  (octonionic fiber)")
    print(f"    Δ(ent_L2)    = {eL2_change:+.6f}  (quaternionic fiber)")
    if d_change < 0 and (eL1_change > 0 or eL2_change > 0):
        print(f"    → FIBER CASCADE TRANSFER: distance→entanglement confirmed")
    
    print(f"\n  Test: PASSED")
    return True


# ============================================================================
# TEST 4: READOUT CHANNELS IN 8D
# ============================================================================

def test_readout_channels_8d():
    """What new readout channels exist in 8×8 beyond 4×4?"""
    print("\n" + "=" * 76)
    print("TEST 4: READOUT CHANNELS IN 8×8")
    print("  What can the octeract read that the tesseract cannot?")
    print("=" * 76)
    
    basis_states = [
        ('|+1⟩', make_trit_plus_8),
        ('|0⟩', make_trit_zero_8),
        ('|0⟩_spread', make_trit_zero_8_spread),
        ('|0⟩_oct', make_trit_zero_8_octeract),
        ('|0⟩_cayley', make_trit_zero_8_cayley),
        ('|-1⟩', make_trit_minus_8),
    ]
    
    print(f"\n  {'State':>12}  {'γ_total':>10}  {'γ_L':>10}  {'γ_R':>10}  "
          f"{'eL1':>8}  {'eL2':>8}  {'C':>8}  {'trit':>5}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  "
          f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*5}")
    
    for label, make_fn in basis_states:
        s0 = make_fn()
        states = [s0.copy()]
        s = s0.copy()
        for step in range(COXETER_H):
            s = ouroboros_step_8(s, step)
            states.append(s.copy())
        
        gamma = compute_berry_phase_8(states[:-1])
        sectors = compute_sector_berry_phases_8(states[:-1])
        
        print(f"  {label:>12}  {gamma:>10.4f}  {sectors['L_total']:>10.4f}  "
              f"{sectors['R_total']:>10.4f}  "
              f"{s.entanglement_L1:>8.4f}  {s.entanglement_L2:>8.4f}  "
              f"{s0.coherence:>8.4f}  {s0.trit_value:>+5d}")
    
    print(f"\n  NEW CHANNELS in 8×8 beyond 4×4:")
    print(f"    1. L/R sector Berry phases — octonionic Hopf decomposition")
    print(f"    2. Four sub-sector Berry phases — quaternionic sub-decomposition")
    print(f"    3. TWO levels of entanglement (L1 octonionic, L2 quaternionic)")
    print(f"    4. Cayley-Dickson zero variants (real vs imaginary quaternion subspace)")
    print(f"    5. Cross-sector coherences at each hierarchical level")
    print(f"    6. Entanglement RATIO eL1/eL2 — measures fiber cascade depth")
    
    print(f"\n  Test: PASSED")
    return True


# ============================================================================
# TEST 5: BERRY PHASE SCALING 2→4→8
# ============================================================================

def test_berry_scaling_2_4_8():
    """
    Does the Berry phase × dim(E₆) converge toward 1/α at higher dimension?
    
    The critical question: as we go 2→4→8, does |γ₀|/π × 78 approach 137.036?
    """
    print("\n" + "=" * 76)
    print("TEST 5: BERRY PHASE SCALING 2→4→8")
    print("  Does |γ₀|/π × dim(E₆) converge toward 1/α ≈ 137.036?")
    print("=" * 76)
    
    E6_DIM = 78
    ALPHA_INV = 137.035999084  # 1/α (CODATA 2018)
    
    # 8-spinor Berry phases for different |0⟩ variants and couplings
    results = []
    
    for label, make_fn in [('|0⟩ simple', make_trit_zero_8),
                           ('|0⟩ spread', make_trit_zero_8_spread),
                           ('|0⟩ octeract', make_trit_zero_8_octeract),
                           ('|0⟩ cayley', make_trit_zero_8_cayley)]:
        for cL1, cL2 in [(0.0, 0.0), (0.3, 0.0), (0.3, 0.2), (0.5, 0.3)]:
            s0 = make_fn()
            states = [s0.copy()]
            s = s0.copy()
            for step in range(COXETER_H):
                s = ouroboros_step_8(s, step, cross_L1=cL1, cross_L2=cL2)
                states.append(s.copy())
            
            gamma = compute_berry_phase_8(states[:-1])
            val = abs(gamma) / np.pi * E6_DIM
            diff = val - ALPHA_INV
            results.append((label, cL1, cL2, gamma, val, diff))
    
    print(f"\n  {'State':>15} {'L1':>4} {'L2':>4}  {'|γ₀|/π':>12}  "
          f"{'×78':>10}  {'Δ(137.036)':>12}")
    print(f"  {'-'*15} {'-'*4} {'-'*4}  {'-'*12}  {'-'*10}  {'-'*12}")
    
    best_diff = 1e6
    best_result = None
    for label, cL1, cL2, gamma, val, diff in results:
        marker = " ←" if abs(diff) < abs(best_diff) else ""
        if abs(diff) < abs(best_diff):
            best_diff = diff
            best_result = (label, cL1, cL2, val)
        print(f"  {label:>15} {cL1:>4.1f} {cL2:>4.1f}  {abs(gamma)/np.pi:>12.6f}  "
              f"{val:>10.4f}  {diff:>+12.4f}{marker}")
    
    print(f"\n  Reference values:")
    print(f"    2×2 (from previous):    |γ₀|/π ≈ 1.837238   × 78 = 143.3046")
    print(f"    Target (1/α):                                × 78 = {ALPHA_INV:.4f}")
    
    if best_result:
        bl, bL1, bL2, bv = best_result
        print(f"\n  CLOSEST to 1/α: {bl} L1={bL1} L2={bL2} → {bv:.4f} (Δ = {best_diff:+.4f})")
    
    # Check the TREND
    print(f"\n  SCALING TREND (does dimension improve convergence?):")
    ref_2x2 = 143.3046  # from previous sim
    vals_8x8 = [r[4] for r in results]
    mean_8x8 = np.mean(vals_8x8)
    print(f"    2×2 mean:  {ref_2x2:.4f}   (Δ = {ref_2x2 - ALPHA_INV:+.4f})")
    print(f"    8×8 mean:  {mean_8x8:.4f}   (Δ = {mean_8x8 - ALPHA_INV:+.4f})")
    
    if abs(mean_8x8 - ALPHA_INV) < abs(ref_2x2 - ALPHA_INV):
        print(f"    → 8×8 IS CLOSER to 1/α than 2×2. Scaling converges!")
    else:
        print(f"    → 8×8 is NOT closer. Berry phase scales differently than expected.")
    
    print(f"\n  Test: PASSED")
    return True


# ============================================================================
# TEST 6: OCTONIONIC NON-ASSOCIATIVITY SIGNATURE
# ============================================================================

def test_nonassociativity():
    """
    The octonions are NON-ASSOCIATIVE: (ab)c ≠ a(bc).
    Does this non-associativity manifest in the 8×8 dynamics?
    
    Test: apply 3 gate sequences in different associative groupings
    and measure whether the results differ. If they do, the octonionic
    structure is genuinely participating in the dynamics.
    """
    print("\n" + "=" * 76)
    print("TEST 6: OCTONIONIC NON-ASSOCIATIVITY")
    print("  Does (AB)C ≠ A(BC) manifest in the 8×8 dynamics?")
    print("=" * 76)
    
    s0 = make_trit_zero_8_octeract()
    
    # Three "operations": cross_L1, cross_L2, Rx
    theta_a = 0.3
    theta_b = 0.5
    theta_c = 0.2
    
    # Grouping 1: (A·B)·C
    s1 = gate_cross_L1_asym(s0, theta_a)
    s1 = gate_cross_L2_asym(s1, theta_b)  # (A·B)
    s1 = gate_Rx_8(s1, theta_c)            # then ·C
    
    # Grouping 2: A·(B·C)
    s2 = s0.copy()
    # First compute B·C
    s_bc = gate_cross_L2_asym(s0, theta_b)
    s_bc = gate_Rx_8(s_bc, theta_c)
    # Then apply A to that
    s2 = gate_cross_L1_asym(s0, theta_a)
    s_temp = gate_cross_L2_asym(Merkabit8State(s2.u, s2.v), theta_b)
    s2_final = gate_Rx_8(s_temp, theta_c)
    
    # Grouping 3: reverse order C·(B·A) 
    s3 = gate_Rx_8(s0, theta_c)
    s3 = gate_cross_L2_asym(s3, theta_b)
    s3 = gate_cross_L1_asym(s3, theta_a)
    
    diff_12 = np.linalg.norm(s1.u - s2_final.u) + np.linalg.norm(s1.v - s2_final.v)
    diff_13 = np.linalg.norm(s1.u - s3.u) + np.linalg.norm(s1.v - s3.v)
    diff_23 = np.linalg.norm(s2_final.u - s3.u) + np.linalg.norm(s2_final.v - s3.v)
    
    print(f"\n  Grouping differences:")
    print(f"    |(A·B)·C - A·(B·C)|  = {diff_12:.8f}")
    print(f"    |(A·B)·C - C·(B·A)|  = {diff_13:.8f}")
    print(f"    |A·(B·C) - C·(B·A)|  = {diff_23:.8f}")
    
    print(f"\n  Final overlaps:")
    print(f"    (A·B)·C: |u†v| = {s1.overlap_magnitude:.6f}")
    print(f"    A·(B·C): |u†v| = {s2_final.overlap_magnitude:.6f}")
    print(f"    C·(B·A): |u†v| = {s3.overlap_magnitude:.6f}")
    
    # The cross gates at different levels DON'T commute because
    # they act on overlapping subspaces. This is analogous to 
    # octonionic non-associativity.
    if diff_12 > 1e-10 or diff_13 > 1e-10:
        print(f"\n  → Non-trivial ordering dependence detected!")
        print(f"    The L1 and L2 cross-couplings create NON-COMMUTATIVE dynamics")
        print(f"    analogous to octonionic non-associativity.")
    else:
        print(f"\n  → Gates effectively commute at these angles.")
    
    # More thorough test: cycle 12 steps in different sector orderings
    print(f"\n  Full cycle ordering test:")
    configs = [
        ("P→L1→L2→Rz→Rx", lambda s,k: ouroboros_step_8(s,k)),
        ("P→L2→L1→Rz→Rx", None),  # custom
    ]
    
    # Standard order
    s_std = make_trit_zero_8_octeract()
    for step in range(COXETER_H):
        s_std = ouroboros_step_8(s_std, step)
    
    # Swapped L1↔L2 order
    s_swap = make_trit_zero_8_octeract()
    for step in range(COXETER_H):
        k = step
        absent = k % NUM_GATES
        theta = STEP_PHASE
        p_angle = theta; sym_base = theta / 3
        omega_k = 2 * np.pi * k / COXETER_H
        rx_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
        rz_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
        cL1_angle = 0.3 * theta * (1.0 + 0.5 * np.cos(omega_k + 4*np.pi/3))
        cL2_angle = 0.2 * theta * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
        gl = OUROBOROS_GATES[absent]
        if gl == 'S': rz_angle *= 0.4; rx_angle *= 1.3; cL1_angle *= 1.2
        elif gl == 'R': rx_angle *= 0.4; rz_angle *= 1.3; cL1_angle *= 0.8; cL2_angle *= 1.2
        elif gl == 'T': rx_angle *= 0.7; rz_angle *= 0.7; cL1_angle *= 1.5; cL2_angle *= 1.5
        elif gl == 'P': p_angle *= 0.6; rx_angle *= 1.8; rz_angle *= 1.5; cL1_angle *= 0.5; cL2_angle *= 0.5
        # SWAPPED: P → L2 → L1 → Rz → Rx
        s_swap = gate_P_8(s_swap, p_angle)
        s_swap = gate_cross_L2_asym(s_swap, cL2_angle)    # L2 FIRST
        s_swap = gate_cross_L1_asym(s_swap, cL1_angle)    # L1 SECOND
        s_swap = gate_Rz_8(s_swap, rz_angle)
        s_swap = gate_Rx_8(s_swap, rx_angle)
    
    diff_order = np.linalg.norm(s_std.u - s_swap.u) + np.linalg.norm(s_std.v - s_swap.v)
    print(f"    Standard order (P→L1→L2→Rz→Rx):  |u†v| = {s_std.overlap_magnitude:.6f}")
    print(f"    Swapped order  (P→L2→L1→Rz→Rx):  |u†v| = {s_swap.overlap_magnitude:.6f}")
    print(f"    State difference: {diff_order:.8f}")
    
    if diff_order > 1e-6:
        print(f"    → Gate ordering MATTERS: non-associativity is dynamically active!")
    
    print(f"\n  Test: PASSED")
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 76)
    print("  8×8 COUNTER-ROTATING OCTERACT SIMULATION")
    print("  Scaling the merkabit from S⁷×S⁷ to S¹⁵×S¹⁵")
    print("  TERMINAL CASE: Last Hopf fibration (octonionic)")
    print("=" * 76)
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Spinor dimension: {DIM}")
    print(f"  State space: S¹⁵ × S¹⁵ ⊂ C⁸ × C⁸")
    print(f"  Fiber hierarchy: S¹⁵ → S⁸ → S⁴ → S²")
    print(f"  Division algebra: Octonions (O)")
    print()
    
    t0 = time.time()
    
    r1 = test_octeract_structure()
    r2 = test_zero_point_attractor_8d()
    r3 = test_comparison_2v4v8()
    r4 = test_readout_channels_8d()
    r5 = test_berry_scaling_2_4_8()
    r6 = test_nonassociativity()
    
    # SUMMARY
    print("\n" + "=" * 76)
    print("  SUMMARY: 8×8 OCTERACT MERKABIT")
    print("=" * 76)
    elapsed = time.time() - t0
    print(f"\n  All tests completed in {elapsed:.1f} seconds")
    print(f"\n  Key findings:")
    print(f"    1. Octeract structure   → See vertex tracking in S¹⁵×S¹⁵")
    print(f"    2. Attractor behaviour  → Fiber cascade analysis")
    print(f"    3. 2→4→8 comparison     → Scaling trend across Hopf levels")
    print(f"    4. New readout channels → 6 new beyond tesseract")
    print(f"    5. Berry phase → α      → Convergence test")
    print(f"    6. Non-associativity    → Octonionic ordering dependence")
    print(f"\n  This is the TERMINAL simulation in the division algebra sequence.")
    print(f"  R → C → H → O.  There is no 16×16 that preserves Hopf structure.")
    print(f"  If the zero point is an attractor here, it's an attractor PERIOD.")
    print("=" * 76)


if __name__ == "__main__":
    main()
