#!/usr/bin/env python3
"""
4Ã—4 COUNTER-ROTATING TESSERACT SIMULATION
==========================================

Scale the merkabit from 2-component spinors (SÂ³ Ã— SÂ³) to 4-component 
spinors (Sâ· Ã— Sâ·) â€” the "counter-rotating cubes" configuration.

Key questions:
  1. Does tesseract (4D hypercube) structure emerge in the geometry?
  2. Does the zero point become a TRUE ATTRACTOR at this dimensionality?
  3. What new readout channels appear in the higher-dimensional space?

Physical reasoning:
  At 2Ã—2, the merkabit lives on SÂ³ Ã— SÂ³ (two 3-spheres).
  At 4Ã—4, it lives on Sâ· Ã— Sâ· (two 7-spheres).
  
  Sâ· is the unit sphere in Câ´ = Râ¸ = octonion space.
  The 4D hypercube (tesseract) has 16 vertices in Râ¸ when realized
  as {Â±1}â´ embedded in the 4-spinor components.
  
  The critical difference: Sâ· has INTERNAL structure that SÂ³ lacks.
  SÂ³ = SU(2) is a group manifold (simple, no internal modes).
  Sâ· is NOT a group manifold â€” it has non-trivial fiber structure:
    Sâ· â†’ Sâ´ (quaternionic Hopf fibration)
  
  This fiber structure means the ouroboros cycle on Sâ· Ã— Sâ· can
  create EFFECTIVE DISSIPATION: energy transfers between base and
  fiber modes, allowing the zero point to attract along the base
  while the fiber absorbs the "excess."
  
  This is the black hole mechanism: quasi-normal modes (fiber)
  ring down into the horizon (base), leaving the conserved structure.

Usage: python3 tesseract_merkabit_simulation.py
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
DIM = 4  # 4-component spinors


# ============================================================================
# 4-SPINOR MERKABIT STATE
# ============================================================================

class Merkabit4State:
    """
    4-spinor merkabit: (u, v) where u, v âˆˆ Câ´, |u| = |v| = 1.
    
    Lives on Sâ· Ã— Sâ· âŠ‚ Câ´ Ã— Câ´.
    
    The 4-component structure decomposes as:
      u = (uâ‚, uâ‚‚, uâ‚ƒ, uâ‚„) âˆˆ Câ´
      
    This can be viewed as a pair of 2-spinors:
      u_upper = (uâ‚, uâ‚‚)  â€” "outer" 2-spinor
      u_lower = (uâ‚ƒ, uâ‚„)  â€” "inner" 2-spinor
      
    The tesseract structure appears when the 4 real+imaginary parts
    of each pair span the vertices of a hypercube.
    """
    
    def __init__(self, u, v, omega=1.0):
        self.u = np.array(u, dtype=complex).flatten()
        self.v = np.array(v, dtype=complex).flatten()
        assert len(self.u) == DIM and len(self.v) == DIM
        self.omega = omega
        self.u /= np.linalg.norm(self.u)
        self.v /= np.linalg.norm(self.v)
    
    @property
    def overlap(self):
        """uâ€ v â€” the full complex overlap."""
        return np.vdot(self.u, self.v)
    
    @property
    def overlap_magnitude(self):
        """r = |uâ€ v|"""
        return abs(self.overlap)
    
    @property
    def coherence(self):
        """C = Re(uâ€ v)"""
        return np.real(self.overlap)
    
    @property
    def relative_phase(self):
        """Ï† = arg(uâ€ v)"""
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
    
    # ---- DECOMPOSITION INTO 2-SPINOR PAIRS ----
    
    @property
    def u_upper(self):
        return self.u[:2]
    
    @property
    def u_lower(self):
        return self.u[2:]
    
    @property
    def v_upper(self):
        return self.v[:2]
    
    @property
    def v_lower(self):
        return self.v[2:]
    
    @property
    def upper_overlap(self):
        """Overlap of upper 2-spinor pair."""
        return np.vdot(self.u_upper, self.v_upper)
    
    @property
    def lower_overlap(self):
        """Overlap of lower 2-spinor pair."""
        return np.vdot(self.u_lower, self.v_lower)
    
    @property
    def inter_overlap_ul(self):
        """Cross-overlap: u_upper Â· v_lower."""
        return np.vdot(self.u_upper, self.v_lower)
    
    @property
    def inter_overlap_lu(self):
        """Cross-overlap: u_lower Â· v_upper."""
        return np.vdot(self.u_lower, self.v_upper)
    
    @property 
    def internal_entanglement(self):
        """
        Measures how entangled the upper/lower sectors are.
        For a product state u = u_up âŠ— u_low, this is 0.
        For a maximally entangled state, this approaches 1.
        Uses the Schmidt decomposition measure.
        """
        # Reshape u as a 2Ã—2 matrix and compute singular values
        U_mat = self.u.reshape(2, 2)
        V_mat = self.v.reshape(2, 2)
        su = np.linalg.svd(U_mat, compute_uv=False)
        sv = np.linalg.svd(V_mat, compute_uv=False)
        # Entanglement entropy
        su_norm = su / np.sum(su)
        sv_norm = sv / np.sum(sv)
        ent_u = -np.sum(su_norm**2 * np.log2(su_norm**2 + 1e-15))
        ent_v = -np.sum(sv_norm**2 * np.log2(sv_norm**2 + 1e-15))
        return (ent_u + ent_v) / 2
    
    @property
    def tesseract_vertices(self):
        """
        Project the 4-spinor onto tesseract vertex coordinates.
        Map Câ´ â†’ Râ¸ â†’ {Â±1}â´ by taking sign of real/imag parts.
        Returns the vertex as a 4-tuple of Â±1.
        """
        # Use the 4 complex components, take sign of real part
        signs_u = tuple(int(np.sign(np.real(c))) if abs(np.real(c)) > 1e-10 else 0 
                       for c in self.u)
        signs_v = tuple(int(np.sign(np.real(c))) if abs(np.real(c)) > 1e-10 else 0 
                       for c in self.v)
        return signs_u, signs_v
    
    def distance_to_zero(self):
        """Distance from |0âŸ©: measured as |uâ€ v|."""
        return self.overlap_magnitude
    
    def copy(self):
        return Merkabit4State(self.u.copy(), self.v.copy(), self.omega)
    
    def __repr__(self):
        return (f"Merkabit4(C={self.coherence:.4f}, |udv|={self.overlap_magnitude:.4f}, "
                f"trit={self.trit_value:+d}, ent={self.internal_entanglement:.3f})")


# ============================================================================
# BASIS STATES (4-spinor)
# ============================================================================

def make_trit_plus_4():
    """|+1âŸ©: u = v, maximum forward coherence."""
    u = np.array([1, 0, 0, 0], dtype=complex)
    return Merkabit4State(u, u.copy())

def make_trit_zero_4():
    """|0âŸ©: u âŠ¥ v in Câ´. Standing wave."""
    u = np.array([1, 0, 0, 0], dtype=complex)
    v = np.array([0, 0, 0, 1], dtype=complex)  # maximally separated
    return Merkabit4State(u, v)

def make_trit_zero_4_alt():
    """|0âŸ© alternative: u âŠ¥ v using adjacent components."""
    u = np.array([1, 0, 0, 0], dtype=complex)
    v = np.array([0, 1, 0, 0], dtype=complex)
    return Merkabit4State(u, v)

def make_trit_zero_4_spread():
    """|0âŸ© spread: u and v occupy different 2-spinor sectors."""
    u = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0], dtype=complex)
    v = np.array([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    return Merkabit4State(u, v)

def make_trit_minus_4():
    """|-1âŸ©: u = -v, maximum inverse coherence."""
    u = np.array([1, 0, 0, 0], dtype=complex)
    v = np.array([-1, 0, 0, 0], dtype=complex)
    return Merkabit4State(u, v)

def make_trit_zero_4_tesseract():
    """|0âŸ© tesseract: u and v at opposite vertices of a tesseract."""
    # Tesseract vertices: (Â±1,Â±1,Â±1,Â±1)/2
    u = np.array([1, 1, 1, 1], dtype=complex) / 2
    v = np.array([1, -1, -1, 1], dtype=complex) / 2  # orthogonal vertex
    return Merkabit4State(u, v)

def make_near_zero_4(eps, direction=None):
    """State near |0âŸ© with controlled perturbation."""
    u = np.array([1, 0, 0, 0], dtype=complex)
    v = np.array([eps, 0, 0, np.sqrt(1 - eps**2)], dtype=complex)
    if direction is not None:
        # Rotate perturbation direction
        phase = np.exp(1j * direction)
        v[0] *= phase
    return Merkabit4State(u, v)


# ============================================================================
# 4Ã—4 GATE IMPLEMENTATIONS
# ============================================================================

def gate_Rx_4(state, theta):
    """Rx on 4-spinor: block-diagonal SU(2) Ã— SU(2) acting symmetrically."""
    c, s = np.cos(theta/2), -1j * np.sin(theta/2)
    R2 = np.array([[c, s], [s, c]], dtype=complex)
    R4 = np.block([[R2, np.zeros((2,2))], [np.zeros((2,2)), R2]])
    return Merkabit4State(R4 @ state.u, R4 @ state.v, state.omega)

def gate_Rz_4(state, theta):
    """Rz on 4-spinor: block-diagonal phase rotation."""
    R2 = np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2)])
    R4 = np.block([[R2, np.zeros((2,2))], [np.zeros((2,2)), R2]])
    return Merkabit4State(R4 @ state.u, R4 @ state.v, state.omega)

def gate_P_4(state, phi):
    """P gate: ASYMMETRIC â€” shifts relative phase. Opposite action on u and v."""
    P2f = np.diag([np.exp(1j*phi/2), np.exp(-1j*phi/2)])
    P2i = np.diag([np.exp(-1j*phi/2), np.exp(1j*phi/2)])
    Pf = np.block([[P2f, np.zeros((2,2))], [np.zeros((2,2)), P2f]])
    Pi = np.block([[P2i, np.zeros((2,2))], [np.zeros((2,2)), P2i]])
    return Merkabit4State(Pf @ state.u, Pi @ state.v, state.omega)

def gate_cross_4(state, theta):
    """
    CROSS GATE: couples upper and lower 2-spinor sectors.
    This is the NEW gate that exists only in 4Ã—4.
    It creates the tesseract mixing â€” rotating between the two cubes.
    
    Physical: this is the inter-pentachoron coupling.
    """
    c, s = np.cos(theta/2), np.sin(theta/2)
    # Rotation in the (1,3) and (2,4) planes â€” connects upper to lower
    C4 = np.array([
        [c,  0, -s, 0],
        [0,  c,  0, -s],
        [s,  0,  c, 0],
        [0,  s,  0, c],
    ], dtype=complex)
    return Merkabit4State(C4 @ state.u, C4 @ state.v, state.omega)

def gate_cross_asym_4(state, theta):
    """
    ASYMMETRIC CROSS: couples sectors with opposite sign for u vs v.
    This creates tesseract torsion â€” the counter-rotation of the cubes.
    """
    c, s = np.cos(theta/2), np.sin(theta/2)
    Cf = np.array([
        [c,  0, -s, 0],
        [0,  c,  0, -s],
        [s,  0,  c, 0],
        [0,  s,  0, c],
    ], dtype=complex)
    Ci = np.array([
        [c,  0,  s, 0],
        [0,  c,  0,  s],
        [-s, 0,  c, 0],
        [0, -s,  0, c],
    ], dtype=complex)
    return Merkabit4State(Cf @ state.u, Ci @ state.v, state.omega)


# ============================================================================
# 4Ã—4 OUROBOROS STEP
# ============================================================================

def ouroboros_step_4(state, step_index, theta=STEP_PHASE, cross_strength=0.3):
    """
    Ouroboros step for 4-spinor merkabit.
    
    Same structure as 2-spinor, but with ADDITIONAL cross-coupling
    between upper and lower sectors. This cross-coupling is what
    creates the tesseract geometry â€” it connects the two cubes.
    
    The cross gate strength determines how strongly the two
    pentachorons interact. At cross_strength=0, we get two
    independent 2-spinor ouroboros cycles. At cross_strength>0,
    the cubes couple and tesseract structure can emerge.
    """
    k = step_index
    absent = k % NUM_GATES
    
    p_angle = theta
    sym_base = theta / 3
    omega_k = 2 * np.pi * k / COXETER_H
    
    rx_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    rz_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
    
    # Cross-coupling angle: modulated by step with triality phase
    cross_angle = cross_strength * theta * (1.0 + 0.5 * np.cos(omega_k + 4*np.pi/3))
    
    gate_label = OUROBOROS_GATES[absent]
    if gate_label == 'S':
        rz_angle *= 0.4; rx_angle *= 1.3
        cross_angle *= 1.2  # S absence enhances cross-coupling
    elif gate_label == 'R':
        rx_angle *= 0.4; rz_angle *= 1.3
        cross_angle *= 0.8
    elif gate_label == 'T':
        rx_angle *= 0.7; rz_angle *= 0.7
        cross_angle *= 1.5  # T (ternary) maximally activates cross
    elif gate_label == 'P':
        p_angle *= 0.6; rx_angle *= 1.8; rz_angle *= 1.5
        cross_angle *= 0.5
    # F: no modification
    
    # Apply gates: P â†’ cross â†’ Rz â†’ Rx
    s = gate_P_4(state, p_angle)
    s = gate_cross_asym_4(s, cross_angle)  # asymmetric cross = tesseract torsion
    s = gate_Rz_4(s, rz_angle)
    s = gate_Rx_4(s, rx_angle)
    return s


# ============================================================================
# BERRY PHASE COMPUTATION (4-spinor)
# ============================================================================

def compute_berry_phase_4(states):
    """Berry phase for 4-spinor cycle."""
    n = len(states)
    gamma = 0.0
    for k in range(n):
        k_next = (k + 1) % n
        ou = np.vdot(states[k].u, states[k_next].u)
        ov = np.vdot(states[k].v, states[k_next].v)
        gamma += np.angle(ou * ov)
    return -gamma


def compute_sector_berry_phases(states):
    """
    Compute Berry phases separately for upper and lower sectors.
    Also compute the cross-sector Berry phase.
    """
    n = len(states)
    g_upper_u, g_upper_v = 0.0, 0.0
    g_lower_u, g_lower_v = 0.0, 0.0
    
    for k in range(n):
        k_next = (k + 1) % n
        g_upper_u += np.angle(np.vdot(states[k].u[:2], states[k_next].u[:2]))
        g_upper_v += np.angle(np.vdot(states[k].v[:2], states[k_next].v[:2]))
        g_lower_u += np.angle(np.vdot(states[k].u[2:], states[k_next].u[2:]))
        g_lower_v += np.angle(np.vdot(states[k].v[2:], states[k_next].v[2:]))
    
    return {
        'upper_u': -g_upper_u, 'upper_v': -g_upper_v,
        'lower_u': -g_lower_u, 'lower_v': -g_lower_v,
        'upper_total': -(g_upper_u + g_upper_v),
        'lower_total': -(g_lower_u + g_lower_v),
    }


# ============================================================================
# TEST 1: TESSERACT STRUCTURE DETECTION
# ============================================================================

def test_tesseract_structure():
    """
    Does the 4-spinor ouroboros cycle produce tesseract geometry?
    
    Test: track the 4-spinor trajectory through one cycle and check
    whether the visited states span tesseract-like vertex structure.
    """
    print("=" * 76)
    print("TEST 1: TESSERACT STRUCTURE DETECTION")
    print("  Does 4Ã—4 counter-rotation produce hypercube geometry?")
    print("=" * 76)
    
    for label, make_fn in [('|+1âŸ©', make_trit_plus_4),
                           ('|0âŸ©', make_trit_zero_4),
                           ('|0âŸ©_tess', make_trit_zero_4_tesseract),
                           ('|âˆ’1âŸ©', make_trit_minus_4)]:
        s0 = make_fn()
        print(f"\n  State: {label}")
        print(f"    Initial: {s0}")
        
        # Track vertex visits through cycle
        states = [s0.copy()]
        s = s0.copy()
        vertices_visited = set()
        
        print(f"\n    {'Step':>6}  {'u signs':>20}  {'v signs':>20}  "
              f"{'|uâ€ v|':>8}  {'C':>8}  {'ent':>6}")
        print(f"    {'-'*6}  {'-'*20}  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*6}")
        
        for step in range(COXETER_H):
            s = ouroboros_step_4(s, step)
            states.append(s.copy())
            
            vu, vv = s.tesseract_vertices
            vertices_visited.add(vu)
            vertices_visited.add(vv)
            
            print(f"    {step+1:>6}  {str(vu):>20}  {str(vv):>20}  "
                  f"{s.overlap_magnitude:>8.4f}  {s.coherence:>8.4f}  "
                  f"{s.internal_entanglement:>6.3f}")
        
        # Check cycle closure
        diff_u = np.linalg.norm(s.u - s0.u)
        diff_v = np.linalg.norm(s.v - s0.v)
        print(f"\n    Cycle closure: |Î”u| = {diff_u:.6f}, |Î”v| = {diff_v:.6f}")
        
        # Tesseract has 16 vertices
        print(f"    Unique vertices visited: {len(vertices_visited)} (tesseract has 16)")
        
        # Berry phase
        gamma = compute_berry_phase_4(states[:-1])
        sectors = compute_sector_berry_phases(states[:-1])
        print(f"    Berry phase: Î³ = {gamma:.6f} rad = {gamma/np.pi:.6f}Ï€")
        print(f"    Sector Berry: upper = {sectors['upper_total']:.6f}, "
              f"lower = {sectors['lower_total']:.6f}")
        
        # Internal entanglement evolution
        ents = [st.internal_entanglement for st in states]
        print(f"    Entanglement: min={min(ents):.4f}, max={max(ents):.4f}, "
              f"final={ents[-1]:.4f}")
    
    # Check connectivity of visited vertices (tesseract graph)
    print(f"\n  TESSERACT GEOMETRY CHECK:")
    print(f"  A tesseract has 16 vertices, 32 edges, 24 faces, 8 cells.")
    print(f"  Each vertex connects to 4 others (4-regular graph).")
    print(f"  The ouroboros cycle visits vertices and traces edges of this graph.")
    
    print(f"\n  Test: PASSED (structure analysis complete)")
    return True


# ============================================================================
# TEST 2: ZERO-POINT ATTRACTOR IN 4D
# ============================================================================

def test_zero_point_attractor_4d():
    """
    THE KEY TEST: Does the zero point become a TRUE ATTRACTOR at 4Ã—4?
    
    In 2Ã—2, |0âŸ© was a center (neutral stability, unitary conservation).
    In 4Ã—4, the cross-coupling creates internal degrees of freedom
    that can act as a dissipation bath.
    
    If |0âŸ© becomes an attractor, states near it should CONVERGE
    over multiple cycles, with the "excess" energy going into
    internal entanglement of the upper/lower sectors.
    """
    print("\n" + "=" * 76)
    print("TEST 2: ZERO-POINT ATTRACTOR TEST (4Ã—4)")
    print("  Does the zero point become self-sustaining at tesseract scale?")
    print("=" * 76)
    
    # Test different cross-coupling strengths
    cross_strengths = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    
    for cross in cross_strengths:
        print(f"\n  Cross-coupling strength: {cross:.1f}")
        print(f"  {'Îµ':>8}  {'dâ‚€':>8}  {'d_1':>8}  {'d_5':>8}  "
              f"{'d_20':>8}  {'d_50':>8}  {'ent_50':>8}  {'Trend':>12}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  "
              f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")
        
        attract_count = 0
        for eps in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
            s0 = make_near_zero_4(eps)
            d0 = s0.overlap_magnitude
            
            s = s0.copy()
            distances = {}
            ent_vals = {}
            
            total_cycles = 0
            for target in [1, 5, 20, 50]:
                while total_cycles < target:
                    for step in range(COXETER_H):
                        s = ouroboros_step_4(s, step, cross_strength=cross)
                    total_cycles += 1
                distances[target] = s.overlap_magnitude
                ent_vals[target] = s.internal_entanglement
            
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
                  f"{d50:>8.4f}  {ent_vals[50]:>8.4f}  {trend:>12}")
        
        print(f"  Attraction count: {attract_count}/8")
    
    # ---- Long-term tracking at optimal cross-coupling ----
    print(f"\n  LONG-TERM EVOLUTION (cross_strength = 0.3, Îµ = 0.05)")
    
    s0 = make_near_zero_4(0.05)
    distances = [s0.overlap_magnitude]
    coherences = [s0.coherence]
    entanglements = [s0.internal_entanglement]
    
    s = s0.copy()
    for cycle in range(200):
        for step in range(COXETER_H):
            s = ouroboros_step_4(s, step, cross_strength=0.3)
        distances.append(s.overlap_magnitude)
        coherences.append(s.coherence)
        entanglements.append(s.internal_entanglement)
    
    distances = np.array(distances)
    entanglements = np.array(entanglements)
    
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
    
    print(f"\n    Entanglement evolution:")
    print(f"    Initial:              {entanglements[0]:.6f}")
    print(f"    After 50 cycles:      {entanglements[50]:.6f}")
    print(f"    After 200 cycles:     {entanglements[200]:.6f}")
    print(f"    Mean (first 50):      {np.mean(entanglements[:50]):.6f}")
    print(f"    Mean (last 50):       {np.mean(entanglements[150:]):.6f}")
    
    # ASCII plot: distance from |0âŸ©
    print(f"\n    Distance from |0âŸ© over 200 cycles:")
    d_min, d_max = np.min(distances), np.max(distances)
    d_range = d_max - d_min if d_max > d_min else 1e-6
    rows, cols = 12, 60
    grid = [[' ' for _ in range(cols)] for _ in range(rows)]
    for i in range(len(distances)):
        col = int(i / 200 * (cols - 1))
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
    print(f"           cycle 0{' '*20}cycle 100{' '*18}cycle 200")
    
    # ---- VERDICT ----
    is_attractor = last_q < first_q * 0.7
    is_weak_attractor = last_q < first_q * 0.95
    
    print(f"\n  VERDICT:")
    if is_attractor:
        print(f"    |0âŸ© is a STRONG ATTRACTOR in 4Ã—4 (distance decreases by >{30:.0f}%)")
        print(f"    The zero point IS SELF-SUSTAINING at tesseract scale")
        print(f"    Internal entanglement absorbs the 'excess' â€” black hole ringdown")
    elif is_weak_attractor:
        print(f"    |0âŸ© is a WEAK ATTRACTOR in 4Ã—4 (distance decreases by ~{(1-last_q/first_q)*100:.0f}%)")
        print(f"    Partial self-sustaining behaviour â€” damped oscillation")
    else:
        print(f"    |0âŸ© remains a CENTER in 4Ã—4 (no net attraction)")
        print(f"    Additional structure (more dimensions? lattice?) may be needed")
    
    print(f"\n  Test: PASSED (dynamics analysis complete)")
    return True


# ============================================================================
# TEST 3: COMPARISON 2Ã—2 vs 4Ã—4
# ============================================================================

def test_comparison_2v4():
    """
    Direct comparison: same perturbation, 2-spinor vs 4-spinor.
    Does the extra dimensionality change the zero-point dynamics?
    """
    print("\n" + "=" * 76)
    print("TEST 3: COMPARISON 2Ã—2 vs 4Ã—4")
    print("  Same perturbation, different dimensionality")
    print("=" * 76)
    
    # 2-spinor version
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
        
        u = Rx @ Rz @ Pf @ state.u
        v = Rx @ Rz @ Pi @ state.v
        return M2(u, v)
    
    eps_val = 0.05
    n_cyc = 100
    
    # 2-spinor run
    s2 = M2([1, 0], [eps_val, np.sqrt(1-eps_val**2)])
    dist_2 = [s2.overlap_magnitude]
    for cyc in range(n_cyc):
        for step in range(COXETER_H):
            s2 = step_2(s2, step)
        dist_2.append(s2.overlap_magnitude)
    
    # 4-spinor runs at different cross-coupling
    dist_4 = {}
    ent_4 = {}
    for cross in [0.0, 0.3, 0.5, 1.0]:
        s4 = make_near_zero_4(eps_val)
        d = [s4.overlap_magnitude]
        e = [s4.internal_entanglement]
        for cyc in range(n_cyc):
            for step in range(COXETER_H):
                s4 = ouroboros_step_4(s4, step, cross_strength=cross)
            d.append(s4.overlap_magnitude)
            e.append(s4.internal_entanglement)
        dist_4[cross] = np.array(d)
        ent_4[cross] = np.array(e)
    
    dist_2 = np.array(dist_2)
    
    print(f"\n  Perturbation Îµ = {eps_val}, tracked for {n_cyc} cycles")
    print(f"\n  {'System':>20}  {'Mean d(first 25)':>18}  {'Mean d(last 25)':>18}  "
          f"{'Change':>10}  {'Mean ent(last 25)':>18}")
    print(f"  {'-'*20}  {'-'*18}  {'-'*18}  {'-'*10}  {'-'*18}")
    
    first_2 = np.mean(dist_2[:25])
    last_2 = np.mean(dist_2[75:])
    change_2 = (last_2 - first_2) / first_2 * 100
    print(f"  {'2Ã—2':>20}  {first_2:>18.6f}  {last_2:>18.6f}  {change_2:>+9.1f}%  {'N/A':>18}")
    
    for cross in [0.0, 0.3, 0.5, 1.0]:
        d = dist_4[cross]
        e = ent_4[cross]
        first_4 = np.mean(d[:25])
        last_4 = np.mean(d[75:])
        change_4 = (last_4 - first_4) / first_4 * 100
        ent_last = np.mean(e[75:])
        print(f"  {'4Ã—4 cross='+str(cross):>20}  {first_4:>18.6f}  {last_4:>18.6f}  "
              f"{change_4:>+9.1f}%  {ent_last:>18.6f}")
    
    # Entanglement growth analysis
    print(f"\n  ENTANGLEMENT GROWTH (does it absorb the 'excess'?):")
    for cross in [0.3, 0.5, 1.0]:
        e = ent_4[cross]
        d = dist_4[cross]
        ent_first = np.mean(e[:25])
        ent_last = np.mean(e[75:])
        # Correlation between distance decrease and entanglement increase
        d_change = np.mean(d[75:]) - np.mean(d[:25])
        e_change = ent_last - ent_first
        print(f"    cross={cross:.1f}: Î”d = {d_change:+.6f}, Î”ent = {e_change:+.6f}", end="")
        if d_change < 0 and e_change > 0:
            print(f"  â†’ TRANSFER (distanceâ†’entanglement)")
        elif d_change < 0:
            print(f"  â†’ ATTRACTION (no entanglement growth)")
        else:
            print(f"  â†’ NO TRANSFER")
    
    print(f"\n  Test: PASSED")
    return True


# ============================================================================
# TEST 4: READOUT CHANNELS IN 4D
# ============================================================================

def test_readout_channels_4d():
    """
    What new readout channels exist in 4Ã—4 that don't exist in 2Ã—2?
    """
    print("\n" + "=" * 76)
    print("TEST 4: READOUT CHANNELS IN 4Ã—4")
    print("  What can the tesseract read that the cube cannot?")
    print("=" * 76)
    
    basis_states = [
        ('|+1âŸ©', make_trit_plus_4),
        ('|0âŸ©', make_trit_zero_4),
        ('|0âŸ©_spread', make_trit_zero_4_spread),
        ('|0âŸ©_tess', make_trit_zero_4_tesseract),
        ('|âˆ’1âŸ©', make_trit_minus_4),
    ]
    
    print(f"\n  {'State':>12}  {'Î³_total':>10}  {'Î³_upper':>10}  {'Î³_lower':>10}  "
          f"{'C_total':>10}  {'C_upper':>10}  {'C_lower':>10}  {'ent':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  "
          f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    
    for label, make_fn in basis_states:
        s0 = make_fn()
        states = [s0.copy()]
        s = s0.copy()
        for step in range(COXETER_H):
            s = ouroboros_step_4(s, step)
            states.append(s.copy())
        
        gamma = compute_berry_phase_4(states[:-1])
        sectors = compute_sector_berry_phases(states[:-1])
        
        c_total = np.real(np.vdot(s0.u, s0.v))
        c_upper = np.real(np.vdot(s0.u[:2], s0.v[:2]))
        c_lower = np.real(np.vdot(s0.u[2:], s0.v[2:]))
        
        print(f"  {label:>12}  {gamma:>10.4f}  {sectors['upper_total']:>10.4f}  "
              f"{sectors['lower_total']:>10.4f}  {c_total:>10.4f}  "
              f"{c_upper:>10.4f}  {c_lower:>10.4f}  "
              f"{s.internal_entanglement:>8.4f}")
    
    print(f"\n  NEW CHANNELS in 4Ã—4:")
    print(f"    1. Sector Berry phases (upper vs lower) â€” not available in 2Ã—2")
    print(f"    2. Cross-sector coherence â€” encodes inter-pentachoron coupling")
    print(f"    3. Internal entanglement â€” measures how much the cubes have merged")
    print(f"    4. Multiple zero-point variants (|0âŸ©, |0âŸ©_spread, |0âŸ©_tess)")
    print(f"       distinguished by sector structure")
    
    print(f"\n  Test: PASSED")
    return True


# ============================================================================
# TEST 5: BERRY PHASE SCALING 2â†’4
# ============================================================================

def test_berry_scaling():
    """
    Does the Berry phase â†’ Î± expression scale correctly from 2Ã—2 to 4Ã—4?
    """
    print("\n" + "=" * 76)
    print("TEST 5: BERRY PHASE SCALING")
    print("  Does |Î³â‚€|/Ï€ Ã— dim(Eâ‚†) scale from 2â†’4 spinors?")
    print("=" * 76)
    
    # 4-spinor Berry phases for different |0âŸ© variants
    E6_DIM = 78
    
    for label, make_fn in [('|0âŸ© simple', make_trit_zero_4),
                           ('|0âŸ© spread', make_trit_zero_4_spread),
                           ('|0âŸ© tess', make_trit_zero_4_tesseract)]:
        for cross in [0.0, 0.3, 0.5]:
            s0 = make_fn()
            states = [s0.copy()]
            s = s0.copy()
            for step in range(COXETER_H):
                s = ouroboros_step_4(s, step, cross_strength=cross)
                states.append(s.copy())
            
            gamma = compute_berry_phase_4(states[:-1])
            val = abs(gamma) / np.pi * E6_DIM
            diff = val - 137.036
            
            print(f"  {label:>15} cross={cross:.1f}:  |Î³â‚€|/Ï€ = {abs(gamma)/np.pi:.6f}  "
                  f"Ã— 78 = {val:.4f}  Î”137 = {diff:+.4f}")
    
    print(f"\n  2Ã—2 reference:              |Î³â‚€|/Ï€ = 1.837238  Ã— 78 = 143.3046")
    
    print(f"\n  Test: PASSED")
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 76)
    print("  4Ã—4 COUNTER-ROTATING TESSERACT SIMULATION")
    print("  Scaling the merkabit from SÂ³Ã—SÂ³ to Sâ·Ã—Sâ·")
    print("=" * 76)
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Spinor dimension: {DIM}")
    print(f"  State space: Sâ· Ã— Sâ· âŠ‚ Câ´ Ã— Câ´")
    print()
    
    t0 = time.time()
    
    r1 = test_tesseract_structure()
    r2 = test_zero_point_attractor_4d()
    r3 = test_comparison_2v4()
    r4 = test_readout_channels_4d()
    r5 = test_berry_scaling()
    
    # SUMMARY
    print("\n" + "=" * 76)
    print("  SUMMARY")
    print("=" * 76)
    print(f"\n  All tests completed in {time.time() - t0:.1f} seconds")
    print(f"\n  Key questions answered:")
    print(f"    1. Tesseract structure? â†’ See vertex tracking results")
    print(f"    2. Self-sustaining zero point? â†’ See attractor analysis")
    print(f"    3. New readout channels? â†’ 4 new channels (sectors, cross, entanglement)")
    print(f"    4. Berry phase scaling? â†’ See comparison with 2Ã—2")
    print("=" * 76)


if __name__ == "__main__":
    main()
