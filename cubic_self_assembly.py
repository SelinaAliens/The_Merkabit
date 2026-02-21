#!/usr/bin/env python3
"""
CUBIC SELF-ASSEMBLY SIMULATION
================================

The fundamental test: does the ouroboros cycle drive two merkabits
into cubic docking from random initial conditions?

The paper claims the torsion tunnel IS the lattice, and the lattice
IS forward cube meeting inverse cube. If this is self-correcting
geometry, it should be self-organising: the drive should produce
the docking, not require it to be pre-engineered.

THE STRUCTURE:
  Each merkabit has 5 gates: R, T, P, F, S
  R = rotation (symmetric, internal — acts on both spinors equally)
  T, P, F, S = inter-channel gates (connect forward to inverse)
  
  R is the self-reference. The other 4 are the connections.
  
  The forward cube = 4 inter-channel gates in forward chirality
  The inverse cube = 4 inter-channel gates in inverse chirality
  
  Cubic docking = A's 4 forward gates aligning with B's 4 inverse gates
  The body-diagonal projection of this cube = hexagonal lattice tile
  
  The ouroboros cycle rotates the absent gate through all 5 positions.
  At each step, 4 gates are active, 1 is absent.
  Forward chirality: absent gate rotates S→R→T→F→P→S
  Inverse chirality: absent gate rotates in opposite direction

THE TEST:
  Start two merkabits with random spinor states and random frequencies.
  Drive both with the ouroboros cycle.
  Track:
    1. Total conductance G = |u_A† v_B|² (overall alignment)
    2. Per-gate channel alignment (which of the 4 inter-channel 
       connections are established)
    3. Number of docked tetrahedral cells (out of 4 per step)
    4. Whether the pattern of docking rotates with the cycle
    5. Frequency convergence: does ω_A + ω_B → 0?

  If the ouroboros drive increases alignment from random starts → 
    self-assembly confirmed
  If alignment stays random → lattice must be engineered from outside

Physical basis:
  Section 8.3.2 (torsion tunnel = cubic junction)
  Section 8.3.2.4 (dynamics of tunnel formation)
  Section 8.3.2.5 (one mechanism at different scales)
  Section 8.5 (ouroboros cycle)

Usage: python3 cubic_self_assembly.py
Requirements: numpy
"""

import numpy as np
import time
import sys

# ============================================================================
# CONSTANTS
# ============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H  # π/6

# 5 pentachoric gates
GATES = ['R', 'T', 'P', 'F', 'S']
INTER_CHANNEL_GATES = ['T', 'P', 'F', 'S']  # the 4 that connect fwd/inv
NUM_GATES = 5

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


# ============================================================================
# MERKABIT STATE
# ============================================================================

class MerkabitState:
    def __init__(self, u, v, omega=1.0):
        self.u = np.array(u, dtype=complex)
        self.v = np.array(v, dtype=complex)
        self.omega = omega
        self.u /= np.linalg.norm(self.u)
        self.v /= np.linalg.norm(self.v)
    
    @property
    def coherence(self):
        return np.real(np.vdot(self.u, self.v))
    
    @property
    def relative_phase(self):
        return np.angle(np.vdot(self.u, self.v))
    
    @property
    def overlap_magnitude(self):
        return abs(np.vdot(self.u, self.v))
    
    def copy(self):
        return MerkabitState(self.u.copy(), self.v.copy(), self.omega)


def make_random_state(rng, omega=None):
    u = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    v = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    if omega is None:
        omega = rng.uniform(-3, 3)
    return MerkabitState(u, v, omega)


# ============================================================================
# SINGLE-MERKABIT GATES
# ============================================================================

def gate_Rx(state, theta):
    c, s = np.cos(theta/2), -1j * np.sin(theta/2)
    R = np.array([[c, s], [s, c]], dtype=complex)
    return MerkabitState(R @ state.u, R @ state.v, state.omega)

def gate_Rz(state, theta):
    R = np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2)])
    return MerkabitState(R @ state.u, R @ state.v, state.omega)

def gate_P(state, phi):
    Pf = np.diag([np.exp(1j*phi/2), np.exp(-1j*phi/2)])
    Pi = np.diag([np.exp(-1j*phi/2), np.exp(1j*phi/2)])
    return MerkabitState(Pf @ state.u, Pi @ state.v, state.omega)

def gate_T(state, theta):
    """Transfer gate: Ry rotation (amplitude exchange between components)."""
    c, s = np.cos(theta/2), np.sin(theta/2)
    R = np.array([[c, -s], [s, c]], dtype=complex)
    return MerkabitState(R @ state.u, R @ state.v, state.omega)

def gate_F(state, delta_omega):
    return MerkabitState(state.u.copy(), state.v.copy(),
                         state.omega + delta_omega)

def gate_S(state, g):
    """Substrate coupling: controlled phase proportional to coupling g."""
    # S(g) = controlled-P(g * phi_0) — environment-dependent phase
    phi_env = g * np.pi / 6  # coupling strength determines phase shift
    Pf = np.diag([np.exp(1j*phi_env/2), np.exp(-1j*phi_env/2)])
    Pi = np.diag([np.exp(-1j*phi_env/2), np.exp(1j*phi_env/2)])
    return MerkabitState(Pf @ state.u, Pi @ state.v, state.omega)


# ============================================================================
# GATE CHANNEL ACTIONS
# ============================================================================

def gate_channel_action(state, gate_label, theta, chirality=+1):
    """
    Apply a specific gate in a given chirality direction.
    
    chirality = +1: forward (standard action)
    chirality = -1: inverse (conjugate action — parameter negated)
    
    This is the key: the forward cube applies gates with chirality +1,
    the inverse cube applies them with chirality -1. Docking means
    the forward action of A aligns with the inverse action of B.
    """
    t = theta * chirality
    
    if gate_label == 'R':
        return gate_Rx(state, t)
    elif gate_label == 'T':
        return gate_T(state, t)
    elif gate_label == 'P':
        return gate_P(state, t)
    elif gate_label == 'F':
        return gate_F(state, t * 0.1)  # scaled for frequency
    elif gate_label == 'S':
        return gate_S(state, t)
    else:
        raise ValueError(f"Unknown gate: {gate_label}")


# ============================================================================
# OUROBOROS STEP (from ouroboros_berry_phase_simulation.py)
# ============================================================================

def ouroboros_step(state, step_index, chirality=+1, theta=STEP_PHASE):
    """
    One step of the ouroboros cycle with chirality-dependent absent gate.
    
    chirality = +1: absent gate rotates S→R→T→F→P (forward)
    chirality = -1: absent gate rotates in reverse P→F→T→R→S (inverse)
    """
    k = step_index
    
    # Absent gate index depends on chirality
    if chirality > 0:
        absent = k % NUM_GATES
    else:
        absent = (-k) % NUM_GATES
    
    absent_label = GATES[absent]
    
    # Asymmetric part: P advances relative phase
    p_angle = theta
    
    # Symmetric part: Rx, Rz modulated by absent gate
    sym_base = theta / 3
    omega_k = 2 * np.pi * k / COXETER_H
    
    rx_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k * chirality))
    rz_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k * chirality + 2*np.pi/3))
    
    # Absent gate modifies balance
    if absent_label == 'S':
        rz_angle *= 0.4; rx_angle *= 1.3
    elif absent_label == 'R':
        rx_angle *= 0.4; rz_angle *= 1.3
    elif absent_label == 'T':
        rx_angle *= 0.7; rz_angle *= 0.7
    elif absent_label == 'P':
        p_angle *= 0.6; rx_angle *= 1.8; rz_angle *= 1.5
    # F absent: no change
    
    s = gate_P(state, p_angle * chirality)
    s = gate_Rz(s, rz_angle)
    s = gate_Rx(s, rx_angle)
    
    return s, absent_label


# ============================================================================
# INTER-MERKABIT ALIGNMENT METRICS
# ============================================================================

def total_conductance(A, B):
    """G_AB = |u_A† v_B|² — overall tunnel conductance."""
    return abs(np.vdot(A.u, B.v)) ** 2

def frequency_detuning(A, B):
    """How far from resonance: |ω_A + ω_B|."""
    return abs(A.omega + B.omega)

def per_gate_alignment(A, B, theta=STEP_PHASE):
    """
    Measure alignment for each of the 4 inter-channel gates.
    
    For each gate g ∈ {T, P, F, S}:
      Apply g in forward chirality to A: A' = g_fwd(A)
      Apply g in inverse chirality to B: B' = g_inv(B)
      Alignment = |⟨u_A'|v_B'⟩|²
    
    If A's forward cube aligns with B's inverse cube,
    all 4 alignments should be high.
    
    R is excluded — it's the internal rotation (self-reference),
    not an inter-channel connection.
    """
    alignments = {}
    
    for g in INTER_CHANNEL_GATES:
        A_fwd = gate_channel_action(A, g, theta, chirality=+1)
        B_inv = gate_channel_action(B, g, theta, chirality=-1)
        
        # Alignment: how well does the forward action on A
        # match the inverse action on B?
        alignment = abs(np.vdot(A_fwd.u, B_inv.v)) ** 2
        alignments[g] = alignment
    
    return alignments

def count_docked_cells(alignments, threshold=0.3):
    """How many of the 4 inter-channel gates are aligned above threshold."""
    return sum(1 for v in alignments.values() if v > threshold)


# ============================================================================
# TEST 1: TWO MERKABITS — RANDOM START, OUROBOROS DRIVE
# ============================================================================

def test_random_pair_drive():
    """
    The central test: start two merkabits at random states.
    Drive both with the ouroboros cycle (opposite chiralities).
    Track whether cubic alignment emerges.
    """
    print("\n" + "=" * 72)
    print("TEST 1: RANDOM PAIR — OUROBOROS DRIVE")
    print("=" * 72)
    print(f"\n  Question: does the ouroboros drive produce cubic docking")
    print(f"  from random initial conditions?")
    
    rng = np.random.default_rng(RANDOM_SEED)
    n_cycles = 10  # 10 full ouroboros cycles = 120 steps
    n_steps = n_cycles * COXETER_H
    
    # Random initial states
    A = make_random_state(rng, omega=rng.uniform(0.5, 2.0))
    B = make_random_state(rng, omega=rng.uniform(-2.0, -0.5))
    
    print(f"\n  Initial states:")
    print(f"    A: φ={A.relative_phase:.4f}, C={A.coherence:.4f}, ω={A.omega:+.4f}")
    print(f"    B: φ={B.relative_phase:.4f}, C={B.coherence:.4f}, ω={B.omega:+.4f}")
    print(f"    |ω_A + ω_B| = {frequency_detuning(A, B):.4f}")
    
    G0 = total_conductance(A, B)
    align0 = per_gate_alignment(A, B)
    cells0 = count_docked_cells(align0)
    
    print(f"    G_AB = {G0:.6f}")
    print(f"    Per-gate: T={align0['T']:.4f}  P={align0['P']:.4f}  "
          f"F={align0['F']:.4f}  S={align0['S']:.4f}")
    print(f"    Docked cells: {cells0}/4")
    
    # Drive and track
    print(f"\n  Driving {n_cycles} ouroboros cycles ({n_steps} steps)...")
    print(f"\n  {'step':>5s}  {'cycle':>5s}  {'G_AB':>8s}  "
          f"{'T':>6s}  {'P':>6s}  {'F':>6s}  {'S':>6s}  "
          f"{'cells':>5s}  {'absent_A':>8s}  {'absent_B':>8s}")
    print(f"  {'─'*5}  {'─'*5}  {'─'*8}  "
          f"{'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  "
          f"{'─'*5}  {'─'*8}  {'─'*8}")
    
    G_history = []
    cells_history = []
    
    for step in range(n_steps):
        # Drive A in forward chirality, B in inverse chirality
        A, absent_A = ouroboros_step(A, step, chirality=+1)
        B, absent_B = ouroboros_step(B, step, chirality=-1)
        
        G = total_conductance(A, B)
        align = per_gate_alignment(A, B)
        cells = count_docked_cells(align)
        
        G_history.append(G)
        cells_history.append(cells)
        
        # Print every 6 steps (twice per cycle) + first and last
        if step % 6 == 0 or step == n_steps - 1:
            cycle = step // COXETER_H
            print(f"  {step:5d}  {cycle:5d}  {G:8.6f}  "
                  f"{align['T']:6.4f}  {align['P']:6.4f}  "
                  f"{align['F']:6.4f}  {align['S']:6.4f}  "
                  f"{cells:5d}  {absent_A:>8s}  {absent_B:>8s}")
    
    # Analysis
    G_initial = G_history[0]
    G_final = np.mean(G_history[-COXETER_H:])
    G_max = max(G_history)
    cells_initial = cells_history[0]
    cells_final_avg = np.mean(cells_history[-COXETER_H:])
    
    print(f"\n  ANALYSIS:")
    print(f"    Initial G:    {G_initial:.6f}")
    print(f"    Final G (avg): {G_final:.6f}")
    print(f"    Max G:        {G_max:.6f}")
    print(f"    G trend:      {'INCREASING ✓' if G_final > G_initial * 1.1 else 'NO TREND ✗' if abs(G_final - G_initial) < G_initial * 0.1 else 'DECREASING'}")
    print(f"    Docked cells: {cells_initial} → {cells_final_avg:.1f} (avg last cycle)")
    
    return G_history, cells_history


# ============================================================================
# TEST 2: ENSEMBLE OF RANDOM PAIRS
# ============================================================================

def test_ensemble():
    """
    Run the self-assembly test across many random initial conditions.
    
    For each pair: track whether G increases over 5 ouroboros cycles.
    Statistics: what fraction of random pairs show increasing alignment?
    """
    print("\n" + "=" * 72)
    print("TEST 2: ENSEMBLE — 500 RANDOM PAIRS")
    print("=" * 72)
    
    rng = np.random.default_rng(RANDOM_SEED + 1)
    n_pairs = 500
    n_cycles = 5
    n_steps = n_cycles * COXETER_H
    
    print(f"\n  {n_pairs} random pairs, {n_cycles} cycles each")
    
    results = {
        'G_initial': [],
        'G_final': [],
        'G_max': [],
        'G_increased': 0,
        'G_decreased': 0,
        'cells_initial': [],
        'cells_final': [],
        'cells_increased': 0,
    }
    
    for pair in range(n_pairs):
        A = make_random_state(rng)
        B = make_random_state(rng)
        
        G0 = total_conductance(A, B)
        align0 = per_gate_alignment(A, B)
        cells0 = count_docked_cells(align0)
        
        # Drive
        for step in range(n_steps):
            A, _ = ouroboros_step(A, step, chirality=+1)
            B, _ = ouroboros_step(B, step, chirality=-1)
        
        Gf = total_conductance(A, B)
        alignf = per_gate_alignment(A, B)
        cellsf = count_docked_cells(alignf)
        
        results['G_initial'].append(G0)
        results['G_final'].append(Gf)
        results['G_max'].append(max(G0, Gf))
        results['cells_initial'].append(cells0)
        results['cells_final'].append(cellsf)
        
        if Gf > G0 * 1.1:
            results['G_increased'] += 1
        elif Gf < G0 * 0.9:
            results['G_decreased'] += 1
        
        if cellsf > cells0:
            results['cells_increased'] += 1
    
    # Statistics
    gi = np.array(results['G_initial'])
    gf = np.array(results['G_final'])
    ci = np.array(results['cells_initial'])
    cf = np.array(results['cells_final'])
    
    print(f"\n  CONDUCTANCE G_AB:")
    print(f"    Initial: mean={np.mean(gi):.4f}, std={np.std(gi):.4f}")
    print(f"    Final:   mean={np.mean(gf):.4f}, std={np.std(gf):.4f}")
    print(f"    Δ(mean): {np.mean(gf) - np.mean(gi):+.4f}")
    print(f"    Pairs where G increased (>10%): {results['G_increased']}/{n_pairs} "
          f"({100*results['G_increased']/n_pairs:.1f}%)")
    print(f"    Pairs where G decreased (>10%): {results['G_decreased']}/{n_pairs} "
          f"({100*results['G_decreased']/n_pairs:.1f}%)")
    
    print(f"\n  DOCKED CELLS (out of 4):")
    print(f"    Initial: mean={np.mean(ci):.2f}")
    print(f"    Final:   mean={np.mean(cf):.2f}")
    print(f"    Pairs where cells increased: {results['cells_increased']}/{n_pairs}")
    
    # Distribution of final G
    print(f"\n  DISTRIBUTION OF FINAL G:")
    bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    for lo, hi in bins:
        count = np.sum((gf >= lo) & (gf < hi))
        bar = '█' * int(count * 40 / n_pairs)
        print(f"    [{lo:.1f}, {hi:.1f}): {count:4d}  {bar}")
    
    # Verdict
    g_trend = np.mean(gf) - np.mean(gi)
    print(f"\n  VERDICT:")
    if g_trend > 0.05:
        print(f"    Conductance INCREASES under ouroboros drive (+{g_trend:.4f})")
        print(f"    → Evidence for self-assembly")
    elif g_trend < -0.05:
        print(f"    Conductance DECREASES under ouroboros drive ({g_trend:.4f})")
        print(f"    → Drive disperses alignment")
    else:
        print(f"    Conductance UNCHANGED under ouroboros drive ({g_trend:+.4f})")
        print(f"    → No evidence for self-assembly from drive alone")
    
    return results


# ============================================================================
# TEST 3: PRE-RESONANT PAIRS (ω_A + ω_B ≈ 0)
# ============================================================================

def test_resonant_pairs():
    """
    What if frequency matching is already satisfied?
    
    The bipartite lattice provides ω_A + ω_B = 0 structurally.
    Given this, does the ouroboros drive produce cubic alignment
    of the 4 inter-channel gates?
    
    This separates the frequency question from the alignment question.
    """
    print("\n" + "=" * 72)
    print("TEST 3: PRE-RESONANT PAIRS (ω_A + ω_B = 0)")
    print("=" * 72)
    print(f"\n  Frequency matching satisfied by lattice structure.")
    print(f"  Question: does the drive align the 4 inter-channel gates?")
    
    rng = np.random.default_rng(RANDOM_SEED + 2)
    n_pairs = 500
    n_cycles = 10
    n_steps = n_cycles * COXETER_H
    
    G_initial = []
    G_final = []
    G_trajectories = []
    
    for pair in range(n_pairs):
        omega = rng.uniform(0.5, 2.0)
        A = make_random_state(rng, omega=omega)
        B = make_random_state(rng, omega=-omega)  # resonance guaranteed
        
        G0 = total_conductance(A, B)
        G_initial.append(G0)
        
        trajectory = [G0]
        for step in range(n_steps):
            A, _ = ouroboros_step(A, step, chirality=+1)
            B, _ = ouroboros_step(B, step, chirality=-1)
            if step % COXETER_H == COXETER_H - 1:
                trajectory.append(total_conductance(A, B))
        
        Gf = total_conductance(A, B)
        G_final.append(Gf)
        G_trajectories.append(trajectory)
    
    gi = np.array(G_initial)
    gf = np.array(G_final)
    
    print(f"\n  {n_pairs} pairs, ω_A = -ω_B (resonance guaranteed)")
    print(f"  Random spinor orientations, {n_cycles} cycles each")
    
    print(f"\n  CONDUCTANCE:")
    print(f"    Initial: mean={np.mean(gi):.4f}, std={np.std(gi):.4f}")
    print(f"    Final:   mean={np.mean(gf):.4f}, std={np.std(gf):.4f}")
    print(f"    Δ(mean): {np.mean(gf) - np.mean(gi):+.4f}")
    
    # Trajectory: average G at each cycle boundary
    n_traj_points = len(G_trajectories[0])
    avg_trajectory = []
    for t in range(n_traj_points):
        vals = [G_trajectories[p][t] for p in range(n_pairs)
                if t < len(G_trajectories[p])]
        avg_trajectory.append(np.mean(vals))
    
    print(f"\n  AVERAGE G TRAJECTORY (by cycle):")
    for t, g in enumerate(avg_trajectory):
        bar_len = int(40 * g)
        bar = '█' * bar_len + '░' * (40 - bar_len)
        label = "initial" if t == 0 else f"cycle {t}"
        print(f"    {label:>10s}: {g:.6f}  {bar}")
    
    # Monotonicity
    increasing = all(avg_trajectory[i+1] >= avg_trajectory[i] - 0.001 
                     for i in range(len(avg_trajectory)-1))
    
    print(f"\n  Trajectory monotonic: {'YES' if increasing else 'NO'}")
    
    # Per-gate analysis at final state
    gate_alignments = {g: [] for g in INTER_CHANNEL_GATES}
    for pair in range(min(n_pairs, 100)):
        omega = rng.uniform(0.5, 2.0)
        A = make_random_state(rng, omega=omega)
        B = make_random_state(rng, omega=-omega)
        for step in range(n_steps):
            A, _ = ouroboros_step(A, step, chirality=+1)
            B, _ = ouroboros_step(B, step, chirality=-1)
        align = per_gate_alignment(A, B)
        for g in INTER_CHANNEL_GATES:
            gate_alignments[g].append(align[g])
    
    print(f"\n  PER-GATE ALIGNMENT (final state, 100 pairs):")
    for g in INTER_CHANNEL_GATES:
        vals = gate_alignments[g]
        print(f"    {g}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")
    
    return gi, gf


# ============================================================================
# TEST 4: CHIRALITY MATTERS — SAME vs OPPOSITE DRIVE
# ============================================================================

def test_chirality_dependence():
    """
    Does opposite-chirality drive produce better alignment than 
    same-chirality drive?
    
    If the cubic docking requires forward-meeting-inverse,
    then driving both merkabits in the SAME chirality should NOT
    produce docking, while opposite chiralities should.
    
    This is the structural prediction: the counter-rotation is 
    essential, not incidental.
    """
    print("\n" + "=" * 72)
    print("TEST 4: CHIRALITY — OPPOSITE vs SAME DRIVE")
    print("=" * 72)
    print(f"\n  If docking requires forward ↔ inverse,")
    print(f"  opposite chirality should dock, same should not.")
    
    rng = np.random.default_rng(RANDOM_SEED + 3)
    n_pairs = 300
    n_cycles = 5
    n_steps = n_cycles * COXETER_H
    
    opposite_G = []
    same_G = []
    
    for pair in range(n_pairs):
        omega = rng.uniform(0.5, 2.0)
        
        # Same initial states for fair comparison
        u_A = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        v_A = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        u_B = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        v_B = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        
        # Opposite chirality (forward + inverse)
        A_opp = MerkabitState(u_A.copy(), v_A.copy(), omega)
        B_opp = MerkabitState(u_B.copy(), v_B.copy(), -omega)
        
        for step in range(n_steps):
            A_opp, _ = ouroboros_step(A_opp, step, chirality=+1)
            B_opp, _ = ouroboros_step(B_opp, step, chirality=-1)
        
        opposite_G.append(total_conductance(A_opp, B_opp))
        
        # Same chirality (forward + forward)
        A_same = MerkabitState(u_A.copy(), v_A.copy(), omega)
        B_same = MerkabitState(u_B.copy(), v_B.copy(), -omega)
        
        for step in range(n_steps):
            A_same, _ = ouroboros_step(A_same, step, chirality=+1)
            B_same, _ = ouroboros_step(B_same, step, chirality=+1)  # same!
        
        same_G.append(total_conductance(A_same, B_same))
    
    opp = np.array(opposite_G)
    sam = np.array(same_G)
    
    print(f"\n  {n_pairs} pairs, ω_A = -ω_B, random spinors")
    print(f"\n  Opposite chirality (fwd + inv):")
    print(f"    mean G = {np.mean(opp):.4f}, std = {np.std(opp):.4f}")
    print(f"  Same chirality (fwd + fwd):")
    print(f"    mean G = {np.mean(sam):.4f}, std = {np.std(sam):.4f}")
    print(f"  Δ(mean) = {np.mean(opp) - np.mean(sam):+.4f}")
    
    if np.mean(opp) > np.mean(sam) + 0.02:
        print(f"\n  → Opposite chirality produces HIGHER alignment ✓")
        print(f"    Counter-rotation is structurally essential")
    elif abs(np.mean(opp) - np.mean(sam)) < 0.02:
        print(f"\n  → No significant difference")
        print(f"    Chirality may not control alignment at this level")
    else:
        print(f"\n  → Same chirality produces higher alignment")
        print(f"    Unexpected — challenges fwd/inv docking model")
    
    return opp, sam


# ============================================================================
# TEST 5: THE R GATE AS SPECTATOR
# ============================================================================

def test_R_spectator():
    """
    R (rotation) is the internal gate — the self-reference.
    It should NOT participate in inter-merkabit alignment.
    
    Test: compare alignment for R vs the 4 inter-channel gates.
    R alignment should be uncorrelated with docking quality.
    """
    print("\n" + "=" * 72)
    print("TEST 5: R GATE AS SPECTATOR")
    print("=" * 72)
    print(f"\n  R = internal rotation (self-reference)")
    print(f"  Should NOT correlate with inter-merkabit alignment")
    
    rng = np.random.default_rng(RANDOM_SEED + 4)
    n_pairs = 300
    n_cycles = 5
    n_steps = n_cycles * COXETER_H
    
    R_alignments = []
    inter_alignments = []
    G_values = []
    
    for pair in range(n_pairs):
        omega = rng.uniform(0.5, 2.0)
        A = make_random_state(rng, omega=omega)
        B = make_random_state(rng, omega=-omega)
        
        for step in range(n_steps):
            A, _ = ouroboros_step(A, step, chirality=+1)
            B, _ = ouroboros_step(B, step, chirality=-1)
        
        G = total_conductance(A, B)
        G_values.append(G)
        
        # R alignment (internal gate)
        A_R = gate_channel_action(A, 'R', STEP_PHASE, chirality=+1)
        B_R = gate_channel_action(B, 'R', STEP_PHASE, chirality=-1)
        R_align = abs(np.vdot(A_R.u, B_R.v)) ** 2
        R_alignments.append(R_align)
        
        # Mean inter-channel alignment
        align = per_gate_alignment(A, B)
        inter_alignments.append(np.mean(list(align.values())))
    
    R_arr = np.array(R_alignments)
    inter_arr = np.array(inter_alignments)
    G_arr = np.array(G_values)
    
    # Correlations
    corr_R_G = np.corrcoef(R_arr, G_arr)[0, 1]
    corr_inter_G = np.corrcoef(inter_arr, G_arr)[0, 1]
    
    print(f"\n  Correlation with total conductance G:")
    print(f"    R (internal):       r = {corr_R_G:+.4f}")
    print(f"    Inter-channel (TPFS): r = {corr_inter_G:+.4f}")
    
    if abs(corr_inter_G) > abs(corr_R_G) + 0.1:
        print(f"\n  → Inter-channel gates correlate MORE with docking than R ✓")
        print(f"    R is structurally a spectator, as predicted")
    elif abs(corr_R_G) > abs(corr_inter_G) + 0.1:
        print(f"\n  → R correlates MORE than inter-channel gates")
        print(f"    Challenges the R-as-spectator hypothesis")
    else:
        print(f"\n  → Similar correlation for both")
        print(f"    May need finer decomposition to distinguish roles")
    
    return corr_R_G, corr_inter_G


# ============================================================================
# TEST 6: DOES ALIGNMENT ROTATE WITH THE CYCLE?
# ============================================================================

def test_rotation_pattern():
    """
    If cubic docking is real, the pattern of WHICH gates are most
    aligned should rotate with the ouroboros cycle.
    
    At step k, the absent gate for A_forward is gates[k mod 5].
    The 4 active gates should show higher alignment than the absent one.
    Does the peak alignment rotate with the cycle?
    """
    print("\n" + "=" * 72)
    print("TEST 6: DOES ALIGNMENT PATTERN ROTATE WITH CYCLE?")
    print("=" * 72)
    
    rng = np.random.default_rng(RANDOM_SEED + 5)
    n_pairs = 200
    n_warmup = 3 * COXETER_H  # 3 cycles warmup
    
    # Track per-gate alignment at each step within one cycle
    step_gate_align = {step: {g: [] for g in GATES} 
                       for step in range(COXETER_H)}
    
    for pair in range(n_pairs):
        omega = rng.uniform(0.5, 2.0)
        A = make_random_state(rng, omega=omega)
        B = make_random_state(rng, omega=-omega)
        
        # Warmup
        for step in range(n_warmup):
            A, _ = ouroboros_step(A, step, chirality=+1)
            B, _ = ouroboros_step(B, step, chirality=-1)
        
        # One measurement cycle
        for step in range(COXETER_H):
            A, absent_A = ouroboros_step(A, n_warmup + step, chirality=+1)
            B, absent_B = ouroboros_step(B, n_warmup + step, chirality=-1)
            
            # Measure all 5 gates
            for g in GATES:
                A_g = gate_channel_action(A, g, STEP_PHASE, chirality=+1)
                B_g = gate_channel_action(B, g, STEP_PHASE, chirality=-1)
                alignment = abs(np.vdot(A_g.u, B_g.v)) ** 2
                step_gate_align[step][g].append(alignment)
    
    # Print the rotation pattern
    print(f"\n  Mean alignment per gate per step ({n_pairs} pairs):")
    print(f"  Absent gate for A_fwd rotates: {' → '.join(GATES)}")
    print(f"\n  {'step':>5s}  {'absent':>7s}  ", end='')
    for g in GATES:
        print(f"{'〈'+g+'〉':>7s}  ", end='')
    print("peak")
    print(f"  {'─'*5}  {'─'*7}  " + "  ".join(['─'*7]*5) + "  ────")
    
    for step in range(COXETER_H):
        absent = GATES[step % NUM_GATES]
        means = {}
        for g in GATES:
            means[g] = np.mean(step_gate_align[step][g])
        
        peak_gate = max(means, key=means.get)
        
        print(f"  {step:5d}  {absent:>7s}  ", end='')
        for g in GATES:
            marker = '▼' if g == absent else (' ' if g != peak_gate else '▲')
            print(f"{means[g]:6.4f}{marker} ", end='')
        print(f"  {peak_gate}")
    
    # Does the minimum follow the absent gate?
    min_follows_absent = 0
    for step in range(COXETER_H):
        absent = GATES[step % NUM_GATES]
        means = {g: np.mean(step_gate_align[step][g]) for g in GATES}
        min_gate = min(means, key=means.get)
        if min_gate == absent:
            min_follows_absent += 1
    
    print(f"\n  Minimum alignment follows absent gate: "
          f"{min_follows_absent}/{COXETER_H} steps")
    
    if min_follows_absent >= COXETER_H * 0.5:
        print(f"  → Alignment pattern ROTATES with absent gate ✓")
        print(f"    The absent gate IS the gap in the cubic docking")
    else:
        print(f"  → Alignment pattern does NOT track absent gate")
        print(f"    Rotation pattern not confirmed at this level")
    
    return min_follows_absent


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(results):
    print("\n" + "=" * 72)
    print("  CUBIC SELF-ASSEMBLY — SUMMARY")
    print("=" * 72)
    
    print(f"""
  THE QUESTION:
    Does the ouroboros drive produce cubic docking between
    two merkabits from random initial conditions?
    
  THE STRUCTURE:
    5 gates: R (internal rotation) + T,P,F,S (inter-channel)
    Forward cube = 4 inter-channel gates in forward chirality
    Inverse cube = 4 inter-channel gates in inverse chirality
    Cubic docking = forward cube of A aligns with inverse cube of B
    
  WHAT WE TESTED:
    1. Single random pair: step-by-step tracking
    2. Ensemble of 500 random pairs: statistics
    3. Pre-resonant pairs (ω_A + ω_B = 0): isolate alignment from frequency
    4. Chirality dependence: opposite vs same drive direction
    5. R as spectator: internal gate vs inter-channel gates
    6. Rotation pattern: does alignment track the absent gate?
""")

    print(f"  RESULTS:")
    for name, result in results.items():
        print(f"    {name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔" + "═" * 70 + "╗")
    print("║  CUBIC SELF-ASSEMBLY SIMULATION" + " " * 37 + "║")
    print("║  Does the ouroboros drive produce docking from random starts?" + " " * 8 + "║")
    print("║  4 forward gates of A ↔ 4 inverse gates of B" + " " * 23 + "║")
    print("╚" + "═" * 70 + "╝")
    
    start = time.time()
    
    results = {}
    
    # Test 1: Single pair tracking
    G_hist, cells_hist = test_random_pair_drive()
    results["Test 1 (single pair)"] = "see trajectory above"
    
    # Test 2: Ensemble statistics
    ens = test_ensemble()
    results["Test 2 (ensemble)"] = f"ΔG = {np.mean(ens['G_final']) - np.mean(ens['G_initial']):+.4f}"
    
    # Test 3: Pre-resonant pairs
    gi, gf = test_resonant_pairs()
    results["Test 3 (resonant)"] = f"ΔG = {np.mean(gf) - np.mean(gi):+.4f}"
    
    # Test 4: Chirality dependence
    opp, sam = test_chirality_dependence()
    results["Test 4 (chirality)"] = f"opp={np.mean(opp):.4f} vs same={np.mean(sam):.4f}"
    
    # Test 5: R as spectator
    corr_R, corr_inter = test_R_spectator()
    results["Test 5 (R spectator)"] = f"r_R={corr_R:+.4f} vs r_inter={corr_inter:+.4f}"
    
    # Test 6: Rotation pattern
    min_follows = test_rotation_pattern()
    results["Test 6 (rotation)"] = f"{min_follows}/12 steps track absent gate"
    
    print_summary(results)
    
    elapsed = time.time() - start
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"  Random seed: {RANDOM_SEED}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
