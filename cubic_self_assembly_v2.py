#!/usr/bin/env python3
"""
CUBIC SELF-ASSEMBLY WITH INTERACTION
======================================

The corrected test: does the ouroboros cycle, combined with the
tunnel conductance acting as a coupling force, produce cubic docking
from random initial conditions?

WHAT WAS MISSING (v1):
  The first simulation drove A and B independently and measured
  alignment passively. But if the tunnel IS the lattice, the
  interaction IS the drive. They cannot be separated.

  The conductance G_ij = |u_i† v_j|² is not just a measurement.
  It is a coupling that feeds back into the dynamics:
  - When u_A overlaps v_B, that overlap PULLS them into alignment
  - Greater alignment → greater conductance → stronger pull
  - This is a positive feedback loop: docking bootstraps itself
  - The ouroboros cycle provides the periodic drive
  - The conductance provides the directional force
  - Together: self-assembly

THE MODEL:
  At each time step, each merkabit experiences:
  
  1. OUROBOROS DRIVE (internal):
     The absent gate rotates with chirality-dependent direction.
     This creates the pentachoric cycle through all 5 gate positions.
     Each step: P(π/6) asymmetric + modulated Rx,Rz symmetric.
  
  2. TUNNEL COUPLING (interaction):
     For each lattice neighbour j of node i:
       G_ij = |u_i† v_j|²                    (tunnel conductance)
       R_ij = exp(−(ω_i + ω_j)²/σ²)         (resonance factor)
       
       Coupling force: rotate u_i toward v_j by angle:
         δθ = λ · G_ij · R_ij · Δt
       where λ is the coupling strength.
       
       This is the physical content: the overlap between the forward
       spinor of one merkabit and the inverse spinor of its neighbour
       creates a restoring force toward cubic alignment.
  
  3. FREQUENCY DRIFT (optional):
     ω_i drifts toward the resonance condition ω_i + ω_j = 0
     proportional to G_ij. This models the F gate as part of the
     docking event, not a separate operation.

  The test: from random initial spinors and frequencies on an
  Eisenstein lattice with bipartite chirality assignment, does the
  system evolve toward cubic docking (high G, 4/4 cells docked)?

LATTICE STRUCTURE (from lattice_scaling_simulation.py):
  7-node Eisenstein cell, radius 1
  Centre: 6 neighbours (interior)
  Periphery: 3 neighbours each (boundary)
  Sublattice colouring: (a+b) mod 3 → chirality {0, +1, −1}
  Bipartite frequency: sublattice 1 gets +ω, sublattice 2 gets −ω

Usage: python3 cubic_self_assembly_v2.py
Requirements: numpy
"""

import numpy as np
from collections import defaultdict
import time
import sys

# ============================================================================
# CONSTANTS
# ============================================================================

RANDOM_SEED = 42

COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H  # π/6

GATES = ['R', 'T', 'P', 'F', 'S']
INTER_CHANNEL_GATES = ['T', 'P', 'F', 'S']
NUM_GATES = 5

# Coupling parameters
COUPLING_LAMBDA = 0.15      # tunnel coupling strength
RESONANCE_WIDTH = 0.5       # σ for resonance Gaussian
FREQ_DRIFT_RATE = 0.02      # frequency drift toward resonance


# ============================================================================
# EISENSTEIN LATTICE (from lattice_scaling_simulation.py)
# ============================================================================

class EisensteinCell:
    UNIT_VECTORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]
    
    def __init__(self, radius):
        self.radius = radius
        self.r_sq = radius * radius
        self.nodes = []
        for a in range(-radius - 1, radius + 2):
            for b in range(-radius - 1, radius + 2):
                if a*a - a*b + b*b <= self.r_sq:
                    self.nodes.append((a, b))
        self.num_nodes = len(self.nodes)
        self.node_index = {n: i for i, n in enumerate(self.nodes)}
        
        node_set = set(self.nodes)
        self.edges = []
        self.neighbours = defaultdict(list)
        for i, (a1, b1) in enumerate(self.nodes):
            for da, db in self.UNIT_VECTORS:
                nb = (a1 + da, b1 + db)
                if nb in node_set:
                    j = self.node_index[nb]
                    if j > i:
                        self.edges.append((i, j))
                    self.neighbours[i].append(j)
        
        self.is_interior = []
        for i, (a, b) in enumerate(self.nodes):
            all_present = all((a+da, b+db) in node_set 
                            for da, db in self.UNIT_VECTORS)
            self.is_interior.append(all_present)
        
        self.sublattice = [(a + b) % 3 for (a, b) in self.nodes]
        self.chirality = []
        for s in self.sublattice:
            self.chirality.append(0 if s == 0 else (+1 if s == 1 else -1))
        
        self.coordination = [len(self.neighbours[i]) 
                            for i in range(self.num_nodes)]


# ============================================================================
# MERKABIT STATE
# ============================================================================

class MerkabitState:
    def __init__(self, u, v, omega=1.0):
        self.u = np.array(u, dtype=complex)
        self.v = np.array(v, dtype=complex)
        self.omega = omega
        nu, nv = np.linalg.norm(self.u), np.linalg.norm(self.v)
        if nu > 1e-15: self.u /= nu
        if nv > 1e-15: self.v /= nv
    
    @property
    def coherence(self):
        return np.real(np.vdot(self.u, self.v))
    
    @property
    def relative_phase(self):
        return np.angle(np.vdot(self.u, self.v))
    
    def copy(self):
        return MerkabitState(self.u.copy(), self.v.copy(), self.omega)


def make_random_state(rng, omega=None):
    u = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    v = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    if omega is None:
        omega = rng.uniform(-3, 3)
    return MerkabitState(u, v, omega)


# ============================================================================
# GATES
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


# ============================================================================
# OUROBOROS STEP (internal drive per merkabit)
# ============================================================================

def ouroboros_step(state, step_index, chirality, theta=STEP_PHASE):
    """
    Internal drive: absent gate rotation with chirality-dependent direction.
    From ouroboros_berry_phase_simulation.py.
    """
    k = step_index
    if chirality > 0:
        absent = k % NUM_GATES
    elif chirality < 0:
        absent = (-k) % NUM_GATES
    else:
        absent = k % NUM_GATES  # reference sublattice: forward default
    
    absent_label = GATES[absent]
    
    p_angle = theta
    sym_base = theta / 3
    omega_k = 2 * np.pi * k / COXETER_H
    
    c_sign = 1 if chirality >= 0 else -1
    rx_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k * c_sign))
    rz_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k * c_sign + 2*np.pi/3))
    
    if absent_label == 'S':
        rz_angle *= 0.4; rx_angle *= 1.3
    elif absent_label == 'R':
        rx_angle *= 0.4; rz_angle *= 1.3
    elif absent_label == 'T':
        rx_angle *= 0.7; rz_angle *= 0.7
    elif absent_label == 'P':
        p_angle *= 0.6; rx_angle *= 1.8; rz_angle *= 1.5
    
    s = gate_P(state, p_angle * c_sign)
    s = gate_Rz(s, rz_angle)
    s = gate_Rx(s, rx_angle)
    return s


# ============================================================================
# TUNNEL INTERACTION (the missing piece)
# ============================================================================

def tunnel_conductance(A, B):
    """G_AB = |u_A† v_B|² — forward-inverse overlap."""
    return abs(np.vdot(A.u, B.v)) ** 2

def resonance_factor(A, B, sigma=RESONANCE_WIDTH):
    """R_AB = exp(−(ω_A + ω_B)² / σ²) — frequency resonance."""
    omega_sum = A.omega + B.omega
    return np.exp(-omega_sum**2 / sigma**2)

def apply_tunnel_coupling(states, cell, lam=COUPLING_LAMBDA, 
                          freq_drift=FREQ_DRIFT_RATE):
    """
    The interaction Hamiltonian: for each edge (i,j), the tunnel
    conductance acts as a coupling force.
    
    Physical picture:
      When u_i overlaps v_j, the overlap creates a restoring force
      that rotates u_i further toward v_j (and v_j toward u_i).
      This is the cubic docking: forward cube meeting inverse cube.
    
    The coupling also drives frequency convergence:
      ω_i drifts toward the value that satisfies ω_i + ω_j = 0,
      proportional to the conductance. Stronger overlap → stronger
      frequency lock. This is F and C-SWAP as one event.
    """
    n = cell.num_nodes
    new_states = [s.copy() for s in states]
    
    for (i, j) in cell.edges:
        A = states[i]
        B = states[j]
        
        G = tunnel_conductance(A, B)
        R = resonance_factor(A, B)
        
        coupling = lam * G * R
        
        if coupling < 1e-12:
            continue
        
        # --- SPINOR COUPLING ---
        # Rotate u_i toward v_j: the forward spinor of i
        # is pulled toward the inverse spinor of j.
        # This IS the tunnel — not a force acting through a tunnel,
        # but the tunnel itself forming.
        
        # Direction: v_j projected perpendicular to u_i
        # (the component that u_i needs to rotate toward)
        u_i = new_states[i].u
        v_j = B.v
        
        # Project v_j onto plane perpendicular to u_i
        proj = v_j - np.vdot(u_i, v_j) * u_i
        proj_norm = np.linalg.norm(proj)
        
        if proj_norm > 1e-12:
            direction = proj / proj_norm
            # Rotate u_i toward v_j by coupling angle
            angle = coupling * np.pi / 4  # bounded rotation
            new_u_i = np.cos(angle) * u_i + np.sin(angle) * direction
            new_u_i /= np.linalg.norm(new_u_i)
            new_states[i] = MerkabitState(new_u_i, new_states[i].v,
                                           new_states[i].omega)
        
        # Same for u_j toward v_i
        u_j = new_states[j].u
        v_i = A.v
        
        proj = v_i - np.vdot(u_j, v_i) * u_j
        proj_norm = np.linalg.norm(proj)
        
        if proj_norm > 1e-12:
            direction = proj / proj_norm
            angle = coupling * np.pi / 4
            new_u_j = np.cos(angle) * u_j + np.sin(angle) * direction
            new_u_j /= np.linalg.norm(new_u_j)
            new_states[j] = MerkabitState(new_u_j, new_states[j].v,
                                           new_states[j].omega)
        
        # --- FREQUENCY COUPLING ---
        # ω_i drifts toward −ω_j (resonance condition)
        # proportional to conductance
        omega_sum = A.omega + B.omega
        drift = freq_drift * G * R
        new_states[i] = MerkabitState(new_states[i].u, new_states[i].v,
                                       new_states[i].omega - drift * omega_sum / 2)
        new_states[j] = MerkabitState(new_states[j].u, new_states[j].v,
                                       new_states[j].omega - drift * omega_sum / 2)
    
    return new_states


# ============================================================================
# METRICS
# ============================================================================

def lattice_metrics(states, cell):
    """Compute alignment metrics across the whole lattice."""
    G_edges = []
    resonant_edges = 0
    freq_detunings = []
    
    for (i, j) in cell.edges:
        G = tunnel_conductance(states[i], states[j])
        G_edges.append(G)
        detuning = abs(states[i].omega + states[j].omega)
        freq_detunings.append(detuning)
        if detuning < RESONANCE_WIDTH:
            resonant_edges += 1
    
    return {
        'mean_G': np.mean(G_edges),
        'max_G': np.max(G_edges),
        'min_G': np.min(G_edges),
        'std_G': np.std(G_edges),
        'resonant_edges': resonant_edges,
        'total_edges': len(cell.edges),
        'mean_detuning': np.mean(freq_detunings),
    }

def per_gate_alignment(A, B, theta=STEP_PHASE):
    """Measure alignment for each of the 4 inter-channel gates."""
    alignments = {}
    for g in INTER_CHANNEL_GATES:
        if g == 'T':
            A_fwd = gate_Rx(A, theta)   # T ≈ Ry, approx as Rx
            B_inv = gate_Rx(B, -theta)
        elif g == 'P':
            A_fwd = gate_P(A, theta)
            B_inv = gate_P(B, -theta)
        elif g == 'F':
            # F shifts frequency — alignment measured by resonance
            alignments[g] = np.exp(-(A.omega + B.omega)**2 / RESONANCE_WIDTH**2)
            continue
        elif g == 'S':
            A_fwd = gate_Rz(A, theta)   # S ≈ controlled-P, approx as Rz
            B_inv = gate_Rz(B, -theta)
        
        if g != 'F':
            alignments[g] = abs(np.vdot(A_fwd.u, B_inv.v)) ** 2
    
    return alignments


# ============================================================================
# TEST 1: LATTICE SELF-ASSEMBLY WITH INTERACTION
# ============================================================================

def test_lattice_assembly():
    """
    The corrected test: 7-node Eisenstein cell with ouroboros drive
    AND tunnel coupling. Random initial states.
    """
    print("\n" + "=" * 72)
    print("TEST 1: LATTICE SELF-ASSEMBLY WITH INTERACTION")
    print("=" * 72)
    print(f"\n  7-node Eisenstein cell, ouroboros drive + tunnel coupling")
    print(f"  λ = {COUPLING_LAMBDA}, freq_drift = {FREQ_DRIFT_RATE}")
    
    rng = np.random.default_rng(RANDOM_SEED)
    cell = EisensteinCell(1)
    
    print(f"  Lattice: {cell.num_nodes} nodes, {len(cell.edges)} edges")
    print(f"  Chiralities: {[cell.chirality[i] for i in range(cell.num_nodes)]}")
    
    # Random initial states with random frequencies
    states = []
    for i in range(cell.num_nodes):
        omega = rng.uniform(-2, 2)
        states.append(make_random_state(rng, omega=omega))
    
    m0 = lattice_metrics(states, cell)
    print(f"\n  Initial: ⟨G⟩={m0['mean_G']:.4f}, resonant={m0['resonant_edges']}/{m0['total_edges']}, "
          f"⟨|Δω|⟩={m0['mean_detuning']:.4f}")
    
    # Drive + interact
    n_cycles = 20
    n_steps = n_cycles * COXETER_H
    
    print(f"\n  {'step':>5s}  {'cycle':>5s}  {'⟨G⟩':>8s}  {'max_G':>8s}  "
          f"{'res_edges':>9s}  {'⟨|Δω|⟩':>8s}")
    print(f"  {'─'*5}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}")
    
    G_trajectory = []
    
    for step in range(n_steps):
        # 1. Ouroboros drive (internal, per merkabit)
        for i in range(cell.num_nodes):
            states[i] = ouroboros_step(states[i], step, cell.chirality[i])
        
        # 2. Tunnel coupling (interaction between neighbours)
        states = apply_tunnel_coupling(states, cell)
        
        # Track
        m = lattice_metrics(states, cell)
        G_trajectory.append(m['mean_G'])
        
        if step % (COXETER_H * 2) == 0 or step == n_steps - 1:
            cycle = step // COXETER_H
            print(f"  {step:5d}  {cycle:5d}  {m['mean_G']:8.4f}  {m['max_G']:8.4f}  "
                  f"{m['resonant_edges']:5d}/{m['total_edges']:3d}  "
                  f"{m['mean_detuning']:8.4f}")
    
    mf = lattice_metrics(states, cell)
    
    print(f"\n  RESULT:")
    print(f"    Initial ⟨G⟩: {m0['mean_G']:.4f}")
    print(f"    Final ⟨G⟩:   {mf['mean_G']:.4f}")
    print(f"    Δ⟨G⟩:        {mf['mean_G'] - m0['mean_G']:+.4f}")
    
    # Trend analysis
    first_quarter = np.mean(G_trajectory[:n_steps//4])
    last_quarter = np.mean(G_trajectory[-n_steps//4:])
    
    if last_quarter > first_quarter + 0.02:
        print(f"    Trend: INCREASING → self-assembly ✓")
    elif last_quarter < first_quarter - 0.02:
        print(f"    Trend: DECREASING")
    else:
        print(f"    Trend: FLAT")
    
    return G_trajectory, states


# ============================================================================
# TEST 2: WITH vs WITHOUT INTERACTION
# ============================================================================

def test_interaction_comparison():
    """
    Compare three conditions:
      A) Ouroboros drive only (no interaction) — the v1 simulation
      B) Interaction only (no drive)
      C) Drive + interaction together
    
    If self-assembly requires both, C should outperform A and B.
    """
    print("\n" + "=" * 72)
    print("TEST 2: DRIVE ONLY vs INTERACTION ONLY vs BOTH")
    print("=" * 72)
    
    rng = np.random.default_rng(RANDOM_SEED + 10)
    cell = EisensteinCell(1)
    n_cycles = 15
    n_steps = n_cycles * COXETER_H
    n_trials = 50
    
    print(f"\n  {n_trials} random lattice configurations, {n_cycles} cycles each")
    
    results = {'drive_only': [], 'interact_only': [], 'both': []}
    
    for trial in range(n_trials):
        # Same initial states for all three conditions
        init_states = []
        for i in range(cell.num_nodes):
            omega = rng.uniform(-2, 2)
            init_states.append(make_random_state(rng, omega=omega))
        
        G0 = lattice_metrics(init_states, cell)['mean_G']
        
        # A) Drive only
        states_A = [s.copy() for s in init_states]
        for step in range(n_steps):
            for i in range(cell.num_nodes):
                states_A[i] = ouroboros_step(states_A[i], step, cell.chirality[i])
        GA = lattice_metrics(states_A, cell)['mean_G']
        results['drive_only'].append(GA - G0)
        
        # B) Interaction only
        states_B = [s.copy() for s in init_states]
        for step in range(n_steps):
            states_B = apply_tunnel_coupling(states_B, cell)
        GB = lattice_metrics(states_B, cell)['mean_G']
        results['interact_only'].append(GB - G0)
        
        # C) Both
        states_C = [s.copy() for s in init_states]
        for step in range(n_steps):
            for i in range(cell.num_nodes):
                states_C[i] = ouroboros_step(states_C[i], step, cell.chirality[i])
            states_C = apply_tunnel_coupling(states_C, cell)
        GC = lattice_metrics(states_C, cell)['mean_G']
        results['both'].append(GC - G0)
    
    for key, label in [('drive_only', 'Drive only'),
                       ('interact_only', 'Interaction only'),
                       ('both', 'Drive + Interaction')]:
        arr = np.array(results[key])
        print(f"\n  {label:>20s}: Δ⟨G⟩ = {np.mean(arr):+.4f} ± {np.std(arr):.4f}")
    
    # Verdict
    d = np.mean(results['drive_only'])
    i = np.mean(results['interact_only'])
    b = np.mean(results['both'])
    
    print(f"\n  VERDICT:")
    if b > d + 0.01 and b > i + 0.01:
        print(f"    Drive + interaction > either alone")
        print(f"    → Self-assembly requires BOTH drive and coupling ✓")
    elif b > max(d, i) + 0.01:
        print(f"    Combined exceeds either component")
    elif i > d + 0.01 and i > b - 0.01:
        print(f"    Interaction dominates; drive adds little")
    elif d > i + 0.01:
        print(f"    Drive dominates; interaction adds little")
    else:
        print(f"    No clear separation between conditions")
    
    return results


# ============================================================================
# TEST 3: BIPARTITE vs RANDOM FREQUENCY ASSIGNMENT
# ============================================================================

def test_bipartite_frequency():
    """
    Does bipartite frequency assignment (the lattice's natural structure)
    produce better self-assembly than random frequencies?
    
    Bipartite: sublattice 1 gets +ω, sublattice 2 gets −ω
    Random: each node gets a random ω
    """
    print("\n" + "=" * 72)
    print("TEST 3: BIPARTITE vs RANDOM FREQUENCY ASSIGNMENT")
    print("=" * 72)
    
    rng = np.random.default_rng(RANDOM_SEED + 20)
    cell = EisensteinCell(1)
    n_cycles = 15
    n_steps = n_cycles * COXETER_H
    n_trials = 80
    
    bipartite_dG = []
    random_dG = []
    
    for trial in range(n_trials):
        base_omega = rng.uniform(0.5, 2.0)
        
        # Bipartite: +ω for sublattice 1, −ω for sublattice 2
        states_bp = []
        for i in range(cell.num_nodes):
            chi = cell.chirality[i]
            omega = base_omega * chi if chi != 0 else base_omega * 0.1
            u = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            v = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            states_bp.append(MerkabitState(u, v, omega))
        
        G0_bp = lattice_metrics(states_bp, cell)['mean_G']
        
        for step in range(n_steps):
            for i in range(cell.num_nodes):
                states_bp[i] = ouroboros_step(states_bp[i], step, cell.chirality[i])
            states_bp = apply_tunnel_coupling(states_bp, cell)
        
        Gf_bp = lattice_metrics(states_bp, cell)['mean_G']
        bipartite_dG.append(Gf_bp - G0_bp)
        
        # Random frequencies
        states_rn = []
        for i in range(cell.num_nodes):
            omega = rng.uniform(-2, 2)
            u = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            v = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            states_rn.append(MerkabitState(u, v, omega))
        
        G0_rn = lattice_metrics(states_rn, cell)['mean_G']
        
        for step in range(n_steps):
            for i in range(cell.num_nodes):
                states_rn[i] = ouroboros_step(states_rn[i], step, cell.chirality[i])
            states_rn = apply_tunnel_coupling(states_rn, cell)
        
        Gf_rn = lattice_metrics(states_rn, cell)['mean_G']
        random_dG.append(Gf_rn - G0_rn)
    
    bp = np.array(bipartite_dG)
    rn = np.array(random_dG)
    
    print(f"\n  {n_trials} trials, {n_cycles} cycles, drive + interaction")
    print(f"\n  Bipartite (ω follows chirality): Δ⟨G⟩ = {np.mean(bp):+.4f} ± {np.std(bp):.4f}")
    print(f"  Random frequencies:               Δ⟨G⟩ = {np.mean(rn):+.4f} ± {np.std(rn):.4f}")
    
    if np.mean(bp) > np.mean(rn) + 0.02:
        print(f"\n  → Bipartite structure aids self-assembly ✓")
        print(f"    The lattice's natural frequency assignment works")
    elif abs(np.mean(bp) - np.mean(rn)) < 0.02:
        print(f"\n  → No significant difference")
        print(f"    Frequency convergence happens regardless of starting point")
    else:
        print(f"\n  → Random frequencies assemble better")
    
    return bp, rn


# ============================================================================
# TEST 4: COUPLING STRENGTH SWEEP
# ============================================================================

def test_coupling_sweep():
    """
    How does the coupling strength λ affect self-assembly?
    
    At λ=0: no interaction → no self-assembly (confirmed by v1)
    At small λ: slow convergence
    At optimal λ: fast convergence
    At large λ: possible instability (overshoot)
    """
    print("\n" + "=" * 72)
    print("TEST 4: COUPLING STRENGTH SWEEP")
    print("=" * 72)
    
    rng = np.random.default_rng(RANDOM_SEED + 30)
    cell = EisensteinCell(1)
    n_cycles = 10
    n_steps = n_cycles * COXETER_H
    n_trials = 40
    
    lambdas = [0.0, 0.01, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 1.00]
    
    print(f"\n  {n_trials} trials × {len(lambdas)} coupling strengths")
    print(f"\n  {'λ':>6s}  {'Δ⟨G⟩':>8s}  {'std':>6s}  bar")
    print(f"  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*40}")
    
    for lam in lambdas:
        dGs = []
        for trial in range(n_trials):
            states = []
            for i in range(cell.num_nodes):
                chi = cell.chirality[i]
                omega = 1.0 * (chi if chi != 0 else 0.1)
                states.append(make_random_state(rng, omega=omega))
            
            G0 = lattice_metrics(states, cell)['mean_G']
            
            for step in range(n_steps):
                for i in range(cell.num_nodes):
                    states[i] = ouroboros_step(states[i], step, cell.chirality[i])
                states = apply_tunnel_coupling(states, cell, lam=lam)
            
            Gf = lattice_metrics(states, cell)['mean_G']
            dGs.append(Gf - G0)
        
        m = np.mean(dGs)
        s = np.std(dGs)
        bar_val = max(0, min(1, (m + 0.3) / 0.6))
        bar = '█' * int(40 * bar_val) + '░' * (40 - int(40 * bar_val))
        print(f"  {lam:6.2f}  {m:+8.4f}  {s:6.4f}  {bar}")
    
    return True


# ============================================================================
# TEST 5: FREQUENCY CONVERGENCE TRACKING
# ============================================================================

def test_frequency_convergence():
    """
    Track frequency detuning over time.
    
    If F and C-SWAP are one event, the frequency convergence
    should be driven by the conductance — stronger overlap
    drives faster frequency lock.
    """
    print("\n" + "=" * 72)
    print("TEST 5: FREQUENCY CONVERGENCE")
    print("=" * 72)
    
    rng = np.random.default_rng(RANDOM_SEED + 40)
    cell = EisensteinCell(1)
    n_cycles = 30
    n_steps = n_cycles * COXETER_H
    
    # Start with random frequencies (far from resonance)
    states = []
    for i in range(cell.num_nodes):
        omega = rng.uniform(-3, 3)
        states.append(make_random_state(rng, omega=omega))
    
    print(f"\n  Initial frequencies:")
    for i in range(cell.num_nodes):
        chi_str = {0: 'ref', 1: '+', -1: '-'}[cell.chirality[i]]
        print(f"    Node {i} (χ={chi_str}): ω = {states[i].omega:+.4f}")
    
    m0 = lattice_metrics(states, cell)
    print(f"\n  Initial ⟨|Δω|⟩ = {m0['mean_detuning']:.4f}")
    print(f"  Initial resonant edges: {m0['resonant_edges']}/{m0['total_edges']}")
    
    print(f"\n  {'cycle':>5s}  {'⟨G⟩':>8s}  {'⟨|Δω|⟩':>8s}  {'res_edges':>9s}  freqs")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*40}")
    
    for step in range(n_steps):
        for i in range(cell.num_nodes):
            states[i] = ouroboros_step(states[i], step, cell.chirality[i])
        states = apply_tunnel_coupling(states, cell)
        
        if step % (COXETER_H * 3) == 0 or step == n_steps - 1:
            m = lattice_metrics(states, cell)
            cycle = step // COXETER_H
            freqs = " ".join(f"{states[i].omega:+.2f}" for i in range(min(7, cell.num_nodes)))
            print(f"  {cycle:5d}  {m['mean_G']:8.4f}  {m['mean_detuning']:8.4f}  "
                  f"{m['resonant_edges']:5d}/{m['total_edges']:3d}  {freqs}")
    
    mf = lattice_metrics(states, cell)
    print(f"\n  Final ⟨|Δω|⟩ = {mf['mean_detuning']:.4f}")
    print(f"  Final resonant edges: {mf['resonant_edges']}/{mf['total_edges']}")
    
    if mf['mean_detuning'] < m0['mean_detuning'] * 0.5:
        print(f"\n  → Frequencies CONVERGING toward resonance ✓")
        print(f"    F gate dynamics emergent from coupling")
    else:
        print(f"\n  → Frequencies not converging significantly")
    
    return True


# ============================================================================
# TEST 6: 19-NODE LATTICE (larger cell)
# ============================================================================

def test_larger_lattice():
    """
    Does self-assembly work on the 19-node cell?
    
    The 19-node cell has 7 interior nodes (full 6-coordination)
    and 12 boundary nodes. If the mechanism works, it should
    scale — and interior nodes should dock better than boundary.
    """
    print("\n" + "=" * 72)
    print("TEST 6: 19-NODE LATTICE")
    print("=" * 72)
    
    rng = np.random.default_rng(RANDOM_SEED + 50)
    cell = EisensteinCell(2)  # 19 nodes
    n_cycles = 15
    n_steps = n_cycles * COXETER_H
    n_trials = 20
    
    print(f"\n  {cell.num_nodes} nodes, {len(cell.edges)} edges")
    print(f"  Interior: {sum(cell.is_interior)}, Boundary: {sum(not x for x in cell.is_interior)}")
    
    dGs = []
    interior_dGs = []
    boundary_dGs = []
    
    for trial in range(n_trials):
        states = []
        for i in range(cell.num_nodes):
            chi = cell.chirality[i]
            omega = 1.0 * (chi if chi != 0 else 0.1)
            states.append(make_random_state(rng, omega=omega))
        
        # Measure initial per-edge G, split by interior/boundary
        G0_int, G0_bnd = [], []
        for (i, j) in cell.edges:
            G = tunnel_conductance(states[i], states[j])
            if cell.is_interior[i] and cell.is_interior[j]:
                G0_int.append(G)
            else:
                G0_bnd.append(G)
        
        G0 = lattice_metrics(states, cell)['mean_G']
        
        for step in range(n_steps):
            for i in range(cell.num_nodes):
                states[i] = ouroboros_step(states[i], step, cell.chirality[i])
            states = apply_tunnel_coupling(states, cell)
        
        Gf = lattice_metrics(states, cell)['mean_G']
        dGs.append(Gf - G0)
        
        Gf_int, Gf_bnd = [], []
        for (i, j) in cell.edges:
            G = tunnel_conductance(states[i], states[j])
            if cell.is_interior[i] and cell.is_interior[j]:
                Gf_int.append(G)
            else:
                Gf_bnd.append(G)
        
        if G0_int:
            interior_dGs.append(np.mean(Gf_int) - np.mean(G0_int))
        if G0_bnd:
            boundary_dGs.append(np.mean(Gf_bnd) - np.mean(G0_bnd))
    
    print(f"\n  Overall Δ⟨G⟩:  {np.mean(dGs):+.4f} ± {np.std(dGs):.4f}")
    if interior_dGs:
        print(f"  Interior Δ⟨G⟩: {np.mean(interior_dGs):+.4f} ± {np.std(interior_dGs):.4f}")
    if boundary_dGs:
        print(f"  Boundary Δ⟨G⟩: {np.mean(boundary_dGs):+.4f} ± {np.std(boundary_dGs):.4f}")
    
    if interior_dGs and boundary_dGs:
        if np.mean(interior_dGs) > np.mean(boundary_dGs) + 0.01:
            print(f"\n  → Interior docks BETTER than boundary ✓")
            print(f"    Full coordination (6 neighbours) aids assembly")
        else:
            print(f"\n  → No interior/boundary difference")
    
    return dGs


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary():
    print("\n" + "=" * 72)
    print("  CUBIC SELF-ASSEMBLY v2 — SUMMARY")
    print("=" * 72)
    print(f"""
  WHAT CHANGED FROM v1:
    v1 drove merkabits independently → no self-assembly (correct null)
    v2 adds tunnel coupling: G_ij feeds back as alignment force
    The tunnel IS the interaction IS the lattice
    
  THE PHYSICAL PICTURE:
    1. Ouroboros cycle provides periodic drive (absent gate rotation)
    2. Tunnel conductance G = |u_A† v_B|² acts as coupling force
    3. Coupling pulls forward spinor toward inverse spinor (docking)
    4. Greater alignment → greater coupling → positive feedback
    5. Frequency drifts toward resonance ω_A + ω_B = 0
    6. F gate and C-SWAP are one event: frequency lock + channel transfer
    
  KEY TESTS:
    1. Full lattice assembly from random starts
    2. Drive alone vs interaction alone vs both
    3. Bipartite vs random frequency assignment
    4. Coupling strength sweep (λ dependence)
    5. Frequency convergence (F gate emergent from coupling)
    6. Larger lattice (19-node, interior vs boundary)
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔" + "═" * 70 + "╗")
    print("║  CUBIC SELF-ASSEMBLY v2 — WITH INTERACTION" + " " * 25 + "║")
    print("║  The tunnel IS the interaction IS the lattice" + " " * 23 + "║")
    print("║  F and C-SWAP as one event" + " " * 42 + "║")
    print("╚" + "═" * 70 + "╝")
    
    start = time.time()
    
    test_lattice_assembly()
    test_interaction_comparison()
    test_bipartite_frequency()
    test_coupling_sweep()
    test_frequency_convergence()
    test_larger_lattice()
    
    print_summary()
    
    elapsed = time.time() - start
    print(f"  Total runtime: {elapsed:.1f}s")
    print(f"  Random seed: {RANDOM_SEED}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
