#!/usr/bin/env python3
"""
NATIVE ALGORITHM BENCHMARKS — MERKABIT vs QUBIT GATE COSTS
============================================================

Addresses the central open question (Section 8.9.5):
  "Identifying specific computational problems where [standing-wave
   zero state, frequency-controlled connectivity, bidirectional time
   evolution] provide advantage is the most important open question
   for practical merkabit computing."

This simulation benchmarks five problem classes where the merkabit's
unique resources — P gate (ternary navigation), F gate (dynamic
connectivity), zero-state C-SWAP, and bidirectional evolution —
provide measurable advantages over qubit implementations.

For each problem we count:
  - Register width (number of computational units)
  - Gate count (single-unit + two-unit gates)
  - Circuit depth (critical path length)
  - Accumulated error (gates × per-gate error rate)
  - SWAP overhead (for fixed-topology qubit lattices)

═══════════════════════════════════════════════════════════════════
BENCHMARK 1: TERNARY SEARCH (Grover analogue)
═══════════════════════════════════════════════════════════════════
  Database of N items indexed by balanced ternary strings.
  Qubit:    ⌈log₂(N)⌉ qubits, O(√N) Grover iterations
  Merkabit: ⌈log₃(N)⌉ merkabits, O(√N) iterations
  
  Advantage: 37% fewer computational units (log₃/log₂ ratio),
  plus information-theoretic gain from ternary oracle (3-way
  branching per query vs 2-way).

═══════════════════════════════════════════════════════════════════
BENCHMARK 2: LONG-RANGE COUPLING (SWAP chain elimination)
═══════════════════════════════════════════════════════════════════
  Circuit requires interaction between units at distance d.
  Qubit:    d SWAP gates × 3 CNOTs each → O(d) depth, O(d) error
  Merkabit: F gate brings pair into resonance → O(1) depth
  
  Advantage: O(d) → O(1) circuit depth. On an n-node lattice with
  diameter D, algorithms requiring all-to-all coupling save O(D)
  per interaction. This is the F gate's killer application.

═══════════════════════════════════════════════════════════════════
BENCHMARK 3: CONSTRAINT SATISFACTION (Graph 3-colouring)
═══════════════════════════════════════════════════════════════════
  Assign 3 colours to vertices such that no edge shares a colour.
  Qubit:    2 qubits per vertex (⌈log₂(3)⌉), custom CNOT checks
  Merkabit: 1 merkabit per vertex (native ternary), P-gate cycling
  
  Advantage: 2× register reduction, plus the P gate natively
  implements the colour-cycling oracle (P(2π/3) cycles trits).

═══════════════════════════════════════════════════════════════════
BENCHMARK 4: REVERSIBLE COMPUTATION (Uncomputation savings)
═══════════════════════════════════════════════════════════════════
  Many quantum algorithms require uncomputing ancilla registers.
  Qubit:    replay circuit backwards → 2× gate count
  Merkabit: inverse spinor always present; Berry phase readout
  
  Advantage: up to 2× gate count reduction for algorithms with
  large uncomputation phases. The bidirectional time evolution
  means the reverse path is a native resource, not a cost.

═══════════════════════════════════════════════════════════════════
BENCHMARK 5: ZERO-STATE CONTROLLED OPERATIONS
═══════════════════════════════════════════════════════════════════
  C-SWAP activates on the zero state — the MOST coherent config.
  Qubit CNOT activates on |1⟩ — no special stability.
  
  Advantage: control operations triggered from the error-corrected
  standing-wave state have intrinsically lower control error rates.
  Quantified via effective error model.

Usage:
  python3 native_algorithm_benchmarks.py

Requirements: numpy
"""

import numpy as np
import math
import time
import sys
from collections import namedtuple

# ============================================================================
# COST MODEL
# ============================================================================

# Gate costs (normalised to single-qubit gate = 1)
# These reflect typical superconducting hardware ratios
QUBIT_COSTS = {
    'single_gate': 1,          # Rₓ, Rz, H, T, etc.
    'CNOT': 10,                # Two-qubit gate (~10× single-gate error)
    'SWAP': 30,                # = 3 CNOTs
    'measurement': 5,          # Destructive readout
    'error_rate_1q': 1e-4,     # Per single-qubit gate
    'error_rate_2q': 1e-3,     # Per CNOT
    'error_rate_meas': 1e-2,   # Per measurement
}

MERKABIT_COSTS = {
    'Rx': 1,                   # Qubit-compatible rotation
    'Rz': 1,                   # Qubit-compatible rotation
    'P': 1,                    # Phase gate (inter-spinor, no qubit analogue)
    'F': 1,                    # Frequency gate (no qubit analogue)
    'C_SWAP': 10,              # Controlled channel swap
    'berry_readout': 1,        # Non-destructive Berry phase readout
    'error_rate_1m': 1e-4,     # Per single-merkabit gate
    'error_rate_2m': 1e-3,     # Per C-SWAP
    'error_rate_readout': 1e-3,# Berry phase (non-destructive, lower error)
    'pi_lock_suppression': 0.5,# Level 1 error suppression factor
}

# Circuit cost record
CircuitCost = namedtuple('CircuitCost', [
    'name', 'register_width', 'gate_count', 'two_unit_gates',
    'circuit_depth', 'swap_overhead', 'total_error', 'description'
])


def format_cost(cost):
    """Format a CircuitCost for display."""
    return (f"    Width: {cost.register_width:>4d} units  |  "
            f"Gates: {cost.gate_count:>6d}  |  "
            f"2-unit: {cost.two_unit_gates:>5d}  |  "
            f"Depth: {cost.circuit_depth:>5d}  |  "
            f"SWAPs: {cost.swap_overhead:>5d}  |  "
            f"Error: {cost.total_error:.2e}")


# ============================================================================
# BENCHMARK 1: TERNARY SEARCH
# ============================================================================

def benchmark_ternary_search():
    """
    Grover search over a database of N items.
    
    Qubit:
      - Register: ⌈log₂(N)⌉ qubits to index N items
      - Iterations: ⌊π/4 · √N⌋ Grover iterations
      - Per iteration: oracle (O(n) gates) + diffusion (O(n) gates + O(n) CNOTs)
      - Total CNOTs per iteration: ~2n (oracle + diffusion)
    
    Merkabit:
      - Register: ⌈log₃(N)⌉ merkabits to index N items
      - Iterations: same O(√N) but with ternary oracle
      - Per iteration: oracle uses P gate for 3-way branching
      - Ternary diffusion about the mean (3 states per unit)
      
    The information-theoretic advantage: each merkabit addresses 3
    values vs qubit's 2, so fewer units are needed. The P gate
    natively implements the ternary phase oracle.
    """
    print("=" * 72)
    print("BENCHMARK 1: TERNARY SEARCH (Grover analogue)")
    print("=" * 72)
    
    results = []
    
    database_sizes = [9, 27, 81, 243, 729, 2187, 6561, 
                      19683, 59049, 177147, 531441]
    
    print(f"\n  {'N':>8s}  {'n_qub':>6s}  {'n_merk':>6s}  "
          f"{'save%':>6s}  {'q_gates':>8s}  {'m_gates':>8s}  "
          f"{'gate_save%':>10s}  {'q_err':>10s}  {'m_err':>10s}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}  "
          f"{'─'*10}  {'─'*10}  {'─'*10}")
    
    for N in database_sizes:
        n_qubits = math.ceil(math.log2(N))
        n_merkabits = math.ceil(math.log(N, 3))
        
        register_savings = 1 - n_merkabits / n_qubits
        
        # Grover iterations
        iterations = max(1, int(np.pi / 4 * np.sqrt(N)))
        
        # Qubit costs per iteration:
        #   Oracle: n_qubits single gates + n_qubits CNOTs (multi-controlled phase)
        #   Diffusion: n_qubits single gates + n_qubits CNOTs
        q_single_per_iter = 4 * n_qubits
        q_cnot_per_iter = 2 * n_qubits
        q_total_gates = iterations * (q_single_per_iter + q_cnot_per_iter)
        q_total_error = iterations * (q_single_per_iter * QUBIT_COSTS['error_rate_1q'] +
                                       q_cnot_per_iter * QUBIT_COSTS['error_rate_2q'])
        
        # Merkabit costs per iteration:
        #   Oracle: n_merkabits P gates (ternary phase marking, 3-way)
        #     + n_merkabits Rz gates (rotation component)
        #   Diffusion: n_merkabits P gates + n_merkabits single gates
        #   C-SWAP for multi-merkabit oracle: ~n_merkabits C-SWAPs
        m_single_per_iter = 4 * n_merkabits   # P + Rz + P + Rx
        m_cswap_per_iter = n_merkabits         # Multi-merkabit oracle
        m_total_gates = iterations * (m_single_per_iter + m_cswap_per_iter)
        
        # Merkabit error with π-lock suppression
        m_total_error = iterations * (
            m_single_per_iter * MERKABIT_COSTS['error_rate_1m'] * 
            MERKABIT_COSTS['pi_lock_suppression'] +
            m_cswap_per_iter * MERKABIT_COSTS['error_rate_2m'] *
            MERKABIT_COSTS['pi_lock_suppression']
        )
        
        gate_savings = 1 - m_total_gates / q_total_gates
        
        print(f"  {N:>8d}  {n_qubits:>6d}  {n_merkabits:>6d}  "
              f"{register_savings:>5.1%}  {q_total_gates:>8d}  {m_total_gates:>8d}  "
              f"{gate_savings:>9.1%}  {q_total_error:>10.2e}  {m_total_error:>10.2e}")
        
        results.append((N, n_qubits, n_merkabits, q_total_gates, 
                        m_total_gates, q_total_error, m_total_error))
    
    # Summary
    avg_register = np.mean([1 - r[2]/r[1] for r in results])
    avg_gate = np.mean([1 - r[4]/r[3] for r in results])
    avg_error = np.mean([r[6]/r[5] for r in results])
    
    print(f"\n  Average register savings:  {avg_register:.1%}")
    print(f"  Average gate count savings: {avg_gate:.1%}")
    print(f"  Average error ratio (m/q):  {avg_error:.3f}")
    print(f"  Sources: log₃/log₂ = {1/math.log2(3):.4f} register ratio, "
          f"P-gate ternary oracle, π-lock suppression")
    
    return results


# ============================================================================
# BENCHMARK 2: LONG-RANGE COUPLING (SWAP CHAIN ELIMINATION)
# ============================================================================

def benchmark_long_range_coupling():
    """
    Circuit requiring interactions between units at varying distances
    on a lattice with limited connectivity.
    
    Qubit (heavy-hex / linear chain):
      - Interaction at distance d requires d SWAP gates
      - Each SWAP = 3 CNOTs
      - Depth: O(d) per long-range interaction
      - Error: accumulates as 3d × ε_CNOT
    
    Merkabit (Eisenstein lattice + F gate):
      - F gate shifts frequency to bring any pair into resonance
      - Tunnel condition: ωₐ + ωᵦ = 0
      - Depth: O(1) per interaction (F + C-SWAP + F⁻¹)
      - Error: constant per interaction (3 gates)
    
    This is the most dramatic advantage: O(d) → O(1).
    For circuits with many long-range interactions (QFT, chemistry
    simulations, optimization), this dominates all other savings.
    """
    print("\n" + "=" * 72)
    print("BENCHMARK 2: LONG-RANGE COUPLING (SWAP chain elimination)")
    print("=" * 72)
    
    # Scenario: n-unit register, circuit requires k interactions at
    # various distances sampled from the register
    
    lattice_sizes = [7, 19, 37, 61, 91, 127, 169]
    n_interactions = 100  # interactions per circuit
    
    print(f"\n  Lattice with {n_interactions} random long-range interactions:\n")
    print(f"  {'n_units':>8s}  {'diam':>5s}  "
          f"{'q_swaps':>8s}  {'q_depth':>8s}  {'q_error':>10s}  "
          f"{'m_depth':>8s}  {'m_error':>10s}  "
          f"{'depth_ratio':>11s}  {'error_ratio':>11s}")
    print(f"  {'─'*8}  {'─'*5}  "
          f"{'─'*8}  {'─'*8}  {'─'*10}  "
          f"{'─'*8}  {'─'*10}  "
          f"{'─'*11}  {'─'*11}")
    
    results = []
    np.random.seed(42)
    
    for n in lattice_sizes:
        # Approximate diameter of hexagonal lattice with n nodes
        # Eisenstein lattice: diameter ≈ 2 × radius ≈ 2√(n/π)
        diameter = max(2, int(2 * np.sqrt(n / np.pi)))
        
        # Generate random interaction pairs and their distances
        # Distance on hex lattice approximated by Manhattan-like metric
        total_swaps = 0
        total_q_depth = 0
        total_m_depth = 0
        
        for _ in range(n_interactions):
            # Random distance between 1 and diameter
            d = np.random.randint(1, diameter + 1)
            
            # Qubit cost: d SWAPs, each = 3 CNOTs
            swaps_needed = max(0, d - 1)  # Adjacent = 0 SWAPs
            q_depth_this = 3 * swaps_needed + 1  # SWAPs + CNOT at destination
            
            # Merkabit cost: F + C-SWAP + F⁻¹ (constant)
            m_depth_this = 3  # F, C-SWAP, F⁻¹
            
            total_swaps += swaps_needed
            total_q_depth += q_depth_this
            total_m_depth += m_depth_this
        
        # Error accumulation
        q_error = (total_swaps * 3 * QUBIT_COSTS['error_rate_2q'] +
                   n_interactions * QUBIT_COSTS['error_rate_2q'])
        m_error = (n_interactions * 2 * MERKABIT_COSTS['error_rate_1m'] +  # 2 F gates
                   n_interactions * MERKABIT_COSTS['error_rate_2m'])          # C-SWAP
        m_error *= MERKABIT_COSTS['pi_lock_suppression']
        
        depth_ratio = total_m_depth / total_q_depth if total_q_depth > 0 else 1
        error_ratio = m_error / q_error if q_error > 0 else 1
        
        print(f"  {n:>8d}  {diameter:>5d}  "
              f"{total_swaps:>8d}  {total_q_depth:>8d}  {q_error:>10.2e}  "
              f"{total_m_depth:>8d}  {m_error:>10.2e}  "
              f"{depth_ratio:>10.3f}×  {error_ratio:>10.3f}×")
        
        results.append((n, diameter, total_swaps, total_q_depth, 
                        total_m_depth, q_error, m_error))
    
    # Asymptotic analysis
    print(f"\n  Scaling behaviour:")
    print(f"    Qubit depth:    O(d) per interaction, d = lattice diameter")
    print(f"    Merkabit depth: O(1) per interaction (F gate → instant resonance)")
    print(f"    As lattice grows: depth ratio → 0 (merkabit advantage increases)")
    print(f"    At n=169, diameter≈14: merkabit uses {results[-1][4]/results[-1][3]:.1%} "
          f"of qubit depth")
    
    return results


# ============================================================================
# BENCHMARK 3: CONSTRAINT SATISFACTION (Graph 3-colouring)
# ============================================================================

def benchmark_graph_colouring():
    """
    Graph 3-colouring: assign colours {0, 1, 2} to vertices such that
    no adjacent pair shares the same colour.
    
    Qubit encoding:
      - 2 qubits per vertex (⌈log₂(3)⌉ = 2)
      - States |00⟩, |01⟩, |10⟩ represent 3 colours
      - |11⟩ is invalid → needs projection/penalty
      - Constraint check: controlled comparison between adjacent pairs
        requires decomposition into ~6 CNOTs per edge
    
    Merkabit encoding:
      - 1 merkabit per vertex (native ternary: +1, 0, -1 = 3 colours)
      - No wasted states, no penalty terms
      - Constraint check: P gate compares ternary values directly
        via coherence interference → ~2 C-SWAPs per edge
      - F gate controls which edges are active (dynamic constraint graph)
    
    Advantages:
      (a) 2× register reduction (1 merkabit vs 2 qubits per vertex)
      (b) No invalid-state penalty (all 3 merkabit states are physical)
      (c) Dynamic edge activation via F gate (incremental solving)
    """
    print("\n" + "=" * 72)
    print("BENCHMARK 3: CONSTRAINT SATISFACTION (Graph 3-colouring)")
    print("=" * 72)
    
    # Test on various graph sizes
    # Random graphs with edge probability p = 3/n (sparse, like real problems)
    graph_sizes = [6, 10, 20, 40, 80, 160, 320]
    
    print(f"\n  Random sparse graphs (edge prob ≈ 3/n):\n")
    print(f"  {'V':>5s}  {'E':>5s}  "
          f"{'q_reg':>6s}  {'m_reg':>6s}  {'reg_save':>8s}  "
          f"{'q_gates':>8s}  {'m_gates':>8s}  {'gate_save':>9s}  "
          f"{'q_err':>10s}  {'m_err':>10s}")
    print(f"  {'─'*5}  {'─'*5}  "
          f"{'─'*6}  {'─'*6}  {'─'*8}  "
          f"{'─'*8}  {'─'*8}  {'─'*9}  "
          f"{'─'*10}  {'─'*10}")
    
    np.random.seed(42)
    results = []
    
    for V in graph_sizes:
        p = min(3.0 / V, 0.5)
        # Generate random graph
        E = 0
        for i in range(V):
            for j in range(i+1, V):
                if np.random.random() < p:
                    E += 1
        E = max(E, V - 1)  # At least a tree
        
        # Qubit: 2 qubits per vertex
        q_register = 2 * V
        
        # Merkabit: 1 merkabit per vertex
        m_register = V
        
        # Grover-like search iterations
        search_space_q = 4**V   # 2 qubits per vertex, 4 states each
        search_space_m = 3**V   # 3 states per merkabit, all valid
        
        # But we only iterate over the valid subspace
        # Effective search space is 3^V for both (qubit wastes encoding)
        iterations = max(1, int(np.pi / 4 * np.sqrt(3**min(V, 15))))
        iterations = min(iterations, 10000)  # Cap for tractability
        
        # Qubit gates per iteration:
        #   Oracle: check E edges, each needs ~6 CNOTs + penalty for |11⟩
        #   Invalid state penalty: ~2 CNOTs per vertex
        #   Diffusion: ~4V single gates + 2V CNOTs
        q_cnots_per_iter = 6 * E + 2 * V + 2 * V
        q_single_per_iter = 4 * V + 2 * V  # Hadamards + phases + penalty
        q_total = iterations * (q_cnots_per_iter + q_single_per_iter)
        q_error = iterations * (q_cnots_per_iter * QUBIT_COSTS['error_rate_2q'] +
                                q_single_per_iter * QUBIT_COSTS['error_rate_1q'])
        
        # Merkabit gates per iteration:
        #   Oracle: check E edges, each needs ~2 C-SWAPs (ternary comparison)
        #   No invalid state penalty (all states valid!)
        #   Diffusion: ~3V single gates (P + Rx + Rz)
        #   F gate for dynamic edge activation: ~E F gates
        m_cswap_per_iter = 2 * E
        m_single_per_iter = 3 * V + E  # P rotations + F gates for edges
        m_total = iterations * (m_cswap_per_iter + m_single_per_iter)
        m_error = iterations * (
            m_cswap_per_iter * MERKABIT_COSTS['error_rate_2m'] +
            m_single_per_iter * MERKABIT_COSTS['error_rate_1m']
        ) * MERKABIT_COSTS['pi_lock_suppression']
        
        reg_save = 1 - m_register / q_register
        gate_save = 1 - m_total / q_total if q_total > 0 else 0
        
        print(f"  {V:>5d}  {E:>5d}  "
              f"{q_register:>6d}  {m_register:>6d}  {reg_save:>7.0%}  "
              f"{q_total:>8d}  {m_total:>8d}  {gate_save:>8.0%}  "
              f"{q_error:>10.2e}  {m_error:>10.2e}")
        
        results.append((V, E, q_register, m_register, q_total, 
                        m_total, q_error, m_error))
    
    print(f"\n  Key advantages for 3-colouring:")
    print(f"    Register: always 50% savings (1 merkabit vs 2 qubits per vertex)")
    print(f"    No invalid states: qubit encoding wastes |11⟩ (25% of Hilbert space)")
    print(f"    P gate cycles colours natively: P(2π/3) → {+1, 0, -1} → {0, -1, +1}")
    print(f"    F gate activates constraints dynamically (incremental solving)")
    
    return results


# ============================================================================
# BENCHMARK 4: REVERSIBLE COMPUTATION (Uncomputation savings)
# ============================================================================

def benchmark_reversible_computation():
    """
    Many quantum algorithms have a structure:
      1. Compute function f(x) into ancilla register
      2. Use result to mark/phase-kick target
      3. Uncompute f(x) to disentangle ancilla
    
    Step 3 replays step 1 backwards — doubling the gate count.
    
    Qubit:
      - Must explicitly uncompute by running circuit in reverse
      - Gate count: 2× the forward computation + marking
    
    Merkabit:
      - Inverse spinor already contains the reverse evolution
      - Berry phase accumulates during forward pass
      - Readout via Berry phase is non-destructive
      - Uncomputation is structural, not a gate sequence
      
    We model this for circuits of varying depth and ancilla width.
    """
    print("\n" + "=" * 72)
    print("BENCHMARK 4: REVERSIBLE COMPUTATION (Uncomputation savings)")
    print("=" * 72)
    
    # Model: function f requires a circuit of depth D with W ancilla
    scenarios = [
        ("Modular exp (Shor's)", 100, 20, 50),
        ("Hamiltonian sim step", 50, 10, 30),
        ("Arithmetic (add/mul)", 30, 8, 15),
        ("Boolean formula eval", 200, 40, 80),
        ("Quantum walk step", 40, 12, 20),
        ("Phase estimation", 150, 30, 60),
    ]
    
    print(f"\n  {'Algorithm':>25s}  {'D':>4s}  {'W':>4s}  "
          f"{'q_gates':>8s}  {'m_gates':>8s}  {'savings':>8s}  "
          f"{'q_depth':>8s}  {'m_depth':>8s}  "
          f"{'q_err':>10s}  {'m_err':>10s}")
    print(f"  {'─'*25}  {'─'*4}  {'─'*4}  "
          f"{'─'*8}  {'─'*8}  {'─'*8}  "
          f"{'─'*8}  {'─'*8}  "
          f"{'─'*10}  {'─'*10}")
    
    results = []
    
    for name, depth, ancilla_width, n_2q_gates in scenarios:
        # Qubit: forward + mark + uncompute
        q_single = 3 * depth * ancilla_width    # Forward + mark + reverse
        q_2qubit = 2 * n_2q_gates + ancilla_width  # Forward + reverse + marking
        q_total = q_single + q_2qubit
        q_depth = 2 * depth + 1  # Forward + reverse + mark
        q_error = (q_single * QUBIT_COSTS['error_rate_1q'] +
                   q_2qubit * QUBIT_COSTS['error_rate_2q'])
        
        # Merkabit: forward + mark (uncompute is structural)
        # The inverse spinor carries the reverse information
        # Berry phase readout extracts the result non-destructively
        m_single = depth * ancilla_width + ancilla_width  # Forward + mark
        m_2merk = n_2q_gates + ancilla_width               # Forward C-SWAPs + marking
        m_total = m_single + m_2merk
        m_depth = depth + 1  # Forward + mark (no uncompute pass)
        
        # Berry phase readout instead of destructive measurement
        m_readout = ancilla_width * MERKABIT_COSTS['error_rate_readout']
        m_error = ((m_single * MERKABIT_COSTS['error_rate_1m'] +
                    m_2merk * MERKABIT_COSTS['error_rate_2m']) *
                   MERKABIT_COSTS['pi_lock_suppression'] + m_readout)
        
        savings = 1 - m_total / q_total
        
        print(f"  {name:>25s}  {depth:>4d}  {ancilla_width:>4d}  "
              f"{q_total:>8d}  {m_total:>8d}  {savings:>7.0%}  "
              f"{q_depth:>8d}  {m_depth:>8d}  "
              f"{q_error:>10.2e}  {m_error:>10.2e}")
        
        results.append((name, q_total, m_total, q_depth, m_depth, 
                        q_error, m_error))
    
    avg_savings = np.mean([1 - r[2]/r[1] for r in results])
    avg_depth_save = np.mean([1 - r[4]/r[3] for r in results])
    avg_error_ratio = np.mean([r[6]/r[5] for r in results])
    
    print(f"\n  Average gate savings:  {avg_savings:.0%}")
    print(f"  Average depth savings: {avg_depth_save:.0%}")
    print(f"  Average error ratio:   {avg_error_ratio:.3f}×")
    print(f"\n  Mechanism: inverse spinor carries reverse evolution natively")
    print(f"  Berry phase readout is non-destructive (no wavefunction collapse)")
    print(f"  Uncomputation cost: qubit = O(D), merkabit = O(1)")
    
    return results


# ============================================================================
# BENCHMARK 5: ZERO-STATE CONTROLLED OPERATIONS
# ============================================================================

def benchmark_zero_state_control():
    """
    The C-SWAP is controlled by the zero state — the standing-wave
    equilibrium, which is the MOST coherent and error-resistant
    configuration. This inverts the qubit paradigm where CNOT is
    controlled by |1⟩, which has no special stability.
    
    We model the effective error rate of controlled operations when
    the control state itself has intrinsic error protection.
    """
    print("\n" + "=" * 72)
    print("BENCHMARK 5: ZERO-STATE CONTROLLED OPERATIONS")
    print("=" * 72)
    
    print(f"\n  Control state error comparison:")
    print(f"  ─────────────────────────────────────────────────────────")
    
    # Qubit: control state |1⟩ has no special stability
    # Error on control preparation: ε_1q
    # Error on CNOT: ε_2q
    # Error on control verification: ε_meas
    # Total control error: ε_1q + ε_2q + ε_meas
    
    q_control_prep = QUBIT_COSTS['error_rate_1q']
    q_gate_error = QUBIT_COSTS['error_rate_2q']
    q_readout_error = QUBIT_COSTS['error_rate_meas']
    q_total_control = q_control_prep + q_gate_error + q_readout_error
    
    # Merkabit: control state |0⟩ is π-locked standing wave
    # π-lock provides first-order insensitivity to phase perturbation
    # Symmetric noise cancels in the standing wave (Level 1)
    # Berry phase readout is non-destructive
    
    m_control_prep = MERKABIT_COSTS['error_rate_1m'] * MERKABIT_COSTS['pi_lock_suppression']
    m_gate_error = MERKABIT_COSTS['error_rate_2m'] * MERKABIT_COSTS['pi_lock_suppression']
    m_readout_error = MERKABIT_COSTS['error_rate_readout']
    m_total_control = m_control_prep + m_gate_error + m_readout_error
    
    print(f"    Qubit |1⟩ control:")
    print(f"      Preparation error:  {q_control_prep:.2e}")
    print(f"      Gate error (CNOT):  {q_gate_error:.2e}")
    print(f"      Readout error:      {q_readout_error:.2e}")
    print(f"      Total per control:  {q_total_control:.2e}")
    
    print(f"\n    Merkabit |0⟩ control (π-locked standing wave):")
    print(f"      Preparation error:  {m_control_prep:.2e}  (π-lock suppressed)")
    print(f"      Gate error (C-SWAP): {m_gate_error:.2e}  (π-lock suppressed)")
    print(f"      Readout error:      {m_readout_error:.2e}  (Berry phase, non-destructive)")
    print(f"      Total per control:  {m_total_control:.2e}")
    
    print(f"\n    Control error ratio: {m_total_control/q_total_control:.3f}×")
    
    # Impact on multi-controlled operations
    print(f"\n  Cascaded controlled operations (n controls in series):")
    print(f"  {'n_controls':>11s}  {'q_error':>12s}  {'m_error':>12s}  {'ratio':>8s}")
    print(f"  {'─'*11}  {'─'*12}  {'─'*12}  {'─'*8}")
    
    for n_controls in [1, 2, 5, 10, 20, 50, 100]:
        q_err = 1 - (1 - q_total_control)**n_controls
        m_err = 1 - (1 - m_total_control)**n_controls
        ratio = m_err / q_err if q_err > 0 else 0
        print(f"  {n_controls:>11d}  {q_err:>12.4e}  {m_err:>12.4e}  {ratio:>7.3f}×")
    
    print(f"\n  Key insight: the zero state (standing wave at π-lock) is the")
    print(f"  most error-resistant configuration in the merkabit's state space.")
    print(f"  Using it as the control trigger means controlled operations are")
    print(f"  intrinsically more reliable than qubit CNOT controlled by |1⟩.")
    print(f"  This advantage compounds exponentially in deep circuits.")
    
    return q_total_control, m_total_control


# ============================================================================
# COMPOSITE ANALYSIS
# ============================================================================

def composite_analysis():
    """
    Combine all benchmarks into an overall comparison showing where
    merkabit advantages are largest and what drives them.
    """
    print("\n" + "=" * 72)
    print("COMPOSITE ANALYSIS: WHERE MERKABITS WIN AND WHY")
    print("=" * 72)
    
    print(f"""
  ┌───────────────────────────┬─────────────────┬──────────────────────┐
  │ Resource                  │ Advantage Type   │ Scaling              │
  ├───────────────────────────┼─────────────────┼──────────────────────┤
  │ P gate (ternary DOF)      │ Register width   │ log₃/log₂ = 0.63×   │
  │                           │ No invalid states│ 3/4 = 0.75× Hilbert │
  ├───────────────────────────┼─────────────────┼──────────────────────┤
  │ F gate (dynamic coupling) │ Circuit depth    │ O(d) → O(1)          │
  │                           │ SWAP elimination │ Saves 3d CNOTs/hop   │
  ├───────────────────────────┼─────────────────┼──────────────────────┤
  │ Zero state (standing wave)│ Control fidelity │ π-lock suppression   │
  │                           │ C-SWAP trigger   │ Compounds in depth   │
  ├───────────────────────────┼─────────────────┼──────────────────────┤
  │ Inverse spinor (bidir.)   │ Uncomputation    │ 2× → 1× gate count  │
  │                           │ Berry readout    │ Non-destructive      │
  ├───────────────────────────┼─────────────────┼──────────────────────┤
  │ F–P duality               │ Gate compilation │ 2 routes per target  │
  │                           │ Time as resource │ Phase ↔ frequency    │
  └───────────────────────────┴─────────────────┴──────────────────────┘
""")
    
    # Problem-resource matching
    print(f"  Problem–resource matching (which advantage dominates):\n")
    print(f"  {'Problem':>30s}  {'Dominant resource':>25s}  {'Scaling gain':>15s}")
    print(f"  {'─'*30}  {'─'*25}  {'─'*15}")
    
    matches = [
        ("Database search",         "P gate (ternary oracle)",   "37% fewer units"),
        ("Long-range interactions",  "F gate (dynamic coupling)", "O(d) → O(1)"),
        ("Graph colouring",          "P gate (native ternary)",   "50% register"),
        ("Shor's / phase estimation","Inverse spinor (bidir.)",   "~50% gates"),
        ("VQE / QAOA",              "F gate + zero-state",        "depth + fidelity"),
        ("Quantum simulation",       "F gate (all-to-all)",       "depth × fidelity"),
        ("Error-sensitive circuits", "Zero-state control",         "exp. in depth"),
        ("Constraint satisfaction",  "P + F combined",             "register + depth"),
    ]
    
    for problem, resource, gain in matches:
        print(f"  {problem:>30s}  {resource:>25s}  {gain:>15s}")
    
    # Where advantages DON'T apply
    print(f"\n  Where merkabit advantages are minimal or absent:")
    print(f"    • Short, shallow circuits with only nearest-neighbour coupling")
    print(f"    • Problems with natural binary structure (bit-level operations)")
    print(f"    • Circuits where all interactions are local (no SWAP overhead)")
    print(f"    • Measurement-heavy protocols (Berry readout advantage is per-shot)")
    
    # The central prediction
    print(f"""
  ═══════════════════════════════════════════════════════════════════
  CENTRAL PREDICTION
  ═══════════════════════════════════════════════════════════════════
  
  The largest advantage comes from the F gate on problems requiring
  long-range interactions. On an n-node lattice with diameter D:
  
    Qubit circuit depth:    O(k · D)    where k = number of long-range ops
    Merkabit circuit depth: O(k)        (F gate eliminates distance dependence)
  
  For quantum chemistry (n-body Hamiltonians), quantum simulation
  (spin chains with long-range terms), and optimisation (dense
  constraint graphs), this translates to:
  
    Depth reduction:  O(D) per interaction → savings grow with system size
    Error reduction:  proportional to depth savings (fewer gates = less noise)
    Fidelity gain:    compounds multiplicatively (each saved gate preserves coherence)
  
  The F gate transforms lattice connectivity from a hardware constraint
  to a software parameter. This is not an incremental improvement —
  it is a qualitative change in the computational model.
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔" + "═" * 70 + "╗")
    print("║  NATIVE ALGORITHM BENCHMARKS — MERKABIT vs QUBIT              " + " " * 5 + "║")
    print("║  Quantifying advantages from P, F, zero-state, bidirectional  " + " " * 2 + "║")
    print("║  Section 8.9.5: 'the most important open question'            " + " " * 4 + "║")
    print("╚" + "═" * 70 + "╝")
    print()
    
    start = time.time()
    
    r1 = benchmark_ternary_search()
    r2 = benchmark_long_range_coupling()
    r3 = benchmark_graph_colouring()
    r4 = benchmark_reversible_computation()
    r5 = benchmark_zero_state_control()
    
    composite_analysis()
    
    elapsed = time.time() - start
    print(f"  Benchmarks completed in {elapsed:.2f}s")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
