#!/usr/bin/env python3
"""
Î±â»Â¹ FROM BERRY PHASE GEOMETRY
==============================

Attempt to derive the fine structure constant Î±â»Â¹ â‰ˆ 137 from the
Berry phase simulation data, providing a THIRD derivation route
independent of:
  Route A: 168 âˆ’ 31 = 137 (configuration counting)
  Route B: N(12 + 5Ï‰) + dim(Dâ‚„) = 109 + 28 = 137 (Casimir + Eisenstein)

The Berry phase route uses the ZERO POINT as the anchor:
  The ouroboros cycle Berry phase encodes the geometric relationship
  between the trit states. The zero point |0âŸ© (uâŠ¥v) is the structurally
  singular state where both readout channels converge. The Berry phase
  geometry on SÂ³Ã—SÂ³, filtered through the Eâ‚† architecture, should
  reproduce the same constant through purely geometric means.

Physical reasoning:
  Î± = coherence frequency = fraction of configurations that are coherent.
  The Berry phase measures the solid angle enclosed by the cycle on SÂ².
  The ratio of the zero-point Berry phase to the total available phase
  space, weighted by the Eâ‚† structure, should give the coherence fraction.

This simulation computes every geometric quantity available from the
ouroboros cycle and systematically searches for relationships to 137.

Usage: python3 alpha_from_berry_phase.py
Requirements: numpy
"""

import numpy as np
import time

# ============================================================================
# CONSTANTS
# ============================================================================

ALPHA_INV = 137.035999084  # CODATA 2018
ALPHA_INV_INT = 137

COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H
OUROBOROS_GATES = ['S', 'R', 'T', 'F', 'P']
NUM_GATES = 5

# Eâ‚† structural constants
E6_RANK = 6
E6_POSITIVE_ROOTS = 36
E6_DIM = 78
E6_WEYL_ORDER = 51840
E6_EXPONENTS = [1, 4, 5, 7, 8, 11]
D4_DIM = 28
E6_OVER_D4 = 50
P24_ORDER = 24


# ============================================================================
# MERKABIT STATE AND GATES (from core simulation)
# ============================================================================

class MerkabitState:
    def __init__(self, u, v, omega=1.0):
        self.u = np.array(u, dtype=complex)
        self.v = np.array(v, dtype=complex)
        self.omega = omega
        self.u /= np.linalg.norm(self.u)
        self.v /= np.linalg.norm(self.v)

    @property
    def relative_phase(self):
        return np.angle(np.vdot(self.u, self.v))

    @property
    def overlap_magnitude(self):
        return abs(np.vdot(self.u, self.v))

    @property
    def coherence(self):
        return np.real(np.vdot(self.u, self.v))

    @property
    def bloch_vector_u(self):
        return np.array([
            2 * np.real(np.conj(self.u[0]) * self.u[1]),
            2 * np.imag(np.conj(self.u[0]) * self.u[1]),
            abs(self.u[0])**2 - abs(self.u[1])**2
        ])

    @property
    def bloch_vector_v(self):
        return np.array([
            2 * np.real(np.conj(self.v[0]) * self.v[1]),
            2 * np.imag(np.conj(self.v[0]) * self.v[1]),
            abs(self.v[0])**2 - abs(self.v[1])**2
        ])

    def copy(self):
        return MerkabitState(self.u.copy(), self.v.copy(), self.omega)


def make_trit_plus():  return MerkabitState([1, 0], [1, 0])
def make_trit_zero():  return MerkabitState([1, 0], [0, 1])
def make_trit_minus(): return MerkabitState([1, 0], [-1, 0])

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


def ouroboros_step(state, step_index, theta=STEP_PHASE):
    k = step_index
    absent = k % NUM_GATES
    p_angle = theta
    sym_base = theta / 3
    omega_k = 2 * np.pi * k / COXETER_H
    rx_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    rz_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))

    gate_label = OUROBOROS_GATES[absent]
    if gate_label == 'S':
        rz_angle *= 0.4; rx_angle *= 1.3
    elif gate_label == 'R':
        rx_angle *= 0.4; rz_angle *= 1.3
    elif gate_label == 'T':
        rx_angle *= 0.7; rz_angle *= 0.7
    elif gate_label == 'P':
        p_angle *= 0.6; rx_angle *= 1.8; rz_angle *= 1.5

    s = gate_P(state, p_angle)
    s = gate_Rz(s, rz_angle)
    s = gate_Rx(s, rx_angle)
    return s


def run_cycle(state, n_cycles=1):
    """Run n ouroboros cycles, return states list and Berry phase."""
    s = state.copy()
    all_states = [s.copy()]
    total_gamma = 0.0

    for _ in range(n_cycles):
        cycle_states = [s.copy()]
        for step in range(COXETER_H):
            s = ouroboros_step(s, step)
            cycle_states.append(s.copy())
        gamma = compute_berry_phase(cycle_states[:-1])
        total_gamma += gamma
        all_states.extend(cycle_states[1:])

    return s, total_gamma, all_states


def compute_berry_phase(states):
    n = len(states)
    gamma = 0.0
    for k in range(n):
        k_next = (k + 1) % n
        ou = np.vdot(states[k].u, states[k_next].u)
        ov = np.vdot(states[k].v, states[k_next].v)
        gamma += np.angle(ou * ov)
    return -gamma


def compute_berry_uv(states):
    """Compute separate Berry phases for u and v spinors."""
    n = len(states)
    gu, gv = 0.0, 0.0
    for k in range(n):
        k_next = (k + 1) % n
        gu += np.angle(np.vdot(states[k].u, states[k_next].u))
        gv += np.angle(np.vdot(states[k].v, states[k_next].v))
    return -gu, -gv


def compute_solid_angle(states, spinor='u'):
    """
    Compute the solid angle enclosed by the Bloch sphere path.
    Uses the spherical excess formula for the polygon on SÂ².
    Î© = Î³_Berry (for a single spinor on SÂ²).
    """
    n = len(states)
    gamma = 0.0
    for k in range(n):
        k_next = (k + 1) % n
        if spinor == 'u':
            overlap = np.vdot(states[k].u, states[k_next].u)
        else:
            overlap = np.vdot(states[k].v, states[k_next].v)
        gamma += np.angle(overlap)
    return -gamma


def compute_path_length(states, spinor='u'):
    """Compute the Fubini-Study path length on SÂ³."""
    n = len(states)
    length = 0.0
    for k in range(n - 1):
        if spinor == 'u':
            fid = abs(np.vdot(states[k].u, states[k+1].u))
        else:
            fid = abs(np.vdot(states[k].v, states[k+1].v))
        # Fubini-Study distance
        fid = min(fid, 1.0)
        length += np.arccos(fid)
    return length


# ============================================================================
# COMPUTATION: EXTRACT ALL GEOMETRIC QUANTITIES
# ============================================================================

def extract_geometry():
    """
    Extract every measurable geometric quantity from the ouroboros cycle
    for all three basis states.
    """
    results = {}

    for label, make_fn in [('plus', make_trit_plus),
                           ('zero', make_trit_zero),
                           ('minus', make_trit_minus)]:
        s0 = make_fn()

        # Run one cycle
        states = [s0.copy()]
        s = s0.copy()
        for step in range(COXETER_H):
            s = ouroboros_step(s, step)
            states.append(s.copy())

        # Berry phases
        gamma_total = compute_berry_phase(states[:-1])
        gamma_u, gamma_v = compute_berry_uv(states[:-1])

        # Solid angles
        omega_u = compute_solid_angle(states[:-1], 'u')
        omega_v = compute_solid_angle(states[:-1], 'v')

        # Path lengths on SÂ³
        len_u = compute_path_length(states, 'u')
        len_v = compute_path_length(states, 'v')

        # Bloch vectors
        bloch_u_start = s0.bloch_vector_u
        bloch_v_start = s0.bloch_vector_v
        bloch_u_end = s.bloch_vector_u
        bloch_v_end = s.bloch_vector_v

        # Bloch sphere separation
        bloch_uv_dot = np.dot(bloch_u_start, bloch_v_start)

        # Interferometric overlap
        overlap_u = np.vdot(s0.u, s.u)
        overlap_v = np.vdot(s0.v, s.v)
        interference = overlap_u * overlap_v
        interf_amp = abs(interference)
        interf_phase = np.angle(interference)

        # Step-by-step Berry connections
        connections = []
        for k in range(COXETER_H):
            k_next = (k + 1) % COXETER_H
            A_u = np.angle(np.vdot(states[k].u, states[k_next].u))
            A_v = np.angle(np.vdot(states[k].v, states[k_next].v))
            connections.append({'A_u': A_u, 'A_v': A_v, 'A_total': A_u + A_v})

        results[label] = {
            'gamma': gamma_total,
            'gamma_u': gamma_u,
            'gamma_v': gamma_v,
            'omega_u': omega_u,
            'omega_v': omega_v,
            'path_u': len_u,
            'path_v': len_v,
            'bloch_uv_dot': bloch_uv_dot,
            'interf_amp': interf_amp,
            'interf_phase': interf_phase,
            'connections': connections,
        }

    return results


# ============================================================================
# SEARCH FOR Î±â»Â¹ = 137
# ============================================================================

def search_for_alpha(geom):
    """
    Systematically combine geometric quantities with Eâ‚† structural
    constants to search for expressions that yield 137.
    """
    print("=" * 76)
    print("  SEARCHING FOR Î±â»Â¹ = 137 FROM BERRY PHASE GEOMETRY")
    print("=" * 76)

    gp = geom['plus']['gamma']
    gz = geom['zero']['gamma']
    gm = geom['minus']['gamma']

    gp_u = geom['plus']['gamma_u']
    gz_u = geom['zero']['gamma_u']
    gm_u = geom['minus']['gamma_u']
    gp_v = geom['plus']['gamma_v']
    gz_v = geom['zero']['gamma_v']
    gm_v = geom['minus']['gamma_v']

    sep = abs(gz - gp)  # |0âŸ© separation from |Â±1âŸ©
    delta_uv_zero = abs(gz_u - gz_v)  # u-v Berry splitting at |0âŸ©
    delta_uv_plus = abs(gp_u - gp_v)  # u-v Berry splitting at |Â±1âŸ©

    path_u_zero = geom['zero']['path_u']
    path_v_zero = geom['zero']['path_v']
    path_u_plus = geom['plus']['path_u']
    path_v_plus = geom['plus']['path_v']

    interf_zero = geom['zero']['interf_amp']
    interf_plus = geom['plus']['interf_amp']
    interf_phase_zero = geom['zero']['interf_phase']
    interf_phase_plus = geom['plus']['interf_phase']

    # --- Print raw data ---
    print(f"\n  RAW BERRY PHASE DATA:")
    print(f"  {'Quantity':>35}  {'|+1âŸ©':>12}  {'|0âŸ©':>12}  {'|âˆ’1âŸ©':>12}")
    print(f"  {'-'*35}  {'-'*12}  {'-'*12}  {'-'*12}")
    print(f"  {'Î³_total (rad)':>35}  {gp:>12.6f}  {gz:>12.6f}  {gm:>12.6f}")
    print(f"  {'Î³_u (rad)':>35}  {gp_u:>12.6f}  {gz_u:>12.6f}  {gm_u:>12.6f}")
    print(f"  {'Î³_v (rad)':>35}  {gp_v:>12.6f}  {gz_v:>12.6f}  {gm_v:>12.6f}")
    print(f"  {'Î³/Ï€':>35}  {gp/np.pi:>12.6f}  {gz/np.pi:>12.6f}  {gm/np.pi:>12.6f}")
    print(f"  {'Î³_u/Ï€':>35}  {gp_u/np.pi:>12.6f}  {gz_u/np.pi:>12.6f}  {gm_u/np.pi:>12.6f}")
    print(f"  {'Î³_v/Ï€':>35}  {gp_v/np.pi:>12.6f}  {gz_v/np.pi:>12.6f}  {gm_v/np.pi:>12.6f}")
    print(f"  {'path_u':>35}  {path_u_plus:>12.6f}  {path_u_zero:>12.6f}  {path_u_plus:>12.6f}")
    print(f"  {'path_v':>35}  {path_v_plus:>12.6f}  {path_v_zero:>12.6f}  {path_v_plus:>12.6f}")
    print(f"  {'|interference|':>35}  {interf_plus:>12.6f}  {interf_zero:>12.6f}  {interf_plus:>12.6f}")
    print(f"  {'arg(interference)':>35}  {interf_phase_plus:>12.6f}  {interf_phase_zero:>12.6f}  {interf_phase_plus:>12.6f}")

    print(f"\n  DERIVED QUANTITIES:")
    print(f"    |Î³â‚€ âˆ’ Î³Â±| (zero-point separation):     {sep:.6f} rad = {sep/np.pi:.6f}Ï€")
    print(f"    |Î³_u âˆ’ Î³_v| at |0âŸ© (u-v splitting):   {delta_uv_zero:.6f} rad")
    print(f"    |Î³_u âˆ’ Î³_v| at |Â±1âŸ©:                  {delta_uv_plus:.6f} rad")

    # --- Eâ‚† structural constants ---
    print(f"\n  Eâ‚† STRUCTURAL CONSTANTS:")
    print(f"    h(Eâ‚†) = {COXETER_H}")
    print(f"    rank = {E6_RANK}")
    print(f"    dim = {E6_DIM}")
    print(f"    positive roots = {E6_POSITIVE_ROOTS}")
    print(f"    |W(Eâ‚†)| = {E6_WEYL_ORDER}")
    print(f"    exponents = {E6_EXPONENTS}")
    print(f"    dim(Dâ‚„) = {D4_DIM}")
    print(f"    dim(Eâ‚†/Dâ‚„) = {E6_OVER_D4}")
    print(f"    |Pâ‚‚â‚„| = {P24_ORDER}")
    print(f"    N(12 + 5Ï‰) = 109")

    # Casimir eigenvalues
    casimir = {m: m * (m + COXETER_H) for m in E6_EXPONENTS}
    print(f"    Casimir: {casimir}")
    casimir_sum = sum(casimir.values())
    casimir_inner = (casimir[5] + casimir[7]) / 2  # = 109
    print(f"    Casimir sum = {casimir_sum}")
    print(f"    Inner pair average = {casimir_inner}")

    # ================================================================
    # SYSTEMATIC SEARCH
    # ================================================================
    print(f"\n  {'='*70}")
    print(f"  ROUTE C: BERRY PHASE GEOMETRY â†’ Î±â»Â¹")
    print(f"  {'='*70}")

    candidates = []

    def check(expr_val, expr_name, tolerance=0.5):
        """Check if expression is close to 137."""
        if abs(expr_val) < 1e-10:
            return
        diff = abs(abs(expr_val) - ALPHA_INV)
        rel_diff = diff / ALPHA_INV
        if diff < tolerance:
            candidates.append((abs(expr_val), expr_name, diff, rel_diff))
            print(f"  *** {expr_name:>55} = {abs(expr_val):>12.4f}  "
                  f"(Î” = {diff:.4f}, {rel_diff*100:.2f}%)")

    def check_exact(expr_val, expr_name, target=ALPHA_INV_INT, tol=0.5):
        if abs(expr_val) < 1e-10:
            return
        diff = abs(abs(expr_val) - target)
        if diff < tol:
            candidates.append((abs(expr_val), expr_name, diff, diff/target))
            print(f"  *** {expr_name:>55} = {abs(expr_val):>12.6f}  "
                  f"(Î” from {target} = {diff:.6f})")

    # ---- Category 1: Pure Berry phase ratios ----
    print(f"\n  Category 1: Pure Berry phase ratios")
    print(f"  {'-'*70}")

    check(gz / gp, "Î³â‚€ / Î³Â±")
    check(sep / gp, "|Î³â‚€ âˆ’ Î³Â±| / Î³Â±")
    check(gz**2 / gp, "Î³â‚€Â² / Î³Â±")
    check(gz * gp, "Î³â‚€ Ã— Î³Â±")
    check(sep / (2 * np.pi), "|Î³â‚€ âˆ’ Î³Â±| / 2Ï€")
    check(sep**2, "|Î³â‚€ âˆ’ Î³Â±|Â²")
    check((gz / np.pi)**2, "(Î³â‚€/Ï€)Â²")
    check(gz**2 / (gp * np.pi), "Î³â‚€Â² / (Î³Â± Ã— Ï€)")

    # ---- Category 2: Berry Ã— Eâ‚† constants ----
    print(f"\n  Category 2: Berry phase Ã— Eâ‚† structural constants")
    print(f"  {'-'*70}")

    check(sep * P24_ORDER, "|Î³â‚€ âˆ’ Î³Â±| Ã— |Pâ‚‚â‚„|")
    check(sep * COXETER_H, "|Î³â‚€ âˆ’ Î³Â±| Ã— h")
    check(sep * E6_POSITIVE_ROOTS, "|Î³â‚€ âˆ’ Î³Â±| Ã— 36")
    check(sep * E6_DIM, "|Î³â‚€ âˆ’ Î³Â±| Ã— 78")
    check(sep * D4_DIM, "|Î³â‚€ âˆ’ Î³Â±| Ã— 28")
    check(sep * E6_RANK, "|Î³â‚€ âˆ’ Î³Â±| Ã— 6")
    check(sep * NUM_GATES, "|Î³â‚€ âˆ’ Î³Â±| Ã— 5")
    check(abs(gz) * P24_ORDER, "|Î³â‚€| Ã— |Pâ‚‚â‚„|")
    check(abs(gz) * COXETER_H, "|Î³â‚€| Ã— h")
    check(abs(gz) * E6_POSITIVE_ROOTS, "|Î³â‚€| Ã— 36")
    check(abs(gz) * E6_DIM, "|Î³â‚€| Ã— 78")
    check(abs(gz) * D4_DIM, "|Î³â‚€| Ã— 28")
    check(abs(gz) / np.pi * E6_DIM, "|Î³â‚€|/Ï€ Ã— 78")
    check(abs(gz) / np.pi * E6_POSITIVE_ROOTS, "|Î³â‚€|/Ï€ Ã— 36")
    check(abs(gz) / np.pi * P24_ORDER, "|Î³â‚€|/Ï€ Ã— 24")
    check(abs(gz) / np.pi * D4_DIM, "|Î³â‚€|/Ï€ Ã— 28")
    check(abs(gz) / np.pi * COXETER_H, "|Î³â‚€|/Ï€ Ã— 12")
    check(abs(gz) / (2*np.pi) * E6_WEYL_ORDER, "|Î³â‚€|/2Ï€ Ã— |W|")
    check(sep / np.pi * P24_ORDER, "|Î³â‚€âˆ’Î³Â±|/Ï€ Ã— 24")
    check(sep / np.pi * COXETER_H, "|Î³â‚€âˆ’Î³Â±|/Ï€ Ã— 12")
    check(sep / np.pi * E6_POSITIVE_ROOTS, "|Î³â‚€âˆ’Î³Â±|/Ï€ Ã— 36")
    check(sep / np.pi * E6_DIM, "|Î³â‚€âˆ’Î³Â±|/Ï€ Ã— 78")
    check(sep / np.pi * D4_DIM, "|Î³â‚€âˆ’Î³Â±|/Ï€ Ã— 28")

    # ---- Category 3: Casimir-weighted Berry phases ----
    print(f"\n  Category 3: Casimir-weighted Berry phases")
    print(f"  {'-'*70}")

    # Weighted sum of Berry phases over Coxeter exponents
    for label in ['zero', 'plus']:
        g = abs(geom[label]['gamma'])
        for m in E6_EXPONENTS:
            C = casimir[m]
            check(g * C, f"|Î³({label})| Ã— C_{m} ({C})")
            check(g / np.pi * C, f"|Î³({label})|/Ï€ Ã— C_{m}")

    check(abs(gz) / np.pi * casimir_inner, "|Î³â‚€|/Ï€ Ã— 109")
    check(sep / np.pi * casimir_inner, "|Î³â‚€âˆ’Î³Â±|/Ï€ Ã— 109")

    # ---- Category 4: Path lengths and solid angles ----
    print(f"\n  Category 4: Path lengths and solid angles")
    print(f"  {'-'*70}")

    total_path_zero = path_u_zero + path_v_zero
    total_path_plus = path_u_plus + path_v_plus
    path_diff = abs(total_path_zero - total_path_plus)

    check(total_path_zero * P24_ORDER, "Lâ‚€_total Ã— 24")
    check(total_path_zero * COXETER_H, "Lâ‚€_total Ã— 12")
    check(total_path_zero * E6_POSITIVE_ROOTS, "Lâ‚€_total Ã— 36")
    check(total_path_zero * E6_DIM, "Lâ‚€_total Ã— 78")
    check(total_path_zero / np.pi * E6_DIM, "Lâ‚€_total/Ï€ Ã— 78")
    check(total_path_zero / np.pi * E6_WEYL_ORDER, "Lâ‚€_total/Ï€ Ã— |W|")
    check(path_diff * E6_POSITIVE_ROOTS, "Î”L Ã— 36")
    check(path_diff * E6_DIM, "Î”L Ã— 78")
    check(path_diff * P24_ORDER, "Î”L Ã— 24")

    # ---- Category 5: Interferometric quantities ----
    print(f"\n  Category 5: Interferometric quantities")
    print(f"  {'-'*70}")

    interf_phase_sep = abs(interf_phase_zero - interf_phase_plus)
    check(interf_phase_sep * P24_ORDER, "|Î”Ï†_interf| Ã— 24")
    check(interf_phase_sep * E6_DIM, "|Î”Ï†_interf| Ã— 78")
    check(interf_phase_sep * COXETER_H, "|Î”Ï†_interf| Ã— 12")
    check(interf_phase_sep / np.pi * E6_DIM, "|Î”Ï†_interf|/Ï€ Ã— 78")
    check(interf_phase_sep / np.pi * E6_POSITIVE_ROOTS, "|Î”Ï†_interf|/Ï€ Ã— 36")

    # ---- Category 6: Combined Berry + Casimir (the inner pair) ----
    print(f"\n  Category 6: Combined Berry + Casimir (inner pair)")
    print(f"  {'-'*70}")

    # The inner pair has Casimir average 109. Berry phase provides the complement.
    check(casimir_inner + D4_DIM, "C_inner + dim(Dâ‚„) [Route B verification]")

    # Can Berry phase reproduce the 28 = dim(Dâ‚„)?
    check(abs(gz) / np.pi * COXETER_H + casimir_inner,
          "|Î³â‚€|/Ï€ Ã— h + 109")
    check(sep / np.pi * COXETER_H + casimir_inner,
          "|Î³â‚€âˆ’Î³Â±|/Ï€ Ã— h + 109")
    check(abs(gp) / np.pi * E6_DIM + casimir_inner,
          "|Î³Â±|/Ï€ Ã— 78 + 109")
    check(abs(gp) * E6_DIM + casimir_inner,
          "|Î³Â±| Ã— 78 + 109")
    check(abs(gp) * P24_ORDER**2 / np.pi + casimir_inner,
          "|Î³Â±| Ã— 24Â²/Ï€ + 109")

    # ---- Category 7: Products of Eâ‚† invariants ----
    print(f"\n  Category 7: Structural products and quotients")
    print(f"  {'-'*70}")

    check(E6_DIM * sep / (2 * np.pi), "78 Ã— |Î³â‚€âˆ’Î³Â±| / 2Ï€")
    check(E6_POSITIVE_ROOTS * abs(gz) / (2 * np.pi), "36 Ã— |Î³â‚€| / 2Ï€")
    check(P24_ORDER * abs(gz) / (2 * np.pi), "24 Ã— |Î³â‚€| / 2Ï€")
    check(E6_WEYL_ORDER * abs(gp) / (2 * np.pi), "|W| Ã— |Î³Â±| / 2Ï€")
    check(E6_POSITIVE_ROOTS * sep / (2 * np.pi), "36 Ã— |Î³â‚€âˆ’Î³Â±| / 2Ï€")
    check(COXETER_H * E6_RANK * sep / (2 * np.pi), "h Ã— rank Ã— |Î³â‚€âˆ’Î³Â±| / 2Ï€")

    # ---- Category 8: Non-linear combinations ----
    print(f"\n  Category 8: Non-linear combinations")
    print(f"  {'-'*70}")

    check(abs(gz)**2 / (2 * np.pi), "Î³â‚€Â² / 2Ï€")
    check(sep**2 / (2 * np.pi * abs(gp)), "sepÂ² / (2Ï€ Ã— |Î³Â±|)")
    check(sep**2 / abs(gp), "sepÂ² / |Î³Â±|")
    check(abs(gz) * sep / np.pi, "|Î³â‚€| Ã— sep / Ï€")
    check(abs(gz) * sep / (2 * np.pi), "|Î³â‚€| Ã— sep / 2Ï€")
    check(np.sqrt(abs(gz) * E6_WEYL_ORDER), "âˆš(|Î³â‚€| Ã— |W|)")
    check((sep / np.pi)**2 * D4_DIM, "(sep/Ï€)Â² Ã— 28")
    check((sep / np.pi)**2 * E6_POSITIVE_ROOTS, "(sep/Ï€)Â² Ã— 36")
    check(abs(gz / gp) * E6_POSITIVE_ROOTS / (2 * np.pi), "|Î³â‚€/Î³Â±| Ã— 36 / 2Ï€")

    # ---- Category 9: Multi-cycle quantities ----
    print(f"\n  Category 9: Multi-cycle Berry phases")
    print(f"  {'-'*70}")

    for n_cyc in [2, 3, 5, 6, 12]:
        _, gz_n, _ = run_cycle(make_trit_zero(), n_cyc)
        _, gp_n, _ = run_cycle(make_trit_plus(), n_cyc)
        sep_n = abs(gz_n - gp_n)

        check(sep_n / n_cyc, f"sep_{n_cyc}cyc / {n_cyc}")
        check(sep_n / (2 * np.pi), f"sep_{n_cyc}cyc / 2Ï€")
        check(sep_n / (2 * np.pi * n_cyc), f"sep_{n_cyc}cyc / (2Ï€ Ã— {n_cyc})")
        check(abs(gz_n) / (2 * np.pi), f"|Î³â‚€_{n_cyc}cyc| / 2Ï€")
        check(abs(gz_n) / np.pi, f"|Î³â‚€_{n_cyc}cyc| / Ï€")

        # Weighted by Eâ‚† constants
        check(sep_n * COXETER_H / (2 * np.pi * n_cyc), f"sep_{n_cyc}cyc Ã— h / (2Ï€n)")
        check(sep_n / (n_cyc * np.pi) * P24_ORDER, f"sep_{n_cyc}cyc/(nÏ€) Ã— 24")
        check(abs(gz_n) / (n_cyc * np.pi) * P24_ORDER, f"|Î³â‚€_{n_cyc}|/(nÏ€) Ã— 24")
        check(abs(gz_n) / (n_cyc * np.pi) * E6_DIM, f"|Î³â‚€_{n_cyc}|/(nÏ€) Ã— 78")
        check(abs(gz_n) / (n_cyc * np.pi) * E6_POSITIVE_ROOTS, f"|Î³â‚€_{n_cyc}|/(nÏ€) Ã— 36")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n  {'='*70}")
    print(f"  CANDIDATE EXPRESSIONS FOR Î±â»Â¹ = 137")
    print(f"  {'='*70}")

    if candidates:
        # Sort by closeness
        candidates.sort(key=lambda x: x[2])
        print(f"\n  {'Expression':>55}  {'Value':>12}  {'Î”':>8}  {'%err':>8}")
        print(f"  {'-'*55}  {'-'*12}  {'-'*8}  {'-'*8}")
        for val, name, diff, rel in candidates[:20]:
            print(f"  {name:>55}  {val:>12.4f}  {diff:>8.4f}  {rel*100:>7.3f}%")
    else:
        print(f"\n  No expressions within tolerance found.")

    return candidates


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 76)
    print("  Î±â»Â¹ FROM BERRY PHASE GEOMETRY")
    print("  Third derivation route through the zero point")
    print("=" * 76)
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Target: Î±â»Â¹ = {ALPHA_INV}")
    print()

    t0 = time.time()

    # Extract geometry
    geom = extract_geometry()

    # Search for Î±
    candidates = search_for_alpha(geom)

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
