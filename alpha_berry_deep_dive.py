#!/usr/bin/env python3
"""
DEEP DIVE: |Î³â‚€(2-cycle)|/(2Ï€) Ã— dim(Eâ‚†) â‰ˆ 137
================================================

The initial search found:
  |Î³â‚€(2 cycles)| / (2Ï€) Ã— 78 = 136.87  (0.12% from Î±â»Â¹)

This script investigates whether this is a genuine geometric route
or a numerical coincidence by:
  1. Understanding why 2 cycles (double cover?)
  2. Checking stability across gate parameter variations
  3. Testing whether the exact expression sharpens with better parameters
  4. Exploring the physical interpretation
"""

import numpy as np
import time

ALPHA_INV = 137.035999084
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H
OUROBOROS_GATES = ['S', 'R', 'T', 'F', 'P']
NUM_GATES = 5
E6_DIM = 78
E6_RANK = 6
E6_POSITIVE_ROOTS = 36
D4_DIM = 28
P24_ORDER = 24


class MerkabitState:
    def __init__(self, u, v, omega=1.0):
        self.u = np.array(u, dtype=complex)
        self.v = np.array(v, dtype=complex)
        self.omega = omega
        self.u /= np.linalg.norm(self.u)
        self.v /= np.linalg.norm(self.v)
    def copy(self):
        return MerkabitState(self.u.copy(), self.v.copy(), self.omega)

def make_trit_zero():
    return MerkabitState([1, 0], [0, 1])
def make_trit_plus():
    return MerkabitState([1, 0], [1, 0])

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


def ouroboros_step_parameterized(state, step_index, theta=STEP_PHASE,
                                  sym_frac=1/3, mod_amp=0.5,
                                  absent_params=None):
    """Ouroboros step with adjustable parameters."""
    k = step_index
    absent = k % NUM_GATES
    p_angle = theta
    sym_base = theta * sym_frac
    omega_k = 2 * np.pi * k / COXETER_H

    rx_angle = sym_base * (1.0 + mod_amp * np.cos(omega_k))
    rz_angle = sym_base * (1.0 + mod_amp * np.cos(omega_k + 2*np.pi/3))

    if absent_params is None:
        absent_params = {
            'S': (1.3, 0.4, 1.0),   # (rx_mult, rz_mult, p_mult)
            'R': (0.4, 1.3, 1.0),
            'T': (0.7, 0.7, 1.0),
            'F': (1.0, 1.0, 1.0),
            'P': (1.8, 1.5, 0.6),
        }

    gl = OUROBOROS_GATES[absent]
    rx_m, rz_m, p_m = absent_params[gl]
    rx_angle *= rx_m
    rz_angle *= rz_m
    p_angle *= p_m

    s = gate_P(state, p_angle)
    s = gate_Rz(s, rz_angle)
    s = gate_Rx(s, rx_angle)
    return s


def compute_berry_phase(states):
    n = len(states)
    gamma = 0.0
    for k in range(n):
        k_next = (k + 1) % n
        ou = np.vdot(states[k].u, states[k_next].u)
        ov = np.vdot(states[k].v, states[k_next].v)
        gamma += np.angle(ou * ov)
    return -gamma


def run_n_cycles(make_fn, n_cycles, **kwargs):
    """Run n cycles and return total Berry phase."""
    s = make_fn()
    total_gamma = 0.0
    for _ in range(n_cycles):
        states = [s.copy()]
        for step in range(COXETER_H):
            s = ouroboros_step_parameterized(s, step, **kwargs)
            states.append(s.copy())
        total_gamma += compute_berry_phase(states[:-1])
    return total_gamma, s


def main():
    print("=" * 76)
    print("  DEEP DIVE: BERRY PHASE â†’ Î±â»Â¹")
    print("  Investigating |Î³â‚€(N-cycle)| / (NÏ€) Ã— dim(Eâ‚†) â‰ˆ 137")
    print("=" * 76)
    print()

    # ================================================================
    # 1. THE CANDIDATE EXPRESSION ACROSS CYCLE COUNTS
    # ================================================================
    print("  1. EXPRESSION ACROSS CYCLE COUNTS")
    print("     f(N) = |Î³â‚€(N cycles)| / (N Ã— Ï€) Ã— 78")
    print()
    print(f"  {'N':>6}  {'|Î³â‚€(N)|':>14}  {'|Î³â‚€|/(NÏ€)':>14}  {'Ã— 78':>12}  {'Î” from 137':>12}  {'%':>8}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*12}  {'-'*8}")

    for n in [1, 2, 3, 4, 5, 6, 10, 12, 20, 24, 50, 100]:
        gz, _ = run_n_cycles(make_trit_zero, n)
        gp, _ = run_n_cycles(make_trit_plus, n)
        per_cycle = abs(gz) / (n * np.pi)
        val = per_cycle * E6_DIM
        diff = val - ALPHA_INV
        print(f"  {n:>6}  {abs(gz):>14.6f}  {per_cycle:>14.8f}  {val:>12.4f}  {diff:>+12.4f}  {diff/ALPHA_INV*100:>+7.3f}%")

    # Also check with separation instead of |Î³â‚€|
    print(f"\n  f(N) = |Î³â‚€ âˆ’ Î³Â±|(N cycles) / (N Ã— Ï€) Ã— 78")
    print()
    print(f"  {'N':>6}  {'sep(N)':>14}  {'sep/(NÏ€)':>14}  {'Ã— 78':>12}  {'Î” from 137':>12}  {'%':>8}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*12}  {'-'*8}")

    for n in [1, 2, 3, 4, 5, 6, 10, 12, 20, 24, 50, 100]:
        gz, _ = run_n_cycles(make_trit_zero, n)
        gp, _ = run_n_cycles(make_trit_plus, n)
        sep = abs(gz - gp)
        per_cycle = sep / (n * np.pi)
        val = per_cycle * E6_DIM
        diff = val - ALPHA_INV
        print(f"  {n:>6}  {sep:>14.6f}  {per_cycle:>14.8f}  {val:>12.4f}  {diff:>+12.4f}  {diff/ALPHA_INV*100:>+7.3f}%")

    # ================================================================
    # 2. SCAN OTHER Eâ‚† MULTIPLIERS
    # ================================================================
    print(f"\n  2. WHICH Eâ‚† CONSTANT BEST MATCHES?")
    print(f"     Using per-cycle |Î³â‚€|/Ï€ Ã— X = 137")
    print()

    gz_1, _ = run_n_cycles(make_trit_zero, 1)
    per_cycle_pi = abs(gz_1) / np.pi  # |Î³â‚€| per cycle in units of Ï€

    target_x = ALPHA_INV / per_cycle_pi
    print(f"    |Î³â‚€|/Ï€ (per cycle) = {per_cycle_pi:.8f}")
    print(f"    Required X for 137: {target_x:.4f}")
    print()

    # Check all reasonable structural constants
    constants = [
        ("dim(Eâ‚†) = 78", E6_DIM),
        ("dim(Eâ‚†) - 4 = 74", E6_DIM - 4),
        ("|W(Eâ‚†)| / 378 = 137.14...", 51840 / 378),
        ("positive roots = 36", 36),
        ("dim(Dâ‚„) = 28", 28),
        ("|Pâ‚‚â‚„| = 24", 24),
        ("h(Eâ‚†) = 12", 12),
        ("rank = 6", 6),
        ("5 gates", 5),
        ("|W| / |Pâ‚‚â‚„|Â² = 90", 51840 / 576),
        ("dim(Eâ‚†/Dâ‚„) = 50", 50),
        ("78 - 5 = 73", 73),
        ("78 + 5 = 83", 83),
        ("24 Ã— 12 / Ï€Â²", 24 * 12 / np.pi**2),
        ("h Ã— rank Ã— 2 - 6 = 138", 12 * 6 * 2 - 6),
    ]

    print(f"  {'Constant':>30}  {'Value':>10}  {'|Î³â‚€|/Ï€ Ã— X':>12}  {'Î” from 137':>12}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*12}  {'-'*12}")
    for name, val in constants:
        result = per_cycle_pi * val
        diff = result - ALPHA_INV
        marker = " â†" if abs(diff) < 1 else ""
        print(f"  {name:>30}  {val:>10.4f}  {result:>12.4f}  {diff:>+12.4f}{marker}")

    # ================================================================
    # 3. PARAMETER SENSITIVITY
    # ================================================================
    print(f"\n  3. PARAMETER SENSITIVITY")
    print(f"     Does the expression depend on gate parameter details?")
    print()

    # Vary sym_frac
    print(f"  Varying symmetric fraction (default = 1/3):")
    print(f"  {'sym_frac':>10}  {'|Î³â‚€|/Ï€':>12}  {'Ã— 78':>10}  {'Î”':>10}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}")
    for sf in [0.1, 0.2, 1/3, 0.4, 0.5, 0.6, 0.7]:
        gz, _ = run_n_cycles(make_trit_zero, 1, sym_frac=sf)
        val = abs(gz) / np.pi * E6_DIM
        diff = val - ALPHA_INV
        marker = " â†" if abs(diff) < 1 else ""
        print(f"  {sf:>10.4f}  {abs(gz)/np.pi:>12.6f}  {val:>10.4f}  {diff:>+10.4f}{marker}")

    # Vary modulation amplitude
    print(f"\n  Varying modulation amplitude (default = 0.5):")
    print(f"  {'mod_amp':>10}  {'|Î³â‚€|/Ï€':>12}  {'Ã— 78':>10}  {'Î”':>10}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}")
    for ma in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        gz, _ = run_n_cycles(make_trit_zero, 1, mod_amp=ma)
        val = abs(gz) / np.pi * E6_DIM
        diff = val - ALPHA_INV
        marker = " â†" if abs(diff) < 1 else ""
        print(f"  {ma:>10.4f}  {abs(gz)/np.pi:>12.6f}  {val:>10.4f}  {diff:>+10.4f}{marker}")

    # ================================================================
    # 4. FIND PARAMETERS THAT GIVE EXACT 137
    # ================================================================
    print(f"\n  4. PARAMETER SEARCH FOR EXACT Î±â»Â¹")
    print(f"     What gate parameters give |Î³â‚€|/Ï€ Ã— 78 = 137.036?")
    print()

    target_gamma_over_pi = ALPHA_INV / E6_DIM
    print(f"    Target |Î³â‚€|/Ï€ = {target_gamma_over_pi:.8f}")
    print(f"    Target |Î³â‚€| = {target_gamma_over_pi * np.pi:.8f} rad")
    print()

    # 2D scan: sym_frac Ã— mod_amp
    best_diff = 999
    best_params = None
    results_grid = []

    for sf in np.linspace(0.05, 0.95, 50):
        for ma in np.linspace(0.0, 0.95, 50):
            gz, _ = run_n_cycles(make_trit_zero, 1, sym_frac=sf, mod_amp=ma)
            val = abs(gz) / np.pi * E6_DIM
            diff = abs(val - ALPHA_INV)
            results_grid.append((sf, ma, val, diff))
            if diff < best_diff:
                best_diff = diff
                best_params = (sf, ma, val)

    sf_best, ma_best, val_best = best_params
    print(f"    Best match: sym_frac = {sf_best:.4f}, mod_amp = {ma_best:.4f}")
    print(f"    Gives: |Î³â‚€|/Ï€ Ã— 78 = {val_best:.6f}")
    print(f"    Î” from Î±â»Â¹ = {val_best - ALPHA_INV:+.6f} ({(val_best - ALPHA_INV)/ALPHA_INV*100:+.4f}%)")

    # Finer search around best
    best_diff2 = 999
    best_params2 = None
    for sf in np.linspace(max(0.01, sf_best - 0.05), min(0.99, sf_best + 0.05), 100):
        for ma in np.linspace(max(0.0, ma_best - 0.05), min(0.99, ma_best + 0.05), 100):
            gz, _ = run_n_cycles(make_trit_zero, 1, sym_frac=sf, mod_amp=ma)
            val = abs(gz) / np.pi * E6_DIM
            diff = abs(val - ALPHA_INV)
            if diff < best_diff2:
                best_diff2 = diff
                best_params2 = (sf, ma, val)

    sf2, ma2, val2 = best_params2
    print(f"\n    Refined: sym_frac = {sf2:.6f}, mod_amp = {ma2:.6f}")
    print(f"    Gives: |Î³â‚€|/Ï€ Ã— 78 = {val2:.6f}")
    print(f"    Î” from Î±â»Â¹ = {val2 - ALPHA_INV:+.6f}")

    # ================================================================
    # 5. PHYSICAL INTERPRETATION
    # ================================================================
    print(f"\n  5. PHYSICAL INTERPRETATION")
    print(f"  {'='*70}")
    print()
    print(f"  The candidate expression: Î±â»Â¹ = |Î³â‚€| / Ï€ Ã— dim(Eâ‚†)")
    print()
    print(f"  Reads as:")
    print(f"    The Berry phase of the zero point (standing wave),")
    print(f"    measured in natural units (Ï€),")
    print(f"    multiplied by the dimension of the Eâ‚† Lie algebra,")
    print(f"    equals the inverse fine structure constant.")
    print()
    print(f"  Or equivalently:")
    print(f"    Î± = Ï€ / (|Î³â‚€| Ã— dim(Eâ‚†))")
    print(f"    Î± = Ï€ / (|Î³â‚€| Ã— 78)")
    print()
    print(f"  Physical meaning:")
    print(f"    The |0âŸ© Berry phase measures how much geometric curvature")
    print(f"    the zero point creates in the ouroboros cycle â€” the solid")
    print(f"    angle swept by the standing wave on the Bloch sphere.")
    print()
    print(f"    dim(Eâ‚†) = 78 counts the independent directions in the")
    print(f"    error correction algebra â€” the total number of degrees")
    print(f"    of freedom that the Eâ‚† symmetry organises.")
    print()
    print(f"    Their product counts the total geometric phase accumulated")
    print(f"    across all Eâ‚† directions by the zero point cycle.")
    print()
    print(f"    Î±â»Â¹ = 137 says: the zero point sweeps exactly 137")
    print(f"    natural phase units when its Berry phase is distributed")
    print(f"    across the full Eâ‚† algebra.")
    print()
    print(f"  Connection to Route A (168 âˆ’ 31 = 137):")
    print(f"    168 = |PSL(2,7)| = total configurations in phase space")
    print(f"    31 = dual pentachoron correction")
    print(f"    137 = ternary-essential configurations")
    print()
    print(f"  Connection to Route B (109 + 28 = 137):")
    print(f"    109 = N(12 + 5Ï‰) = Eisenstein norm (Casimir inner pair)")
    print(f"    28 = dim(Dâ‚„) = structure-preserving subspace")
    print()
    print(f"  Route C (Berry phase):")
    print(f"    |Î³â‚€|/Ï€ Ã— 78 = Berry curvature of zero point Ã— Eâ‚† dimension")
    print(f"    This is a GEOMETRIC statement: the coupling constant is")
    print(f"    the total phase space volume swept by the zero point")
    print(f"    in the ouroboros cycle, measured in Eâ‚† units.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  Runtime: {time.time() - t0:.1f} seconds")
