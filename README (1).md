# Pentachoric Code Simulation

Computational validation of the pentachoric error correction code, a new quantum error correction framework based on dual-spinor (merkabit) architecture on the Eisenstein lattice ℤ[ω].

Accompanies: Stenberg, S. (2026). *The Merkabit: Dual-Spinor Architecture for Quantum Computation.*

## What this computes

The merkabit's error correction operates at three levels. This simulation validates all three:

- **Level 1 — Symmetric noise cancellation.** Two counter-rotating spinors cancel common-mode noise exactly (algebraic identity). Monte Carlo confirms 2–3× suppression at realistic symmetric noise fractions.

- **Level 2 — Pentachoric gate complementarity.** Five gates on a hexagonal lattice, each node missing one. Adjacent nodes must jointly cover all five. Errors break closure and are detected structurally. Exhaustive enumeration of all 78,125 gate assignments on the 7-node cell.

- **Level 3 — E₆ syndrome space.** Generates all 36 positive roots, verifies Coxeter structure, confirms the P₂₄ tensor product ring is Z₃-graded (error algebra).

The key result is the **dynamic simulation**: the ouroboros gate cycle rotates each node's gate schedule, giving each edge multiple independent closure checks per cycle. Detection jumps from 70% (static) to **95.1%** (dynamic), yielding **41–68× composite error suppression** with zero qubit overhead.

## Quick start

```bash
# Static simulation (all three levels)
python3 pentachoric_code_simulation.py

# Dynamic simulation (ouroboros gate rotation)
python3 dynamic_pentachoric_simulation.py
```

Requires Python 3 and NumPy. Runs in under 15 seconds.

## Key results

| | Static (τ=1) | Dynamic (τ≥5) | Paper prediction |
|---|---|---|---|
| Central node detection | 91.0% | 100.0% | > 95% |
| Peripheral node detection | 66.4% | 94.3% | ~ 90% |
| Overall detection | 69.9% | 95.1% | > 90% |
| Composite suppression | 7–11× | 41–68× | 20–70× |

The static model tests one frozen snapshot. The dynamic model accounts for the ouroboros gate rotation (Section 9.8 of the paper), where each node's absent gate cycles through all five values. The condition for dynamic protection is τ ≥ 5 gate steps — satisfied by a factor of 100–1,000 in all candidate platforms.

## Files

| File | Description |
|---|---|
| `pentachoric_code_simulation.py` | Static simulation: Levels 1–3, full enumeration, Monte Carlo, E₆ roots, P₂₄ tensor products |
| `dynamic_pentachoric_simulation.py` | Dynamic simulation: ouroboros gate rotation, τ-dependence, composite analysis |

## Licence

The simulation code is released under the MIT License.

The paper and appendices are © Selina Stenberg 2026, all rights reserved.
