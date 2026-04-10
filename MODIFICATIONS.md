# PCOA Modifications Log

This document tracks every modification made to the base Pine Cone Optimization Algorithm (PCOA), explaining *what* was changed, *why*, and the *expected effect*.

---

## Base Algorithm (PCOA)

The original PCOA by Anaraki & Farzin (2024) consists of:
- **Pollination phase** (exploration) — Algorithm 1 with SHADE-like adaptive memory + Eq 12 alternative
- **Animal dispersal phase** (exploitation) — Algorithm 2 with 4 operators (QP, Lévy, tree-based, centroid-based)
- **Shrinking boundaries** (Eqs 7-11) — search space contracts around the best solution

### Known weaknesses (from baseline tests):
| Function | Base PCOA Result | Issue |
|----------|-----------------|-------|
| Sphere (unimodal) | `3.58e-12` | ✅ Excellent |
| Rastrigin (multimodal) | `1.69e+01` | ❌ Trapped in local optima |
| Rosenbrock (valley) | `1.33e+03` | ❌ Slow convergence in narrow valleys |

The base PCOA struggles with **multimodal functions** (local optima trapping) and **narrow valleys** (insufficient step diversity).

---

## Modification 1: Lévy Flight in Pollination Phase

**Commit**: `Add Lévy flight to pollination phase`
**File**: `src/mpcoa.py`

### What was changed

In Algorithm 1 (wind pollination), the position update equation:

**Before (base PCOA, line 26):**
```
CX_new = CX + W1 × (Tbest - CX_r1) + W2 × (CX_r1 - Tpopall_r2)
```

**After (modified):**
```
CX_new = CX + W1 × Lévy(β) × (Tbest - CX_r1) + W2 × (CX_r1 - Tpopall_r2)
```

Where Lévy(β) is generated using Mantegna's algorithm:
```
Lévy(β) = u / |v|^(1/β)
u ~ N(0, σ²),  v ~ N(0, 1)
σ = [Γ(1+β) × sin(πβ/2) / (Γ((1+β)/2) × β × 2^((β-1)/2))]^(1/β)
β = 1.5
```

The same Lévy scaling is applied to the second mutation strategy (lines 28-29).

### Why this modification

The base PCOA uses **uniform random numbers** to scale the difference vectors in pollination. This produces steps of roughly uniform magnitude — fine for local search, but:

1. **Uniform steps can't escape deep local optima** — they don't generate the occasional large jump needed to leap out of a basin
2. **Real pollen dispersal follows Lévy patterns** — wind-blown particles in nature follow heavy-tailed distributions, not uniform ones
3. **Lévy flights are mathematically optimal for blind search** — proven to be the most efficient random search strategy for sparse targets (Viswanathan et al., 1999)

### Expected effect

| Aspect | Before (uniform) | After (Lévy) |
|--------|-----------------|--------------|
| Step size distribution | Roughly equal magnitude | Mostly small, occasionally very large |
| Local search | Good | Good (small Lévy steps behave like local search) |
| Escaping local optima | Poor | **Much better** (large jumps can escape basins) |
| Multimodal functions | Struggles | Should improve significantly |
| Convergence speed | Normal | May be slightly slower on unimodal (trade-off for robustness) |

### How it helps on specific function types

- **Rastrigin** (many local optima): Large Lévy jumps help escape the 10^n local minima
- **Ackley** (flat regions with deep central minimum): Heavy-tailed steps explore flat areas more effectively
- **Composition functions** (CEC F23-F30): Multiple basins require diverse step sizes to navigate between them

---

## Modification 2: Chaotic Map Initialization (Logistic Map)

**Commit**: `Add chaotic map initialization`
**File**: `src/mpcoa.py`

### What was changed

The population initialization now uses the **Logistic Map** chaotic sequence instead of pseudo-random numbers.

**Before (base PCOA):**
```python
cone_pos[idx] = LbS + np.random.rand(dim) * (UbS - LbS)
```

**After (modified):**
```python
chaos = logistic_map(pop_size, dim)    # chaotic sequence in [0, 1]
cone_pos[idx] = LbS + chaos[idx] * (UbS - LbS)
```

Where the Logistic Map is:
```
x_{n+1} = r × x_n × (1 - x_n),  r = 4
```
Starting from a random seed in (0.1, 0.9) per dimension.

### Why this modification

Pseudo-random number generators (like `np.random.rand()`) can produce **clusters and gaps** in the initial population, especially in high dimensions. This means:

1. **Some regions of the search space may have no initial agents** — if the global optimum is there, it may never be found
2. **Multiple agents may start very close together** — wasting evaluations on redundant search
3. **The Logistic Map at r=4 is ergodic** — it visits every sub-interval of [0,1] with equal frequency, giving guaranteed uniform coverage

### Expected effect

| Aspect | Before (pseudo-random) | After (chaotic) |
|--------|----------------------|-----------------|
| Space coverage | Probabilistically uniform, but may cluster | Deterministically more uniform |
| Initial population quality | Variable | More consistently spread |
| Reproducibility | Seed-dependent | More robust across seeds |
| High-dimensional spaces | Gaps increase with dimension | Chaotic sequences maintain coverage |
| Computational cost | Same | Same (negligible overhead) |

### How it helps

- **Better starting positions** → algorithm finds good regions faster in early generations
- **Reduced sensitivity to random seed** → more consistent results across runs (lower std)
- **Particularly beneficial for narrow-basin functions** (Rosenbrock, Schwefel) where initial placement near the valley matters

---

## Modification 3: Nonlinear Adaptive Boundary Decay

**Commit**: `Add nonlinear boundary decay`
**File**: `src/mpcoa.py`

### What was changed

The boundary shrinking weight `W` (Eq 9) now uses a **cosine-based nonlinear curve** instead of a linear ramp.

**Before (base PCOA, Eq 9):**
```
W = min(fes / max_fes, 0.5)
```
This is a straight line from 0 to 0.5 — boundaries shrink at a constant rate.

**After (modified):**
```
t = fes / max_fes
W = min(0.5 × (1 − cos(π × t)), 0.5)
```

The cosine curve produces an **S-shaped** transition:
- **Early phase (t ≈ 0)**: W ≈ 0 → boundaries stay wide → full exploration
- **Mid phase (t ≈ 0.5)**: W grows fastest → rapid transition to exploitation
- **Late phase (t ≈ 1)**: W ≈ 0.5 → boundaries stabilize → fine-tuning

### Why this modification

The base PCOA's linear boundary shrinking has a fundamental problem:

1. **Too aggressive early on** — at t=0.2, the linear W is already 0.2, shrinking bounds by 20%. This can cut off unexplored regions before the algorithm finds promising areas
2. **Too slow at the end** — the linear ramp provides equal shrinkage rate at all stages, giving no special fine-tuning phase
3. **No adaptation to problem difficulty** — hard problems need more exploration time; the cosine curve naturally provides this with its slow start

### Expected effect

| Phase | Linear (base) | Cosine (modified) | Benefit |
|-------|--------------|-------------------|---------|
| t = 0.1 | W = 0.10 (10% shrink) | W = 0.02 (2% shrink) | **8× wider** exploration early |
| t = 0.3 | W = 0.30 (30% shrink) | W = 0.15 (15% shrink) | **2× wider** — still exploring |
| t = 0.5 | W = 0.50 (max) | W = 0.50 (max) | Same at midpoint |
| t = 0.7 | W = 0.50 (max) | W = 0.50 (max) | Same — full exploitation |

- **Multimodal functions**: Longer exploration phase → better chance of finding the global basin
- **Unimodal functions**: No harm — the late-phase exploitation is equally strong

## Modification 4: Dynamic Lévy Flight Scaling & Stagnation Burst (Fix A)

**Commit**: `Add dynamic Lévy scaling and stagnation detection`
**File**: `src/mpcoa.py`

### What was changed

In the original v1 modification, Lévy flights were applied with a constant magnitude multiplier. In v2, the Lévy step magnitude is now **dynamically scaled** across the iteration budget:
1. **Cosine Annealing Schedule:** The scaling factor `alpha` decays nonlinearly from a maximum (`levy_alpha_max` = 1.5) to a minimum (`levy_alpha_min` = 0.01) as evaluations progress.
2. **Stagnation Burst:** A `stag_counter` monitors the global best fitness. If fitness fails to improve for 30 consecutive iterations (`stag_limit`), the algorithm forces a temporary "burst" by multiplying the Lévy magnitude by `stag_boost` (7.0) to violently eject trapped populations.

### Why this modification

The stagnation problem observed on multi-modal functions like CEC2014 F4 (where variance dropped to near zero) occurs because constant large Lévy flights disrupt fine-tuning convergence, while strictly small flights fail to escape deep basins. 
- Early iterations require massive explosive jumps.
- Late iterations require microscopic precision steps.
- If completely stuck, the algorithm needs a "panic button" to break out of a hopeless local crater.

### Expected effect

| Phase | Before (Constant Lévy) | After (Dynamic Levy + Burst) |
|-------|-----------------------|------------------------------|
| Early Search | Medium random jumps | Explosive, space-spanning jumps |
| Late Search | Disruptive jumps | Micro-fine precision tuning |
| Stagnation Event | Permanent entrapment | Violent ejection to new basins |
| CEC2014 F4 Variance | Near zero | High — guaranteed continued exploration |

---

## Modification 5: Restricted Segment Communication (Fix B)

**Commit**: `Add exponential decay to segment communication`
**File**: `src/mpcoa.py`

### What was changed

The cooperative mechanism sharing information between different pine trees/segments was heavily restricted.
1. **Exponential Decay Probability:** Segments no longer share information 100% of the time. The probability of communication decays exponentially ($\approx e^{-6t}$), starting at ~90% and ending at ~5%.
2. **Gated Acceptance:** A segment only adopts the global best if it probabilistically decides to; otherwise, it strictly adheres to its own local segment best.

### Why this modification

The "Parent Problem" caused MPCOA to underperform the base PCOA on functions like CEC2014 F1. Because the parallel populations were sharing information every single iteration, they were converging onto each other's sub-optimal vectors far too soon. 
Restricting communication ensures parallel populations evolve independently and maintain high diversity, only sharing breakthroughs when mathematically advantageous.

### Expected effect

- **Diversity Maintenance:** Parallel sub-populations will no longer violently collapse into a single point prematurely.
- **Improved F1 Performance:** Independent exploration paths prevent the algorithm from being dragged down by early, false-positive global bests.

---

## Modification 6: Personal-Best (pBest) Memory & Swarm Blending (Fix C)

**Commit**: `Add pBest individual memory and PSO blending`
**File**: `src/mpcoa.py`

### What was changed

Inspired by Particle Swarm Optimization (PSO), every individual cone now strictly maintains a historical record of its personal-best position (`pBest`) and fitness. The Animal Dispersal equations (exploitation phase) were blended with PSO velocity terms:
```python
cognitive_term = c1 * r1 * (pBest - current_pos)
social_term = c2 * r2 * (gBest - current_pos)
```
The cognitive coefficient `c1` decays linearly (2.0 → 0.5) while the social coefficient `c2` grows (0.5 → 2.0).

### Why this modification

Nature-inspired leaders like PSO and SCA continuously dictate performance benchmarks because they exploit independent individual memory. PCOA originally only had a single "global best" and "segment best" memory. Without `pBest`, individual cones had zero "fine-tuning" memory precision. Blending PSO mechanics into the animal dispersal phase mathematically secures MPCOA the exact precision that makes PSO S-tier.

### Expected effect

- **Superior Exploitation:** Deep, surgical fine-tuning on unimodal functions and exact basin extraction.
- **Leaderboard Dominance:** Closing the precision gap against PSO and SCA on functions where exact coordinate precision inherently dictates the final fitness score.

---

*More modifications will be added below as they are implemented.*
