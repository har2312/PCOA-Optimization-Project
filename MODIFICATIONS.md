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

*More modifications will be added below as they are implemented.*
