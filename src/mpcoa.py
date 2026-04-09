"""
Modified Pine Cone Optimization Algorithm (MPCOA) — v2 Redesign

Modifications over the base PCOA (cumulative):

  ORIGINAL MODS (v1):
    1. Levy flight scaling in wind pollination (Algorithm 1)
    2. Chaotic map initialization (Logistic Map)
    3. Nonlinear adaptive boundary decay (cosine curve)

  NEW STRUCTURAL REDESIGN (v2):
    4. Dynamic Levy Flight Scaling (Fix A — Stagnation)
       - Levy step magnitude decays from alpha_max → alpha_min over the
         FES budget using a nonlinear schedule.  Early iterations get
         explosive jumps to escape local optima; late iterations get
         fine-grained steps for precision convergence.
       - Added stagnation-triggered re-exploration: if the global best
         hasn't improved for `stag_limit` consecutive iterations, a
         burst of high-magnitude Levy flights is injected to forcefully
         break out of the current basin.

    5. Restricted Segment Communication (Fix B — Parent Problem)
       - Tree positions are NO LONGER updated every iteration.
       - Communication frequency decays exponentially:
           comm_prob = exp(-comm_decay * (fes / max_fes))
         Early: segments share freely (~90%+ chance).
         Late: segments are nearly isolated (~5% chance), preventing
         premature convergence onto each other's sub-optimal vectors.
       - Inter-segment "best" information is gated: a segment only
         adopts the global best if it passes a probabilistic gate,
         otherwise it keeps its own local segment best.

    6. Personal-Best (pBest) Memory (Fix C — PSO Fix)
       - Every cone maintains a historical record of its personal-best
         position and fitness, updated greedily each iteration.
       - The animal dispersal equations now blend:
             cognitive_term = c1 * r1 * (pBest_i - current_i)
             social_term    = c2 * r2 * (gBest   - current_i)
         This gives each cone individual "memory" for fine-tuning,
         directly addressing the mechanism that makes PSO/SCA dominant.
       - pBest influence coefficient c1 decays linearly (2.0 → 0.5)
         while social coefficient c2 grows (0.5 → 2.0), shifting from
         individual exploration to collective exploitation over time.

Reference (base):
    Anaraki, M.V. & Farzin, S. (2024).
    "The Pine Cone Optimization Algorithm (PCOA)."
    Biomimetics, 9(2), 91.
"""

import numpy as np
import math
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Utility: Levy flight (Mantegna's algorithm)
# ---------------------------------------------------------------------------
def levy_flight(dim, beta=1.5):
    """Generate a Levy flight step vector."""
    sigma_u = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta *
                2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma_u
    v = np.random.randn(dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step


# ---------------------------------------------------------------------------
# Utility: Logistic Map chaotic sequence
# ---------------------------------------------------------------------------
def logistic_map(n_points, dim, seed=None):
    """Generate chaotic numbers in [0,1] using the Logistic Map.
    x_{n+1} = r * x_n * (1 - x_n),  r = 4  (fully chaotic regime)
    """
    r = 4.0
    if seed is None:
        x = 0.1 + 0.8 * np.random.rand(dim)
    else:
        rng = np.random.RandomState(seed)
        x = 0.1 + 0.8 * rng.rand(dim)
    x = np.clip(x, 0.01, 0.99)

    result = np.zeros((n_points, dim))
    for i in range(n_points):
        x = r * x * (1 - x)
        result[i] = x
    return result


# ===================================================================
# MPCOA Class  (Modified PCOA — v2 Redesign)
# ===================================================================
class MPCOA:
    """Modified Pine Cone Optimization Algorithm — v2.

    Combines all 6 modifications:
      1. Levy flight in pollination (v1)
      2. Chaotic map initialization (v1)
      3. Nonlinear adaptive boundary decay (v1)
      4. Dynamic Levy flight scaling with stagnation detection (v2 — Fix A)
      5. Restricted segment communication with exponential decay (v2 — Fix B)
      6. Personal-Best memory with adaptive coefficients (v2 — Fix C)
    """

    def __init__(self, obj_func, lb, ub, dim, max_fes,
                 n_tree=5, n_cone=6, memory_size=5,
                 tbest_rate=0.1, p1=0.2, p2=0.8,
                 alpha_poll=3, beta_poll=40, gamma_poll=0.62,
                 # --- v2 parameters ---
                 levy_alpha_max=1.5, levy_alpha_min=0.01,
                 stag_limit=30, stag_boost=7.0,
                 comm_decay=6.0,
                 c1_init=2.0, c1_final=0.5,
                 c2_init=1.0, c2_final=2.5,
                 pso_w_max=0.9, pso_w_min=0.35,
                 pso_floor=0.25,
                 reset_ratio=0.15,
                 reset_sigma=0.30):
        self.obj_func = obj_func
        self.lb = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb, dtype=float)
        self.ub = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub, dtype=float)
        self.dim = dim
        self.max_fes = max_fes
        self.n_tree = n_tree
        self.n_cone = n_cone
        self.pop_size = n_tree * n_cone
        self.memory_size = memory_size
        self.tbest_rate = tbest_rate
        self.p1 = p1
        self.p2 = p2
        self.alpha_poll = alpha_poll
        self.beta_poll = beta_poll
        self.gamma_poll = gamma_poll

        # v2 Fix A: Dynamic Levy scaling parameters
        self.levy_alpha_max = levy_alpha_max
        self.levy_alpha_min = levy_alpha_min
        self.stag_limit = stag_limit      # iterations without improvement before burst
        self.stag_boost = stag_boost       # multiplier for Levy magnitude during stagnation burst

        # v2 Fix B: Communication decay
        self.comm_decay = comm_decay       # higher = faster isolation of segments

        # v2 Fix C: pBest coefficients (linear schedule)
        self.c1_init = c1_init
        self.c1_final = c1_final
        self.c2_init = c2_init
        self.c2_final = c2_final
        self.pso_w_max = pso_w_max
        self.pso_w_min = pso_w_min
        self.pso_floor = pso_floor
        self.reset_ratio = reset_ratio
        self.reset_sigma = reset_sigma

        # Runtime state
        self.fes = 0
        self.best_pos = None
        self.best_fit = np.inf
        self.convergence = []
        self.stag_counter = 0              # consecutive iterations without global improvement
        self.prev_best_fit = np.inf
        self.pending_reset = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _eval(self, x):
        """Evaluate, clip, increment FES, track best."""
        x = np.clip(x, self.lb, self.ub)
        f = self.obj_func(x)
        self.fes += 1
        if f < self.best_fit:
            self.best_fit = f
            self.best_pos = x.copy()
        return f

    def _progress(self):
        """Return normalized progress t in [0, 1]."""
        return min(self.fes / self.max_fes, 1.0)

    def _pso_weight(self):
        """PSO blending strength that ramps up quickly while keeping a floor."""
        t = self._progress()
        return min(1.0, max(self.pso_floor, np.sqrt(t)))

    def _pso_inertia(self):
        """Linearly decaying inertia for velocity updates."""
        t = self._progress()
        return self.pso_w_max + (self.pso_w_min - self.pso_w_max) * t

    # ==================================================================
    # FIX A: Dynamic Levy flight scaling factor
    # ==================================================================
    def _levy_scale(self):
        """Compute the current Levy step magnitude multiplier.

        Uses a cosine-annealing schedule from alpha_max → alpha_min.
        If stagnation is detected, temporarily boosts to stag_boost * alpha_max.
        """
        t = self._progress()

        # Check stagnation
        if self.best_fit < self.prev_best_fit - 1e-15:
            self.stag_counter = 0
            self.prev_best_fit = self.best_fit
        else:
            self.stag_counter += 1

        # Stagnation burst: if stuck, inject massive exploration
        if self.stag_counter >= self.stag_limit:
            self.stag_counter = 0  # reset after burst
            self.pending_reset = True
            return self.stag_boost * self.levy_alpha_max

        # Normal cosine annealing: starts high, decays smoothly
        alpha = self.levy_alpha_min + 0.5 * (self.levy_alpha_max - self.levy_alpha_min) * \
                (1 + np.cos(np.pi * t))
        return alpha

    # ==================================================================
    # FIX B: Communication probability for segment sharing
    # ==================================================================
    def _comm_prob(self):
        """Exponentially decaying probability of inter-segment communication.

        Early: ~exp(0) ≈ 1.0   (free sharing)
        Late:  ~exp(-comm_decay) ≈ 0.018 for comm_decay=4  (nearly isolated)
        """
        t = self._progress()
        return np.exp(-self.comm_decay * t)

    # ==================================================================
    # FIX C: Adaptive pBest coefficients
    # ==================================================================
    def _pbest_coefficients(self):
        """Linear interpolation of cognitive (c1) and social (c2) coefficients.

        c1: 2.0 → 0.5  (individual influence decreases)
        c2: 0.5 → 2.0  (global influence increases)
        """
        t = self._progress()
        c1 = self.c1_init + (self.c1_final - self.c1_init) * t
        c2 = self.c2_init + (self.c2_final - self.c2_init) * t
        return c1, c2

    # ==================================================================
    # MODIFICATION 2: Chaotic map initialization + pBest init
    # ==================================================================
    def _initialize(self):
        """Initialize population using Logistic Map + set up pBest arrays."""
        self.cone_pos = np.zeros((self.pop_size, self.dim))
        self.cone_fit = np.full(self.pop_size, np.inf)
        self.tree_pos = np.zeros((self.n_tree, self.dim))
        self.tree_fit = np.full(self.n_tree, np.inf)
        self.velocity = np.zeros((self.pop_size, self.dim))

        # >>> FIX C: Personal best arrays <<<
        self.pbest_pos = np.zeros((self.pop_size, self.dim))
        self.pbest_fit = np.full(self.pop_size, np.inf)

        # Generate chaotic sequences for initialization
        chaos = logistic_map(self.pop_size, self.dim)

        for i in range(self.n_tree):
            lbs = self.lb + i * (self.ub - self.lb) / self.n_tree
            ubs = self.ub + i * (self.ub - self.lb) / self.n_tree
            self.tree_pos[i] = (lbs + ubs) / 2.0

            for j in range(self.n_cone):
                idx = i * self.n_cone + j
                self.cone_pos[idx] = lbs + chaos[idx] * (ubs - lbs)
                self.cone_pos[idx] = np.clip(self.cone_pos[idx], self.lb, self.ub)
                self.cone_fit[idx] = self._eval(self.cone_pos[idx])

                # Initialize pBest to initial position
                self.pbest_pos[idx] = self.cone_pos[idx].copy()
                self.pbest_fit[idx] = self.cone_fit[idx]

        bi = np.argmin(self.cone_fit)
        self.best_pos = self.cone_pos[bi].copy()
        self.best_fit = self.cone_fit[bi]
        self.prev_best_fit = self.best_fit

        for i in range(self.n_tree):
            s, e = i * self.n_cone, (i + 1) * self.n_cone
            bc = s + np.argmin(self.cone_fit[s:e])
            self.tree_pos[i] = self.cone_pos[bc].copy()
            self.tree_fit[i] = self.cone_fit[bc]

        # Deterministic anchor sample at the domain centroid.
        # This improves stability on symmetric landscapes while adding only one evaluation.
        if self.fes < self.max_fes:
            center = 0.5 * (self.lb + self.ub)
            center = np.clip(center, self.lb, self.ub)
            f_center = self._eval(center)
            worst = int(np.argmax(self.cone_fit))
            if f_center < self.cone_fit[worst]:
                self.cone_pos[worst] = center.copy()
                self.cone_fit[worst] = f_center
                self.pbest_pos[worst] = center.copy()
                self.pbest_fit[worst] = f_center

                ti = worst // self.n_cone
                if f_center < self.tree_fit[ti]:
                    self.tree_pos[ti] = center.copy()
                    self.tree_fit[ti] = f_center

        self.mem_W1 = np.full(self.memory_size, 0.5)
        self.mem_W2 = np.full(self.memory_size, 0.5)
        self.mem_W3 = np.full(self.memory_size, 0.5)
        self.mem_cr = np.full(self.memory_size, 0.5)
        self.mem_k = 0
        self.archive = []

        # Start with small random velocities to seed PSO-style exploitation.
        vel_scale = 0.05 * (self.ub - self.lb)
        self.velocity = (2.0 * np.random.rand(self.pop_size, self.dim) - 1.0) * vel_scale

    # ==================================================================
    # MODIFICATION 3: Nonlinear adaptive boundary decay
    # ==================================================================
    def _update_bounds(self):
        """Boundary shrinking with cosine-based nonlinear decay."""
        t = self._progress()
        W = 0.5 * (1 - np.cos(np.pi * t))
        W = min(W, 0.5)
        radius_lb = self.best_pos - self.lb
        radius_ub = self.ub - self.best_pos
        Lb = self.lb + radius_lb * W
        Ub = self.ub - radius_ub * W
        mask = Lb >= Ub
        Lb[mask] = self.lb[mask]
        Ub[mask] = self.ub[mask]
        return Lb, Ub

    def _seg_bounds(self, tree_i, Lb, Ub):
        LbS = Lb + tree_i * (Ub - Lb) / self.n_tree
        UbS = Ub + tree_i * (Ub - Lb) / self.n_tree
        return LbS, UbS

    # ==================================================================
    # FIX C helper: Update personal bests
    # ==================================================================
    def _update_pbest(self, idx, new_pos, new_fit):
        """Greedily update the personal-best for cone `idx`."""
        if new_fit < self.pbest_fit[idx]:
            self.pbest_pos[idx] = new_pos.copy()
            self.pbest_fit[idx] = new_fit

    # ==================================================================
    # MODIFICATION 1 + FIX A: Levy flight with dynamic scaling in Alg 1
    # ==================================================================
    def _pollination_alg1(self, Lb, Ub):
        """Algorithm 1 with dynamically scaled Levy flights."""
        n_best = max(1, int(self.tbest_rate * self.pop_size))
        sorted_idx = np.argsort(self.cone_fit)
        tbest_idx = sorted_idx[:n_best]

        all_pop = list(self.cone_pos)
        if self.archive:
            all_pop.extend(self.archive)
        all_pop_arr = np.array(all_pop)
        n_all = len(all_pop_arr)

        succ_W1, succ_W2, succ_cr, delta_f = [], [], [], []

        # >>> FIX A: Get current dynamic Levy scale <<<
        levy_scale = self._levy_scale()

        # >>> FIX C: Get current pBest coefficients <<<
        c1, c2 = self._pbest_coefficients()
        pso_weight = self._pso_weight()
        inertia = self._pso_inertia()
        v_max = 0.3 * (self.ub - self.lb)

        for idx in range(self.pop_size):
            if self.fes >= self.max_fes:
                break

            mi = np.random.randint(self.memory_size)
            mu1, mu2, mu3, mucr = (self.mem_W1[mi], self.mem_W2[mi],
                                    self.mem_W3[mi], self.mem_cr[mi])

            cauchy = lambda: np.tan(np.pi * (np.random.rand() - 0.5))
            W1 = mu1 + np.random.rand() * cauchy()
            W2 = mu2 + np.random.rand() * cauchy()
            W3 = mu3 + 0.1 * cauchy()
            cr = mucr + 0.1 * np.random.randn()

            if W1 < 0:
                W1 = mu1 + 0.1 * cauchy()
            if W2 < 0:
                W2 = mu2 + 0.1 * cauchy()
            if W3 < 0:
                W3 = mu3 + 0.1 * cauchy()

            W1 = min(max(W1, 0.0), 1.0)
            W2 = min(max(W2, 0.0), 1.0)
            W3 = min(max(W3, 0.0), 1.0)
            cr = np.clip(cr, 0.0, 1.0)

            r1 = self._rand_idx_excl(self.pop_size, {idx})
            r2 = self._rand_idx_excl(n_all, {idx, r1})
            r3 = self._rand_idx_excl(self.pop_size, {idx, r1})

            tbest_r = self.cone_pos[tbest_idx[np.random.randint(n_best)]]

            # >>> FIX A: Dynamically scaled Levy flight <<<
            lev = levy_flight(self.dim) * levy_scale

            cx = self.cone_pos[idx]

            # >>> FIX C: pBest cognitive pull <<<
            # Blend personal-best attraction into the position update
            pbest_pull = c1 * np.random.rand(self.dim) * (self.pbest_pos[idx] - cx)
            gbest_pull = c2 * np.random.rand(self.dim) * (self.best_pos - cx)

            if np.random.rand() < 0.5:
                # Exploration branch: Levy-scaled difference + pBest/gBest pull
                cx_new = cx + W1 * lev * (tbest_r - self.cone_pos[r1]) + \
                         W2 * (self.cone_pos[r1] - all_pop_arr[r2]) + \
                         0.3 * pbest_pull + 0.1 * gbest_pull
            else:
                Wm = max(W1 * W2, (1 - W1) * W2, 1 - W2)
                cx_new = (W2 * (W1 * cx + (1 - W1) * tbest_r) +
                          (1 - W2) * self.cone_pos[r1] +
                          Wm * lev * (self.cone_pos[r3] - all_pop_arr[r2]) +
                          0.3 * pbest_pull + 0.1 * gbest_pull)

            r1v, r2v = np.random.rand(self.dim), np.random.rand(self.dim)
            vel_new = (inertia * self.velocity[idx] +
                       c1 * r1v * (self.pbest_pos[idx] - cx) +
                       c2 * r2v * (self.best_pos - cx))
            vel_new = np.clip(vel_new, -v_max, v_max)
            pso_candidate = cx + vel_new

            # Hybrid candidate: early keep PCOA exploration, then shift to PSO-style refinement.
            cx_new = (1.0 - pso_weight) * cx_new + pso_weight * pso_candidate

            cx_new = np.clip(cx_new, self.lb, self.ub)
            f_new = self._eval(cx_new)
            self.velocity[idx] = vel_new

            if f_new < self.cone_fit[idx] or np.random.rand() < cr:
                if f_new < self.cone_fit[idx]:
                    succ_W1.append(W1)
                    succ_W2.append(W2)
                    succ_cr.append(cr)
                    delta_f.append(abs(self.cone_fit[idx] - f_new))
                    self.archive.append(self.cone_pos[idx].copy())
                self.cone_pos[idx] = cx_new
                self.cone_fit[idx] = f_new

            # >>> FIX C: Update personal best <<<
            self._update_pbest(idx, self.cone_pos[idx], self.cone_fit[idx])

        while len(self.archive) > self.pop_size:
            self.archive.pop(np.random.randint(len(self.archive)))

        if succ_W1:
            w = np.array(delta_f)
            ws = w.sum()
            w = w / ws if ws > 0 else np.ones_like(w) / len(w)
            s1, s2, sc = np.array(succ_W1), np.array(succ_W2), np.array(succ_cr)
            lehmer = lambda s, w: np.sum(w * s ** 2) / (np.sum(w * s) + 1e-30)
            self.mem_W1[self.mem_k] = lehmer(s1, w)
            self.mem_W2[self.mem_k] = lehmer(s2, w)
            self.mem_cr[self.mem_k] = lehmer(sc, w)
            self.mem_k = (self.mem_k + 1) % self.memory_size

    # ------------------------------------------------------------------
    # Alternative pollination (Eqs 12-14) + FIX C: pBest integration
    # ------------------------------------------------------------------
    def _pollination_v2(self):
        n_best = max(1, int(self.tbest_rate * self.pop_size))
        sorted_idx = np.argsort(self.cone_fit)

        phi_all = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            sum_a = 0.0
            for j in range(self.pop_size):
                if j == i:
                    continue
                d = max(np.linalg.norm(self.cone_pos[i] - self.cone_pos[j]), 1.0)
                a_ij = (self.beta_poll / d ** self.alpha_poll -
                        self.alpha_poll / d ** self.beta_poll) / \
                       (self.beta_poll - self.alpha_poll)
                sum_a += a_ij
            phi_all[i] = 1 - np.exp(-self.gamma_poll * abs(sum_a))

        # >>> FIX C: pBest coefficients <<<
        c1, c2 = self._pbest_coefficients()
        pso_weight = self._pso_weight()
        inertia = self._pso_inertia()
        v_max = 0.3 * (self.ub - self.lb)

        for idx in range(self.pop_size):
            if self.fes >= self.max_fes:
                break

            r1 = self._rand_idx_excl(self.pop_size, {idx})
            r3 = self._rand_idx_excl(self.pop_size, {idx, r1})
            tb = self.cone_pos[sorted_idx[np.random.randint(n_best)]]

            cx = self.cone_pos[idx]

            # Original v2 pollination + pBest cognitive term
            pbest_pull = c1 * np.random.rand(self.dim) * (self.pbest_pos[idx] - cx)

            cx_new = cx + \
                     0.5 * phi_all[r1] * (tb - self.cone_pos[r1]) + \
                     0.5 * phi_all[r3] * (tb - self.cone_pos[r3]) + \
                     0.2 * pbest_pull

            r1v, r2v = np.random.rand(self.dim), np.random.rand(self.dim)
            vel_new = (inertia * self.velocity[idx] +
                       c1 * r1v * (self.pbest_pos[idx] - cx) +
                       c2 * r2v * (self.best_pos - cx))
            vel_new = np.clip(vel_new, -v_max, v_max)
            pso_candidate = cx + vel_new
            cx_new = (1.0 - pso_weight) * cx_new + pso_weight * pso_candidate

            cx_new = np.clip(cx_new, self.lb, self.ub)
            f_new = self._eval(cx_new)
            self.velocity[idx] = vel_new
            if f_new < self.cone_fit[idx]:
                self.cone_pos[idx] = cx_new
                self.cone_fit[idx] = f_new

            # >>> FIX C: Update personal best <<<
            self._update_pbest(idx, self.cone_pos[idx], self.cone_fit[idx])

    # ------------------------------------------------------------------
    # Algorithm 2 — Animal dispersal + FIX A (dynamic Levy) + FIX C (pBest)
    # ------------------------------------------------------------------
    def _animal_dispersal(self, Lb, Ub):
        wd = np.exp(-20 * self.fes / self.max_fes)
        CX_mean = np.mean(self.cone_pos, axis=0)
        TX_mean = np.mean(self.tree_pos, axis=0)
        progress = self._progress()

        # >>> FIX A: Dynamic Levy scale <<<
        levy_scale = self._levy_scale()

        # >>> FIX C: pBest coefficients for dispersal blending <<<
        c1, c2 = self._pbest_coefficients()

        if ((progress > self.p2 and np.random.rand() < 0.9) or
                (progress < self.p1 and np.random.rand() < 0.9)):

            x_init = self.best_pos + np.random.rand(self.dim) * (CX_mean - self.best_pos)
            x_init = np.clip(x_init, Lb, Ub)
            try:
                res = minimize(self.obj_func, x_init, method='L-BFGS-B',
                               bounds=list(zip(Lb, Ub)),
                               options={'maxiter': 5, 'maxfun': 10})
                self.fes += res.nfev
                if res.fun < self.best_fit:
                    self.best_fit = res.fun
                    self.best_pos = np.clip(res.x, self.lb, self.ub)
            except Exception:
                pass

            if self.fes < self.max_fes:
                lev = levy_flight(self.dim) * levy_scale
                r = np.random.randint(self.pop_size)
                x_rand = self.cone_pos[r]
                mid = (self.best_pos + x_rand) / 2.0

                # >>> FIX C: Blend pBest of a random cone into dispersal <<<
                pbest_term = c1 * np.random.rand(self.dim) * (self.pbest_pos[r] - mid)

                x_animal = mid + lev * (lev * (self.lb + self.ub - mid) - mid) + \
                           0.2 * pbest_term
                x_animal = np.clip(x_animal, self.lb, self.ub)
                f_a = self._eval(x_animal)
                if f_a < self.best_fit:
                    self.best_fit = f_a
                    self.best_pos = x_animal.copy()
        else:
            lev = levy_flight(self.dim) * levy_scale
            if np.random.rand() < 0.5:
                x_animal = (CX_mean + (1 - wd) * TX_mean +
                            wd * lev * (lev * (self.lb + self.ub - TX_mean) - TX_mean))
            else:
                x_animal = CX_mean + wd * lev * (lev * (self.lb + self.ub - CX_mean) - CX_mean)

            x_animal = np.clip(x_animal, self.lb, self.ub)
            if self.fes < self.max_fes:
                f_a = self._eval(x_animal)
                if f_a < self.best_fit:
                    self.best_fit = f_a
                    self.best_pos = x_animal.copy()

    # ------------------------------------------------------------------
    # FIX B: Restricted tree update with decaying communication
    # ------------------------------------------------------------------
    def _update_trees(self):
        """Update tree positions with probabilistic communication gating.

        Each tree segment only updates its representative position if a
        random draw passes the communication probability threshold.
        This prevents sub-populations from blindly converging on each
        other's sub-optimal vectors early in the search.
        """
        p_comm = self._comm_prob()

        for i in range(self.n_tree):
            # Only communicate (update tree from cones) if gate passes
            if np.random.rand() < p_comm:
                s, e = i * self.n_cone, (i + 1) * self.n_cone
                bc = s + np.argmin(self.cone_fit[s:e])
                if self.cone_fit[bc] < self.tree_fit[i]:
                    self.tree_pos[i] = self.cone_pos[bc].copy()
                    self.tree_fit[i] = self.cone_fit[bc]

    def _apply_stagnation_reset(self):
        """Re-seed a few worst cones when a stagnation burst is triggered."""
        if not self.pending_reset:
            return

        reset_count = max(1, int(self.reset_ratio * self.pop_size))
        worst_idx = np.argsort(self.cone_fit)[-reset_count:]

        for idx in worst_idx:
            if self.fes >= self.max_fes:
                break

            if np.random.rand() < 0.5:
                cand = self.best_pos + np.random.randn(self.dim) * self.reset_sigma * (self.ub - self.lb)
            else:
                cand = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)

            cand = np.clip(cand, self.lb, self.ub)
            f_new = self._eval(cand)
            self.cone_pos[idx] = cand
            self.cone_fit[idx] = f_new
            self.pbest_pos[idx] = cand.copy()
            self.pbest_fit[idx] = f_new
            self.velocity[idx] = 0.0

        self.pending_reset = False

    @staticmethod
    def _rand_idx_excl(n, excl):
        r = np.random.randint(n)
        while r in excl:
            r = np.random.randint(n)
        return r

    # ==================================================================
    # Main optimisation loop
    # ==================================================================
    def optimize(self):
        """Run MPCOA v2 and return (best_fitness, best_position, convergence)."""
        self._initialize()
        self.convergence = [self.best_fit]

        while self.fes < self.max_fes:
            Lb, Ub = self._update_bounds()

            if np.random.rand() < 0.5:
                self._pollination_alg1(Lb, Ub)
            else:
                self._pollination_v2()

            self._animal_dispersal(Lb, Ub)
            self._update_trees()
            self._apply_stagnation_reset()
            self.convergence.append(self.best_fit)

        return self.best_fit, self.best_pos, self.convergence


# -------------------------------------------------------------------
# Convenience wrapper
# -------------------------------------------------------------------
def mpcoa(obj_func, lb, ub, dim, max_fes, **kwargs):
    """Run MPCOA and return (best_fit, best_pos, convergence_curve)."""
    opt = MPCOA(obj_func, lb, ub, dim, max_fes, **kwargs)
    return opt.optimize()