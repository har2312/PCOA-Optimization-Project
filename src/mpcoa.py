"""
Modified Pine Cone Optimization Algorithm (MPCOA)

Modifications over the base PCOA:
  1. Levy flight scaling in wind pollination (Algorithm 1)
     - Heavy-tailed steps for better exploration on multimodal landscapes.
  2. Chaotic map initialization (Logistic Map)
     - Replaces pseudo-random init with chaotic sequences for better
       initial coverage of the search space.
  3. Nonlinear adaptive boundary decay (cosine curve)
     - Replaces linear W with cosine-based decay for smoother
       exploration→exploitation transition.

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

    Returns an (n_points, dim) array of values in (0, 1).
    Chaotic sequences provide more uniform space coverage than
    pseudo-random numbers, reducing initial clustering.
    """
    r = 4.0
    if seed is None:
        x = 0.1 + 0.8 * np.random.rand(dim)  # start in (0.1, 0.9)
    else:
        rng = np.random.RandomState(seed)
        x = 0.1 + 0.8 * rng.rand(dim)

    # Avoid fixed points at 0, 0.25, 0.5, 0.75, 1.0
    x = np.clip(x, 0.01, 0.99)

    result = np.zeros((n_points, dim))
    for i in range(n_points):
        x = r * x * (1 - x)
        result[i] = x
    return result


# ===================================================================
# MPCOA Class  (Modified PCOA)
# ===================================================================
class MPCOA:
    """Modified Pine Cone Optimization Algorithm.

    Modification 1: Levy flight in pollination
        Heavy-tailed step scaling for better exploration.
    Modification 2: Chaotic map initialization
        Logistic Map sequences replace pseudo-random init.

    Parameters  (same as base PCOA)
    ----------
    obj_func, lb, ub, dim, max_fes, n_tree, n_cone, memory_size,
    tbest_rate, p1, p2, alpha_poll, beta_poll, gamma_poll
    """

    def __init__(self, obj_func, lb, ub, dim, max_fes,
                 n_tree=5, n_cone=6, memory_size=5,
                 tbest_rate=0.1, p1=0.2, p2=0.8,
                 alpha_poll=3, beta_poll=40, gamma_poll=0.62):
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

        # Runtime state
        self.fes = 0
        self.best_pos = None
        self.best_fit = np.inf
        self.convergence = []

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

    # ==================================================================
    # MODIFICATION 2: Chaotic map initialization
    # ==================================================================
    def _initialize(self):
        """Initialize population using Logistic Map chaotic sequences.

        CHANGED from base PCOA:
        - Cone positions use chaotic sequences instead of pseudo-random
          numbers.  The Logistic Map x_{n+1} = 4*x*(1-x) generates
          numbers that appear random but cover the space more uniformly,
          reducing the chance of initial clustering and improving the
          starting quality of the population.
        """
        self.cone_pos = np.zeros((self.pop_size, self.dim))
        self.cone_fit = np.full(self.pop_size, np.inf)
        self.tree_pos = np.zeros((self.n_tree, self.dim))
        self.tree_fit = np.full(self.n_tree, np.inf)

        # Generate chaotic sequences for initialization
        chaos = logistic_map(self.pop_size, self.dim)

        for i in range(self.n_tree):
            lbs = self.lb + i * (self.ub - self.lb) / self.n_tree
            ubs = self.ub + i * (self.ub - self.lb) / self.n_tree
            self.tree_pos[i] = (lbs + ubs) / 2.0

            for j in range(self.n_cone):
                idx = i * self.n_cone + j
                # Use chaotic values instead of np.random.rand()
                self.cone_pos[idx] = lbs + chaos[idx] * (ubs - lbs)
                self.cone_pos[idx] = np.clip(self.cone_pos[idx], self.lb, self.ub)
                self.cone_fit[idx] = self._eval(self.cone_pos[idx])

        bi = np.argmin(self.cone_fit)
        self.best_pos = self.cone_pos[bi].copy()
        self.best_fit = self.cone_fit[bi]

        for i in range(self.n_tree):
            s, e = i * self.n_cone, (i + 1) * self.n_cone
            bc = s + np.argmin(self.cone_fit[s:e])
            self.tree_pos[i] = self.cone_pos[bc].copy()
            self.tree_fit[i] = self.cone_fit[bc]

        self.mem_W1 = np.full(self.memory_size, 0.5)
        self.mem_W2 = np.full(self.memory_size, 0.5)
        self.mem_W3 = np.full(self.memory_size, 0.5)
        self.mem_cr = np.full(self.memory_size, 0.5)
        self.mem_k = 0
        self.archive = []

    # ==================================================================
    # MODIFICATION 3: Nonlinear adaptive boundary decay
    # ==================================================================
    def _update_bounds(self):
        """Boundary shrinking with cosine-based nonlinear decay.

        CHANGED from base PCOA:
        - Base uses linear:  W = min(fes/max_fes, 0.5)
        - Modified uses cosine: W = 0.5 * (1 - cos(π * t))
          where t = fes / max_fes

        The cosine curve starts slow (preserving wide search space for
        exploration), accelerates in the middle, and decelerates near
        the end (fine-grained exploitation).  This avoids the base's
        problem of shrinking too fast early on, which can trap the
        algorithm before it discovers promising regions.
        """
        t = self.fes / self.max_fes
        W = 0.5 * (1 - np.cos(np.pi * t))   # range [0, 1], capped below
        W = min(W, 0.5)                       # same cap as base
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
    # MODIFICATION 1: Levy flight in Algorithm 1 (wind pollination)
    # ==================================================================
    def _pollination_alg1(self, Lb, Ub):
        """Algorithm 1 with Levy flight scaling.

        CHANGED from base PCOA:
        - Line 26:  added  levy_flight()  scaling to the difference vector
                    (Tbest - CX_r1).  This makes step sizes follow a
                    heavy-tailed distribution instead of uniform.
        - Lines 28-29:  added levy scaling to the Wm difference term.
        """
        n_best = max(1, int(self.tbest_rate * self.pop_size))
        sorted_idx = np.argsort(self.cone_fit)
        tbest_idx = sorted_idx[:n_best]

        all_pop = list(self.cone_pos)
        if self.archive:
            all_pop.extend(self.archive)
        all_pop_arr = np.array(all_pop)
        n_all = len(all_pop_arr)

        succ_W1, succ_W2, succ_cr, delta_f = [], [], [], []

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

            # ============================================================
            # >>> MODIFICATION 1: Levy flight scaling <<<
            # Generate a Levy step for this agent
            lev = levy_flight(self.dim)
            # ============================================================

            cx = self.cone_pos[idx]
            if np.random.rand() < 0.5:
                # Base:  cx + W1*(tbest - cx_r1) + W2*(cx_r1 - pop_r2)
                # Mod:   cx + W1*levy*(tbest - cx_r1) + W2*(cx_r1 - pop_r2)
                #         ^^^^ Levy scales the exploration term toward best
                cx_new = cx + W1 * lev * (tbest_r - self.cone_pos[r1]) + \
                         W2 * (self.cone_pos[r1] - all_pop_arr[r2])
            else:
                Wm = max(W1 * W2, (1 - W1) * W2, 1 - W2)
                # Base:  weighted combo + Wm*(cx_r3 - pop_r2)
                # Mod:   weighted combo + Wm*levy*(cx_r3 - pop_r2)
                #         ^^^^ Levy scales the diversity term
                cx_new = (W2 * (W1 * cx + (1 - W1) * tbest_r) +
                          (1 - W2) * self.cone_pos[r1] +
                          Wm * lev * (self.cone_pos[r3] - all_pop_arr[r2]))

            cx_new = np.clip(cx_new, self.lb, self.ub)
            f_new = self._eval(cx_new)

            if f_new < self.cone_fit[idx] or np.random.rand() < cr:
                if f_new < self.cone_fit[idx]:
                    succ_W1.append(W1)
                    succ_W2.append(W2)
                    succ_cr.append(cr)
                    delta_f.append(abs(self.cone_fit[idx] - f_new))
                    self.archive.append(self.cone_pos[idx].copy())
                self.cone_pos[idx] = cx_new
                self.cone_fit[idx] = f_new

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
    # Alternative pollination  (Eqs 12-14)  — unchanged from base
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

        for idx in range(self.pop_size):
            if self.fes >= self.max_fes:
                break

            r1 = self._rand_idx_excl(self.pop_size, {idx})
            r3 = self._rand_idx_excl(self.pop_size, {idx, r1})
            tb = self.cone_pos[sorted_idx[np.random.randint(n_best)]]

            cx_new = self.cone_pos[idx] + \
                     0.5 * phi_all[r1] * (tb - self.cone_pos[r1]) + \
                     0.5 * phi_all[r3] * (tb - self.cone_pos[r3])

            cx_new = np.clip(cx_new, self.lb, self.ub)
            f_new = self._eval(cx_new)
            if f_new < self.cone_fit[idx]:
                self.cone_pos[idx] = cx_new
                self.cone_fit[idx] = f_new

    # ------------------------------------------------------------------
    # Algorithm 2 — Animal dispersal  — unchanged from base
    # ------------------------------------------------------------------
    def _animal_dispersal(self, Lb, Ub):
        wd = np.exp(-20 * self.fes / self.max_fes)
        CX_mean = np.mean(self.cone_pos, axis=0)
        TX_mean = np.mean(self.tree_pos, axis=0)
        progress = self.fes / self.max_fes

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
                lev = levy_flight(self.dim)
                r = np.random.randint(self.pop_size)
                x_rand = self.cone_pos[r]
                mid = (self.best_pos + x_rand) / 2.0
                x_animal = mid + lev * (lev * (self.lb + self.ub - mid) - mid)
                x_animal = np.clip(x_animal, self.lb, self.ub)
                f_a = self._eval(x_animal)
                if f_a < self.best_fit:
                    self.best_fit = f_a
                    self.best_pos = x_animal.copy()
        else:
            lev = levy_flight(self.dim)
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
    # Tree update  — unchanged
    # ------------------------------------------------------------------
    def _update_trees(self):
        for i in range(self.n_tree):
            s, e = i * self.n_cone, (i + 1) * self.n_cone
            bc = s + np.argmin(self.cone_fit[s:e])
            if self.cone_fit[bc] < self.tree_fit[i]:
                self.tree_pos[i] = self.cone_pos[bc].copy()
                self.tree_fit[i] = self.cone_fit[bc]

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
        """Run MPCOA and return (best_fitness, best_position, convergence)."""
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
            self.convergence.append(self.best_fit)

        return self.best_fit, self.best_pos, self.convergence


# -------------------------------------------------------------------
# Convenience wrapper
# -------------------------------------------------------------------
def mpcoa(obj_func, lb, ub, dim, max_fes, **kwargs):
    """Run MPCOA and return (best_fit, best_pos, convergence_curve)."""
    opt = MPCOA(obj_func, lb, ub, dim, max_fes, **kwargs)
    return opt.optimize()
