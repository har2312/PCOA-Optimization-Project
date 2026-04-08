"""
Pine Cone Optimization Algorithm (PCOA) - Base Implementation

Reference:
    Anaraki, M.V. & Farzin, S. (2024).
    "The Pine Cone Optimization Algorithm (PCOA)."
    Biomimetics, 9(2), 91. DOI: 10.3390/biomimetics9020091
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


# ===================================================================
# PCOA Class
# ===================================================================
class PCOA:
    """Pine Cone Optimization Algorithm (baseline).

    Parameters
    ----------
    obj_func : callable
        Objective function f(x) -> float (minimization).
    lb, ub : float or array-like
        Lower / upper bounds of the search space.
    dim : int
        Dimensionality.
    max_fes : int
        Maximum number of function evaluations.
    n_tree : int
        Number of pine trees.
    n_cone : int
        Number of cones per tree  (pop_size = n_tree * n_cone).
    memory_size : int
        Size of the adaptive parameter memory (H).
    tbest_rate : float
        Fraction of top solutions used as Tbest.
    p1, p2 : float
        Phase-switch thresholds for Algorithm 2.
    alpha_poll, beta_poll, gamma_poll : float
        Constants for the pollination-effect model (Eqs 13-14).
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

    # ------------------------------------------------------------------
    # Initialization  (Eqs 4-6, control_param = 0)
    # ------------------------------------------------------------------
    def _initialize(self):
        self.cone_pos = np.zeros((self.pop_size, self.dim))
        self.cone_fit = np.full(self.pop_size, np.inf)
        self.tree_pos = np.zeros((self.n_tree, self.dim))
        self.tree_fit = np.full(self.n_tree, np.inf)

        for i in range(self.n_tree):
            # Segment bounds (Eqs 5-6 with original lb/ub)
            lbs = self.lb + i * (self.ub - self.lb) / self.n_tree
            ubs = self.lb + (i + 1) * (self.ub - self.lb) / self.n_tree
            self.tree_pos[i] = (lbs + ubs) / 2.0

            for j in range(self.n_cone):
                idx = i * self.n_cone + j
                # Eq 4 (control_param=0): scatter around tree
                self.cone_pos[idx] = lbs + np.random.rand(self.dim) * (ubs - lbs)
                self.cone_fit[idx] = self._eval(self.cone_pos[idx])

        # Best
        bi = np.argmin(self.cone_fit)
        self.best_pos = self.cone_pos[bi].copy()
        self.best_fit = self.cone_fit[bi]

        # Update tree positions to best cone per segment
        for i in range(self.n_tree):
            s, e = i * self.n_cone, (i + 1) * self.n_cone
            bc = s + np.argmin(self.cone_fit[s:e])
            self.tree_pos[i] = self.cone_pos[bc].copy()
            self.tree_fit[i] = self.cone_fit[bc]

        # Adaptive memory for W1, W2, W3, cr  (Algorithm 1)
        self.mem_W1 = np.full(self.memory_size, 0.5)
        self.mem_W2 = np.full(self.memory_size, 0.5)
        self.mem_W3 = np.full(self.memory_size, 0.5)
        self.mem_cr = np.full(self.memory_size, 0.5)
        self.mem_k = 0  # circular index

        # External archive  (Tpopall)
        self.archive = []

    # ------------------------------------------------------------------
    # Boundary update  (Eqs 7-11)
    # ------------------------------------------------------------------
    def _update_bounds(self):
        W = min(self.fes / self.max_fes, 0.5)              # Eq 9
        radius_lb = self.best_pos - self.lb                 # Eq 10
        radius_ub = self.ub - self.best_pos                 # Eq 11
        Lb = self.lb + radius_lb * W                        # Eq 7
        Ub = self.ub - radius_ub * W                        # Eq 8
        # Ensure Lb < Ub
        mask = Lb >= Ub
        Lb[mask] = self.lb[mask]
        Ub[mask] = self.ub[mask]
        return Lb, Ub

    def _seg_bounds(self, tree_i, Lb, Ub):
        """Segment bounds for tree i  (Eqs 5-6)."""
        LbS = Lb + tree_i * (Ub - Lb) / self.n_tree
        UbS = Lb + (tree_i + 1) * (Ub - Lb) / self.n_tree
        return LbS, UbS

    # ------------------------------------------------------------------
    # Algorithm 1 — Wind pollination  (exploration)
    # ------------------------------------------------------------------
    def _pollination_alg1(self, Lb, Ub):
        n_best = max(1, int(self.tbest_rate * self.pop_size))
        sorted_idx = np.argsort(self.cone_fit)
        tbest_idx = sorted_idx[:n_best]

        # Combined pool = current pop + archive
        all_pop = list(self.cone_pos)
        if self.archive:
            all_pop.extend(self.archive)
        all_pop_arr = np.array(all_pop)
        n_all = len(all_pop_arr)

        succ_W1, succ_W2, succ_cr, delta_f = [], [], [], []

        for idx in range(self.pop_size):
            if self.fes >= self.max_fes:
                break

            # --- sample from memory (lines 3-7) ---
            mi = np.random.randint(self.memory_size)
            mu1, mu2, mu3, mucr = (self.mem_W1[mi], self.mem_W2[mi],
                                    self.mem_W3[mi], self.mem_cr[mi])

            # Generate W1,W2,W3,cr  (lines 8-11, Cauchy + normal)
            cauchy = lambda: np.tan(np.pi * (np.random.rand() - 0.5))
            W1 = mu1 + np.random.rand() * cauchy()
            W2 = mu2 + np.random.rand() * cauchy()
            W3 = mu3 + 0.1 * cauchy()
            cr = mucr + 0.1 * np.random.randn()

            # Repair negatives (lines 12-20)
            if W1 < 0:
                W1 = mu1 + 0.1 * cauchy()
            if W2 < 0:
                W2 = mu2 + 0.1 * cauchy()
            if W3 < 0:
                W3 = mu3 + 0.1 * cauchy()

            # Clip (line 21)
            W1 = min(max(W1, 0.0), 1.0)
            W2 = min(max(W2, 0.0), 1.0)
            W3 = min(max(W3, 0.0), 1.0)
            cr = np.clip(cr, 0.0, 1.0)

            # Random indices (line 22)
            r1 = self._rand_idx_excl(self.pop_size, {idx})
            r2 = self._rand_idx_excl(n_all, {idx, r1})
            r3 = self._rand_idx_excl(self.pop_size, {idx, r1})

            tbest_r = self.cone_pos[tbest_idx[np.random.randint(n_best)]]

            # Position update (lines 25-30)
            cx = self.cone_pos[idx]
            if np.random.rand() < 0.5:
                # Eq line 26
                cx_new = cx + W1 * (tbest_r - self.cone_pos[r1]) + \
                         W2 * (self.cone_pos[r1] - all_pop_arr[r2])
            else:
                # Eq lines 28-29
                Wm = max(W1 * W2, (1 - W1) * W2, 1 - W2)
                cx_new = (W2 * (W1 * cx + (1 - W1) * tbest_r) +
                          (1 - W2) * self.cone_pos[r1] +
                          Wm * (self.cone_pos[r3] - all_pop_arr[r2]))

            cx_new = np.clip(cx_new, self.lb, self.ub)
            f_new = self._eval(cx_new)

            # Greedy selection with crossover (lines 32-34)
            if f_new < self.cone_fit[idx] or np.random.rand() < cr:
                if f_new < self.cone_fit[idx]:
                    succ_W1.append(W1)
                    succ_W2.append(W2)
                    succ_cr.append(cr)
                    delta_f.append(abs(self.cone_fit[idx] - f_new))
                    self.archive.append(self.cone_pos[idx].copy())
                self.cone_pos[idx] = cx_new
                self.cone_fit[idx] = f_new

        # Trim archive (line 35)
        while len(self.archive) > self.pop_size:
            self.archive.pop(np.random.randint(len(self.archive)))

        # Update memory — weighted Lehmer mean (line 36)
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
    # Alternative pollination  (Eqs 12-14)
    # ------------------------------------------------------------------
    def _pollination_v2(self):
        n_best = max(1, int(self.tbest_rate * self.pop_size))
        sorted_idx = np.argsort(self.cone_fit)

        for idx in range(self.pop_size):
            if self.fes >= self.max_fes:
                break

            # phi_i  (Eq 13-14)
            sum_a = 0.0
            for j in range(self.pop_size):
                if j == idx:
                    continue
                d = max(np.linalg.norm(self.cone_pos[idx] - self.cone_pos[j]), 1.0)
                a_ij = (self.beta_poll / d ** self.alpha_poll -
                        self.alpha_poll / d ** self.beta_poll) / \
                       (self.beta_poll - self.alpha_poll)
                sum_a += a_ij
            phi = 1 - np.exp(-self.gamma_poll * abs(sum_a))

            r1 = self._rand_idx_excl(self.pop_size, {idx})
            r3 = self._rand_idx_excl(self.pop_size, {idx, r1})
            tb = self.cone_pos[sorted_idx[np.random.randint(n_best)]]

            # Eq 12
            cx_new = self.cone_pos[idx] + \
                     0.5 * phi * (tb - self.cone_pos[r1]) + \
                     0.5 * phi * (tb - self.cone_pos[r3])

            cx_new = np.clip(cx_new, self.lb, self.ub)
            f_new = self._eval(cx_new)
            if f_new < self.cone_fit[idx]:
                self.cone_pos[idx] = cx_new
                self.cone_fit[idx] = f_new

    # ------------------------------------------------------------------
    # Algorithm 2 — Animal dispersal  (exploitation)
    # ------------------------------------------------------------------
    def _animal_dispersal(self, Lb, Ub):
        wd = np.exp(-20 * self.fes / self.max_fes)           # Eq 18
        CX_mean = np.mean(self.cone_pos, axis=0)
        TX_mean = np.mean(self.tree_pos, axis=0)
        progress = self.fes / self.max_fes

        if ((progress > self.p2 and np.random.rand() < 0.9) or
                (progress < self.p1 and np.random.rand() < 0.9)):

            # --- Operator 1: Quadratic-programming local search (Eq 15) ---
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

            # --- Operator 2: Levy-based dispersal (Eq 16) ---
            if self.fes < self.max_fes:
                lev = levy_flight(self.dim)
                r = np.random.randint(self.pop_size)
                x_a2 = self.cone_pos[r]
                x_animal = (self.best_pos + x_a2 +
                            lev * (lev * (self.lb + self.ub - self.best_pos + x_a2)
                                   - self.best_pos + x_a2))
                x_animal = np.clip(x_animal, self.lb, self.ub)
                f_a = self._eval(x_animal)
                if f_a < self.best_fit:
                    self.best_fit = f_a
                    self.best_pos = x_animal.copy()
        else:
            lev = levy_flight(self.dim)
            if np.random.rand() < 0.5:
                # --- Operator 3 (Eq 17) ---
                x_animal = (CX_mean + (1 - wd) * TX_mean +
                            wd * lev * (lev * (self.lb + self.ub - TX_mean) - TX_mean))
            else:
                # --- Operator 4 (Eq 19) ---
                x_animal = CX_mean + wd * lev * (lev * (self.lb + self.ub - CX_mean) - CX_mean)

            x_animal = np.clip(x_animal, self.lb, self.ub)
            if self.fes < self.max_fes:
                f_a = self._eval(x_animal)
                if f_a < self.best_fit:
                    self.best_fit = f_a
                    self.best_pos = x_animal.copy()

    # ------------------------------------------------------------------
    # Tree update
    # ------------------------------------------------------------------
    def _update_trees(self):
        for i in range(self.n_tree):
            s, e = i * self.n_cone, (i + 1) * self.n_cone
            bc = s + np.argmin(self.cone_fit[s:e])
            if self.cone_fit[bc] < self.tree_fit[i]:
                self.tree_pos[i] = self.cone_pos[bc].copy()
                self.tree_fit[i] = self.cone_fit[bc]

    # ------------------------------------------------------------------
    # Random index helper
    # ------------------------------------------------------------------
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
        """Run PCOA and return (best_fitness, best_position, convergence)."""
        self._initialize()
        self.convergence = [self.best_fit]

        while self.fes < self.max_fes:
            # Update shrinking boundaries (Eqs 7-11)
            Lb, Ub = self._update_bounds()

            # Pollination — exploration
            if np.random.rand() < 0.5:
                self._pollination_alg1(Lb, Ub)
            else:
                self._pollination_v2()

            # Animal dispersal — exploitation
            self._animal_dispersal(Lb, Ub)

            # Update tree positions
            self._update_trees()

            self.convergence.append(self.best_fit)

        return self.best_fit, self.best_pos, self.convergence


# -------------------------------------------------------------------
# Convenience wrapper (functional API)
# -------------------------------------------------------------------
def pcoa(obj_func, lb, ub, dim, max_fes, **kwargs):
    """Run PCOA and return (best_fit, best_pos, convergence_curve)."""
    opt = PCOA(obj_func, lb, ub, dim, max_fes, **kwargs)
    return opt.optimize()
