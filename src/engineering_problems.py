"""
Engineering Optimization Benchmarks

Implements 4 standard constrained engineering design problems.
Constraints are handled seamlessly using a static penalty function approach,
so that unconstrained solvers (like PCOA and mealpy models) can solve them.
"""

import numpy as np

class BaseEngineeringProblem:
    def __init__(self, ndim=None):
        self.f_bias = 0.0

    def evaluate(self, x):
        # Clip strictly to bounds to ensure valid evaluation
        x_clipped = np.clip(x, self.lb, self.ub)
        fx = self.obj_func(x_clipped)
        
        # Calculate penalty for constraint violations
        g = self.constraints(x_clipped)
        penalty = 0.0
        for gi in g:
            # g(x) <= 0 is the standard. If gi > 0, it's a violation.
            if gi > 0:
                penalty += 1e10 * (gi ** 2)
                
        return fx + penalty

    def obj_func(self, x):
        return 0.0

    def constraints(self, x):
        return []

class PressureVessel(BaseEngineeringProblem):
    def __init__(self, ndim=4):
        super().__init__(ndim)
        self.lb = np.array([0.0, 0.0, 10.0, 10.0])
        self.ub = np.array([99.0, 99.0, 200.0, 200.0])
        
    def obj_func(self, x):
        return 0.6224 * x[0] * x[2] * x[3] + 1.7781 * x[1] * (x[2] ** 2) + 3.1661 * (x[0] ** 2) * x[3] + 19.84 * (x[0] ** 2) * x[2]
        
    def constraints(self, x):
        g1 = -x[0] + 0.0193 * x[2]
        g2 = -x[1] + 0.00954 * x[2]
        g3 = -np.pi * (x[2] ** 2) * x[3] - (4/3) * np.pi * (x[2] ** 3) + 1296000.0
        g4 = x[3] - 240.0
        return [g1, g2, g3, g4]

class SpringDesign(BaseEngineeringProblem):
    def __init__(self, ndim=3):
        super().__init__(ndim)
        self.lb = np.array([0.05, 0.25, 2.0])
        self.ub = np.array([2.0, 1.3, 15.0])
        
    def obj_func(self, x):
        return (x[2] + 2) * x[1] * (x[0] ** 2)
        
    def constraints(self, x):
        g1 = 1.0 - (x[1] ** 3 * x[2]) / (71785.0 * (x[0] ** 4) + 1e-10)
        numerator = 4 * (x[1] ** 2) - x[0] * x[1]
        denominator = 12566.0 * (x[1] * (x[0] ** 3) - (x[0] ** 4)) + 1e-10
        g2 = numerator / denominator + 1.0 / (5108.0 * (x[0] ** 2) + 1e-10) - 1.0
        g3 = 1.0 - (140.45 * x[0]) / ((x[1] ** 2) * x[2] + 1e-10)
        g4 = (x[0] + x[1]) / 1.5 - 1.0
        return [g1, g2, g3, g4]

class SpeedReducer(BaseEngineeringProblem):
    def __init__(self, ndim=7):
        super().__init__(ndim)
        self.lb = np.array([2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0])
        self.ub = np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5])
        
    def obj_func(self, x):
        return 0.7854 * x[0] * (x[1]**2) * (3.3333 * (x[2]**2) + 14.9334*x[2] - 43.0934) \
               - 1.508 * x[0] * (x[5]**2 + x[6]**2) + 7.4777 * (x[5]**3 + x[6]**3) \
               + 0.7854 * (x[3] * (x[5]**2) + x[4] * (x[6]**2))
               
    def constraints(self, x):
        g1 = 27.0 / (x[0] * (x[1]**2) * x[2] + 1e-10) - 1.0
        g2 = 39.75 / (x[0] * (x[1]**2) * x[2] + 1e-10) - 1.0
        g3 = 1.93 * (x[3]**3) / (x[1] * x[2] * (x[5]**4) + 1e-10) - 1.0
        g4 = 1.93 * (x[4]**3) / (x[1] * x[2] * (x[6]**4) + 1e-10) - 1.0
        g5 = np.sqrt((745.0 * x[3] / (x[1] * x[2] + 1e-10))**2 + 16.9e6) / (110.0 * (x[5]**3) + 1e-10) - 1.0
        g6 = np.sqrt((745.0 * x[4] / (x[1] * x[2] + 1e-10))**2 + 157.5e6) / (85.0 * (x[6]**3) + 1e-10) - 1.0
        g7 = x[1] * x[2] / 40.0 - 1.0
        g8 = 5.0 * x[1] / x[0] - 1.0
        g9 = x[0] / (12.0 * x[1] + 1e-10) - 1.0
        g10 = (1.5 * x[5] + 1.9) / x[3] - 1.0
        g11 = (1.1 * x[6] + 1.9) / x[4] - 1.0
        return [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11]

class WeldedBeam(BaseEngineeringProblem):
    def __init__(self, ndim=4):
        super().__init__(ndim)
        self.lb = np.array([0.1, 0.1, 0.1, 0.1])
        self.ub = np.array([2.0, 10.0, 10.0, 2.0])
        
    def obj_func(self, x):
        return 1.10471 * (x[0]**2) * x[1] + 0.04811 * x[2] * x[3] * (14.0 + x[1])
        
    def constraints(self, x):
        P = 6000.0
        L = 14.0
        E = 30e6
        G = 12e6
        t_max = 13600.0
        s_max = 30000.0
        d_max = 0.25
        
        tau_prime = P / (np.sqrt(2) * x[0] * x[1] + 1e-10)
        M = P * (L + x[1] / 2.0)
        R = np.sqrt(x[1]**2 / 4.0 + ((x[0] + x[2]) / 2.0)**2)
        J = 2 * (np.sqrt(2) * x[0] * x[1] * (x[1]**2 / 12.0 + ((x[0] + x[2]) / 2.0)**2))
        tau_double_prime = M * R / (J + 1e-10)
        tau = np.sqrt(tau_prime**2 + tau_double_prime**2 + x[1] * tau_prime * tau_double_prime / (R + 1e-10))
        
        sigma = 6 * P * L / (x[3] * x[2]**2 + 1e-10)
        delta = 4 * P * L**3 / (E * x[3] * x[2]**3 + 1e-10)
        Pc = 4.013 * E * np.sqrt(x[2]**2 * x[3]**6 / 36.0) / (L**2) * (1 - x[2] * np.sqrt(E / (4 * G)) / (2 * L))
        
        g1 = tau - t_max
        g2 = sigma - s_max
        g3 = x[0] - x[3]
        g4 = 0.10471 * x[0]**2 + 0.04811 * x[2] * x[3] * (14.0 + x[1]) - 5.0
        g5 = 0.125 - x[0]
        g6 = delta - d_max
        g7 = P - Pc
        return [g1, g2, g3, g4, g5, g6, g7]

ENGINEERING_PROBS = {
    1: ("Pressure Vessel Design", PressureVessel),
    2: ("Tension/Compression Spring Design", SpringDesign),
    3: ("Speed Reducer Design", SpeedReducer),
    4: ("Welded Beam Design", WeldedBeam)
}

def get_engineering_functions():
    """Returns list of (fid, func, info) matching CEC interface."""
    results = []
    for fid, (name, cls) in ENGINEERING_PROBS.items():
        obj = cls()
        
        info = {
            "name": name,
            "lb": obj.lb,
            "ub": obj.ub,
            "f_bias": obj.f_bias,
            "dim": len(obj.lb)
        }
        
        def evaluate(x, _obj=obj):
            return _obj.evaluate(x)
            
        results.append((fid, evaluate, info))
        
    return results
