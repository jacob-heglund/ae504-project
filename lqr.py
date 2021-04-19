import numpy as np
from control import StateSpace, lqr

class LQR():
    def __init__(self, A, B, C, D, Q, R):
        sys = StateSpace(A, B, C, D)
        K, S, E = lqr(sys, Q, R)
        self.K = K
    
    def apply_state_controller(self, K, x):
        # feedback controller
        u = -np.dot(K, x)   # u = -Kx
        if u > 0:
            return 1, u     # if force_dem > 0 -> move cart right
        else:
            return 0, u     # if force_dem <= 0 -> move cart left