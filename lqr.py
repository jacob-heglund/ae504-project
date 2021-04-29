import numpy as np
from control import StateSpace, lqr
from scipy import linalg

class LQR():
    def __init__(self, A, B, C, D, Q, R):
        sys = StateSpace(A, B, C, D)
        K, S, E = lqr(sys, Q, R)

        P = linalg.solve_continuous_are(A, B, Q, R)
        K = np.dot(np.linalg.inv(R), np.dot(B.T, P))
        self.K = K
    
    def apply_state_controller(self, K, x):
        # feedback controller
        u = -np.dot(K, x)   # u = -Kx
        if u > 0:
            return 1, u     # if force_dem > 0 -> move cart right
        else:
            return 0, u     # if force_dem <= 0 -> move cart left