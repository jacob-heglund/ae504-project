import numpy as np
from control import StateSpace, lqr

class LQG():
    def __init__(self, A, B, C, D, Q, R, Vd, Vn):
        self.P = np.eye(4) * 0.1
        self.x_hat = np.ones((4,1))
        sys = StateSpace(A,B,C,D)

        #lqr gains
        K, S, E = lqr(sys, Q, R)
        self.K = K

    def kalman_gain(self, A, obs, C, Vd):
        x_hat_minus = A @ self.x_hat
        P_minus = A @ self.P @ A.T
        y = C @ obs
        e = y - C @ x_hat_minus
        S = C @ P_minus @ C.T + Vd

        Kf = P_minus @ C.T @ np.linalg.inv(S)
        self.x_hat = x_hat_minus + Kf @ e
        self.P = (np.eye(4) - Kf @ C) @ P_minus
        return Kf

    def add_noise(self):
        noise = np.random.normal(loc=0, scale=(0.010))
        noise = np.ones((4,1)) * noise
        return noise

    def get_state_estimate(self, Kf, obs, state, C, A):
        state = np.array([state])
        pred_state = state + Kf@((obs) - C @ state)
        return pred_state

    def apply_state_controller(self, K, x):
        u = -np.dot(K, x)
        if u > 0:
            return 1, u
        else:
            return 0, u
