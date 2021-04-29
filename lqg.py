import numpy as np
from control import StateSpace, lqr

class LQG():
    def __init__(self, A, B, C, D, Q, R, Vd, Vn):
        self.P = np.eye(4) * 0.1
        self.x_hat = np.zeros((4,1))
        sys = StateSpace(A,B,C,D)

        #lqr gains
        K, S, E = lqr(sys, Q, R)
        self.K = K

        #kalman filter gains
        #L, P, E = lqe(A, np.eye(4), C, Vd, Vn)
        #self.Kf = L

    def kalman_gain(self, A, obs, C, Vd):
        x_hat_minus = A @ self.x_hat
        P_minus = A @ self.P @ A.T
        y = C @ obs.T
        e = y - C @ x_hat_minus
        S = C @ P_minus @ C.T + Vd

        K = P_minus @ C.T @ np.linalg.inv(S)
        self.x_hat = x_hat_minus + K @ e
        self.P = (np.eye(4) - K @ C) @ P_minus
        return K

    def add_noise(self):
        noise = np.random.normal(loc=0, scale=np.sqrt(0.10))
        noise = np.ones((1,4)) * noise
        return noise

    def get_state_estimate(self, Kf, obs, state, C, A):
        state = np.array([state])
        pred_state = A@state.T + Kf@((obs.T) - C @ state.T)
        return pred_state

    def apply_state_controller(self, K, x):
        u = -np.dot(K, x)
        if u > 0:
            return 1, u
        else:
            return 0, u
