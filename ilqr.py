import numpy as np
import gym
import cartpole_noise
from gym import wrappers

class iLQR():
    def __init__(self, Q, R, Q_terminal, x_goal):
        self.J_hist = []
        self.cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = 1e10
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._k = np.zeros((N, 1))
        self._K = np.zeros((N, 1, 5))
    
    def reduce_state(self, state):
        # [x, x', sin(theta), cos(theta), theta'] -> [x, x', theta, theta']
        if state.ndim == 1:
            x, x_dot, sin_theta, cos_theta, theta_dot = state
        else:
            x = state[..., 0].reshape(-1, 1)
            x_dot = state[..., 1].reshape(-1, 1)
            sin_theta = state[..., 2].reshape(-1, 1)
            cos_theta = state[..., 3].reshape(-1, 1)
            theta_dot = state[..., 4].reshape(-1, 1)

        theta = np.arctan2(sin_theta, cos_theta)
        return np.hstack([x, x_dot, theta, theta_dot])
    
    def augment_state(self, state):
        # [x, x', theta, theta'] -> [x, x', sin(theta), cos(theta), theta']
        if state.ndim == 1:
            x, x_dot, theta, theta_dot = state
        else:
            x = state[..., 0].reshape(-1, 1)
            x_dot = state[..., 1].reshape(-1, 1)
            theta = state[..., 2].reshape(-1, 1)
            theta_dot = state[..., 3].reshape(-1, 1)

        return np.hstack([x, x_dot, np.sin(theta), np.cos(theta), theta_dot])
    
    def fit(self, env, x0, us_init, n_iterations, tol=1e-6):
        self.dynamics = env
        # Reset regularization term.
        self._mu = 1.0
        self._delta = self._delta_0

        # Backtracking line search candidates 0 < alpha <= 1.
        alphas = 1.1**(-np.arange(10)**2)

        us = us_init.copy()
        k = self._k
        K = self._K

        changed = True
        converged = False
        for iteration in range(n_iterations):
            accepted = False

            # Forward rollout only if it needs to be recomputed.
            if changed:
                (xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux,
                 F_uu) = self._forward_rollout(x0, us)
                J_opt = L.sum()
                changed = False

            try:
                # Backward pass.
                k, K = self._backward_pass(F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu,
                                           F_xx, F_ux, F_uu)

                # Backtracking line search.
                for alpha in alphas:
                    xs_new, us_new = self._control(xs, us, k, K, alpha)
                    J_new = self._trajectory_cost(xs_new, us_new)

                    if J_new < J_opt:
                        if np.abs((J_opt - J_new) / J_opt) < tol:
                            converged = True

                        J_opt = J_new
                        xs = xs_new
                        us = us_new
                        changed = True

                        # Decrease regularization term.
                        self._delta = min(1.0, self._delta) / self._delta_0
                        self._mu *= self._delta
                        if self._mu <= self._mu_min:
                            self._mu = 0.0

                        # Accept this.
                        accepted = True
                        break
            except np.linalg.LinAlgError as e:
                # Quu was not positive-definite and this diverged.
                # Try again with a higher regularization term.
                warnings.warn(str(e))

            if not accepted:
                # Increase regularization term.
                self._delta = max(1.0, self._delta) * self._delta_0
                self._mu = max(self._mu_min, self._mu * self._delta)
                if self._mu_max and self._mu >= self._mu_max:
                    warnings.warn("exceeded max regularization term")
                    break

            self.info(iteration, xs, us, J_opt, accepted, converged)

            if converged:
                break

        # Store fit parameters.
        self._k = k
        self._K = K
        self._nominal_xs = xs
        self._nominal_us = us

        return xs, us

    def info(self, iteration_count, xs, us, J_opt, accepted, converged):
        self.J_hist.append(J_opt)
        info = "converged" if converged else ("accepted" if accepted else "failed")
        final_state = self.reduce_state(xs[-1])
        print("iteration", iteration_count, info, J_opt, final_state)
    
    def _control(self, xs, us, k, K, alpha=1.0):
        xs_new = np.zeros_like(xs)
        us_new = np.zeros_like(us)
        xs_new[0] = xs[0].copy()

        for i in range(self.N):
            # Eq (12).
            us_new[i] = us[i] + alpha * k[i] + K[i].dot(xs_new[i] - xs[i])

            # Eq (8c).
            env.reset(self.reduce_state(xs_new[i]))
            xs_new[i + 1] = self.augment_state(self.dynamics.step(us_new[i][..., 0]))

        return xs_new, us_new

    def _trajectory_cost(self, xs, us):
        J = map(lambda args: self.cost.l(*args), zip(xs[:-1], us, range(self.N)))
        return sum(J) + self.cost.l(xs[-1], None, self.N, terminal=True)

    def _forward_rollout(self, x0, us):
        state_size = 5
        action_size = 1
        N = us.shape[0]

        xs = np.empty((N + 1, state_size))
        F_x = np.empty((N, state_size, state_size))
        F_u = np.empty((N, state_size, action_size))
        F_xx = None
        F_ux = None
        F_uu = None

        L = np.empty(N + 1)
        L_x = np.empty((N + 1, state_size))
        L_u = np.empty((N, action_size))
        L_xx = np.empty((N + 1, state_size, state_size))
        L_ux = np.empty((N, action_size, state_size))
        L_uu = np.empty((N, action_size, action_size))

        xs[0] = x0
        for i in range(N):
            x = xs[i]
            u = us[i]

            env.reset(self.reduce_state(x))
            xs[i + 1] = self.dynamics.step(u[..., 0])
            F_x[i] = self.dynamics.f_x(x, u, i)
            F_u[i] = self.dynamics.f_u(x, u, i)

            L[i] = self.cost.l(x, u, i, terminal=False)
            L_x[i] = self.cost.l_x(x, u, i, terminal=False)
            L_u[i] = self.cost.l_u(x, u, i, terminal=False)
            L_xx[i] = self.cost.l_xx(x, u, i, terminal=False)
            L_ux[i] = self.cost.l_ux(x, u, i, terminal=False)
            L_uu[i] = self.cost.l_uu(x, u, i, terminal=False)

        x = xs[-1]
        L[-1] = self.cost.l(x, None, N, terminal=True)
        L_x[-1] = self.cost.l_x(x, None, N, terminal=True)
        L_xx[-1] = self.cost.l_xx(x, None, N, terminal=True)

        return xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux, F_uu

    def _backward_pass(self, F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu, F_xx=None, F_ux=None, F_uu=None):
        V_x = L_x[-1]
        V_xx = L_xx[-1]

        k = np.empty_like(self._k)
        K = np.empty_like(self._K)

        for i in range(self.N - 1, -1, -1):
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                    L_u[i], L_xx[i], L_ux[i],
                                                    L_uu[i], V_x, V_xx)

            # Eq (6).
            k[i] = -np.linalg.solve(Q_uu, Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)

            # Eq (11b).
            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            # Eq (11c).
            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.

        return np.array(k), np.array(K)

    def _Q(self, f_x, f_u, l_x, l_u, l_xx, l_ux, l_uu, V_x, V_xx, f_xx=None, f_ux=None, f_uu=None):
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu

class QRCost(Cost):
    """Quadratic Regulator Instantaneous Cost."""
    def __init__(self, Q, R, Q_terminal=None, x_goal=None, u_goal=None):
        self.Q = np.array(Q)
        self.R = np.array(R)

        if Q_terminal is None:
            self.Q_terminal = self.Q
        else:
            self.Q_terminal = np.array(Q_terminal)

        if x_goal is None:
            self.x_goal = np.zeros(Q.shape[0])
        else:
            self.x_goal = np.array(x_goal)

        if u_goal is None:
            self.u_goal = np.zeros(R.shape[0])
        else:
            self.u_goal = np.array(u_goal)

        assert self.Q.shape == self.Q_terminal.shape, "Q & Q_terminal mismatch"
        assert self.Q.shape[0] == self.Q.shape[1], "Q must be square"
        assert self.R.shape[0] == self.R.shape[1], "R must be square"
        assert self.Q.shape[0] == self.x_goal.shape[0], "Q & x_goal mismatch"
        assert self.R.shape[0] == self.u_goal.shape[0], "R & u_goal mismatch"

        # Precompute some common constants.
        self._Q_plus_Q_T = self.Q + self.Q.T
        self._R_plus_R_T = self.R + self.R.T
        self._Q_plus_Q_T_terminal = self.Q_terminal + self.Q_terminal.T

        super(QRCost, self).__init__()

    def l(self, x, u, i, terminal=False):
        Q = self.Q_terminal if terminal else self.Q
        R = self.R
        x_diff = x - self.x_goal
        squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

        if terminal:
            return squared_x_cost

        u_diff = u - self.u_goal
        return squared_x_cost + u_diff.T.dot(R).dot(u_diff)

    def l_x(self, x, u, i, terminal=False):
        Q_plus_Q_T = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
        x_diff = x - self.x_goal
        return x_diff.T.dot(Q_plus_Q_T)

    def l_u(self, x, u, i, terminal=False):
        if terminal:
            return np.zeros_like(self.u_goal)

        u_diff = u - self.u_goal
        return u_diff.T.dot(self._R_plus_R_T)

    def l_xx(self, x, u, i, terminal=False):
        return self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T

    def l_ux(self, x, u, i, terminal=False):
        return np.zeros((self.R.shape[0], self.Q.shape[0]))

    def l_uu(self, x, u, i, terminal=False):
        if terminal:
            return np.zeros_like(self.R)

        return self._R_plus_R_T

def main():
    Q = np.eye(5)
    Q[0, 0] = 1.0
    Q[1, 1] = Q[4, 4] = 0.0
    Q[0, 2] = Q[2, 0] = 1.0
    Q[2, 2] = Q[3, 3] = 2.0
    Q_terminal = 100 * np.eye(5)
    R = np.array([[0.1]])
    x_goal = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    N = 500
    ilqr = iLQR(Q, R, Q_terminal, x_goal)

    env = gym.make('cp-cont-v0')

    env = wrappers.Monitor(env, './videos/' + str(time()) + '/')
    env.env.seed(1)     # seed for reproducibility
    obs = env.reset(init_state=[0, 0, np.pi, 0])
    us_init = np.random.uniform(-1, 1, (N, 1))
    xs, us = ilqr.fit(env, obs, us_init, n_iterations=N)
    '''
    for i in range(N):
        env.render()
        
        # get force direction (action) and force value (force)
        action, force = lqr.apply_state_controller(lqr.K, obs)
        
        # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
        abs_force = abs(float(np.clip(force, -10, 10)))
        
        # change magnitute of the applied force in CartPole
        env.env.force_mag = abs_force

        # apply action
        obs, reward, done, _ = env.step(action)
        if done:
            print(f'Terminated after {i+1} iterations.')
            break
    '''

    env.close()

if __name__ == '__main__':
    main()