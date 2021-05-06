import cvxpy
import numpy as np
import time
import pdb


def mpc_control(x0, Q, R, sys_params):
    #TODO don't hard-define these here
    nx = 4 # number of states
    nu = 1 # number of inputs
    T = 10 # time horizon
    delta_t = 0.1 # time step

    x = cvxpy.Variable((nx, T + 1))
    u = cvxpy.Variable((nu, T))

    A, B = get_model_matrix_mpc(nx, delta_t, sys_params)

    cost = 0.0
    constr = []

    for t in range(T):
        # define cost over time horizon
        cost += cvxpy.quad_form(x[:, t + 1], Q)
        cost += cvxpy.quad_form(u[:, t], R)

        # define dynamical constraints over time horizon
        constr += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]

    constr += [x[:, 0] == x0[:, 0]]

    # solve the optimization over time horizon
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)
    start = time.time()
    prob.solve(verbose=False)
    elapsed_time = time.time() - start
    # print("calc time:{0} [sec]".format(elapsed_time))

    if prob.status == cvxpy.OPTIMAL:
        ox = get_nparray_from_matrix(x.value[0, :])
        dx = get_nparray_from_matrix(x.value[1, :])
        theta = get_nparray_from_matrix(x.value[2, :])
        dtheta = get_nparray_from_matrix(x.value[3, :])

        ou = get_nparray_from_matrix(u.value[0, :])
    # necessary magic gain to make the control input strong enough
    # and have the right sign
    #TODO something is wrong about the handling of signs, and it really
    # messes things up when you approach the top from one direction and not the other
    magic_gain = -20
    return magic_gain * ou[1]


def get_model_matrix_mpc(nx, delta_t, sys_params):
    m_cart = sys_params[0] # [kg]
    m_bar = sys_params[1] # [kg]
    l_bar = sys_params[2] # [meters]
    g = sys_params[3] # [m/s^2]

    # model dynamics matrices
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, m_bar * g / m_cart, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (m_cart + m_bar) / (l_bar * m_cart), 0.0]
    ])
    A = np.eye(nx) + delta_t * A

    B = np.array([
        [0.0],
        [1.0 / m_cart],
        [0.0],
        [1.0 / (l_bar * m_cart)]
    ])
    B = delta_t * B

    return A, B


def get_model_matrix_lqr():
    # state matrix
    A = np.array([[0, 1, 0, 0],
                  [0, 0, 7.8878, 0],
                  [0, 0, 0, 1],
                  [0, 0, 7.8878, 0]])

    # input matrix
    # B = np.array([[0], [0.9755], [0], [1.46]])
    B = np.array([[0], [0.9091], [0], [-0.8049]])

    return A, B


def energy_shaping_control(state, sys_params):
    # based on https://github.com/tsitsimis/underactuated/blob/master/examples/cartpole_energy_shaping.py
    # https://homes.cs.washington.edu/~todorov/courses/amath579/Tedrake_notes.pdf
    x, x_dot, theta, theta_dot = state
    m_cart = sys_params[0] # [kg]
    m_bar = sys_params[1] # [kg]
    l_bar = sys_params[2] # [meters]
    g = sys_params[3] # [m/s^2]

    # gains for energy-shaping and PD controllers
    k_e, k_p, k_d = 15, 20, 10

    # define energy terms
    K = 0.5 * m_bar * l_bar**2 * theta_dot**2
    U = m_bar * g * l_bar * np.cos(theta)
    E = K + U

    # define energy set-point
    ## system energy for vertical pole, no cart motion
    E_d = m_bar * g * l_bar

    # both of these u_energy_shaping work, but one includes more terms from the system parameters
    # u_energy_shaping = k_e * theta_dot * np.cos(theta) * (E - E_d)
    t1 = m_bar * l_bar**2 * theta_dot * np.cos(theta)
    t2 = m_bar * l_bar * np.sin(theta) * (g - l_bar * theta_dot)
    u_energy_shaping = k_e * (-1 * t1 + t2) * (E - E_d)

    # add PD controller to keep cart near 0
    u_desired = u_energy_shaping - k_p * x - k_d * x_dot

    # apply collocated partial feedback linearization
    term_1 = (m_cart - m_bar * np.cos(theta) ** 2) * u_desired
    term_2 = m_bar * g * np.sin(theta) * np.cos(theta) + m_bar * l_bar * np.sin(theta) * theta_dot ** 2
    u = term_1 - term_2

    return u


def get_nparray_from_matrix(x):
    """
    get build-in list from matrix
    """
    return np.array(x).flatten()


def standardize_angle(theta):
    theta = np.rad2deg(theta)

    if theta > 180:
        theta -= 360
    elif theta < -180:
        theta += 360
    return np.deg2rad(theta)