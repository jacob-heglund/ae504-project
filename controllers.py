import cvxpy
import numpy as np
import time
import pdb


def mpc_control(x0, Q, R, sys_params):
    #TODO don't hard-define these here
    nx = 4 # number of states
    nu = 1 # number of inputs
    T = 5 # time horizon
    delta_t = 0.1 # time step

    x = cvxpy.Variable((nx, T + 1))
    u = cvxpy.Variable((nu, T))

    A, B = get_model_matrix(nx, delta_t, sys_params)

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
    return 1*ou[1]


def get_model_matrix(nx, delta_t, sys_params):
    M = sys_params[0] # [kg]
    m = sys_params[1] # [kg]
    l_bar = sys_params[2] # [meters]
    g = sys_params[3] # [m/s^2]

    # Model Parameter
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]
    ])
    A = np.eye(nx) + delta_t * A

    B = np.array([
        [0.0],
        [1.0 / M],
        [0.0],
        [1.0 / (l_bar * M)]
    ])
    B = delta_t * B

    return A, B

def get_nparray_from_matrix(x):
    """
    get build-in list from matrix
    """
    return np.array(x).flatten()
