import numpy as np
import control
from lqr import LQR

import gym
import cartpole_noise
from gym import wrappers
from time import time
import matplotlib
import matplotlib.pyplot as plt
import pdb
# matplotlib.use("macOSX")

def plotting(x):
    plt.plot(x[0, :], label="x")
    plt.plot(x[1, :], label="x_dot")
    plt.plot(x[2, :], label="theta")
    plt.plot(x[3, :], label="theta_dot")
    plt.legend()
    plt.show()

def main(init_state, end_state, Q, R, sys_parms, render=False):
    # state matrix
    A = np.array([[0, 1, 0, 0],
                  [0, 0, 7.8878, 0],
                  [0, 0, 0, 1],
                  [0, 0, 7.8878, 0]])

    # input matrix
    # B = np.array([[0], [0.9755], [0], [1.46]])
    B = np.array([[0], [0.9091], [0], [-0.8049]])
    C = np.eye(4)
    D = np.zeros((4, 1))
    lqr = LQR(A, B, C, D, Q, R)

    # env = gym.make('cp-cont-v0')
    # env = gym.make('CartPole-v0')
    env = gym.make("cp-custom-v0")
    if render:
        env = wrappers.Monitor(env, './videos/' + str(time()) + '/')
        env.env.seed(1)     # seed for reproducibility
    else:
        env.seed(1)

    # obs = env.reset(init_state=[0, 0, 0.1, 0])
    obs = env.reset(init_state=init_state,
                    end_state=end_state,
                    action_noise_std=sys_params[0],
                    obs_noise_std=sys_params[1]
                    )

    states = []

    for i in range(1000):
        env.render()

        # get force direction (action) and force value (force)
        action, force = lqr.apply_state_controller(lqr.K, obs)

        # apply action
        obs, reward, done, _ = env.step(force)
        states.append(obs)
        if done:
            print(f'Terminated after {i+1} iterations.')
            break
    states = np.array(states)
    # plotting(states)
    env.close()

if __name__ == '__main__':
    # define system parameters
    action_noise_std = 0.0
    obs_noise_std = 0.0
    sys_params = (action_noise_std, obs_noise_std)

    # define initial and end states
    x0 = np.array([-1, 0.0, 1.1, 0.0]).reshape(-1, 1)
    xf = np.array([0, 0, 0, 0]).reshape(-1, 1)

    # define reward
    Q = np.diag([100.0, 1.0, 10.0, 10.0])
    R = np.diag([1])

    render=True

    main(x0, xf, Q, R, sys_params, render=render)
