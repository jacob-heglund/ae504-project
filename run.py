import numpy as np
import control
from lqr import LQR

import gym
import cartpole_noise
from gym import wrappers
from time import time

def main():
    # state matrix
    A = np.array([[0, 1, 0, 0],
                  [0, 0, 0.716, 0],
                  [0, 0, 0, 1],
                  [0, 0, 15.76, 0]])

    # input matrix
    B = np.array([[0], [0.9755], [0], [1.46]])
    C = np.eye(4)
    D = np.zeros((4, 1))
    Q = np.array([[10, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 10, 0],
                  [0, 0, 0, 1]])
    R = np.array([10])
    lqr = LQR(A, B, C, D, Q, R)

    # env = gym.make('CartPole-v0')
    env = gym.make('cpn-v0')
    env = wrappers.Monitor(env, './videos/' + str(time()) + '/')
    env.env.seed(1)     # seed for reproducibility
    obs = env.reset()

    for i in range(1000):
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

    env.close()

if __name__ == '__main__':
    main()