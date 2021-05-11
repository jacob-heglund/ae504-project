import numpy as np
import control
from lqg import LQG
import matplotlib.pyplot as plt

import gym
import cartpole_noise
from gym.wrappers.monitoring import video_recorder
from gym import wrappers
import time


def main():
    # state matrix
    A = np.array([[0, 1, 0, 0],
                  [0, 0, 7.8878, 0],
                  [0, 0, 0, 1],
                  [0, 0, 7.8878, 0]])

    # input matrix
    B = np.array([[0], [0.9091], [0], [-0.8049]])
    C = np.eye(4)
    D = np.zeros((4, 1))
    Q = np.array([[100, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 10, 0],
                  [0, 0, 0, 1]])
    R = np.array([10])

    #covariance matrices for disturbance and noise
    Vd = .001*np.eye(4);
    Vn = .001;

    #build lqg regulator
    lqg = LQG(A, B, C, D, Q, R, Vd, Vn)

    env = gym.make('cp-custom-v0')
    vid = video_recorder.VideoRecorder(env, './videos/' + str(time.time()) + '.mp4')
    env.seed(1)  # seed for reproducibility
    obs = env.reset(init_state=np.array([0.0, 0.0, 0.2, 0.0]).reshape(-1, 1),
                    end_state=np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1, 1))

    for i in range(1000):
        #time.sleep(.1)
        env.render()
        vid.capture_frame()

        noise = lqg.add_noise()

        noisy_obs = obs + noise

        Kf = lqg.kalman_gain(A, noisy_obs, C, Vd)

        pred_state = lqg.get_state_estimate(Kf, noisy_obs, obs, C, A)

        # get force direction (action) and force value (force)
        action, force = lqg.apply_state_controller(lqg.K, pred_state)


        # apply action
        obs, reward, done, _ = env.step(force)
        if done:
            print(f'Terminated after {i + 1} iterations.')
            break
    env.close()


if __name__ == '__main__':
    main()
