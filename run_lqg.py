import numpy as np
import control
from lqg import LQG
import matplotlib.pyplot as plt

import gym
import cartpole_noise
from gym.wrappers.monitoring import video_recorder
from gym import wrappers
import time

def plotting(x, x_dot, theta, theta_dot, u):
    fig, axs = plt.subplots(nrows=2, figsize=(8, 5), dpi=100)
    ax = axs[0]
    ax.plot(x, label="x")
    ax.plot(x_dot, label="x_dot")
    ax.plot(theta, label="theta")
    ax.plot(theta_dot, label="theta_dot")
    ax.legend(loc='lower right')

    ax = axs[1]
    ax.plot(u, label="input")
    ax.legend()

    plt.show()

    fig.savefig("state_and_input_vs_time.png")

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
    action_noise_std = 0.1
    obs_noise_std = 0.05
    Vd = action_noise_std*np.eye(4)
    Vn = obs_noise_std

    #build lqg regulator
    lqg = LQG(A, B, C, D, Q, R, Vd, Vn)

    env = gym.make('cp-custom-v0')
    # vid = video_recorder.VideoRecorder(env, './videos/' + str(time.time()) + '.mp4')
    env.seed(1)  # seed for reproducibility
    obs = env.reset(init_state=np.array([0.0, 0.0, 0.5, 0.0]).reshape(-1, 1),
                    end_state=np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1, 1),
                    action_noise_std=action_noise_std,
                    obs_noise_std=obs_noise_std)

    x = []
    x_dot = []
    theta = []
    theta_dot = []
    forces = []

    for i in range(1000):
        #time.sleep(.1)
        env.render()
        # vid.capture_frame()

        noise = lqg.add_noise()

        noisy_obs = obs + noise

        Kf, x_hat = lqg.kalman_gain(A, noisy_obs, C, Q, Vd)

        pred_state = lqg.get_state_estimate(Kf, noisy_obs, obs, C, A)

        # get force direction (action) and force value (force)
        action, force = lqg.apply_state_controller(lqg.K, x_hat)
        forces.append(np.asscalar(force))


        # apply action
        obs, reward, done, _ = env.step(force)
        x.append(obs[0,])
        x_dot.append(obs[1,])
        theta.append(obs[2,])
        theta_dot.append(obs[3,])
        if done:
            print(f'Terminated after {i + 1} iterations.')
            break
    x = np.array(x)
    x_dot = np.array(x_dot)
    theta = np.array(theta)
    theta_dot = np.array(theta_dot)
    forces = np.array(forces)
    plotting(x, x_dot, theta, theta_dot, forces)
    env.close()


if __name__ == '__main__':
    main()
