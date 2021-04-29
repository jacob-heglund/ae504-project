import numpy as np
from time import time
import gym
import cartpole_noise
from gym.wrappers.monitoring import video_recorder
import matplotlib.pyplot as plt
import pdb

from controllers import mpc_control


def main(init_state, end_state, Q, R, sys_params, render=True):
    """Renders MPC-controlled actions on a cartpole system

    Args:
    """
    # initialize environment
    env = gym.make('cp-custom-v0')
    video_path = './videos/' + str(time()).replace(".", "") + '.mp4'
    vid = video_recorder.VideoRecorder(env, path=video_path)
    env.seed(1) # seed for reproducibility

    obs = env.reset(init_state=init_state,
                    end_state=end_state,
                    action_noise_std=sys_params[4],
                    obs_noise_std=sys_params[5]
                    )

    n_states = obs.shape[0]
    done = False

    n_steps = 1000
    x_OUT = np.zeros((n_steps, 4))
    u_OUT = np.zeros((n_steps))

    for i in range(n_steps):
        if render:
            env.render()
            vid.capture_frame()

        #TODO figure out where in the cost function I can include magic gains to actually make it work nicely (you need a scaling like this for the inputs to be anywhere close to adequate)
        #TODO figure out where this negative is coming from (I need it for anything to work)
        action = -5*mpc_control(obs, Q, R, sys_params)

        # if i % 10 == 0:
        #     print(action)

        # apply action
        obs, reward, done, _ = env.step(action)
        x_OUT[i, :] = obs.squeeze()
        u_OUT[i] = action

        if done:
            print(f'Terminated after {i+1} iterations \n Final State: {obs}\n Target State: {xf}')
            break

    return x_OUT, u_OUT


if __name__ == '__main__':
    # define system parameters
    mass_cart = 1.0  # [kg]
    mass_bar = 0.3  # [kg]
    length_bar = 0.5  # length of bar
    gravity = 9.8  # [m/s^2]
    action_noise_std = 0.0
    obs_noise_std = 0.0
    sys_params = (mass_cart, mass_bar, length_bar, gravity, action_noise_std, obs_noise_std)

    # define initial and end states
    # state consists of x, x_dot, theta, theta_dot
    # theta = 0 is when the bar is vertical
    x0 = np.array([0.5, 0, -0.1, 0]).reshape(-1, 1)
    xf = np.array([0, 0, 0, 0]).reshape(-1, 1)

    # define reward
    Q = np.diag([1.0, 1.0, 10.0, 10.0])
    R = np.diag([1])

    x_OUT, u_OUT = main(x0, xf, Q, R, sys_params)
    np.save("x_OUT.npy", x_OUT)
    np.save("u_OUT.npy", u_OUT)


