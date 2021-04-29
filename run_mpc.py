import numpy as np
from datetime import datetime
from pathlib import Path
import gym
import cartpole_noise
from gym.wrappers.monitoring import video_recorder
import matplotlib.pyplot as plt
import pdb

from controllers import mpc_control


def main(init_state, end_state, Q, R, sys_params, curr_time_str, results_dir,render=False):
    """Renders MPC-controlled actions on a cartpole system

    Args:
    """
    # initialize environment

    env = gym.make('cp-custom-v0')
    if render:
        video_path = results_dir + '/vid.mp4'
        vid = video_recorder.VideoRecorder(env, path=video_path)
    env.seed(1) # seed for reproducibility

    obs = env.reset(init_state=init_state,
                    end_state=end_state,
                    action_noise_std=sys_params[4],
                    obs_noise_std=sys_params[5]
                    )

    n_states = obs.shape[0]
    x_OUT = np.zeros((n_states, 1))
    u_OUT = np.zeros((1))
    done = False
    step_count = 0
    max_steps = 1000

    while True:
        if render:
            env.render()
            vid.capture_frame()

        #TODO figure out where in the cost function I can include magic gains to actually make it work nicely (you need a scaling like this for the inputs to be anywhere close to adequate)
        #TODO figure out where this negative is coming from (I need it for anything to work)
        action = -5*mpc_control(obs, Q, R, sys_params)

        if step_count % 100 == 0:
            print(step_count)

        # apply action
        obs, reward, done, _ = env.step(action)
        x_OUT = np.append(x_OUT, obs, axis=1)
        u_OUT = np.append(u_OUT, action)
        step_count += 1

        if (done) or (step_count >= max_steps):
            print(f'Terminated after {step_count+1} iterations \n Final State: {obs}\n Target State: {xf}')
            break

    env.close()
    return x_OUT[:, 1:], u_OUT[1:]


if __name__ == '__main__':
    now = datetime.now()
    curr_time_str = now.strftime("%H-%M-%S--%m-%d-%Y")
    # curr_time_str = str(time()).replace(".", "")
    render = True
    results_dir = "./results/" + curr_time_str + "/"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # define system parameters
    mass_cart = 1.0  # [kg]
    mass_bar = 0.3  # [kg]
    length_bar = 0.5  # length of bar
    gravity = 9.8  # [m/s^2]
    action_noise_std = 0.0
    obs_noise_std = 0.0
    sys_params = (mass_cart, mass_bar, length_bar, gravity, action_noise_std, obs_noise_std)

    # define initial and end states
    x0 = np.array([0.5, 0, -0.2, 0]).reshape(-1, 1)
    xf = np.array([0, 0, 0, 0]).reshape(-1, 1)

    # define reward
    Q = np.diag([100.0, 1.0, 10.0, 10.0])
    R = np.diag([1])

    x_OUT, u_OUT = main(x0, xf, Q, R, sys_params, curr_time_str, results_dir, render=render)
    np.save(results_dir + "x_OUT.npy", x_OUT)
    np.save(results_dir + "u_OUT.npy", u_OUT)

