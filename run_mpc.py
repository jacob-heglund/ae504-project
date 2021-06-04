import numpy as np
from datetime import datetime
from pathlib import Path
import gym
import time
import cartpole_noise
from gym.wrappers.monitoring import video_recorder
import matplotlib.pyplot as plt
import pdb

from controllers import mpc_control, energy_shaping_control, get_model_matrix_mpc, get_model_matrix_lqr, standardize_angle
from plotting import state_and_input_vs_time
from lqr import LQR


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

    A, B, C, D = get_model_matrix_lqr()
    lqr = LQR(A, B, C, D, Q, R)
    controller_time = []
    while True:
        if render:
            env.render()
            vid.capture_frame()

        # check if bar is near the top "stable" region
        theta_std = standardize_angle(obs[2])

        # uncomment if you want to do a swing up
        """
        if (-0.5 < theta_std < 0.5):
            # action = mpc_control(obs, Q, R, sys_params)
            _, action = lqr.apply_state_controller(lqr.K, obs)
        else:
            action = energy_shaping_control(obs, sys_params)
        """

        start_time = time.time()
        action = mpc_control(obs, Q, R, sys_params)
        # _, action = lqr.apply_state_controller(lqr.K, obs)
        controller_time.append(time.time() - start_time)

        # if step_count % 100 == 0:
        #     print(step_count)

        # apply action
        obs, reward, done, end_state_reached, _ = env.step(action)
        x_OUT = np.append(x_OUT, obs, axis=1)
        u_OUT = np.append(u_OUT, action)
        step_count += 1

        if (done) or (step_count >= max_steps):
            # print(f'Terminated after {step_count+1} iterations \n Final State: {obs}\n Target State: {xf}')
            # print(np.mean(controller_time))
            break

    env.close()
    return x_OUT[:, 1:], u_OUT[1:], end_state_reached, step_count+1


if __name__ == '__main__':
    now = datetime.now()
    curr_time_str = now.strftime("%H-%M-%S--%m-%d-%Y")
    print(curr_time_str)
    # curr_time_str = str(time()).replace(".", "")
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


    """
    # Single run of control system
    # define reward
    Q = np.diag([100.0, 1.0, 10.0, 10.0])
    R = np.diag([1])
    render = False

    # define initial and end states
    # x0 = np.array([0, 0.0, -0.5, 0.0]).reshape(-1, 1)
    # xf = np.array([0, 0, 0, 0]).reshape(-1, 1)

    x0 = np.array([-1, 0.0, 0.9, 0.0]).reshape(-1, 1)
    xf = np.array([0, 0, 0, 0]).reshape(-1, 1)

    x_OUT, u_OUT, done = main(x0, xf, Q, R, sys_params, curr_time_str, results_dir, render=render)

    # state_and_input_vs_time(x_OUT, u_OUT, save_loc=f"{results_dir}/state_and_input_vs_time.png")
    np.save(results_dir + "x_OUT.npy", x_OUT)
    np.save(results_dir + "u_OUT.npy", u_OUT)
    np.savetxt(results_dir + "Q.txt", Q)
    np.savetxt(results_dir + "R.txt", R)
    np.savetxt(results_dir + "x0.txt", x0)
    np.savetxt(results_dir + "xf.txt", xf)
    """

    # Sweep over Q and R (separately) to see sensitivity
    # define reward
    Q_vals_0 = np.arange(91, 100, step=1)
    Q_vals_2 = np.arange(1, 10, step=1)
    R_vals = [0.3, 1, 10, 100]
    n_iterations = []

    # for i in Q_vals_0:
    #     Q = np.diag([i, 1.0, 10.0, 10.0])
    #     R = np.diag([1.0])

    # for i in Q_vals_2:
    #     Q = np.diag([100.0, 1.0, i, 10.0])
    #     R = np.diag([1.0])

    for i in R_vals:
        Q = np.diag([100.0, 1.0, 10.0, 10.0])
        R = np.diag([i])


        print(i)
        render = False

        x0 = np.array([-1, 0.0, 0.9, 0.0]).reshape(-1, 1)
        xf = np.array([0, 0, 0, 0]).reshape(-1, 1)

        x_OUT, u_OUT, done, iterations = main(x0, xf, Q, R, sys_params, curr_time_str, results_dir, render=render)

        n_iterations.append(iterations)

    # state_and_input_vs_time(x_OUT, u_OUT, save_loc=f"{results_dir}/state_and_input_vs_time.png")
    np.savetxt(results_dir + "n_iterations.txt", n_iterations)
    np.savetxt(results_dir + "Q_vals_0.txt", Q_vals_0)
    np.savetxt(results_dir + "Q_vals_2.txt", Q_vals_2)
    np.savetxt(results_dir + "R_vals.txt", R_vals)


    """
    # Sweep over theta, theta_dot to get region of attraction
    theta_limit = np.pi
    theta_dot_limit = np.pi
    step = 0.1
    theta_range = np.arange(-theta_limit, theta_limit, step)
    theta_dot_range = np.arange(-theta_dot_limit, theta_dot_limit, step)
    converged_array = np.zeros((len(theta_range), len(theta_dot_range)))

    x0 = np.array([0.0, 0.0, np.pi, -0.1]).reshape(-1, 1)
    xf = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)

    for i in range(len(theta_range)):
        theta0 = theta_range[i]
        print(f"Theta = {theta0} ({i+1}/{len(theta_range)})")
        for j in range(len(theta_dot_range)):
            theta_dot0 = theta_dot_range[j]

            x0 = np.array([0.0, 0.0, theta0, theta_dot0]).reshape(-1, 1)
            xf = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)

            x_OUT, u_OUT, done = main(x0, xf, Q, R, sys_params, curr_time_str, results_dir, render=render)

            converged_array[i, j] = done

    plt.matshow(converged_array)
    plt.show()
    np.save(results_dir + "converged_OUT.npy", converged_array)
    np.save(results_dir + "theta_range.npy", theta_range)
    np.save(results_dir + "theta_dot_range.npy", theta_dot_range)
    np.savetxt(results_dir + "Q.txt", Q)
    np.savetxt(results_dir + "R.txt", R)

    """

