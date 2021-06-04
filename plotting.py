import numpy as np
import matplotlib.pyplot as plt
import pdb


def state_and_input_vs_time(x_OUT, u_OUT, save_loc):
    # good size for full-width figures
    fig, axs = plt.subplots(nrows=2, figsize=(7.1, 2.5), dpi=100)
    fontsize = 8

    ax = axs[0]
    ax.plot(x_OUT[0, :], label="x")
    ax.plot(x_OUT[1, :], label="x_dot")
    ax.plot(x_OUT[2, :], label="theta")
    ax.plot(x_OUT[3, :], label="theta_dot")
    ax.axes.xaxis.set_ticklabels([])
    ax.set_ylabel("States", fontsize=fontsize)
    ax.legend(loc="lower right", fontsize = fontsize)

    ax = axs[1]
    ax.plot(u_OUT, label="input")
    ax.set_xlabel("Time Steps", fontsize=fontsize)
    ax.set_ylabel("Input", fontsize=fontsize)
    ax.legend(loc="lower right", fontsize = fontsize)

    plt.tight_layout()
    fig.savefig(save_loc)


def convergence(converged_OUT, theta_range, theta_dot_range, save_loc):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    fontsize=8
    ax.imshow(converged_OUT)
    ax.set_xlabel("$\\theta_0$", fontsize=fontsize)
    ax.set_ylabel("$\\dot{\\theta_0}$", fontsize=fontsize)
    step = 10


    ticks = np.arange(0, len(theta_range), step=step)

    ax.set_xticks(ticks)
    ax.set_xticklabels(np.round(theta_range[0:-1:step], 1), fontsize=fontsize)
    ax.set_yticks(ticks)
    ax.set_yticklabels(np.round(theta_dot_range[0:-1:step], 1),fontsize=fontsize)
    plt.savefig(save_loc, bbox_inches="tight")


def iter_vs_vals(n_iters, values, val_name, save_loc):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    fontsize=8
    ax.semilogx(values, n_iters)
    # ax.plot(values, n_iters)
    ax.set_xlabel(val_name, fontsize=fontsize)
    ax.set_ylabel("Number of Iterations", fontsize=fontsize)
    plt.savefig(save_loc, bbox_inches="tight")


if __name__ == "__main__":
    """
    # for MPC states vs time plot
    # datetime_load = "16-03-39--05-06-2021"
    # results_dir = f"./results/{datetime_load}/"
    # x_OUT = np.load(results_dir + "x_OUT.npy")
    # u_OUT = np.load(results_dir + "u_OUT.npy")
    # state_and_input_vs_time(x_OUT, u_OUT, save_loc=f"{results_dir}/state_and_input_vs_time.png")
    """

    """
    # LQR region of attraction in \theta, \theta_dot phase space
    # datetime_load = "16-20-24--05-11-2021"
    results_dir = f"./results/{datetime_load}/"
    theta_range = np.load(results_dir + "theta_range.npy")
    theta_dot_range = np.load(results_dir + "theta_dot_range.npy")
    converged_OUT = np.load(results_dir + "converged_OUT.npy")
    convergence(converged_OUT, theta_range, theta_dot_range, save_loc=f"{results_dir}/convergence.png")
    """
    # for varying Q_vals_0
    # datetime_load="20-04-35--05-11-2021"
    # results_dir = f"./results/{datetime_load}/"
    # val = "Q[0]"
    # n_iters = np.loadtxt(results_dir + "n_iterations.txt")
    # values = np.loadtxt(results_dir + "Q_vals_0.txt")
    # iter_vs_vals(n_iters, values, val, f"{results_dir}/{val}.png")

    # for varying Q_vals_2
    # datetime_load="20-05-16--05-11-2021"
    # results_dir = f"./results/{datetime_load}/"
    # val = "Q[2]"
    # n_iters = np.loadtxt(results_dir + "n_iterations.txt")
    # values = np.loadtxt(results_dir + "Q_vals_2.txt")
    # iter_vs_vals(n_iters, values, val, f"{results_dir}/{val}.png")

    # for varying R_vals
    datetime_load="20-05-35--05-11-2021"
    results_dir = f"./results/{datetime_load}/"
    val = "R"
    n_iters = np.loadtxt(results_dir + "n_iterations.txt")
    values = np.loadtxt(results_dir + "R_vals.txt")
    iter_vs_vals(n_iters, values, val, f"{results_dir}/{val}.png")


