import numpy as np
import matplotlib.pyplot as plt
import pdb


def state_and_input_vs_time(x_OUT, u_OUT, save_loc):
    # good size for full-width figures
    fig, axs = plt.subplots(nrows=2, figsize=(7.1, 2.5), dpi=100)
    fontsize = 8

    #TODO
    # make ticks invisible for this plot
    # make tick fontsize same as legend fontsize
    # make axis labels

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

if __name__ == "__main__":
    datetime_load = "16-03-39--05-06-2021"
    results_dir = f"./results/{datetime_load}/"
    x_OUT = np.load(results_dir + "x_OUT.npy")
    u_OUT = np.load(results_dir + "u_OUT.npy")
    state_and_input_vs_time(x_OUT, u_OUT, save_loc=f"{results_dir}/state_and_input_vs_time.png")
