import numpy as np
import matplotlib.pyplot as plt
import pdb

datetime_load = "09-40-26--04-29-2021"
load_dir = f"./results/{datetime_load}/"

x_OUT = np.load(load_dir + "x_OUT.npy")
u_OUT = np.load(load_dir + "u_OUT.npy")

def state_and_input_vs_time(x_OUT, u_OUT):
    fig, axs = plt.subplots(nrows=2, figsize=(8, 5), dpi=100)

    ax = axs[0]
    ax.plot(x_OUT[0, :], label="x")
    ax.plot(x_OUT[1, :], label="x_dot")
    ax.plot(x_OUT[2, :], label="theta")
    ax.plot(x_OUT[3, :], label="theta_dot")
    ax.legend()

    ax = axs[1]
    ax.plot(u_OUT, label="input")
    ax.legend()

    fig.savefig(load_dir + "state_and_input_vs_time.png", )

plotting(x_OUT, u_OUT)
