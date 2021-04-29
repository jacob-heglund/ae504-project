import numpy as np
import matplotlib.pyplot as plt

x_OUT = np.load("x_OUT.npy")
u_OUT = np.load("u_OUT.npy")

def plotting(x_OUT, u_OUT):
    plt.plot(x_OUT[0, :], label="x")
    plt.plot(x_OUT[1, :], label="x_dot")
    plt.plot(x_OUT[2, :], label="theta")
    plt.plot(x_OUT[3, :], label="theta_dot")
    plt.legend()
    plt.show()

plotting(x_OUT, u_OUT)
