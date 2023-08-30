# https://towardsdatascience.com/computational-fluid-dynamics-using-python-modeling-laminar-flow-272dad1ebec
# Original github: https://github.com/gauravsdeshmukh/FlowPy 

import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from flowPy import *
from parameters_htm_cfd import length, breadth, colPoints, rowPoints


def visualize(dir_path):
    # go through files in the directory and store filenames
    filenames = []
    iterations = []
    for root, dirs, files in os.walk(dir_path):
        for datafile in files:
            if "PUV" in datafile:
                filenames.append(datafile)
                no_ext_file = datafile.replace(".csv", "").strip()
                iter_no = int(no_ext_file.split("V")[-1])
                iterations.append(iter_no)
    # discern the final iteration and interval
    initial_iter = np.amin(iterations)
    final_iter = np.amax(iterations)
    inter = (final_iter - initial_iter) / (len(iterations) - 1)
    number_of_frames = len(iterations)
    sorted_iterations = np.sort(iterations)

    # create mesh for X and Y inputs to the figure
    x = np.linspace(0, length, colPoints)
    y = np.linspace(0, breadth, rowPoints)
    [X, Y] = np.meshgrid(x, y)

    # determine indexing for stream plot (10 points only)
    index_cut_x = int(colPoints / 10)
    index_cut_y = int(rowPoints / 10)

    # create blank figure
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(xlim=(0, length), ylim=(0, breadth))

    # create initial contour and stream plot as well as color bar
    p_p, u_p, v_p = read_datafile(0, dir_path)
    ax.set_xlim([0, length])
    ax.set_ylim([0, breadth])
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title("Frame No: 0")
    cont = ax.contourf(X, Y, p_p)
    stream = ax.streamplot(X[::index_cut_y, ::index_cut_x], Y[::index_cut_y, ::index_cut_x],
                           u_p[::index_cut_y, ::index_cut_x], v_p[::index_cut_y, ::index_cut_x], color="k")
    fig.colorbar(cont)
    fig.tight_layout()

    def animate(i):
        # print frames left to be added to the animation
        sys.stdout.write("\rFrames remaining: {0:03d}".format(len(sorted_iterations) - i))
        sys.stdout.flush()
        # get iterations in a sequential manner through sorted_iterations
        iteration = sorted_iterations[i]
        # use the read_datafile function to get pressure and velocities
        p_p, u_p, v_p = read_datafile(iteration, dir_path)
        # clear previous plot and make contour and stream plots for current iteration
        ax.clear()
        ax.set_xlim([0, length])
        ax.set_ylim([0, breadth])
        ax.set_xlabel("$x$", fontsize=12)
        ax.set_ylabel("$y$", fontsize=12)
        ax.set_title("Frame No: {0}".format(i))
        cont = ax.contourf(X, Y, p_p)
        stream = ax.streamplot(X[::index_cut_y, ::index_cut_x],
                               Y[::index_cut_y, ::index_cut_x],
                               u_p[::index_cut_y, ::index_cut_x],
                               v_p[::index_cut_y, ::index_cut_x],
                               color="k")
        if i == 15:
            plt.savefig("cfd.svg")
        return cont, stream

    print("######## Making FlowPy Animation ########")
    print("#########################################")
    anim = animation.FuncAnimation(fig, animate, frames=number_of_frames, interval=50, blit=False)
    movie_path = os.path.join(dir_path, "FluidFlowAnimation.mp4")
    anim.save(r"{0}".format(movie_path))
    print("\nAnimation saved as FluidFlowAnimation.mp4 in result")
