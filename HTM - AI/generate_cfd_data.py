import os

from flowPySim import runCFD
from flowPyVisualizer import visualize
from parameters_htm_cfd import u_in

def main():
    cwd = os.getcwd()  # safe the current working directory in cwd

    dir_path = os.path.join(cwd, "result/u_in_{}".format(u_in))  # create "result" folder
    print(u_in)
    runCFD(u_in, dir_path)  # run CFD simulation & create/save the data
    visualize(dir_path)  # visualize the simulation


if __name__ == "__main__":
    main()
