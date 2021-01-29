from __future__ import division

import torch
import model_caller
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import viz
import time
import os
import copy
import data_utils
import forward_kinematics

def main():
    parent, offset, rotInd, expmapInd = forward_kinematics._some_variables()


    # directions, 1, 1, 180, 8, 10, 10 flips out

    angles_to_wiggle = range(5)

    # === Plot and animate ===
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ob = viz.Ax3DPose(ax)

    # Plot the prediction
    for i in angles_to_wiggle:
        time = 0
        pose = np.zeros((99))
        while time < 7:
            pose[i] = np.sin(time)
            xyz = forward_kinematics.fkl(pose, parent, offset, rotInd, expmapInd)
            time += 0.05

            ob.update(xyz, rcolor="#f06090", lcolor="#6090b0")
            plt.show(block=False)
            fig.canvas.draw()

            plt.pause(0.02)

if __name__ == '__main__':
    main()
