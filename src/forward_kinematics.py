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

def fkl( angles, parent, offset, rotInd, expmapInd ):
  """
  Convert joint angles and bone lenghts into the 3d points of a person.
  Based on expmap2xyz.m, available at
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

  Args
    angles: 99-long vector with 3d position and 3d joint angles in expmap format
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  Returns
    xyz: 32x3 3d points that represent a person in 3d space
  """

  assert len(angles) == 99

  # Structure that indicates parents for each joint
  njoints   = 32
  xyzStruct = [dict() for x in range(njoints)]

  for i in np.arange( njoints ):

    if not rotInd[i] : # If the list is empty
      xangle, yangle, zangle = 0, 0, 0
    else:
      xangle = angles[ rotInd[i][0]-1 ]
      yangle = angles[ rotInd[i][1]-1 ]
      zangle = angles[ rotInd[i][2]-1 ]

    r = angles[ expmapInd[i] ]

    thisRotation = data_utils.expmap2rotmat(r)
    thisPosition = np.array([xangle, yangle, zangle])

    if parent[i] == -1: # Root node
      xyzStruct[i]['rotation'] = thisRotation
      xyzStruct[i]['xyz']      = np.reshape(offset[i,:], (1,3)) + thisPosition
    else:
      xyzStruct[i]['xyz'] = (offset[i,:] + thisPosition).dot( xyzStruct[ parent[i] ]['rotation'] ) + xyzStruct[ parent[i] ]['xyz']
      xyzStruct[i]['rotation'] = thisRotation.dot( xyzStruct[ parent[i] ]['rotation'] )

  xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
  xyz = np.array( xyz ).squeeze()
  xyz = xyz[:,[0,2,1]]
  # xyz = xyz[:,[2,0,1]]


  return np.reshape( xyz, [-1] )

def revert_coordinate_space(channels, R0, T0):
  """
  Bring a series of poses to a canonical form so they are facing the camera when they start.
  Adapted from
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m

  Args
    channels: n-by-99 matrix of poses
    R0: 3x3 rotation for the first frame
    T0: 1x3 position for the first frame
  Returns
    channels_rec: The passed poses, but the first has T0 and R0, and the
                  rest of the sequence is modified accordingly.
  """
  n, d = channels.shape

  channels_rec = copy.copy(channels)
  R_prev = R0
  T_prev = T0
  rootRotInd = np.arange(3,6)

  # Loop through the passed posses
  for ii in range(n):
    R_diff = data_utils.expmap2rotmat( channels[ii, rootRotInd] )
    R = R_diff.dot( R_prev )

    channels_rec[ii, rootRotInd] = data_utils.rotmat2expmap(R)
    T = T_prev + ((R_prev.T).dot( np.reshape(channels[ii,:3],[3,1]))).reshape(-1)
    channels_rec[ii,:3] = T
    T_prev = T
    R_prev = R

  return channels_rec


def _some_variables():
  """
  We define some variables that are useful to run the kinematic tree

  Args
    None
  Returns
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  """

  parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9,10, 1,12,13,14,15,13,
                    17,18,19,20,21,20,23,13,25,26,27,28,29,28,31])-1

  offset = np.array([0.000000,0.000000,0.000000,-132.948591,0.000000,0.000000,0.000000,-442.894612,0.000000,0.000000,-454.206447,0.000000,0.000000,0.000000,162.767078,0.000000,0.000000,74.999437,132.948826,0.000000,0.000000,0.000000,-442.894413,0.000000,0.000000,-454.206590,0.000000,0.000000,0.000000,162.767426,0.000000,0.000000,74.999948,0.000000,0.100000,0.000000,0.000000,233.383263,0.000000,0.000000,257.077681,0.000000,0.000000,121.134938,0.000000,0.000000,115.002227,0.000000,0.000000,257.077681,0.000000,0.000000,151.034226,0.000000,0.000000,278.882773,0.000000,0.000000,251.733451,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999627,0.000000,100.000188,0.000000,0.000000,0.000000,0.000000,0.000000,257.077681,0.000000,0.000000,151.031437,0.000000,0.000000,278.892924,0.000000,0.000000,251.728680,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999888,0.000000,137.499922,0.000000,0.000000,0.000000,0.000000])
  offset = offset.reshape(-1,3)

  rotInd = [[5, 6, 4],
            [8, 9, 7],
            [11, 12, 10],
            [14, 15, 13],
            [17, 18, 16],
            [],
            [20, 21, 19],
            [23, 24, 22],
            [26, 27, 25],
            [29, 30, 28],
            [],
            [32, 33, 31],
            [35, 36, 34],
            [38, 39, 37],
            [41, 42, 40],
            [],
            [44, 45, 43],
            [47, 48, 46],
            [50, 51, 49],
            [53, 54, 52],
            [56, 57, 55],
            [],
            [59, 60, 58],
            [],
            [62, 63, 61],
            [65, 66, 64],
            [68, 69, 67],
            [71, 72, 70],
            [74, 75, 73],
            [],
            [77, 78, 76],
            []]

  expmapInd = np.split(np.arange(4,100)-1,32)

  return parent, offset, rotInd, expmapInd


def main():
  taylor = False
  display_uncertainty = True
  use_sampling = False
  if not taylor:
    if not display_uncertainty:
        # Load all the data
        parent, offset, rotInd, expmapInd = _some_variables()

        # numpy implementation
        with h5py.File( 'samples.h5', 'r' ) as h5f:
            expmap_gt = h5f['expmap/gt/directions_5'][:]
            expmap_pred = h5f['expmap/preds/directions_5'][:]

        nframes_gt, nframes_pred = expmap_gt.shape[0], expmap_pred.shape[0]

        # Put them together and revert the coordinate space
        expmap_all = revert_coordinate_space( np.vstack((expmap_gt, expmap_pred)), np.eye(3), np.zeros(3) )
        expmap_gt   = expmap_all[:nframes_gt,:]
        expmap_pred = expmap_all[nframes_gt:,:]

        # Compute 3d points for each frame
        xyz_gt, xyz_pred = np.zeros((nframes_gt, 96)), np.zeros((nframes_pred, 96))
        for i in range( nframes_gt ):
            xyz_gt[i,:] = fkl( expmap_gt[i,:], parent, offset, rotInd, expmapInd )
        for i in range( nframes_pred ):
            xyz_pred[i,:] = fkl( expmap_pred[i,:], parent, offset, rotInd, expmapInd )

        # === Plot and animate ===
        fig = plt.figure()
        ax = plt.gca(projection='3d')
        ob = viz.Ax3DPose(ax)

        # Plot the conditioning ground truth
        for i in range(nframes_gt)[-25:]:
            ob.update( xyz_gt[i,:] )
            plt.show(block=False)
            fig.canvas.draw()
            plt.pause(0.04)

        # Plot the prediction
        for i in range(nframes_pred)[:10]:
            ob.update( xyz_pred[i,:], rcolor="#f06090", lcolor="#6090b0" )
            plt.show(block=False)
            fig.canvas.draw()
            plt.pause(0.04)
    else:
        if use_sampling:
            parent, offset, rotInd, expmapInd = _some_variables()
            action = "directions"
            subject = 1
            subaction = 2
            target_frame = 190
            history = 8
            true_frames = 50
            pred_frames = 5
            model_dir = "model_results/model_all_5_8000"
            data = data_utils.load_data(os.path.normpath("./data/h3.6m/dataset"), [subject], [action], False)
            data = data[0][(subject, action, subaction, "even")]

            model = torch.load(model_dir)
            poses_in = data[target_frame-true_frames:target_frame]

            ### Code to test zeroing everything except the arms, does alright (for the arms)  ###
            #itt = 51
            #
            #for i in ( [*range(itt)] ):
            #    if abs(poses_in[0][i]) > .000001:
            #        poses_in[:, i] = np.random.normal(0, 0.003, (poses_in.shape[0]))
            #####################################################################################

            (poses, covars) = model_caller.get_both(100, model, poses_in, true_frames-1)

            xyz_gt, xyz_pred = np.zeros((true_frames, 96)), np.zeros((pred_frames, 96))
            for i in range(true_frames):
                xyz_gt[i, :] = fkl(data[target_frame - true_frames:target_frame][i, :], parent, offset, rotInd, expmapInd)
            for i in range(pred_frames):
                xyz_pred[i, :] = fkl(poses[i, :], parent, offset, rotInd, expmapInd)

            # === Plot and animate ===
            fig = plt.figure()
            ax = plt.gca(projection='3d')
            ob = viz.Ax3DPose(ax)

            # Plot the conditioning ground truth
            for i in range(true_frames):
                ob.update(xyz_gt[i, :])
                plt.show(block=False)
                fig.canvas.draw()
                plt.pause(0.06)

            # Plot the prediction
            for i in range(pred_frames):
                for j in range(15):
                    sample_pose = np.zeros((99))
                    for k in range(33):
                        sample_pose[k*3:k*3+3] += np.random.multivariate_normal(poses[i, k*3:k*3+3], covars[i, k])
                    xyz_sample = fkl(sample_pose, parent, offset, rotInd, expmapInd)
                    ob.update(xyz_sample, rcolor="#ffa0c0", lcolor="#a0c0e0")
                    plt.show(block=False)
                    fig.canvas.draw()
                    plt.pause(0.03)
                ob.update(xyz_pred[i, :], rcolor="#f06090", lcolor="#6090b0")
                plt.show(block=False)
                fig.canvas.draw()
                plt.pause(0.2)
        else:
            parent, offset, rotInd, expmapInd = _some_variables()
            action = "directions"
            subject = 1
            subaction = 1
            target_frame = 190
            history = 8
            true_frames = 50
            pred_frames = 5
            model_dir = "model_results/model_uncertainty_directions_25_5_10000"
            data = data_utils.load_data(os.path.normpath("./data/h3.6m/dataset"), [subject], [action], False)
            data = data[0][(subject, action, subaction, "even")]

            model = torch.load(model_dir)
            poses_in = data[target_frame - true_frames:target_frame]

            means, sigmas = model_caller.predict(model, poses_in, true_frames - 1, use_noise=False)
            xyz_gt, xyz_pred = np.zeros((true_frames, 96)), np.zeros((pred_frames, 96))
            for i in range(true_frames):
                xyz_gt[i, :] = fkl(data[target_frame - true_frames:target_frame][i, :], parent, offset, rotInd,
                                   expmapInd)
            for i in range(pred_frames):
                xyz_pred[i, :] = fkl(means[i, :], parent, offset, rotInd, expmapInd)

            # === Plot and animate ===
            fig = plt.figure()
            ax = plt.gca(projection='3d')
            ob = viz.Ax3DPose(ax)

            # Plot the conditioning ground truth
            for i in range(true_frames):
                ob.update(xyz_gt[i, :])
                plt.show(block=False)
                fig.canvas.draw()
                plt.pause(0.06)

            # Plot the prediction
            for i in range(pred_frames):
                for j in range(15):
                    sample_pose = np.random.normal(means[i], sigmas[i])
                    xyz_sample = fkl(sample_pose, parent, offset, rotInd, expmapInd)
                    ob.update(xyz_sample, rcolor="#ffa0c0", lcolor="#a0c0e0")
                    plt.show(block=False)
                    fig.canvas.draw()
                    plt.pause(0.03)
                ob.update(xyz_pred[i, :], rcolor="#f06090", lcolor="#6090b0")
                plt.show(block=False)
                fig.canvas.draw()
                plt.pause(0.2)
  else:
    parent, offset, rotInd, expmapInd = _some_variables()
    #directions, 1, 1, 180, 8, 10, 10 flips out
    action = "directions"
    subject = 1
    subaction = 1
    target_frame = 180
    history = 8
    true_frames = 10
    pred_frames = 10
    data = data_utils.load_data(os.path.normpath("./data/h3.6m/dataset"), [subject], [action], False)
    data = data[0][(subject, action, subaction, "even")]

    preds = []
    derivs = model_caller.estimate_derivatives(data[target_frame-history:target_frame], 2, 1)
    preds = model_caller.taylor_approximation(derivs, pred_frames)
    
    xyz_gt, xyz_pred = np.zeros((true_frames, 96)), np.zeros((pred_frames, 96))
    for i in range(true_frames):
        xyz_gt[i, :] = fkl(data[target_frame-true_frames:target_frame][i, :], parent, offset, rotInd, expmapInd)
    for i in range(pred_frames):
        xyz_pred[i, :] = fkl(preds[i, :], parent, offset, rotInd, expmapInd)

    # === Plot and animate ===
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ob = viz.Ax3DPose(ax)

    # Plot the conditioning ground truth
    for i in range(true_frames):
        ob.update(xyz_gt[i, :])
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.12)

    # Plot the prediction
    for i in range(pred_frames):
        if i > 0:
            #print( [ ((xyz_pred[i, j] - xyz_pred[i-1, j]) if (xyz_pred[i, j] - xyz_pred[i-1, j] > .1) else 0) for j in range(96)]  )
            print("#########################################")
            print(np.argmax( (xyz_pred[i] - xyz_pred[i-1])) )
            print(max(xyz_pred[i] - xyz_pred[i-1]))
            print(xyz_pred[i][np.argmax( (xyz_pred[i] - xyz_pred[i-1]) )])
        ob.update(xyz_pred[i, :], rcolor="#f06090", lcolor="#6090b0")
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.12)

if __name__ == '__main__':
  main()
