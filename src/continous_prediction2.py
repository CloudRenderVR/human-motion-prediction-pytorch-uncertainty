#Note: Some prints are commented out by Ismet, no needed for debug. 
import numpy as np
import math
import torch

import os
import sys

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


import time
import scipy as sc
from scipy import spatial
import data_utils
import model_caller
import forward_kinematics
import test_pyplot
import translate
from time import sleep

import struct
import pickle

if os.name == "nt":
    import win32pipe
    import win32file
else:
    from ipcqueue import posixmq
    from ipcqueue.serializers import RawSerializer

import sampling

#here we attempt to write code...
class printPose(object):  
    def createKinematicPose(self):
        return forward_kinematics._some_variables()

    def __init__(self, ax):
        """
        Create a 3d pose visualizer that can be updated with new poses.
        Args
        ax: 3d axis to plot the 3d pose on
        lcolor: String. Colour for the left part of the body
        rcolor: String. Colour for the right part of the body
        """
        # previous position for calculating the difference:
        self.prevPosition = [None,None,None]
        
        # Start and endpoints of our representation
        self.I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
        self.J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
        # Left / right indicator
        self.LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        self.ax = ax

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        self.parent, self.offset, self.rotInd, self.expmapInd = self.createKinematicPose()

        self.drawer = test_pyplot.AnActuallySaneWayOfDrawingThings(ax, -500, -500, -500, 500, 500, 500)

        self.lastHeadRotation = np.identity(3)

    def getLines(self, xyzPose):
        # Make connection matrix
        lines = []
        
        
        for i in range( len(self.I) ):
            start_point = ( xyzPose[self.I[i]*3 + 0],
                            xyzPose[self.I[i]*3 + 1],
                            xyzPose[self.I[i]*3 + 2] )

            end_point  =  ( xyzPose[self.J[i]*3 + 0],
                            xyzPose[self.J[i]*3 + 1],
                            xyzPose[self.J[i]*3 + 2] )                                 
            lines.append((start_point, end_point))

        return lines

    def expmapToKinematicPose(self, true_frames, pred_frames, target_frame, data, means, index):
        #xyz_gt, xyz_pred = np.zeros((true_frames+pred_frames, 96)), np.zeros((pred_frames, 96))
        xyz_gt, xyz_pred = np.zeros((true_frames, 96)), np.zeros((pred_frames, 96))
        # print("Start: {}, end: {}".format(target_frame - true_frames, target_frame+pred_frames)) # commented out by Ismet, no needed for debug.
        #for i in range(true_frames+pred_frames):
        for i in range(true_frames):
            #xyz_gt[i, :] = forward_kinematics.fkl(data[target_frame - true_frames:target_frame+pred_frames][i, :], self.parent, self.offset, self.rotInd, self.expmapInd)
            xyz_gt[i, :] = forward_kinematics.fkl(data[index:true_frames + index][i, :], self.parent, self.offset, self.rotInd, self.expmapInd)
        
        #for i in range(true_frames):
        #    xyz_gt[i, :] = forward_kinematics.fkl(data[index:index+true_frames][i, :], self.parent, self.offset, self.rotInd, self.expmapInd)

        for i in range(pred_frames):
            xyz_pred[i, :] = forward_kinematics.fkl(means[i, :], self.parent, self.offset, self.rotInd, self.expmapInd)

        return xyz_gt, xyz_pred

    def drawPose(self, kinematicPose):
        lines = self.getLines(kinematicPose)

        # Determine final world space position of the head.
        # The kinematic bone chain is a hierarchy of local transforms, so we need to multiply along the chain to get the final orientation.
        parentTransform = np.identity(4)

        # Affine transformation info: https://www.euclideanspace.com/maths/geometry/affine/matrix4x4/index.htm

        # List of indices into joint angles for bone hierarchy from root to head.
        headBoneChain = [
            3,  4,  5,   # Root
            36, 37, 38,  # Lower spine
            39, 40, 41,  # Upper spine
            42, 43, 44,  # Neck
            45, 46, 47   # Head
        ]

        for i in range(int(len(headBoneChain) / 3)):
            baseIndex = i * 3
            # kinematicPose is a vector of AoS euler angles, zyxzyxzyx...
            jointAngles = [ kinematicPose[headBoneChain[baseIndex + 0]], kinematicPose[headBoneChain[baseIndex + 1]], kinematicPose[headBoneChain[baseIndex + 2]] ]
            localRotation = sc.spatial.transform.Rotation.from_euler("zyx", jointAngles, degrees=True)
            localRotation = localRotation.as_matrix()  # localRotation is a 3x3 matrix.
            localRotation = np.array(localRotation).reshape(3, 3)  # 3x3 numpy array.
            transform = np.resize(localRotation, (4, 4))  # Expand to a 4x4.

            # If we create a line originating at the parent joint's location, and extending towards the local rotation a distance of the bone length,
            # then the endpoint should be the new world space location of the local joint.

            parentJointPosition = parentTransform[0:3, 3]
            parentJointRotation = parentTransform[0:3, 0:3]

            # Use the bone length to create a line, which we can then rotate around the origin.
            boneLength = self.offset[i, :]  # Index is wrong?
            jointOffset = np.array(boneLength)
            # print("Bone index {} length: {}".format(i, np.linalg.norm(jointOffset))) # commented out by Ismet, no needed for debug.
            # Rotate the extended point in space around the origin.
            # Sure we need to add the parent joint position here? We mat mul so this might be a double add
            localJointPosition = parentJointPosition + np.matmul(jointOffset, parentJointRotation)

            x = localJointPosition[0]
            y = localJointPosition[1]
            z = localJointPosition[2]

            # Stitch the translation info into the affine transform.
            # [ r00, r01, r02, x ]
            # [ r10, r11, r12, y ]
            # [ r20, r21, r22, z ]
            # [ 0,   0,   0,   1 ]

            transform[:, 3] = [ x, y, z, 1 ]
            transform[3, 0:3] = [ 0, 0, 0 ]
            
            parentTransform = np.matmul(parentTransform, transform)

        # We now have the fully transformed orientation matrix of the head joint.
        headPosition = transform[0:3, 3]
        headRotation = transform[0:3, 0:3]
        
        return lines, headPosition, headRotation



    def printHead2(self, pose, flag_GT):
        #print head, calculate mouse delta
        lines, headPosition, headRotation = self.drawPose(pose[0])
        # print("Head location: {}, rotation: {}".format(headPosition, headRotation))  # commented out by Ismet, no needed for debug.

        # Disabled drawing.
        # Draw the head look direction
        
        self.drawer.draw_look_direction([ lines[8], lines[9] ])

        yaw, pitch, roll = [ 0., 0., 0. ]

        if not flag_GT:
            # Find rotation delta between headRotation and self.lastHeadRotation.
            deltaRotation = sc.spatial.transform.Rotation.from_matrix(np.array(headRotation) * np.linalg.inv(self.lastHeadRotation))
            yaw, pitch, roll = deltaRotation.as_euler("zyx")
            # print("pitch yaw roll: {}, {}, {}".format(pitch, yaw, roll))  # commented out by Ismet, no needed for debug.

        else:
            self.lastHeadRotation = headRotation

        return headPosition, sc.spatial.transform.Rotation.from_matrix(np.array(headRotation)).as_euler("zyx", degrees=True), [ yaw, pitch ]

def createOutputFeed():
    if os.name == "nt":
        return win32pipe.CreateNamedPipe(
            r"\\.\pipe\CVROutputFeed",
            win32pipe.PIPE_ACCESS_OUTBOUND,
            win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT,
            1,
            1024,
            1024,
            10000000,
            None
        )
    else:
        return posixmq.Queue("/cvr_predictions", serializer=(RawSerializer())) #, maxsize=1024, maxmsgsize=256, serializer=(RawSerializer())

#create input feed, assuming will only run on linux machines

win32PipeName = r"\\.\pipe\CVRInputFeed"

def createInputFeed():
    if os.name == "nt":
        #return win32pipe.CreateNamedPipe(
        #    win32PipeName,
        #    win32pipe.PIPE_ACCESS_INBOUND,
        #    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT,
        #    1,
        #    1024,
        #    1024,
        #    10000000,
        #    None
        #)
        return win32file.CreateFile(win32PipeName, win32file.GENERIC_READ, 0, None, win32file.OPEN_EXISTING, 0, 0)
    else:
        return posixmq.Queue("/cvr_input") #, maxsize=1024, maxmsgsize=256, serializer=(RawSerializer())

def comparisonMath(directionGT, directionPred, positionGT, positionPred):
    return (directionGT - directionPred),(positionGT - positionPred)
    
def Streaming(model, pastHistoryFrames, ob, pipe):
    historicalPoses = np.zeros(shape=(pastHistoryFrames, 99)).to(device)
    frame = 0
    pdr = open("poseDataRaw.txt", "w+")
    # ripped plotting code from forward_kinematics.py:295
    ###########################################################
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    import test_pyplot
    drawer = test_pyplot.AnActuallySaneWayOfDrawingThings(ax, -500, -500, -500, 500, 500, 500)
    def get_lines(xyz):        
        I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1 #
        J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1 #
        lines_to_ret = []
        for j in range(len(I)):
            start_point = ( xyz[I[j]*3 + 0],
                            xyz[I[j]*3 + 1],
                            xyz[I[j]*3 + 2] )
            end_point  =  ( xyz[J[j]*3 + 0],
                            xyz[J[j]*3 + 1],
                            xyz[J[j]*3 + 2] )
            lines_to_ret.append((start_point, end_point))
        return lines_to_ret
    ###########################################################

    while True:
        # Load expmap from the pipe
        buffer = None
        try:
            result, buffer = win32file.ReadFile(pipe, 2048*64, None)
            if result != 0:
                print(f"ERROR: failed to read from named pipe in input streaming with code {result}")
                exit(1)
            print(f"Loaded {len(buffer)} bytes from the input pipe!")
        except win32file.error as e:
            if e.winerror == 536:
                print("Waiting for sending process to open the pipe...")
                sleep(1)
                continue
            else:
                raise(e)

        buffer = pickle.loads(buffer)
        # TEMP: value seem way too small, what happens if we scale up
        #buffer = buffer * 1000.0
        pdr.write(f"{frame}\n")
        pdr.write(str(buffer))
        pdr.write("\n")
        pdr.flush()

        if frame < historicalPoses.shape[0]:
            historicalPoses[frame] = buffer
        else:
            historicalPoses = np.roll(historicalPoses, -1)
            # replace the oldest pose
            historicalPoses[-1] = buffer

        # Render the ground truth
        parent, offset, rotInd, expmapInd = forward_kinematics._some_variables()
        bufferFkl = forward_kinematics.fkl(buffer, parent, offset, rotInd, expmapInd)
        lines = get_lines(buffer)
        drawer.clear()
        drawer.draw_lines(lines, [(1, 0, 0) if lr else (0, 0, 1) for lr in LR])
        drawer.show(f"streamedPose_frame{frame}")
        plt.pause(0.2)
        plt.savefig(f"tempPics/f{frame}.png")

        poses_in = historicalPoses
        means, sigmas = model_caller.predict(model, poses_in, pastHistoryFrames - 1, use_noise=False)
        if means is None or sigmas is None:
            continue # Bad data
        discretePoses = sampling.generateSamples(means, sigmas, ob)
        predHeadPos, predHeadRot, predictionDeltas = ob.printHead2(discretePoses[0], False)

        print("Frame {}, predicted head pos: {}, rot: {}".format(frame, predHeadPos, predHeadRot))
        frame += 1

def main():
    positionStreaming = False
    if "--streaming" in sys.argv:
        positionStreaming = True
        print("Started in streaming mode")
    pastHistoryFrames = 50  # How many frames in the past to sample for future predictions. ( I think? check )
    predictedFrames = 10  # How many frames in advance to speculate.
    model_dir = "model_results/discussion_10_mle"

    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu"
    device = torch.device(dev) 

    # Load the testing dataset, replace this with pose streaming via OpenVR?
    action = "discussion"
    subject = 5
    subaction = 1
    target_frame = 230  # What is this?
    true_frames = pastHistoryFrames
    pred_frames = predictedFrames
    if not positionStreaming:
        t = time.time()
        data = data_utils.load_data(os.path.normpath("./data/h3.6m/dataset"), [subject], [action], False)
        print("Dataset load time: {:.3f}s".format(time.time() - t))
        data = data[0][(subject, action, subaction, "even")]

    # Load the model
    t = time.time()
    model = torch.load(model_dir)
    print("Model load time: {:.3f}s".format(time.time() - t))

    
    #define vars for printing
    fig = plt.figure()
    #ax = plt.gca(projection='3d')
    ax = plt.axes(projection='3d')
    ob = printPose(ax)
    
    translate.flags.translate_loss_func = "mle"

    pipeHandle = createOutputFeed()
    #TODO: make queue of poses
    inputHandle = None
    if positionStreaming:
        inputHandle = createInputFeed()
    poseSequence = []

    if positionStreaming:
        Streaming(model, pastHistoryFrames, ob, inputHandle)
        if os.name == "nt":
            win32file.CloseHandle(inputHandle)
        return

    print("Total test frames: {}".format(data.shape[0]))
    
    #TODO: change to while inputHandle is receiving data
    for i in range(40):  # Range incorrect?
        print(data.shape[0])
        newMs = time.time()*1000.0
        print("start time: ",datetime.utcnow())
        #TODO: Create prediction.
        """
        if(len(poseSequence) >= pastHistoryFrames):
            poseSequence.pop(0)
        currPose = inputHandle.get(block=(False))
        poseSequence.append(currPose)
        """
        #poses_in = data[target_frame-pastHistoryFrames+i:target_frame+i]

        poses_in = data[i : i + true_frames]

        # print(f"poses_in shape: {poses_in.shape}, true_frames={true_frames}, i={i}") # commented out by Ismet, no needed for debug.
        
        #poses_in = data[target_frame - true_frames + i:target_frame+pred_frames + i]
        print("Model source seq len: {}, model input size: {}".format(model.source_seq_len, model.input_size)) # commented out by Ismet, no needed for debug.
        
        # print("before predict",datetime.utcnow())
        # t = time.time()
        # means, sigmas = model_caller.predict(model, poses_in, true_frames - 1, use_noise=False)
        # print("after predict:",datetime.utcnow())
        # print("prediction: {:.3f}s".format(time.time() - t))
        
        for i in range(1):
            # print("before predict",datetime.utcnow())
            t = time.time()
            means, sigmas = model_caller.predict(model, poses_in, true_frames - 1, use_noise=False)
            # print("after predict:",datetime.utcnow())
            print("prediction: {:.3f}s".format(time.time() - t))

        count = 0
        for supertempvar in poses_in[0]:
            if(supertempvar != 0):
                count+=1
                
        # print("there were nonZero: " + str(count)) # commented out by Ismet, no needed for debug.
        
        # Generate our target poses.
        t = time.time()
        discretePoses = sampling.generateSamples(means, sigmas, ob)
        print("generateSamples: {:.3f}s".format(time.time() - t))
        
        #means, sigmas = model_caller.predict(model, poses_in, model.source_seq_len + 1, use_noise=False)
        print("Generated {} samples in: {:.3f}s".format(len(discretePoses), time.time() - t))

        # Transform the joint data (expmap) to kinematic poses for rendering.
        t = time.time()
        # xyz_gt, xyz_pred = ob.expmapToKinematicPose(true_frames, pred_frames, target_frame, data, means, i)
        print("Transformed model data to forward kinematic pose data in {:.3f}s".format(time.time() - t))
        
        
        t = time.time()
        xyz_gt_future = np.zeros((1, 96))
        
        xyz_gt_future[0, :] = forward_kinematics.fkl(data[i+48:true_frames + i+48][0, :], ob.parent, ob.offset, ob.rotInd, ob.expmapInd)
        
        # Render the future ground truth.
        gtHeadPos, gtHeadRot, _ = ob.printHead2(xyz_gt_future, True)
        
        predictedHeadPositions = []
        predictedHeadRotations = []
        predictedHeadDeltas = []

        # Render the predicted frames.
        #for i in range(predictedFrames):
        #    arr = np.zeros((1, 96))
        #    arr[0, :] = xyz_pred[i]
        #    predHeadPos, predHeadRot, predictionDeltas = ob.printHead2(arr, False)
        #    predictedHeadPositions.append(predHeadPos)
        #    predictedHeadRotations.append(predHeadRot)
        #    predictedHeadDeltas.append(predictionDeltas)
        
        # Render the predicted frames (from our custom sampling).
        for i in range(len(discretePoses)):
            predHeadPos, predHeadRot, predictionDeltas = ob.printHead2(discretePoses[i], False)
            predictedHeadPositions.append(predHeadPos)
            predictedHeadRotations.append(predHeadRot)
            predictedHeadDeltas.append(predictionDeltas)
        
        #compare GT (next GT) to predictions:
        xyz_gt_comparison = np.zeros((1, 96))
        xyz_gt_comparison[0, :] = forward_kinematics.fkl(data[i+49:true_frames + i+49][0, :], ob.parent, ob.offset, ob.rotInd, ob.expmapInd)
        gt_comparison_position, gt_comparison_direction, _ = ob.printHead2(xyz_gt_comparison,False)
        
        
        print("until next frame: {:.3f}s".format(time.time() - t))

        #compare with all predictions
        #declare acceptable difference

        acceptableDiffDirection = 50
        acceptableDiffPosition = 50

        correctFrames = []
        """
        for i in range(predictedFrames):
            #do the comparison math
            deltaDirection, deltaPosition = comparisonMath(gt_comparison_direction,predictedHeadRotations[i],gt_comparison_position,predictedHeadPositions[i])
            testVal = sum(deltaDirection)
            if(testVal <= acceptableDiffDirection):
                print("correctly predicted Direction")
            if(sum(deltaPosition) <= acceptableDiffPosition):
                print("correctly predicted Position")
            if(sum(deltaPosition) <= acceptableDiffPosition) and (testVal <= acceptableDiffDirection):
                correctFrames.append(i)
            print("next predicted pose")
        """

        print("Next Frame :)")

        """
        commented out for running on xavier
        # Disabled drawing.
        ob.drawer.show()
        plt.pause(.01)
        ob.drawer.clear()
        """
        ms = time.time()*1000.0
        print("Time between frames - before it's piped {:.2f}".format(ms-newMs))
        print("Current date:",datetime.utcnow())
        
        # Send the pose data to the client.
        if pipeHandle:
            formatStr = "=Hffffff" + len(discretePoses) * "ff"  # Each prediction includes a delta xy, will include position as well later.
            unpackedPredictions = [ value for sub in predictedHeadDeltas for value in sub ]
            payload = struct.pack(formatStr, len(discretePoses), *gtHeadPos, *gtHeadRot, *unpackedPredictions)
            print(f"Generated {len(payload)} byte prediction payload with {len(discretePoses)} poses\n\tgtHeadPos={gtHeadPos}")
            if os.name == "nt":
                try:
                    win32file.WriteFile(pipeHandle, payload)
                    print("Wrote to pipe in " + str(ms))
                except Exception as e:
                    print("Failed to write to pipe: {}".format(e))
            else:
                try:
                    pipeHandle.put(payload, block=False)
                except Exception as e:
                    print("Failed to write to pipe: {}".format(e))
                    
        ms = time.time()*1000.0
        print("Time between frames {:.2f}".format(ms-newMs))
        
    # if pipeHandle:
    #     if os.name == "nt":
    #         win32file.CloseHandle(pipeHandle)
    #     else:
    #         #close both
    #         pipeHandle.close()
    #         inputHandle.close()

if __name__ == '__main__':
  main()
