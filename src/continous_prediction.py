import numpy as np
import math
import torch
import matplotlib.pyplot as plt

import os
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
        print("Start: {}, end: {}".format(target_frame - true_frames, target_frame+pred_frames))
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
            print("Bone index {} length: {}".format(i, np.linalg.norm(jointOffset)))
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
        print("Head location: {}, rotation: {}".format(headPosition, headRotation))

        # Disabled drawing.
        # Draw the head look direction
        
        self.drawer.draw_look_direction([ lines[8], lines[9] ])

        yaw, pitch, roll = [ 0., 0., 0. ]

        if not flag_GT:
            # Find rotation delta between headRotation and self.lastHeadRotation.
            deltaRotation = sc.spatial.transform.Rotation.from_matrix(np.array(headRotation) * np.linalg.inv(self.lastHeadRotation))
            yaw, pitch, roll = deltaRotation.as_euler("zyx")
            print("pitch yaw roll: {}, {}, {}".format(pitch, yaw, roll))

        else:
            self.lastHeadRotation = headRotation

        return headPosition, sc.spatial.transform.Rotation.from_matrix(np.array(headRotation)).as_euler("zyx", degrees=True), [ yaw, pitch ]

def createOutputFeed():
    if os.name == "nt":
        return win32pipe.CreateNamedPipe(
            r"\\.\pipe\CVRInputFeed",
            win32pipe.PIPE_ACCESS_OUTBOUND,
            win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT,
            1,
            1024,
            1024,
            10000000,
            None
        )
    else:
        return posixmq.Queue("/cvr_predictions") #, maxsize=1024, maxmsgsize=256, serializer=(RawSerializer())

#create input feed, assuming will only run on linux machines

def createInputFeed():
    return posixmq.Queue("/cvr_input") #, maxsize=1024, maxmsgsize=256, serializer=(RawSerializer())

def comparisonMath(directionGT, directionPred, positionGT, positionPred):
    return (directionGT - directionPred),(positionGT - positionPred)
    
def main():
    pastHistoryFrames = 50  # How many frames in the past to sample for future predictions. ( I think? check )
    predictedFrames = 10  # How many frames in advance to speculate.
    model_dir = "model_results/discussion_10_mle"

    # Load the testing dataset, replace this with pose streaming via OpenVR?
    action = "discussion"
    subject = 5
    subaction = 1
    target_frame = 230  # What is this?
    true_frames = pastHistoryFrames
    pred_frames = predictedFrames
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
    ax = plt.gca(projection='3d')
    ob = printPose(ax)
    
    translate.flags.translate_loss_func = "mle"

    print("Total test frames: {}".format(data.shape[0]))

    #pipeHandle = createOutputFeed()
    #TODO: make queue of poses
    #inputHandle = createInputFeed()
    poseSequence = []
    
    #TODO: change to while inputHandle is receiving data
    for i in range(data.shape[0]):  # Range incorrect?
        newMs = time.time()*1000.0
        #TODO: Create prediction.
        """
        if(len(poseSequence) >= pastHistoryFrames):
            poseSequence.pop(0)
        currPose = inputHandle.get(block=(False))
        poseSequence.append(currPose)
        """
        #poses_in = data[target_frame-pastHistoryFrames+i:target_frame+i]
        poses_in = data[i : i + true_frames]
        
        #poses_in = data[target_frame - true_frames + i:target_frame+pred_frames + i]
        print("Model source seq len: {}, model input size: {}".format(model.source_seq_len, model.input_size))
        
        means, sigmas = model_caller.predict(model, poses_in, true_frames - 1, use_noise=False)
        count = 0
        for supertempvar in poses_in[0]:
            if(supertempvar != 0):
                count+=1 
                
        print("there were nonZero: " + str(count))
        
        # Generate our target poses.
        discretePoses = sampling.generateSamples(means, sigmas, ob)
        
        #means, sigmas = model_caller.predict(model, poses_in, model.source_seq_len + 1, use_noise=False)
        print("Generated {} samples in: {:.3f}s".format(len(discretePoses), time.time() - t))

        # Transform the joint data (expmap) to kinematic poses for rendering.
        t = time.time()
        xyz_gt, xyz_pred = ob.expmapToKinematicPose(true_frames, pred_frames, target_frame, data, means, i)
        print("Transformed model data to forward kinematic pose data in {:.3f}s".format(time.time() - t))
        
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
        '''
        # Send the pose data to the client.
        if pipeHandle:
            formatStr = "=Hffffff" + len(discretePoses) * "ff"  # Each prediction includes a delta xy, will include position as well later.
            unpackedPredictions = [ value for sub in predictedHeadDeltas for value in sub ]
            payload = struct.pack(formatStr, len(discretePoses), *gtHeadPos, *gtHeadRot, *unpackedPredictions)
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
                    '''
        ms = time.time()*1000.0
        print("Time between frames {:.2f}".format(ms-newMs))
        
    if pipeHandle:
        if os.name == "nt":
            win32file.CloseHandle(pipeHandle)
        else:
            #close both
            pipeHandle.close()
            inputHandle.close()

if __name__ == '__main__':
  main()
