import numpy as np
import random

import forward_kinematics

sampleCount = 10#1000
discreteSamples = 4
# List of indices into the model output.
headBoneChain = [  # Note: I think these are the correct indices?
    3,  4,  5,   # Root
    36, 37, 38,  # Lower spine
    39, 40, 41,  # Upper spine
    42, 43, 44,  # Neck
    45, 46, 47   # Head
]

def generateSamples(means, sigmas, poseObject):
    poses = np.empty((sampleCount, 1, 96))  # Note: forward kinematics converts a 99 wide vector to 96, no idea why.
    for i in range(sampleCount):
        poseAngles = np.zeros((99))
        for j in range(len(headBoneChain)):
            # Generate a single sample for each angle of each relevant bone.
            # Use just the first set of means/sigmas, as that's the least delay from ground truth.
            sample = np.random.normal(means[0, headBoneChain[j]], sigmas[0, headBoneChain[j]], (1, 1, 1))
            poseAngles[j] = sample
        # Forward kinematics constructs a full body pose, but we're only interested in the head chain from the root.
        # Note: We need to reshape from a (96, ) into a (1, 96) array.
        poses[i,:,:] = np.reshape(forward_kinematics.fkl(poseAngles, poseObject.parent, poseObject.offset, poseObject.rotInd, poseObject.expmapInd), (1, 96))

    # Now we have all our pose samples, select a few that are going to cover the most space.
    # ...
    
    # TEMP: Right now just take 4 random poses from the distribution. This is not ideal.
    return random.choices(poses, k=discreteSamples)