import torch

import data_utils
import numpy as np

def predict(model, poses_in, current):
    """Use a saved model on the sequence of input poses.
    Args
        model: the saved pytorch model
        poses_in: The sequence of input poses. Expects an (source_seq_length, 99)? numpy array.
        current: The index of the current pose in the sequence to predict from.
    Returns
        poses_out: A numpy array of size (target_seq_length, 99)? output. target_seq_length
         will be taken from the saved model.
    """

    #This standardizes from mean and std of all the data. Might not be best to use on a single case
    data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(poses_in)

    #data_utils expects a dict, nothing fancy, immediately deconvert it afterwards
    poses_dict = {0:poses_in}
    #this also removes the 0's from the input data, reducing to 54 dimensions
    poses_in = data_utils.normalize_data( poses_dict, data_mean, data_std, dim_to_use, [], False )[0]
    
    encoder_inputs = np.zeros((model.source_seq_len - 1, model.input_size), dtype=float)
    decoder_inputs = np.zeros((model.target_seq_len, model.input_size), dtype=float)

    #source_seq_len - 1 preceding frames to current frame (ex: len = 3, current = 10, gives 8, 9, leaves 10 for decoder
    encoder_inputs[:, :] = poses_in[current-model.source_seq_len+1:current, :]
    #Only need to feed in the current pose to start the feedback loop
    #For some reason, decoder_inputs doesn't usually do anything
    #for the future recurrent nodes unless you just feed ground truth to the network.
    decoder_inputs[0, :] = poses_in[current, :]

    encoder_inputs = torch.from_numpy(encoder_inputs).float()
    decoder_inputs = torch.from_numpy(decoder_inputs).float()
    
    encoder_inputs = torch.unsqueeze(encoder_inputs, 0)
    decoder_inputs = torch.unsqueeze(decoder_inputs, 0)

    encoder_inputs += torch.normal(0.0, 0.05, encoder_inputs.shape)
    decoder_inputs += torch.normal(0.0, 0.05, decoder_inputs.shape)
    
    encoder_inputs = encoder_inputs.cuda()
    decoder_inputs = decoder_inputs.cuda()

    out = model(encoder_inputs, decoder_inputs)
    
    #probably the slow step honestly... try gpu or batch?
    out = out.cpu().data.numpy()
    out = out.transpose([1,0,2])

    out_reverted = data_utils.revert_output_format(out, data_mean, data_std, dim_to_ignore, [], False)
    #Looks like this is built to handle batches too...
    return out_reverted[0]


def get_covars(poses):
    """Get the covariance matrices from the samples of the network output.
    Args
        poses: The samples of output poses. Dimensions (n_samples, target_sequence_length, 99)
    Returns
        covars: Covariance matrices. Dimensions (target_sequence_length, 33, 3, 3)
    """
    covars = np.zeros((poses.shape[1], 33, 3, 3), float)
    for i in range(0, 33):
        for j in range(poses.shape[0]):
            poses_subsection = poses[:, j, i*3 : i*3+3]
            covars[j, i, :, :] = np.cov(poses_subsection, rowvar=False)
    return covars

def get_both(n_samples, model, poses_in, current_frame):
    """Use a saved model on the sequence of input poses.
    Args
        n_samples: number of random samples to take
        model: The pre-saved model
        poses_in: The sequence of input poses. Expects an (n, 99)? numpy array.
        current_frame: The index of the current pose in the sequence to predict from.
    Returns
        poses_out: A numpy array of size (seq_length, 99)? output. seq_length
         will be taken from the saved model.
        covars: Covariance matrices. Dimensions (target_sequence_length, 33, 3, 3)
    """
    pose_samples = np.zeros((n_samples, model.target_seq_len, 99), float)

    for i in range(n_samples):
        pose_samples[i, :, :] = predict(model, poses_in, current_frame)

    poses_out = np.mean(pose_samples, 0)
    covars = get_covars(pose_samples)

    return (poses_out, covars)

def estimate_derivatives(poses, which_derivatives, order):
    #make sure we have enough poses
    assert(poses.shape[0] >= which_derivatives+order)
    #coefficients table, indexed by table[derivative][accuracy order]
    table = [       [],
                    [
                         [0.0],
                         [1.0, -1.0],
                         [3.0/2.0, -2.0, 1.0/2.0],
                         [11.0/6.0, -3.0, 3.0/2.0, -1.0/3.0],
                         [25.0/12.0, -4.0, 3.0, -4.0/3.0, 1.0/4.0]],
                    [
                         [0.0, 0.0],
                         [1.0, -2.0, 1.0],
                         [2.0, -5.0, 4.0, -1.0],
                         [35.0/12.0, -26.0/3.0, 19.0/2.0, -14.0/3.0, 11.0/12.0],
                         [15.0/4.0, -77.0/6.0, 107.0/6.0, -13.0, 61.0/12.0, -5.0/6.0]],
                    [
                         [0.0, 0.0, 0.0],
                         [1.0, -3.0, 3.0, -1.0],
                         [5.0/2.0, -9.0, 12.0, -7.0, 3.0/2.0],
                         [17.0/4.0, -71.0/4.0, 59.0/2.0, -49.0/2.0, 41.0/4.0, -7.0/4.0],
                         [49.0/8.0, -29.0, 461.0/8.0, -62.0, 307.0/8.0, -13.0, 15.0/8.0]]

                     ]
    assert(table[which_derivatives][order])

    derivs = []
    #0'th derivative
    derivs.append(poses[-1,:])

    #TODO, check in place here also?

    for deriv in range(1, which_derivatives+1):
        poses_slice = poses[-(deriv+order):, : ]
        table_row = np.flip(np.array(table[deriv][order]))
        derivs.append(np.matmul(poses_slice.T, table_row).T)
    return derivs

def taylor_approximation(derivatives, steps):
    out_poses = []
    for step in range(1, steps+1):
        out_pose = derivatives[0]
        for i in range(1, len(derivatives)):
            #TODO, check not in place:
            out_pose = out_pose + ( (derivatives[i] / np.math.factorial(i))  *  np.power(step, i) )
        out_poses.append(out_pose)
    return np.array(out_poses)
#Hardcoded as a test
if __name__ == "__main__":
    filename  = "./data/h3.6m/dataset/S1/walking_1.txt"
    model_location = "model_testing"
    action_sequence = data_utils.readCSVasFloat(filename)
    model = torch.load(model_location)
    prediction = get_both(5, model, action_sequence, 70)
    pass
