import torch

import data_utils
import numpy as np

def predict(model, poses_in, current):
    """Use a saved model on the sequence of input poses.
    Args
        model: the saved pytorch model
        poses_in: The sequence of input poses. Expects an (n, 99)? numpy array.
        current: The index of the current pose in the sequence to predict from.
    Returns
        poses_out: A numpy array of size (seq_length, 99)? output. seq_length
         will be taking from the saved model.
        covariances: A numpy array of size (seq_length, 33, 3, 3) specifying the
         covariance matrix for each corresponding pose in poses_out
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


#poses: (samples, output_frames, 99)
#covars: (output_frames, 33, 3, 3)
def get_covars(poses):
    covars = np.zeros((poses.shape[1], 33, 3, 3), float)
    for i in range(0, 33):
        for j in range(poses.shape[0]):
            poses_subsection = poses[:, j, i*3 : i*3+3]
            covars[j, i, :, :] = np.cov(poses_subsection, rowvar=False)
    return covars

def get_both(n_samples, model, poses_in, current_frame):
    pose_samples = np.zeros((n_samples, model.target_seq_len, 99), float)

    for i in range(n_samples):
        pose_samples[i, :, :] = predict(model, poses_in, current_frame)

    poses_out = np.mean(pose_samples, 0)
    covars = get_covars(pose_samples)

    return (poses_out, covars)


#Hardcoded as a test
if __name__ == "__main__":
    filename  = "./data/h3.6m/dataset/S1/walking_1.txt"
    model_location = "./experiments/walking/out_25/iterations_10000/tied/sampling_based/omit_one_hot/depth_1/size_1024/lr_0.005/residual_vel/model_10000"
    action_sequence = data_utils.readCSVasFloat(filename)
    model = torch.load(model_location)
    prediction = get_both(5, model, action_sequence, 70)
    pass
