import torch

from src import data_utils
import
import numpy as np

def predict(model, poses_in, current):
    """Use a saved model on the sequence of input poses.
    Args
        model: the saved pytorch model
        poses_in: The sequence of input poses. Expects an (n, 99)? numpy array.
        current: The index of the current pose in the sequence to predict from.
    Returns
        poses_out: A numpy array of size (seq_length, 33, 3)? output. seq_length
         will be taking from the saved model.
        covariances: A numpy array of size (seq_length, 33, 3, 3) specifying the
         covariance matrix for each corresponding pose in poses_out
    """

    #This standardizes from mean and std of all the data. Might not be best to use on a single case
    data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(poses_in)

    poses_dict = {0:poses_in}




    poses_in = data_utils.normalize_data( poses_dict, data_mean, data_std, dim_to_use, [], False )[0]

    encoder_inputs = np.zeros((model.source_seq_len - 1, model.input_size), dtype=float)
    decoder_inputs = np.zeros((model.target_seq_len, model.input_size), dtype=float)

    encoder_inputs[:, :] = poses_in[0:model.source_seq_len - 1, :]

    out = model(encoder_inputs, decoder_inputs)

#Hardcoded as a test
if __name__ == "__main__":
    filename  = "./data/h3.6m/dataset/S1/walking_1.txt"
    model_location = "./experiments/walking/out_25/iterations_10000/tied/sampling_based/one_hot/depth_1/size_1024/lr_0.005/residual_vel/model_10000"
    import pdb; pdb.set_trace()
    action_sequence = data_utils.readCSVasFloat(filename)
    model = torch.load(model_location)
    predict(model, action_sequence, 30)