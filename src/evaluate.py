
"""Simple code for training an RNN for motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin

import data_utils
import seq2seq_model
import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse

# Learning
parser = argparse.ArgumentParser(description='Train RNN for human pose estimation')
parser.add_argument('--learning_rate', dest='learning_rate',
                  help='Learning rate',
                  default=0.005, type=float)
parser.add_argument('--learning_rate_decay_factor', dest='learning_rate_decay_factor',
                  help='Learning rate is multiplied by this much. 1 means no decay.',
                  default=0.95, type=float)
parser.add_argument('--learning_rate_step', dest='learning_rate_step',
                  help='Every this many steps, do decay.',
                  default=10000, type=int)
parser.add_argument('--batch_size', dest='batch_size',
                  help='Batch size to use during training.',
                  default=16, type=int)
parser.add_argument('--max_gradient_norm', dest='max_gradient_norm',
                  help='Clip gradients to this norm.',
                  default=5, type=float)
parser.add_argument('--iterations', dest='iterations',
                  help='Iterations to train for.',
                  default=1e5, type=int)
parser.add_argument('--test_every', dest='test_every',
                  help='',
                  default=1000, type=int)
# Architecture
parser.add_argument('--architecture', dest='architecture',
                  help='Seq2seq architecture to use: [basic, tied].',
                  default='tied', type=str)
parser.add_argument('--loss_to_use', dest='loss_to_use',
                  help='The type of loss to use, supervised or sampling_based',
                  default='sampling_based', type=str)
parser.add_argument('--residual_velocities', dest='residual_velocities',
                  help='Add a residual connection that effectively models velocities',action='store_true',
                  default=False)
parser.add_argument('--size', dest='size',
                  help='Size of each model layer.',
                  default=1024, type=int)
parser.add_argument('--num_layers', dest='num_layers',
                  help='Number of layers in the model.',
                  default=1, type=int)
parser.add_argument('--seq_length_in', dest='seq_length_in',
                  help='Number of frames to feed into the encoder. 25 fp',
                  default=50, type=int)
parser.add_argument('--seq_length_out', dest='seq_length_out',
                  help='Number of frames that the decoder has to predict. 25fps',
                  default=10, type=int)
parser.add_argument('--omit_one_hot', dest='omit_one_hot',
                  help='', action='store_true',
                  default=False)
parser.add_argument('--taylor', dest='finite_taylor_extrapolate',
                    help='Whether to augment the network with a taylor series extrapolation from a finite difference scheme of the previous frames', action='store_true',
                    default=False)
# Directories
parser.add_argument('--data_dir', dest='data_dir',
                  help='Data directory',
                  default=os.path.normpath("./data/h3.6m/dataset"), type=str)
parser.add_argument('--train_dir', dest='train_dir',
                  help='Training directory',
                  default=os.path.normpath("./experiments/"), type=str)
parser.add_argument('--action', dest='action',
                  help='The action to train on. all means all the actions, all_periodic means walking, eating and smoking',
                  default='all', type=str)
parser.add_argument('--use_cpu', dest='use_cpu',
                  help='', action='store_true',
                  default=False)
parser.add_argument('--load', dest='load',
                  help='Try to load a previous checkpoint.',
                  default=0, type=int)
parser.add_argument('--sample', dest='sample',
                  help='Set to True for sampling.', action='store_true',
                  default=False)
parser.add_argument('--distribution_output_direct', dest='distribution_output_direct',
                  default=False)

args = parser.parse_args()

train_dir = os.path.normpath(os.path.join( args.train_dir, args.action,
  'out_{0}'.format(args.seq_length_out),
  'iterations_{0}'.format(args.iterations),
  args.architecture,
  args.loss_to_use,
  'omit_one_hot' if args.omit_one_hot else 'one_hot',
  'depth_{0}'.format(args.num_layers),
  'size_{0}'.format(args.size),
  'lr_{0}'.format(args.learning_rate),
  'residual_vel' if args.residual_velocities else 'not_residual_vel'))

print(train_dir)
os.makedirs(train_dir, exist_ok=True)

def create_model(actions, sampling=False):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      args.architecture,
      args.seq_length_in if not sampling else 50,
      args.seq_length_out if not sampling else 100,
      args.size, # hidden layer size
      args.num_layers,
      args.max_gradient_norm,
      args.batch_size,
      args.learning_rate,
      args.learning_rate_decay_factor,
      args.loss_to_use if not sampling else "sampling_based",
      len( actions ),
      not args.omit_one_hot,
      args.residual_velocities,
      args.finite_taylor_extrapolate,
      output_as_normal_distribution = args.distribution_output_direct,
      dtype=torch.float32)

  if args.load <= 0:
    return model

  print("Loading model")
  model = torch.load(train_dir + '/model_' + str(args.load))
  if sampling:
    model.source_seq_len = 50
    model.target_seq_len = 100
  return model

def clean_batch(batch):
    encoder_inputs, decoder_inputs, decoder_outputs = batch
    encoder_inputs = torch.from_numpy(encoder_inputs).float()
    decoder_inputs = torch.from_numpy(decoder_inputs).float()
    decoder_outputs = torch.from_numpy(decoder_outputs).float()
    if not args.use_cpu:
        encoder_inputs = encoder_inputs.cuda()
        decoder_inputs = decoder_inputs.cuda()
        decoder_outputs = decoder_outputs.cuda()
    encoder_inputs = Variable(encoder_inputs)
    decoder_inputs = Variable(decoder_inputs)
    decoder_outputs = Variable(decoder_outputs)
    return (encoder_inputs, decoder_inputs, decoder_outputs)


import flags
def get_loss(output, truth):
    if flags.translate_loss_func == "mse":
        return ( (output-truth)**2 ).mean()
    if flags.translate_loss_func == "me":
        return ( np.abs(output-truth) ).mean()
    if flags.translate_loss_func == "mle":
        assert(output.shape[-1] == truth.shape[-1] * 2)
        means  = output[..., :int(truth.shape[-1])]
        sigmas = output[..., int(truth.shape[-1]):]
        #print("################################")
        neg_log_likelihood = torch.sum(torch.log(torch.pow(sigmas, 2))) / 2.0

        p1 = (means - truth)
        p2 = p1 / sigmas
        p3 = torch.pow(p2, 2)
        #print("Sigma likelihood cont:", neg_log_likelihood)
        neg_log_likelihood += torch.numel(means) / 2.0 * np.log(2.0*3.1415926)
        neg_log_likelihood += torch.sum(p3) / 2.0
        #print("Max Means:", torch.max(means))
        #print("Min Sigmas:", torch.min(sigmas))
        #print("p1 max:", torch.max(torch.abs(p1)))
        #print("p1 avg:", torch.mean(torch.abs(p1)))
        #print("p2 max:", torch.max(torch.abs(p2)))
        #print("p3 max:", torch.max(p3))
        #print("p3 min:", torch.min(p3))
        #print("p3 avg:", torch.mean(p3))
        #print("likelihood:", neg_log_likelihood)
        return neg_log_likelihood

def train():
  """Train a seq2seq model on human motion"""

  actions = define_actions( args.action )

  number_of_actions = len( actions )

  #these will all be expangles
  train_set, test_set, experiment_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
    actions, args.seq_length_in, args.seq_length_out, args.data_dir, not args.omit_one_hot )

  # Limit TF to take a fraction of the GPU memory

  if True:
    model = create_model(actions, args.sample)
    if not args.use_cpu:
        model = model.cuda()

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not args.omit_one_hot )

    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if args.load <= 0 else args.load + 1
    previous_losses = []

    step_time, loss = 0, 0
    optimiser = optim.SGD(model.parameters(), lr=args.learning_rate)
    #optimiser = optim.Adam(model.parameters(), lr=learning_rate, betas = (0.9, 0.999))

    for _ in range( args.iterations ):
      optimiser.zero_grad()
      model.train()

      start_time = time.time()

      # Actual training

      # === Training step ===
      encoder_inputs, decoder_inputs, decoder_outputs = clean_batch(model.get_batch( train_set, not args.omit_one_hot ))
      preds = model(encoder_inputs, decoder_inputs)

      step_loss = get_loss(preds, decoder_outputs)

      # Actual backpropagation
      step_loss.backward()
      optimiser.step()

      step_loss = step_loss.cpu().data.numpy()
      # TODO:
      preds = preds[..., :54]
      #if current_step % 100 == 0:
      #  print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss ))

      step_time += (time.time() - start_time) / args.test_every
      loss += step_loss / args.test_every
      current_step += 1
      # === step decay ===
      if current_step % args.learning_rate_step == 0:
        args.learning_rate = args.learning_rate*args.learning_rate_decay_factor
        optimiser = optim.Adam(model.parameters(), lr=args.learning_rate, betas = (0.9, 0.999))
        print("Decay learning rate. New value at " + str(args.learning_rate))

      #cuda.empty_cache()

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % args.test_every == 0:
        model.eval()

        # === Validation with randomly chosen seeds ===
        encoder_inputs, decoder_inputs, decoder_outputs = clean_batch(model.get_batch( test_set, not args.omit_one_hot ))

        preds = model(encoder_inputs, decoder_inputs)
        #mse_loss = torch.mean( (preds[..., :54] - decoder_outputs)**2)
        val_loss = get_loss(preds, decoder_outputs)
        # TODO:
        preds = preds[..., :54]
        print()
        print("{0: <16} \t|".format("milliseconds"), end="")
        for ms in [80, 160, 320, 400, 560, 1000]:
          print(" {0:5d} \t|".format(ms), end="")
        print()
        model.batch_size = 256 #256
        # === Validation with srnn's seeds ===
        for action in actions:

          # Evaluate the model on the test batches
          #### Evaluate model on action

          encoder_inputs, decoder_inputs, decoder_outputs = clean_batch(model.get_batch(experiment_set, action))
          mle_string = "   lik "+action+"\t\t|"
          mse_string = "   mse "+action+"\t\t|"
          sig_string = "   sig "+action+"\t\t|"

          preds = model(encoder_inputs, decoder_inputs)

          #Exp here means experiment, not expmap
          for steps in range(2, 16, 2):
              preds_exp = preds[ :, steps-1, : ]
              decoder_out_exp = decoder_outputs[ :, steps-1, : ]

              mse_exp = torch.mean( (preds_exp[..., :54] - decoder_out_exp)**2)
              #Actually does a whole batch at once, so this needs to divide to get actual number
              mle_exp = get_loss(preds_exp, decoder_out_exp)/model.batch_size
              sig_exp = torch.mean(preds_exp[..., 54:])

              mse_string += " {0:.5f} \t|".format(mse_exp)
              mle_string += " {0:.2f} \t|".format(mle_exp)
              sig_string += " {0:.5f} \t|".format(sig_exp)
          #print(mse_string)
          print(mle_string)
          #print(sig_string)

          if True:
              srnn_poses = model(encoder_inputs, decoder_inputs)


              srnn_loss = get_loss(srnn_poses, decoder_outputs)
              #TODO:
              srnn_poses = srnn_poses.cpu().data.numpy()
              sigmas = srnn_poses[...,54:]
              srnn_poses = srnn_poses[...,:54]

              srnn_poses = srnn_poses.transpose([1,0,2])

              srnn_loss = srnn_loss.cpu().data.numpy()
              # Denormalize the output
              srnn_pred_expmap = data_utils.revert_output_format( srnn_poses,
                data_mean, data_std, dim_to_ignore, actions, not args.omit_one_hot )

              experiment_predicted_means = np.array(srnn_pred_expmap)
              experiment_truth           = np.array(data_utils.revert_output_format( decoder_outputs.cpu().data.numpy(),
                data_mean, data_std, dim_to_ignore, actions, not args.omit_one_hot ))

              sigmas_reverted = data_utils.revert_output_format(sigmas,
                np.zeros(data_mean.shape), data_std, dim_to_ignore, actions, False)
              sigmas_reverted = np.array(sigmas_reverted)
              experiment_sigmas = sigmas_reverted.copy()

              # Save the errors here
              mean_errors = np.zeros( (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]) )

              # Training is done in exponential map, but the error is reported in
              # Euler angles, as in previous work.
              # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-247769197

              #Literally just here for idx_to_use right now
              N_SEQUENCE_TEST = 8
              for i in np.arange(N_SEQUENCE_TEST):
                eulerchannels_pred = srnn_pred_expmap[i]

                # Convert from exponential map to Euler angles
                if not flags.convert_to_euler_first:
                  for j in np.arange( eulerchannels_pred.shape[0] ):
                    for k in np.arange(3,97,3):
                      eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(
                        data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

                # The global translation (first 3 entries) and global rotation
                # (next 3 entries) are also not considered in the error, so the_key
                # are set to zero.
                # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
                gt_i=np.copy(srnn_gts_euler[action][i])
                gt_i[:,0:6] = 0

                # Now compute the l2 error. The following is numpy port of the error
                # function provided by Ashesh Jain (in matlab), available at
                # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
                idx_to_use = np.where( np.std( gt_i, 0 ) > 1e-4 )[0]

                euc_error = np.power( gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
                euc_error = np.sum(euc_error, 1)
                euc_error = np.sqrt( euc_error )
                mean_errors[i,:] = euc_error

              #Select same indices of sigmas
              sigmas_reverted = sigmas_reverted.mean(1)
              sigmas_reverted = sigmas_reverted[:,idx_to_use]
              sigmas_reverted = sigmas_reverted.mean(1)

              experiment_truth = experiment_truth[:, :, idx_to_use]
              experiment_predicted_means = experiment_predicted_means[:, :, idx_to_use]
              experiment_predicted_means = np.transpose(experiment_predicted_means, (1, 0, 2))


              SMSEs = np.zeros(experiment_truth.shape[0])

              smse_sig = experiment_sigmas.copy()
              smse_mean = experiment_predicted_means.copy()
              if flags.evaluate_do_SMSE:
                  smse_sig = smse_sig[:, 0, :]
                  smse_mean = smse_mean[:, 0, :]

                  smse_sig = smse_sig[:, idx_to_use]

                  SMSEmean = np.zeros((100, 14))
                  for u in range(100):
                  
                    K = 50

                    samples = np.random.normal(smse_mean, smse_sig, (K, 14, 48))

                    MAEs = np.zeros((K, smse_mean.shape[0]))

                    for q in range(K):
                        MAEs[q] = np.sqrt(np.sum( (experiment_truth[:, 0, :] - samples[q])**2 , 1))
                    SMSEs = MAEs
                    #SMSEs = SMSEs.mean(2)
                    SMSEs = SMSEs.min(0)
                    SMSEmean[u] = SMSEs
                  SMSEs = SMSEmean.mean(0)

              experiment_eucerror     = np.sqrt(np.sum( (experiment_truth - experiment_predicted_means)**2 , 2))
              experiment_meaneucerror = np.mean(experiment_eucerror, 1)

              # This is simply the mean error over the N_SEQUENCE_TEST examples
              mean_mean_errors = np.mean( mean_errors, 0 )
              # Pretty print of the results for 80, 160, 240, 320, 400, 480, 560 and 1000 ms
              #print("{0: <16} |".format(action), end="")
              #for ms in [1,3,5,7,9,11,13,24]:
              #  if args.seq_length_out >= ms+1:
              #    print(" {0:.3f} |".format( mean_mean_errors[ms] ), end="")
              #  else:
              #    print("   n/a |", end="")
              #print()
              print("{0: <16} |".format(action), end="")
              for ms in [1, 3, 5, 7, 9, 11, 13, 24]:
                  if args.seq_length_out >= ms + 1:
                      print(" {0:.3f} |".format(experiment_meaneucerror[ms]), end="")
                  else:
                      print("   n/a |", end="")
              print()
              print("{0: <16} |".format("S-MSEs"), end="")
              for ms in [1, 3, 5, 7, 9, 11, 13, 24]:
                  if args.seq_length_out >= ms + 1:
                      print(" {0:.3f} |".format(SMSEs[ms]), end="")
                  else:
                      print("   n/a |", end="")
              print()
              print("{0: <16} |".format("sigmas"), end="")
              for ms in [1, 3, 5, 7, 9, 11, 13, 24]:
                  if args.seq_length_out >= ms + 1:
                      print(" {0:.3f} |".format(sigmas_reverted[ms]), end="")
                  else:
                      print("   n/a |", end="")
              print()
        model.batch_size = 16
        print()
        print("============================\n"
              "Global step:         %d\n"
              "Learning rate:       %.4f\n"
              "Step-time (ms):     %.4f\n"
              "Train loss avg:      %.4f\n"
              "--------------------------\n"
              "Val loss:            %.4f\n"
              "============================" % (current_step,
              args.learning_rate, step_time*1000, loss,
              val_loss))
        with open("training_out.txt", 'a+') as f:
            f.write(action + " " + str(current_step)+": "+str(val_loss)+"\n")
        torch.save(model, train_dir + '/model_' + str(current_step))

        print()
        previous_losses.append(loss)

        # Reset global time and loss
        step_time, loss = 0, 0

        sys.stdout.flush()


def get_srnn_gts( actions, model, test_set, data_mean, data_std, dim_to_ignore, one_hot, to_euler=True ):
  """
  Get the ground truths for srnn's sequences, and convert to Euler angles (by default).
  (the error is always computed in Euler angles).

  Args
    actions: a list of actions to get ground truths for.
    model: training model we are using (we only use the "get_batch" method).
    test_set: dictionary with normalized training data.
    data_mean: d-long vector with the mean of the training data.
    data_std: d-long vector with the standard deviation of the training data.
    dim_to_ignore: dimensions that we are not using to train/predict.
    one_hot: whether the data comes with one-hot encoding indicating action.
    to_euler: whether to convert the angles to Euler format or keep thm in exponential map

  Returns
    srnn_gts_euler: a dictionary where the keys are actions, and the values
      are the ground_truth, denormalized expected outputs of srnns's seeds.
  """
  srnn_gts_euler = {}

  for action in actions:

    srnn_gt_euler = []
    _, _, srnn_expmap = model.get_batch_srnn( test_set, action )

    # expmap -> rotmat -> euler
    for i in np.arange( srnn_expmap.shape[0] ):
      denormed = data_utils.unNormalizeData(srnn_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )

      if to_euler:
        for j in np.arange( denormed.shape[0] ):
          for k in np.arange(3,97,3):
            denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3] ))

      srnn_gt_euler.append( denormed );

    # Put back in the dictionary
    srnn_gts_euler[action] = srnn_gt_euler

  return srnn_gts_euler


def sample():
  """Sample predictions for srnn's seeds"""
  actions = define_actions( args.action )

  if True:
    # === Create the model ===
    print("Creating %d layers of %d units." % (args.num_layers, args.size))
    sampling     = True
    model = create_model(actions, sampling)
    if not args.use_cpu:
        model = model.cuda()
    print("Model created")

    # Load all the data
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
      actions, args.seq_length_in, args.seq_length_out, args.data_dir, not args.omit_one_hot )

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_expmap = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not args.omit_one_hot, to_euler=False )
    srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not args.omit_one_hot )
    # Clean and create a new h5 file of samples
    SAMPLES_FNAME = 'samples.h5'
    try:
      os.remove( SAMPLES_FNAME )
    except OSError:
      pass

    # Predict and save for each action
    for action in actions:

      # Make prediction with srnn' seeds. These will just be in expangles.
      encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn( test_set, action )

      encoder_inputs = torch.from_numpy(encoder_inputs).float()
      decoder_inputs = torch.from_numpy(decoder_inputs).float()
      decoder_outputs = torch.from_numpy(decoder_outputs).float()
      if not args.use_cpu:
        encoder_inputs = encoder_inputs.cuda()
        decoder_inputs = decoder_inputs.cuda()
        decoder_outputs = decoder_outputs.cuda()
      encoder_inputs = Variable(encoder_inputs)
      decoder_inputs = Variable(decoder_inputs)
      decoder_outputs = Variable(decoder_outputs)

      srnn_poses = model(encoder_inputs, decoder_inputs)

      srnn_loss = (srnn_poses[..., :54] - decoder_outputs)**2
      srnn_loss.cpu().data.numpy()
      srnn_loss = srnn_loss.mean()

      srnn_poses = srnn_poses.cpu().data.numpy()
      srnn_poses = srnn_poses.transpose([1,0,2])

      srnn_loss = srnn_loss.cpu().data.numpy()
      # denormalizes too
      srnn_pred_expmap = data_utils.revert_output_format(srnn_poses[..., :54], data_mean, data_std, dim_to_ignore, actions, not args.omit_one_hot )

      # Save the samples
      with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
        for i in np.arange(8):
          # Save conditioning ground truth
          node_name = 'expmap/gt/{1}_{0}'.format(i, action)
          hf.create_dataset( node_name, data=srnn_gts_expmap[action][i] )
          # Save prediction
          node_name = 'expmap/preds/{1}_{0}'.format(i, action)
          hf.create_dataset( node_name, data=srnn_pred_expmap[i] )

      # Compute and save the errors here
      mean_errors = np.zeros( (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]) )

      for i in np.arange(8):

        eulerchannels_pred = srnn_pred_expmap[i]

        for j in np.arange( eulerchannels_pred.shape[0] ):
          for k in np.arange(3,97,3):
            eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(
              data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

        eulerchannels_pred[:,0:6] = 0

        # Pick only the dimensions with sufficient standard deviation. Others are ignored.
        idx_to_use = np.where( np.std( eulerchannels_pred, 0 ) > 1e-4 )[0]

        euc_error = np.power( srnn_gts_euler[action][i][:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
        euc_error = np.sum(euc_error, 1)
        euc_error = np.sqrt( euc_error )
        mean_errors[i,:] = euc_error

      mean_mean_errors = np.mean( mean_errors, 0 )
      print( action )
      print( ','.join(map(str, mean_mean_errors.tolist() )) )

      with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
        node_name = 'mean_{0}_error'.format( action )
        hf.create_dataset( node_name, data=mean_mean_errors )

  return


def define_actions( action ):
  """
  Define the list of actions we are using.

  Args
    action: String with the passed action. Could be "all"
  Returns
    actions: List of strings of actions
  Raises
    ValueError if the action is not included in H3.6M
  """

  actions = ["walking", "eating", "smoking", "discussion",  "directions",
              "greeting", "phoning", "posing", "purchases", "sitting",
              "sittingdown", "takingphoto", "waiting", "walkingdog",
              "walkingtogether"]

  if action in actions:
    return [action]

  if action == "all":
    return actions

  if action == "all_srnn":
    return ["walking", "eating", "smoking", "discussion"]

  raise( ValueError, "Unrecognized action: %d" % action )


def read_all_data(actions, seq_length_in, seq_length_out, data_dir, one_hot):
    """
    Loads data for training/testing and normalizes it ALSO REMOVING UNUSED DIMENSIONS AS DEFINED IN
    normalization_stats()!
    Does nothing to rotation format.

    Args
      actions: list of strings (actions) to load
      seq_length_in: number of frames to use in the burn-in sequence
      seq_length_out: number of frames to use in the output sequence
      data_dir: directory to load the data from
      one_hot: whether to use one-hot encoding per action
    Returns
      train_set: dictionary with normalized training data
      test_set: dictionary with test data
      data_mean: d-long vector with the mean of the training data
      data_std: d-long vector with the standard dev of the training data
      dim_to_ignore: dimensions that are not used becaused stdev is too small
      dim_to_use: dimensions that we are actually using in the model
    """

    # === Read training data ===
    print("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
        seq_length_in, seq_length_out))

    train_subject_ids = [1, 6, 7, 8, 9, 11]
    test_subject_ids  = [5]
    experiment_subject_ids   = [5]

    train_set, complete_train = data_utils.load_data(data_dir, train_subject_ids, actions, one_hot)
    test_set, complete_test = data_utils.load_data(data_dir, test_subject_ids, actions, one_hot)
    experiment_set, experiment_val = data_utils.load_data(data_dir, experiment_subject_ids, actions, one_hot)
    # Convert to euler angles here I guess?

    # Compute normalization stats
    data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train)

    # Normalize -- subtract mean, divide by stdev
    train_set = data_utils.normalize_data(train_set, data_mean, data_std, dim_to_use, actions, one_hot)
    test_set = data_utils.normalize_data(test_set, data_mean, data_std, dim_to_use, actions, one_hot)
    experiment_set = data_utils.normalize_data(experiment_set, data_mean, data_std, dim_to_use, actions, one_hot)
    print("done reading data.")

    return train_set, test_set, experiment_set, data_mean, data_std, dim_to_ignore, dim_to_use



def main():
  if args.sample:
    sample()
  else:
    import sys
    with open("training_out.txt", 'a+') as f:
      f.write("============================================================\n"+str(sys.argv)+"\n")

    train()

if __name__ == "__main__":
    main()
