# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data_single as input_data
import onlyfc as c3d_model
import numpy as np
import csv
import argparse

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 2
flags.DEFINE_integer('batch_size', 10 , 'Batch size.')
FLAGS = flags.FLAGS

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder

def _variable_on_cpu(name, shape, initializer):
  #with tf.device('/cpu:%d' % cpu_id):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
  return var

def run_test(model_step):
  # model_name = "./sports1m_finetuning_ucf101.model"
  # model_name = "./c3d_ucf101_finetune_whole_iter_20000_TF.model"
  # test_list_file = 'list/test.list'
  # test_list_file = 'list/all_data.list'
  test_list_file = 'list/bg-flow-all.list'
  num_test_videos = len(list(open(test_list_file,'r')))
  print("Number of test videos={}".format(num_test_videos))

  # Get the sets of images and labels for training, validation, and
  images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
  with tf.variable_scope('var_name') as var_scope:
    weights = {
            'out1': _variable_with_weight_decay('wout1', [12544, 4096], 0.04, 0.0005),
            'out2': _variable_with_weight_decay('wout2', [4096, c3d_model.NUM_CLASSES], 0.04, 0.0005)
            }
    biases = {
            'out1': _variable_with_weight_decay('bout1', [4096], 0.04, 0.0),
            'out2': _variable_with_weight_decay('bout2', [c3d_model.NUM_CLASSES], 0.04, 0.0)
            }
  logits = []
  for gpu_index in range(0, gpu_num):
    with tf.device('/cpu:%d' % gpu_index):
      logit = c3d_model.inference_c3d(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:], 0.6, FLAGS.batch_size, weights, biases)
      logits.append(logit)
  logits = tf.concat(logits,0)
  norm_score = tf.nn.softmax(logits)
  # saver = tf.train.Saver()
  # 600 the best
  model_dir = 'models/finetune-models/step_' + str(model_step) + '/'
  saver = tf.train.import_meta_graph(model_dir + 'c3d_bridgestone-' + str(model_step) + '.meta')
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  saver.restore(sess, tf.train.latest_checkpoint(model_dir))
  init = tf.global_variables_initializer()
  sess.run(init)
  # Create a saver for writing training checkpoints.
  # saver.restore(sess, model_name)

  # And then after everything is built, start the training loop.
  bufsize = 0
  write_file = open('predict_ret.txt', 'w+')
  next_start_pos = 0
  all_steps = int((num_test_videos - 1) / (FLAGS.batch_size * gpu_num) + 1)
  tp, fp, tn, fn = 0, 0, 0, 0
  for step in xrange(all_steps):
    # Fill a feed dictionary with the actual set of images and labels
    # for this particular training step.
    start_time = time.time()
    test_images, test_labels, next_start_pos, _, valid_len = \
            input_data.read_clip_and_label(
                    test_list_file,
                    FLAGS.batch_size * gpu_num,
                    num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                    start_pos=next_start_pos
                    )
    predict_score = norm_score.eval(
            session=sess,
            feed_dict={images_placeholder: test_images}
            )
    print(predict_score)
    for i in range(0, valid_len):
      true_label = test_labels[i],
      top1_predicted_label = np.argmax(predict_score[i])
      # Write results: true label, class prob for true label, predicted label, class prob for predicted label
      write_file.write('{}, {}, {}, {}\n'.format(
              true_label[0],
              predict_score[i][true_label],
              top1_predicted_label,
              predict_score[i][top1_predicted_label]))
      
      true_label, top1_predicted_label = int(true_label[0]), int(top1_predicted_label)
      if true_label == 0:
          if top1_predicted_label == true_label:
              tp += 1
          else:
              fn += 1
      else:
          if top1_predicted_label == true_label:
              tn += 1
          else:
              fp += 1
  write_file.close()

  print(tp, fp, tn, fn)
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  f1_score = 2 / ( 1 / precision + 1 / recall )
  with open('result.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([str(model_step), str(f1_score), str(precision), str(recall), str(tp), str(fp), str(tn), str(fn)])
      

    print("done")

def main(_):
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-step", help="the step of the pretrained model")
  args = parser.parse_args()
  model_step = int(args.model_step)

  print("Current model_step: ", model_step)
  run_test(model_step)

if __name__ == '__main__':
  tf.app.run()
