'''
The training code for our plane recovery network in ECCV-2018 paper

Recovering 3D Planes from a Single Image via Convolutional Neural Networks

@author -- Fengting Yang
@created time -- Feb.01. 2018
@final modification -- Jul.11.2018

@Usage:
    Train our plane recovering network with chosen parameter
    The path of dataset and check_file should be modified in different machine

The code was tested on Tensorflow 1.4.1.
As Tensorflow changed the tf.app.flags module in higher version, related modification is required.

if the default weight does not works for your for any reason
try 0.08 0.092 0.11 0.13, or a line search in [0.08, 0.13] may be help 
'''

from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
from RecoverPlane import RecoverPlane

import os

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("log_dir", "", "Dataset directory")
flags.DEFINE_string("init_checkpoint_file",'', "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.99, "Momentum term of adam")
flags.DEFINE_float("beta2", 0.9999, "Momentum term of adam")
flags.DEFINE_float("plane_weight",0.1, "Weight for plane regularization")
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 192, "Image height")
flags.DEFINE_integer("img_width", 320, "Image width")
flags.DEFINE_integer("max_steps",500000 , "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 1000, "Logging every log_freq iterations")
flags.DEFINE_integer("num_plane",5, "The estimated number of planes in the scenario")
flags.DEFINE_integer("save_latest_freq", 5000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_boolean("debug", True, "debug mode?")
flags.DEFINE_string("gpu", "0", "GPU ID")


check_file = flags.FLAGS.log_dir + '/' + flags.FLAGS.dataset_dir.split("/")[-1] +\
            '_lr=' + str(flags.FLAGS.learning_rate) +\
            '_b1=' + str(flags.FLAGS.beta1) + '_b2=' + str(flags.FLAGS.beta2) +\
             '_weight=' + str(flags.FLAGS.plane_weight)  +\
             '_n_plane=' + str(flags.FLAGS.num_plane)


flags.DEFINE_string("checkpoint_dir", check_file, "Directory name to save the checkpoints")

FLAGS = flags.FLAGS  # this is used to transfer all the params need during the app.run

def main(_):
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags) #change this to pp.pprint(tf.app.flags.FLAGS.flag_values_dict()) for tensorflow 1.5 or higher
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
        
    planeRecover = RecoverPlane()
    planeRecover.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()

