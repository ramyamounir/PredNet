import tensorflow as tf
from tensorboard import program
import numpy as np
import cv2
import os, sys, time, argparse, shutil
from tqdm import tqdm

################################### HELPER FUNCTIONS ###################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default="my_model")
    parser.add_argument('-gpu', type=str, default="0")
    parser.add_argument('-path', type=str, default="./")
    parser.add_argument('-drop', type=float, default=0.0)
    parser.add_argument('-reg', type=float, default=0.0)
    parser.add_argument('-lr', type=float, default=0.001)
    args = parser.parse_args()

    return args

def tb_init(logdir = "./logs/"):

    if os.path.exists(logdir):
        shutil.rmtree(logdir)

    train_writer = tf.summary.create_file_writer(logdir + "train/")
    test_writer = tf.summary.create_file_writer(logdir+ "test/")

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', "./logs/", '--samples_per_plugin', "images=0"])
    url = tb.launch()
    print("TensorBoard started at URL: {}".format(url))

    return train_writer, test_writer

args = parse_args() #options [model, gpu, path, drop, reg, lr]
model_name = args.model
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

train_writer, test_writer = tb_init("./logs/" + model_name + "/")

####################################### DEFINE MODELS #######################################

class PredNet (tf.keras.Model):
	def __init__(self):

		super(PredNet, self).__init__()

		# layer 1
		self.ConvLSTM1 = tf.keras.layers.convLSTM2D(filters = 2048, kernel_size = (3,3), strides = 1, padding = 'same')
		self.targetConv1 = tf.keras.layers.Conv2D(filters = 2048, kernel_size = (3,3), strides = 1, padding = 'same')
		self.predConv1 = tf.keras.layers.Conv2D(filters = 2048, kernel_size = (3,3), strides = 1, padding = 'same')
		self.maxPool1 = tf.keras.layers.MaxPool2D(pool_size = (2,2))
		self.relu1 = tf.keras.layers.ReLU()

		# layer 2
		self.ConvLSTM2 = tf.keras.layers.convLSTM2D(filters = 2048, kernel_size = (3,3), strides = 1, padding = 'same')
		self.targetConv2 = tf.keras.layers.Conv2D(filters = 2048, kernel_size = (3,3), strides = 1, padding = 'same')
		self.predConv2 = tf.keras.layers.Conv2D(filters = 2048, kernel_size = (3,3), strides = 1, padding = 'same')
		self.maxPool2 = tf.keras.layers.MaxPool2D(pool_size = (2,2))
		self.relu2 = tf.keras.layers.ReLU()
		self.unpool2 = tf.keras.layers.Upsampling2D(size = (2,2))

		# layer 3
		self.ConvLSTM3 = tf.keras.layers.convLSTM2D(filters = 2048, kernel_size = (3,3), strides = 1, padding = 'same')
		self.targetConv3 = tf.keras.layers.Conv2D(filters = 2048, kernel_size = (3,3), strides = 1, padding = 'same')
		self.predConv3 = tf.keras.layers.Conv2D(filters = 2048, kernel_size = (3,3), strides = 1, padding = 'same')
		self.maxPool3 = tf.keras.layers.MaxPool2D(pool_size = (2,2))
		self.relu3 = tf.keras.layers.ReLU()

	def call(self, x, E1, E2, E3, R1, R2, R3):

		# top down updates
		output_3 , hidden_state_3 , cell_state_3 = self.ConvLSTM3(tf.concat([E3, R3]))
		output_2 , hidden_state_2 , cell_state_2 = self.ConvLSTM2(tf.concat([E2, R2, self.unpool(output_3)]))
		output_1 , hidden_state_1 , cell_state_1 = self.ConvLSTM1(tf.concat([E1, R1, self.unpool(output_2)]))

		# Regular Updates layer 1
		pred1 = self.predConv1(output_1)
		pred1 = self.relu1(pred1)

		E1 = self.relu1(tf.subtract(pred1, x))
		E1_target = self.maxPool1(self.relu1(self.targetConv1(E1)))

		# Regular Updates layer 1
		pred2 = self.predConv2(output_2)
		pred2 = self.relu2(pred2)

		E2 = self.relu2(tf.subtract(pred2, E1_target))
		E2_target = self.maxPool2(self.relu2(self.targetConv2(E2)))	

		# Regular Updates layer 1
		pred3 = self.predConv2(output_3)
		pred3 = self.relu2(pred3)

		E3 = self.relu3(tf.subtract(pred3, E2_target))

		return E1, E2, E3, R1, R2, R3

####################################### BUILD MODELS ########################################

# PredNet model
predictModel = PredNet()

# Optimizer for training
optimizer = tf.keras.optimizers.Adam(learning_rate = args.lr)


####################################### TRAIN STEP #########################################

@tf.function
def train_step(input_seq):

	loss = 0

	with tf.GradientTape() as tape:

		for i in input_seq:

			E1, E2, E3, R1, R2, R3 = predictModel(i, E1, E2, E3, R1, R2, R3)
			loss += E1
			loss += E2
			loss += E3

	training_variables = predictModel.training_variables
	gradients = tape.gradient(loss, training_variables)
	optimizer.apply_gradients(zip(gradients, trainable_variables))

	return loss

####################################### PREPARE DATASET #####################################





