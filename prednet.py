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

        self.batch_len = 64
        self.hidden_units = 2048

        # layer 1
        self.ConvLSTM1 = tf.keras.layers.ConvLSTM2D(filters = 6, kernel_size = (3,3), strides = 1, padding = 'same', stateful = False)
        self.targetConv1 = tf.keras.layers.Conv2D(filters = 3, kernel_size = (3,3), strides = 1, padding = 'same')
        self.predConv1 = tf.keras.layers.Conv2D(filters = 3, kernel_size = (3,3), strides = 1, padding = 'same')

        # layer 2
        self.ConvLSTM2 = tf.keras.layers.ConvLSTM2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', stateful = False)
        self.targetConv2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same')
        self.predConv2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same')


        # layer 3
        self.ConvLSTM3 = tf.keras.layers.ConvLSTM2D(filters = 256, kernel_size = (3,3), strides = 1, padding = 'same', stateful = False)
        self.targetConv3 = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same')
        self.predConv3 = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same')


        self.relu = tf.keras.layers.ReLU()
        self.unpool = tf.keras.layers.UpSampling2D(size = (2,2))
        self.pool = tf.keras.layers.MaxPool2D(pool_size = (2,2))

    def call(self, x, E1, E2, E3, R1, R2, R3):

        # top down updates
        output_3  = self.ConvLSTM3(tf.concat([E3, R3], axis = -1))
        output_2  = self.ConvLSTM2(tf.concat([E2, R2, tf.expand_dims(self.unpool(output_3), axis = 0)], axis = -1))
        output_1  = self.ConvLSTM1(tf.concat([E1, R1, tf.expand_dims(self.unpool(output_2), axis = 0)], axis = -1))

        # Regular Updates layer 1
        pred1 = self.predConv1(output_1)
        pred1 = self.relu(pred1)
        pred1_clip = tf.clip_by_value(pred1, 0.0, 1.0)

        E1 = self.relu(tf.subtract(pred1, x))

        # Regular Updates layer 2
        pred2 = self.predConv2(output_2)
        pred2 = self.relu(pred2)

        E2 = self.relu(tf.subtract(pred2, self.pool(self.relu(self.targetConv2(E1)))))

        # Regular Updates layer 3
        pred3 = self.predConv3(output_3)
        pred3 = self.relu(pred3)
        E3 = self.relu(tf.subtract(pred3, self.pool(self.relu(self.targetConv3(E2)))))




        return tf.expand_dims(E1, axis = 0), tf.expand_dims(E2, axis = 0), tf.expand_dims(E3, axis = 0), tf.expand_dims(output_1, axis = 0), tf.expand_dims(output_2, axis = 0), tf.expand_dims(output_3, axis = 0), tf.reduce_mean(pred1_clip, axis = 0)

    def reset_states(self):

        E1 = tf.random.normal((1, 1, 512, 512, 3))
        E2 = tf.random.normal((1, 1, 256, 256, 64))
        E3 = tf.random.normal((1, 1, 128, 128, 128))
        R1 = tf.random.normal((1, 1, 512, 512, 6))
        R2 = tf.random.normal((1, 1, 256, 256, 128))
        R3 = tf.random.normal((1, 1, 128, 128, 256))

        return E1, E2, E3, R1, R2, R3

####################################### BUILD MODELS ########################################

# PredNet model
predictModel = PredNet()

# Optimizer for training
optimizer = tf.keras.optimizers.Adam(learning_rate = args.lr)


####################################### TRAIN STEP #########################################

# @tf.function
def train_step(weight, img, E1, E2, E3, R1, R2, R3):
    loss = 0

    with tf.GradientTape() as tape:

        E1, E2, E3, R1, R2, R3, pred = predictModel(img, E1, E2, E3, R1, R2, R3)
        loss = weight * (tf.reduce_mean(E1) + 0.0 * tf.reduce_mean(E2) + 0.0 * tf.reduce_mean(E3))

    trainable_variables = predictModel.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return 1.0, loss, E1, E2, E3, R1, R2, R3, pred

####################################### PREPARE DATASET #####################################

frame_count = 10
snippet_size = 10
vidcap = cv2.VideoCapture(args.path)
num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
steps = int(num_frames/(snippet_size))
weight = 0.0

E1, E2, E3, R1, R2, R3 = predictModel.reset_states()

training_pbar = tqdm(range(steps), desc='Train Step {:>2}/{} (loss _.___)'.format(0, steps), unit = 'steps', total = steps)

for curr_step in training_pbar:

    # Read one snippet
    video_snippet = []
    for i in range(snippet_size):
        success, img = vidcap.read()
        img = cv2.resize(img, (512,512))
        if success:
            video_snippet.append(img)
    if not success:
        break

    for i in range(snippet_size):

        # print(tf.image.per_image_standardization(tf.cast(video_snippet[i], tf.float32)))

        weight, loss, E1, E2, E3, R1, R2, R3, pred = train_step(weight, tf.image.per_image_standardization(tf.cast(video_snippet[i], tf.float32)), E1, E2, E3, R1, R2, R3)

    with train_writer.as_default():
        tf.summary.scalar("Loss", loss, step = frame_count)
        tf.summary.image("Prediction", tf.expand_dims(pred * 255, axis = 0) , step = frame_count)
        tf.summary.image("Target", tf.expand_dims(video_snippet[i], axis = 0) , step = frame_count)
        train_writer.flush()

    frame_count += snippet_size





train_writer.close()
test_writer.close()
vidcap.release()
print('All done', flush=True)
