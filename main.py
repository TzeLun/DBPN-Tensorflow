import tensorflow as tf
from tensorflow.python.keras import Input
import numpy as np
from DBPNM import *
from DDBPN import *
from Data_Eval import *


#gpu_available = tf.test.is_gpu_available()
#is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
#
# print(gpu_available)
# print(is_cuda_gpu_available)

# Parameters
choice = "DBPN-M"  # DBPN-M (default), DBPN-M NEF, D-DBPN
sf = 4  # default at 2, otherwise 4 or 8

# Write code to bring in data. DIV2K: https://ieeexplore.ieee.org/document/8014884
# add data augmentation if possible
# should have x_train, x_test, y_train, y_test

# Training the model
batch_size = 16
width = 0
height = 0
channel = 0
input_shape = (width, height, channel)
lr = 0.0004  # learning rate
alpha = 0.9  # momentum
epochs = 100000  # author used 1000000!!
split = 0.2  # ratio of dataset for validation

# Model:
model_input = Input(shape=input_shape, name=choice)
# DBPN-M with error feedback (default)
output = DBPNM(scale_factor=sf)(model_input)
if choice is "DBPN-M NEF":
    # DBPNM without error feedback
    output = DBPNM_WithoutEF(scale_factor=sf)(model_input)
elif choice is "D-DBPN":
    # D-DBPN
    output = DDBPN(scale_factor=sf)(model_input)

model = tf.keras.Model(model_input, output)
model.compile(
    loss=l1_loss,
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=alpha)
)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=split)
test_result = model.evaluate(x_test, y_test)
print("Test Loss:", test_result[0])

# save the trained model
model.save(choice + "_@epoch_" + str(epochs) + ".h5")
