import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from DBPNM import *
from DDBPN import *
from Data_Eval import *


# Checks for available gpu
gpu_available = tf.test.is_gpu_available()
is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
print(gpu_available)
print(is_cuda_gpu_available)

# Parameters
choice = "D-DBPN"  # DBPN-M (default), DBPN-M NEF, D-DBPN
sf = 4  # default at 2, otherwise 4 or 8

# Data generation:
dataset_path = ["C:/Users/Tze Lun/DIV2K/x_train/X4_40x40/",
                "C:/Users/Tze Lun/DIV2K/x_test/ModelEval/",
                "C:/Users/Tze Lun/DIV2K/x_valid/X4_40x40/",
                "C:/Users/Tze Lun/DIV2K/y_train/HR_160x160/",
                "C:/Users/Tze Lun/DIV2K/y_test/ModelEval/",
                "C:/Users/Tze Lun/DIV2K/y_valid/HR_160x160/"]

[x_train, x_test, x_valid, y_train, y_test, y_valid] = makeDataset(dataset_path)
input_dim = x_train.shape

# Training the model
batch_size = 16
width = input_dim[1]
height = input_dim[2]
channel = input_dim[3]
input_shape = [height, width, channel]
lr = 0.0004  # learning rate
alpha = 0.9  # momentum
epochs = 100  # author used 1000000!!

# Model:
model_input = Input(shape=input_shape, name=choice)
output = 0
if choice == "DBPN-M":
    # DBPN-M with error feedback (default)
    output = DBPNM(scale_factor=sf)(model_input)
elif choice == "DBPN-M_NEF":
    # DBPNM without error feedback
    output = DBPNM_WithoutEF(scale_factor=sf)(model_input)
elif choice == "D-DBPN":
    # D-DBPN
    output = DDBPN(scale_factor=sf)(model_input)

model = Model(model_input, output)
print(model.summary())
model.compile(
    loss='mean_absolute_error',
    optimizer=Adam(learning_rate=lr, beta_1=alpha)
)

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid), verbose=1)
# save the trained model
model.save(choice + "_@epoch_" + str(epochs) + ".h5")


