# VisualMediaAssignment
This is the assignment for the 4840-1014 Visual Media module, a graduate level module in the University of Tokyo <br />

## Deep Back-ProjectiNetwork for Single Image Super-Resolution [1]
This is the chosen journal to study and work on for my assignment. The chosen journal is under IEEE TPAMI. <br />
Credit goes to the authors of this paper, available under the reference section. <br />

## Details of research
In general, this research is about Single Image Super-Resolution (SISR), which is using mainly neural network or deep learning to transform a low resolution (LR) image to a high resolution (HR) one. The details on the novelty of the author's work are explained quite thoroughly in a summary report I made. Either read the original paper (highly recommended) or my summary report to understand the implementation. <br /> <br />
Kindly check out the summary report of this research which includes the explanation of the results of my reimplementation which is available [here](https://github.com/TzeLun/VisualMediaAssignment/tree/main/Summary%20Report). <br />

## What is this repository for?
Besides fulfilling the assignment requirement for my class, it would be nice to have a very simple and intuitive program to replicate the author's SISR model. The program is written with the goal of having users understand the workings of DBPN, more so than to show one's own technical flair. I hope whoever interested in SISR research could learn something from my report and program. If you are looking for something very robust for the DBPN implementation, I highly recommend you to check the author's GitHub repository [here](https://github.com/alterzero/DBPN-Pytorch), which was implemented with PyTorch. <br />

## How to use?
The code was written with TensorFlow 2.3 with GPU enabled and Python 3.8. Besides NumPy, Python Imaging Library (PIL) or Pillow is used to process images.
In this reimplementation work, three SISR models are chosen from the DBPN paper. Two of them are the DBPN variants, which are the DBPN-M and D-DBPN. The third model is also DBPN-M with the error feedback disabled in every projection unit.

### Training
To start training the model, run the file ```training.py```. Before doing so, make sure to edit the script to your desired configuration: <br />
```Python
# Parameters
choice = "D-DBPN"  # DBPN-M (default), DBPN-M NEF, D-DBPN
sf = 4  # default at 2, otherwise 4 or 8

# Data generation:
dataset_path = ["Path to folder for x_train/",
                "Path to folder for x_test/",
                "Path to folder for x_valid/",
                "Path to folder for y_train/",
                "Path to folder for y_test/",
                "Path to folder for y_valid/"]

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
```
If you prefer to use your own data preprocessing function, feel free not to use the ```makeDataset()``` function. Otherwise, using the function is very simple. By organising your image dataset in the right folder and compile a list of path to the dataset in your favorite order, the ```makeDataset()``` automatically loads the dataset for you in the order of the path you specified. Each dataset is a numpy array in the following format: <br />
```Python
np.array().shape: (Num_images, Height, Width, Channels)
```
In case if it wasn't detailed before, the y-dataset comprises of HR images, whereas the x-dataset comprises of LR images. Both HR and LR images have a 1:1 aspect ratio but their sizes differ by the scalar multiple (scaling factor, sf) you choose. So make sure to prepare the dataset accordingly. <br />

Also, each of the three models have their pre-trained model for 100 epoch and they are accessible through [here](https://github.com/TzeLun/VisualMediaAssignment/tree/main/Trained%20Models). As the training was done with limited data and without data augmentation, it is advisable to use them only as a quick demo. <br />

### SISR transformation
Once the model is trained, it should automatically be saved in a ".h5" file. To perform LR-to-HR transformation using the trained model, run the ```testing.py``` file.
Below shows the contents of the aforementioned python file: <br />
```Python
from Data_Eval import *
from tensorflow.keras.models import load_model

# Load the data for SISR convertion
dataset_path = ["Path to folder of test images/"]
[x_test] = makeDataset(dataset_path)

# Load the trained model of your choice
model = load_model("Path to folder/a_trained_model.h5")

# Post-processing of test data: SISR conversion
y_pred = LR_to_HR_transform(model, x_test)
saveIMAGE("Path to folder/", y_pred)
```

### Evaluation
To evaluate the model, average Peak Signal-to-Noise Ratio (PSNR) is used by calculating the PSNR values for all the HR-transformed test images. To do so, run the ```Evaluation.py``` file. Make sure to edit the path to match your workspace. The average PSNR is calculated with the ```PSNR()```
function. The file first executes the LR-to-HR conversion of test images, using the ```LR_to_HR_transform()```
function, then calls the function to evaluate PSNR using the transformed test image with their actual HR counterparts. The PSNR function is shown below:
```Python
# Get the average peak signal-to-noise ratio for each model
PSNR_bicubic = PSNR(bicubic, y_test, 255)
PSNR_DBPNM = PSNR(y_pred_DBPNM, y_test, 255)
PSNR_DBPNM_NEF = PSNR(y_pred_DBPNM_NEF, y_test, 255)
PSNR_DDBPN = PSNR(y_pred_DDBPN, y_test, 255)
```

## Results
With minimal training and no data augmentation, below shows the comparison of the HR images produced by DBPN-M with error feedback, DBPN-M without error feedback and D-DBPN using LR images with scaling factor of 4x. Bicubic interpolation is used to resize the LR image to its HR counterpart to approximate the LR image without any SISR transform. <br />
![Comparison of DBPN-M with error feedback, DBPN-M without error feedback and D-DBPN](https://github.com/TzeLun/VisualMediaAssignment/blob/main/Summary%20Report/Results.png)

## Reference
[1] M. Haris, G. Shakhnarovich and N. Ukita, "Deep Back-ProjectiNetworks for Single Image Super-Resolution," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 12, pp. 4323-4337, 1 Dec. 2021, doi: 10.1109/TPAMI.2020.3002836.
<br /> <br />
Directly access the paper [here](https://ieeexplore.ieee.org/document/9119166)
