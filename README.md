# VisualMediaAssignment
This is the assignment for the 4840-1014 Visual Media module, a graduate level module in the University of Tokyo

## Chosen Journal
Deep Back-ProjectiNetwork for Single Image Super-Resolution [1] (IEEE TPAMI).<br />
Credit goes to the authors of this paper.

## Details of Research
In general, this research is about Single Image Super-Resolution (SISR), which is using mainly neural network or deep learning to transform a low resolution (LR) image to a high resolution (HR) one. The details on the novelty of the author's work are explained quite thoroughly in a summary report I made. Either read the original paper (highly recommended) or my summary report to understand the implementation. <br /> <br />
Kindly check out the summary report of this research which includes the explanation of the results of my reimplementation which is available [here](https://github.com/TzeLun/VisualMediaAssignment/tree/main/Summary%20Report). <br />

## How to Use?
The code was written with TensorFlow 2.3 with GPU enabled and Python 3.8. Besides NumPy, Python Imaging Library (PIL) or Pillow is used to process images. <br />
### Training
### SISR Transformation
### Evaluation

## Results
With minimal training and no data augmentation, below shows the comparison of the HR images produced by DBPN-M with error feedback, DBPN-M without error feedback and D-DBPN using LR images with scaling factor of 4x. Bicubic interpolation is used to resize the LR image to its HR counterpart to approximate the LR image without any SISR transform. <br />
![Comparison of DBPN-M with error feedback, DBPN-M without error feedback and D-DBPN](https://github.com/TzeLun/VisualMediaAssignment/blob/main/Summary%20Report/Results.png)

## Reference
[1] M. Haris, G. Shakhnarovich and N. Ukita, "Deep Back-ProjectiNetworks for Single Image Super-Resolution," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 12, pp. 4323-4337, 1 Dec. 2021, doi: 10.1109/TPAMI.2020.3002836.
<br /> <br />
Directly access the paper [here](https://ieeexplore.ieee.org/document/9119166)
