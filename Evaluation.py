from Data_Eval import *
from tensorflow.keras.models import load_model

# Load the data for SISR convertion
dataset_path = ["C:/Users/Tze Lun/DIV2K/x_test/ModelEval/",
                "C:/Users/Tze Lun/DIV2K/y_test/ModelEval/"]
[x_test, y_test] = makeDataset(dataset_path)

# Load the trained model of your choice
[bicubic] = makeDataset(["C:/Users/Tze Lun/DIV2K/x_test/ScaledLR/"])
DBPNM = load_model("C:/Users/Tze Lun/VisualMediaAssignment/DBPN-M_@epoch_100.h5")
DBPNM_NEF = load_model("C:/Users/Tze Lun/VisualMediaAssignment/DBPN-M_NEF_@epoch_100.h5")
DDBPN = load_model("C:/Users/Tze Lun/VisualMediaAssignment/D-DBPN_@epoch_100.h5")

# Post-processing of test data: SISR conversion of LR test images
y_pred_DBPNM = LR_to_HR_transform(DBPNM, x_test)
y_pred_DBPNM_NEF = LR_to_HR_transform(DBPNM_NEF, x_test)
y_pred_DDBPN = LR_to_HR_transform(DDBPN, x_test)

# Get the average peak signal-to-noise ratio for each model
PSNR_bicubic = PSNR(bicubic, y_test, 255)
PSNR_DBPNM = PSNR(y_pred_DBPNM, y_test, 255)
PSNR_DBPNM_NEF = PSNR(y_pred_DBPNM_NEF, y_test, 255)
PSNR_DDBPN = PSNR(y_pred_DDBPN, y_test, 255)

print("Avg PSNR for Bicubic: ", end="")
print(PSNR_bicubic)
print("Avg PSNR for DBPN-M: ", end="")
print(PSNR_DBPNM)
print("Avg PSNR for DBPN-M Without Error Feedback: ", end="")
print(PSNR_DBPNM_NEF)
print("Avg PSNR for D-DBPN: ", end="")
print(PSNR_DDBPN)
