from Data_Eval import *
from tensorflow.keras.models import load_model

# Load the data for SISR convertion
dataset_path = ["C:/Users/Tze Lun/DIV2K/x_test/ModelEval/"]
[x_test] = makeDataset(dataset_path)

# Load the trained model of your choice
model = load_model("C:/Users/Tze Lun/VisualMediaAssignment/D-DBPN_@epoch_100.h5")

# Post-processing of test data: SISR conversion and image pixels restoration
y_pred = LR_to_HR_transform(model, x_test)
saveIMAGE("C:/Users/Tze Lun/DIV2K test results/test_D-DBPN_epoch_100/", y_pred)
#print("Average L1 Loss (Test dataset):", evaluate(model, x_test, y_test))