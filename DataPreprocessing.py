from Data_Eval import crop

# Folders containing original images from the DIV2K
list_to_convert = ["C:/Users/Tze Lun/DIV2K/x_train/X4/",
                   "C:/Users/Tze Lun/DIV2K/x_valid/X4/",
                   "C:/Users/Tze Lun/DIV2K/y_train/HR/",
                   "C:/Users/Tze Lun/DIV2K/y_valid/HR/"]

# Folders to contain the cropped images
list_to_save = ["C:/Users/Tze Lun/DIV2K/x_train/X4_40x40/",
                "C:/Users/Tze Lun/DIV2K/x_valid/X4_40x40/",
                "C:/Users/Tze Lun/DIV2K/y_train/HR_160x160/",
                "C:/Users/Tze Lun/DIV2K/y_valid/HR_160X160/"]

# Folders containing 5 test image from Set5, available on author's Github:
# https://github.com/alterzero/DBPN-Pytorch/tree/master/Input/Set5_LR_x4
dir_test_file = "C:/Users/Tze Lun/DIV2K/x_test/LR_original/"
dir_save = "C:/Users/Tze Lun/DIV2K/x_test/X4_40X40/"

# re-crop the images to the desired size
for i in range(len(list_to_convert)):
    if i <= 1:
        size = (40, 40)  # width, height
        crop(list_to_convert[i], list_to_save[i], size)
    else:
        size = (160, 160)
        crop(list_to_convert[i], list_to_save[i], size)

size = (40, 40)
crop(dir_test_file, dir_save, size)
