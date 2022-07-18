from tensorflow.python.keras.losses import mean_absolute_error


# The L1 loss function
def l1_loss(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    loss = mae * len(y_true)
    return loss

