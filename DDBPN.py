from tensorflow.keras.layers import Concatenate
from Models import *


# The D-DBPN model. It has densely connected BP layers. This is a lightweighted version that comprises of a 7 stage
# BP layers, instead of the 10 stages version. The performance doesn't differ much from the latter.
class DDBPN:
    def __init__(self, scale_factor=2, bias=True, bias_init='zeros'):
        # default scaling factor of 2
        kernel_size = 6
        stride = 2
        padding = 2

        if scale_factor == 4:
            kernel_size = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel_size = 12
            stride = 8
            padding = 2

        # Feature extraction stage
        self.f0 = CONV(256, 3, 1, 1, bias, bias_init, True)
        self.f1 = CONV(64, 1, 1, 0, bias, bias_init, True)
        # Back Projection stage, D-DBPN has a total of 7 BP stages, last stage only has an up-projection
        self.up1 = DenseUpProjection(64, kernel_size, stride, padding, 1, bias, bias_init)
        self.down1 = DenseDownProjection(64, kernel_size, stride, padding, 1, bias, bias_init)
        self.up2 = DenseUpProjection(64, kernel_size, stride, padding, 1, bias, bias_init)
        self.down2 = DenseDownProjection(64, kernel_size, stride, padding, 2, bias, bias_init)
        self.up3 = DenseUpProjection(64, kernel_size, stride, padding, 2, bias, bias_init)
        self.down3 = DenseDownProjection(64, kernel_size, stride, padding, 3, bias, bias_init)
        self.up4 = DenseUpProjection(64, kernel_size, stride, padding, 3, bias, bias_init)
        self.down4 = DenseDownProjection(64, kernel_size, stride, padding, 4, bias, bias_init)
        self.up5 = DenseUpProjection(64, kernel_size, stride, padding, 4, bias, bias_init)
        self.down5 = DenseDownProjection(64, kernel_size, stride, padding, 5, bias, bias_init)
        self.up6 = DenseUpProjection(64, kernel_size, stride, padding, 5, bias, bias_init)
        self.down6 = DenseDownProjection(64, kernel_size, stride, padding, 6, bias, bias_init)
        self.up7 = DenseUpProjection(64, kernel_size, stride, padding, 6, bias, bias_init)

        # Reconstruction
        self.reconstruction = CONV(1, 1, 1, 0, bias, bias_init, False)

    def __call__(self, x):

        # Feature Extraction
        x = self.f0(x)
        x = self.f1(x)
        # BP stage
        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)

        h_cat = Concatenate(axis=1)([h2, h1])
        l2 = self.down2(h_cat)

        l_cat = Concatenate(axis=1)([l2, l1])
        h = self.up3(l_cat)

        h_cat = Concatenate(axis=1)([h, h_cat])
        l = self.down3(h_cat)

        l_cat = Concatenate(axis=1)([l, l_cat])
        h = self.up4(l_cat)

        h_cat = Concatenate(axis=1)([h, h_cat])
        l = self.down4(h_cat)

        l_cat = Concatenate(axis=1)([l, l_cat])
        h = self.up5(l_cat)

        h_cat = Concatenate(axis=1)([h, h_cat])
        l = self.down5(h_cat)

        l_cat = Concatenate(axis=1)([l, l_cat])
        h = self.up6(l_cat)

        h_cat = Concatenate(axis=1)([h, h_cat])
        l = self.down6(h_cat)

        l_cat = Concatenate(axis=1)([l, l_cat])
        h = self.up7(l_cat)

        h_cat = Concatenate(axis=1)([h, h_cat])

        # Reconstruction stage
        x = self.reconstruction(h_cat)
        return x

