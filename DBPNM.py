from Models import *


# The DBPN-M model. It has no densely connected BP layers. Lightweight but not the best in performance
# By default, all layers uses PReLU activation function, according to the author's work.
class DBPNM:
    def __init__(self, scale_factor=2, bias=True, bias_init='zeros'):
        kernel_size = 0
        stride = 0
        padding = 0

        if scale_factor == 2:
            # default scaling factor of 2
            kernel_size = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel_size = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel_size = 12
            stride = 8
            padding = 2

        # Feature extraction stage
        self.f0 = CONV(128, 3, 1, 1, bias, bias_init, True)
        self.f1 = CONV(32, 1, 1, 0, bias, bias_init, True)
        # Back Projection stage, DBPN-M has a total of 4 BP stages, last stage only has an up-projection
        self.up1 = UpProjection(32, kernel_size, stride, padding, bias, bias_init)
        self.down1 = DownProjection(32, kernel_size, stride, padding, bias, bias_init)
        self.up2 = UpProjection(32, kernel_size, stride, padding, bias, bias_init)
        self.down2 = DownProjection(32, kernel_size, stride, padding, bias, bias_init)
        self.up3 = UpProjection(32, kernel_size, stride, padding, bias, bias_init)
        self.down3 = DownProjection(32, kernel_size, stride, padding, bias, bias_init)
        self.up4 = UpProjection(32, kernel_size, stride, padding, bias, bias_init)
        # Reconstruction
        self.reconstruction = CONV(3, 1, 1, 0, bias, bias_init, False)

    def __call__(self, x):

        # Feature Extraction
        x = self.f0(x)
        x = self.f1(x)
        # BP stage
        h = self.up1(x)
        l = self.down1(h)
        h = self.up2(l)
        l = self.down2(h)
        h = self.up3(l)
        l = self.down3(h)
        h = self.up4(l)
        # Reconstruction stage
        x = self.reconstruction(h)
        return x


# Just for comparison. In general, this model will not be used in Super Resolution application
class DBPNM_WithoutEF:
    def __init__(self, scale_factor, bias=True, bias_init='zeros'):
        kernel_size = 0
        stride = 0
        padding = 0

        if scale_factor == 2:
            # default scaling factor of 2
            kernel_size = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel_size = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel_size = 12
            stride = 8
            padding = 2

        # Feature extraction stage
        self.f0 = CONV(128, 3, 1, 1, bias, bias_init, True)
        self.f1 = CONV(32, 1, 1, 0, bias, bias_init, True)
        # Back Projection stage, DBPN-M has a total of 4 BP stages, last stage only has an up-projection
        self.up1 = UpProjectionWithoutEF(32, kernel_size, stride, padding, bias, bias_init)
        self.down1 = DownProjectionWithoutEF(32, kernel_size, stride, padding, bias, bias_init)
        self.up2 = UpProjectionWithoutEF(32, kernel_size, stride, padding, bias, bias_init)
        self.down2 = DownProjectionWithoutEF(32, kernel_size, stride, padding, bias, bias_init)
        self.up3 = UpProjectionWithoutEF(32, kernel_size, stride, padding, bias, bias_init)
        self.down3 = DownProjectionWithoutEF(32, kernel_size, stride, padding, bias, bias_init)
        self.up4 = UpProjectionWithoutEF(32, kernel_size, stride, padding, bias, bias_init)
        # Reconstruction
        self.reconstruction = CONV(3, 1, 1, 0, bias, bias_init, False)

    def __call__(self, x):

        # Feature Extraction
        x = self.f0(x)
        x = self.f1(x)
        # BP stage
        h = self.up1(x)
        l = self.down1(h)
        h = self.up2(l)
        l = self.down2(h)
        h = self.up3(l)
        l = self.down3(h)
        h = self.up4(l)
        # Reconstruction stage
        x = self.reconstruction(h)
        return x
