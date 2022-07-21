from tensorflow.keras.layers import Conv2D, Conv2DTranspose, PReLU, ZeroPadding2D, Subtract, Add


class CONV:
    def __init__(self, num_filters, kernel_size, strides, padding, bias, bias_init, activation=True):
        self.padding = padding
        self.activation = activation
        self.f = Conv2D(num_filters, kernel_size, strides=strides, padding='same',
                        activation=None, use_bias=bias, bias_initializer=bias_init)

    def __call__(self, x):
        x = self.f(x)
        if self.activation:
            x = PReLU()(x)
        return x


class DownProjection:
    def __init__(self, num_filters, kernel_size, strides, padding, bias, bias_init):
        self.padding = padding
        self.H_to_L0 = Conv2D(num_filters, kernel_size, strides=strides, padding='same',
                              activation=None, use_bias=bias, bias_initializer=bias_init)
        self.L0_to_H0 = Conv2DTranspose(num_filters, kernel_size, strides=strides, padding='same',
                                        activation=None, use_bias=bias, bias_initializer=bias_init)
        self.eH0_to_eL0 = Conv2D(num_filters, kernel_size, strides=strides, padding='same',
                                 activation=None, use_bias=bias, bias_initializer=bias_init)

    def __call__(self, x):
        l0 = self.H_to_L0(x)
        l0 = PReLU()(l0)
        h0 = self.L0_to_H0(l0)
        h0 = PReLU()(h0)
        eh0 = Subtract()([h0, x])
        el0 = self.eH0_to_eL0(eh0)
        el0 = PReLU()(el0)
        return Add()([l0, el0])


class UpProjection:
    def __init__(self, num_filters, kernel_size, strides, padding, bias, bias_init):
        self.padding = padding
        self.L_to_H0 = Conv2DTranspose(num_filters, kernel_size, strides=strides, padding='same',
                                       activation=None, use_bias=bias, bias_initializer=bias_init)
        self.H0_to_L0 = Conv2D(num_filters, kernel_size, strides=strides, padding='same',
                               activation=None, use_bias=bias, bias_initializer=bias_init)
        self.eL0_to_eH0 = Conv2DTranspose(num_filters, kernel_size, strides=strides, padding='same',
                                          activation=None, use_bias=bias, bias_initializer=bias_init)

    def __call__(self, x):
        h0 = self.L_to_H0(x)
        h0 = PReLU()(h0)
        l0 = self.H0_to_L0(h0)
        l0 = PReLU()(l0)
        el0 = Subtract()([l0, x])
        eh0 = self.eL0_to_eH0(el0)
        eh0 = PReLU()(eh0)
        return Add()([h0, eh0])


# Only used to evaluate the significance of error feedback (EF)
class DownProjectionWithoutEF:
    def __init__(self, num_filters, kernel_size, strides, padding, bias, bias_init):
        self.padding = padding
        self.H_to_L0 = Conv2D(num_filters, kernel_size, strides=strides, padding='same',
                              activation=None, use_bias=bias, bias_initializer=bias_init)
        self.L0_to_H0 = Conv2DTranspose(num_filters, kernel_size, strides=strides, padding='same',
                                        activation=None, use_bias=bias, bias_initializer=bias_init)
        self.eH0_to_eL0 = Conv2D(num_filters, kernel_size, strides=strides, padding='same',
                                 activation=None, use_bias=bias, bias_initializer=bias_init)

    def __call__(self, x):
        l0 = self.H_to_L0(x)
        l0 = PReLU()(l0)
        h0 = self.L0_to_H0(l0)
        h0 = PReLU()(h0)
        eh0 = Subtract()([h0, x])
        el0 = self.eH0_to_eL0(eh0)
        el0 = PReLU()(el0)
        return l0


# Only used to evaluate the significance of error feedback (EF)
class UpProjectionWithoutEF:
    def __init__(self, num_filters, kernel_size, strides, padding, bias, bias_init):
        self.padding = padding
        self.L_to_H0 = Conv2DTranspose(num_filters, kernel_size, strides=strides, padding='same',
                                       activation=None, use_bias=bias, bias_initializer=bias_init)
        self.H0_to_L0 = Conv2D(num_filters, kernel_size, strides=strides, padding='same',
                               activation=None, use_bias=bias, bias_initializer=bias_init)
        self.eL0_to_eH0 = Conv2DTranspose(num_filters, kernel_size, strides=strides, padding='same',
                                          activation=None, use_bias=bias, bias_initializer=bias_init)

    def __call__(self, x):
        h0 = self.L_to_H0(x)
        h0 = PReLU()(h0)
        l0 = self.H0_to_L0(h0)
        l0 = PReLU()(l0)
        el0 = Subtract()([l0, x])
        eh0 = self.eL0_to_eH0(el0)
        eh0 = PReLU()(eh0)
        return h0


class DenseDownProjection:
    def __init__(self, num_filters, kernel_size, strides, padding, n_stage, bias, bias_init):
        self.padding = padding
        self.conv = Conv2D(num_filters, kernel_size=1, strides=1, padding='same',
                           activation=None, use_bias=bias, bias_initializer=bias_init)
        self.H_to_L0 = Conv2D(num_filters, kernel_size, strides=strides, padding='same',
                              activation=None, use_bias=bias, bias_initializer=bias_init)
        self.L0_to_H0 = Conv2DTranspose(num_filters, kernel_size, strides=strides, padding='same',
                                        activation=None, use_bias=bias, bias_initializer=bias_init)
        self.eH0_to_eL0 = Conv2D(num_filters, kernel_size, strides=strides, padding='same',
                                 activation=None, use_bias=bias, bias_initializer=bias_init)

    def __call__(self, x):
        x = self.conv(x)
        x = PReLU()(x)
        l0 = self.H_to_L0(x)
        l0 = PReLU()(l0)
        h0 = self.L0_to_H0(l0)
        h0 = PReLU()(h0)
        eh0 = Subtract()([h0, x])
        el0 = self.eH0_to_eL0(eh0)
        el0 = PReLU()(el0)
        return Add()([l0, el0])


class DenseUpProjection:
    def __init__(self, num_filters, kernel_size, strides, padding, n_stage, bias, bias_init):
        self.padding = padding
        self.conv = Conv2D(num_filters, kernel_size=1, strides=1, padding='same',
                           activation=None, use_bias=bias, bias_initializer=bias_init)
        self.L_to_H0 = Conv2DTranspose(num_filters, kernel_size, strides=strides, padding='same',
                                       activation=None, use_bias=bias, bias_initializer=bias_init)
        self.H0_to_L0 = Conv2D(num_filters, kernel_size, strides=strides, padding='same',
                               activation=None, use_bias=bias, bias_initializer=bias_init)
        self.eL0_to_eH0 = Conv2DTranspose(num_filters, kernel_size, strides=strides, padding='same',
                                          activation=None, use_bias=bias, bias_initializer=bias_init)

    def __call__(self, x):
        x = self.conv(x)
        x = PReLU()(x)
        h0 = self.L_to_H0(x)
        h0 = PReLU()(h0)
        l0 = self.H0_to_L0(h0)
        l0 = PReLU()(l0)
        el0 = Subtract()([l0, x])
        eh0 = self.eL0_to_eH0(el0)
        eh0 = PReLU()(eh0)
        return Add()([h0, eh0])
