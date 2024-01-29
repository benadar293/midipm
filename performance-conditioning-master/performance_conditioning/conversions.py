import numpy as np

def midi2hertz(m):
    # return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
    return 440. * (2. ** ((m - 69.) / 12.))

def hertz2midi(h):
    return 12. * np.log2(h / (440.)) + 69.

def midi2spec(m):
    # return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
    return midi2hertz(m) / 8

def spec2midi(s):
    return hertz2midi(8 * s)

def hertz2mel(m):
    return 2595. * np.log10(1 + m / 700.)

def mel2hertz(h):
    return 700 * (np.power(10, h / 2595.) - 1)