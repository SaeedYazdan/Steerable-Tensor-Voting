import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, ifftshift

try:
    from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
# Otherwise use the normal scipy fftpack ones instead (~2-3x slower!)
except ImportError:
    import warnings
    warnings.warn("""
Module 'pyfftw' (FFTW Python bindings) could not be imported. To install it, try
running 'pip install pyfftw' from the terminal. Falling back on the slower
'fftpack' module for 2D Fourier transforms.""")
    from scipy.fftpack import fft2, ifft2



def imshow(img):
    plt.imshow(img)
    plt.show()


def c(m, s, be):
    #s: stickness field, be:orientation field
    return s * np.exp(-1j * m * be)


def w(m, h, u, sigma):
    
    w2 = np.ceil(u / 2)
    h2 = np.ceil(h / 2)

    #xx = np.linspace(0, 1, u)
    xx = np.linspace(0, u, u, endpoint=False) + 1
    #yy = np.linspace(0, 1, h)
    yy = np.linspace(0, h, h, endpoint=False) + 1
    x, y = np.meshgrid(xx, yy)

    kernel = np.exp(-(((x - w2) ** 2 + (y - h2) ** 2) / (2 * sigma ** 2))) * (((x - w2) + 1j * (y - h2)) / (np.sqrt((x - w2) ** 2 + (y - h2) ** 2))) ** m

    kernel[int(h2) - 1, int(w2) - 1] = 1

    kernel = kernel.round(decimals=3, out=None)

    return kernel
    
    

def vote(s, be, sigma):

    [height, width] = s.shape

    c0 = c(0, s, be)
    c2 = c(2, s, be)
    c4 = c(4, s, be)
    c6 = c(6, s, be)

    c2bar = np.conj(c2)

    w0 = w(0, height, width, sigma)
    w2 = w(2, height, width, sigma)
    w4 = w(4, height, width, sigma)
    w6 = w(6, height, width, sigma)
    w8 = w(8, height, width, sigma)

    c0_f = fft2(c0)
    c2_f = fft2(c2)	
    c4_f = fft2(c4)		
    c6_f = fft2(c6)
    c2bar_f = fft2(c2bar)

    w0_f = fft2(w0)
    w2_f = fft2(w2)	
    w4_f = fft2(w4)		
    w6_f = fft2(w6)
    w8_f = fft2(w8)

    w0_c2bar = w0_f * c2bar_f #eight convolutions required
    w2_c0 = w2_f * c0_f 
    w4_c2 = w4_f * c2_f
    w6_c4 = w6_f * c4_f
    w8_c6 = w8_f * c6_f
    w0_c0 = w0_f * c0_f
    w2_c2 = w2_f * c2_f
    w4_c4 = w4_f * c4_f


    U_minus2 = ifftshift(ifft2((w0_c2bar) + 4 * (w2_c0) + 6 * (w4_c2) + 4 * (w6_c4) + (w8_c6)))
    U_2 = np.conj(U_minus2)
    U_0 = np.real(ifftshift(ifft2(6 * (w0_c0) + 8 * (w2_c2) + 2 * (w4_c4))))


    saliency = abs(U_minus2)
    ballness = 0.5 * (U_0 - abs(U_2))
    #orientation = 0.5 * arg(U_minus2)
    orientation = 0.5 * np.angle(U_minus2)

    #Uncomment to show normalized saliency field:
    imshow((saliency - np.min(saliency)) / np.max(saliency))
    saliency = (saliency - np.min(saliency)) / np.max(saliency)

    return saliency, ballness, orientation 
    
    
