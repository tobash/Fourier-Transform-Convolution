import numpy as np
from imageio import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.ndimage import convolve
from scipy import signal

GRASCALE_REPRE = 1
COLOR_LEVEL = 256
CONVOLVE_VAl = np.array([[1,0,-1]])

def read_image(filename, representation):

    '''
    A function that converts the image to a desired representation, and with
    intesities normalized to the range of [0,1]
    :param filename: the filename of an image on disk, could be grayscale or
    RGB
    :param representation: representation code, either 1 or 2 defining whether
    the output should be a grayscale image (1) or an RGB image (2)
    :return: an image in the desired representation.
    '''

    im = imread(filename)
    if representation == GRASCALE_REPRE:
        im = rgb2gray(im)
    im_float = im.astype(np.float64)
    im_float /= (COLOR_LEVEL - 1)
    return im_float


def DFT(signal):

    '''
    Function that transform a 1D discrete signal to its Fourier representation
    using a given formula
    :param signal: an array of dtype float64 with shape (N,1)
    :return: complex Fourier signal as an array of dtype complex128 with the
    same shape
    '''

    num_row = np.shape(signal)[0]
    raise_val = np.exp(-2j*np.pi/num_row)
    multp_mat = np.fromfunction(lambda i,j: raise_val**(i*j), (num_row,num_row))
    return np.dot(multp_mat,signal).astype(np.complex128)


def IDFT(fourier_signal):

    '''
    Function that transform a 1D discrete Fourier signal to its origin signal
    representation using a given formula
    :param fourier_signal: an array of dtype complex128 with shape (N,1)
    :return: complex signal as an array of dtype complex128 with the same shape
    '''

    num_row = np.shape(fourier_signal)[0]
    raise_val = np.exp(2j * np.pi / num_row)
    multp_mat = np.fromfunction(lambda i, j: raise_val ** (i * j),(num_row, num_row))
    return (np.dot(multp_mat, fourier_signal).astype(np.complex128)) / num_row

def DFT2(image):

    '''
    Function that convert a 2D discrete signal to its Fourier representation
    using a given formula.
    :param image: a grayscale image of dtype float64
    :return: complex Fourier signal as an array of dtype complex128 with the
    same shape
    '''

    row_dft = DFT(image)
    return DFT(row_dft.transpose()).transpose()

def IDFT2(fourier_image):

    '''
    Function that transform a 2D discrete Fourier signal to its origin signal
    representation using a given formula
    :param fourier_image: complex Fourier signal as 2D array of dtype complex128
    :return: a grayscale image of dtype float64 with the same shape
    '''

    row_dft = IDFT(fourier_image)
    return IDFT(row_dft.transpose()).transpose()

def conv_der(im):

    '''
    Function that computes the magnitude of image derivatives. Doing so by
    deriving the image in each direction separately (vertical and horizontal)
    using simple convolution with [1, 0,−1] as a row and column vectors. Then
    using the derivative images to compute the magnitude image.
    :param im: grayscale images of type float64
    :return: magnitude of the derivative, with the same dtype and shape
    '''

    der_im_x = convolve(im,CONVOLVE_VAl)
    der_im_y = convolve(im,CONVOLVE_VAl.T)
    magnitude = np.sqrt(np.abs(der_im_x) ** 2 + np.abs(der_im_y) ** 2)
    return magnitude.astype(np.float64)


def fourier_der(im):
    '''
    Function that computes the magnitude of image derivatives using Fourier
    transform
    :param im: float64 grayscale images
    :return: float64 grayscale images
    '''

    const = (2j*np.pi)/np.shape(im)[0]

    mult_mat_x = np.arange(-np.shape(im)[0]/2,np.shape(im)[0]/2)[:,None]
    der_FT_x = np.fft.fftshift(DFT2(im))
    der_FT_x = der_FT_x * mult_mat_x
    der_FT_x = IDFT2(np.fft.ifftshift(der_FT_x*const))


    mult_mat_y = np.arange(-np.shape(im)[1] / 2, np.shape(im)[1] / 2)[None,:]
    der_FT_y = np.fft.fftshift(DFT2(im))
    der_FT_y = der_FT_y * mult_mat_y
    der_FT_y = IDFT2(np.fft.ifftshift(der_FT_y*const))

    der_FT_y = IDFT2(der_FT_y)

    magnitude = np.sqrt(np.abs(der_FT_y) ** 2 + np.abs(der_FT_x) ** 2)
    return (magnitude.real).astype(np.float64)


def blur_spatial (im, kernel_size):

    '''
    function that performs image blurring using 2D convolution between the
    image and a gaussian kernel.
    :param im: image to be blurred (grayscale float64 image).
    :param kernel_size: size of the gaussian kernel in each dimension
    (an odd integer).
    :return: blurry image (grayscale float64 image)
    '''

    kernel_mat = np.array([1,1],np.float64)
    conv_mat = np.array([1,1])

    for i in range(kernel_size):
        kernel_mat = np.convolve(kernel_mat,conv_mat)

    kernel_mat = (np.expand_dims(kernel_mat,axis=0)).transpose()
    conv_mat = np.expand_dims(conv_mat,axis=0)

    for i in range(kernel_size):
        kernel_mat = signal.convolve2d(kernel_mat,conv_mat)

    kernel_mat *= 1/np.sum(kernel_mat)
    return convolve(im,kernel_mat)

def blur_fourier (im, kernel_size):

    '''
    Function that performs image blurring with gaussian kernel in Fourier space
    :param im: image to be blurred (grayscale float64 image).
    :param kernel_size: size of the gaussian kernel in each dimension
    :return: blurry image (grayscale float64 image)
    '''

    kernel_mat = np.array([1,1],np.float64)
    conv_mat = np.array([1,1])

    for i in range(kernel_size):
        kernel_mat = np.convolve(kernel_mat,conv_mat)

    kernel_mat = (np.expand_dims(kernel_mat,axis=0)).transpose()
    conv_mat = np.expand_dims(conv_mat,axis=0)

    for i in range(kernel_size):
        kernel_mat = signal.convolve2d(kernel_mat,conv_mat)

    fourier_im = DFT2(im)
    paded_kernel = np.zeros(im.shape)
    star_y = int(abs(fourier_im.shape[1]-kernel_mat.shape[1])/2) #check that the in is good
    star_x = int(abs(fourier_im.shape[0]-kernel_mat.shape[0])/2)
    paded_kernel[star_x:kernel_mat.shape[0]+star_x,star_y:kernel_mat.shape[1] + star_y] = kernel_mat
    paded_kernel = DFT2(np.fft.ifftshift(paded_kernel))

    return IDFT2(np.multiply(fourier_im,paded_kernel)).real