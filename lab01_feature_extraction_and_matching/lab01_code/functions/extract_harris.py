import numpy as np
from scipy import signal
from scipy import ndimage
import cv2

def extract_harris(img, sigma=1.0, k=0.05, thresh=1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:        (h, w) numpy array storing the corner strength
    '''
    # 1 - convert to float
    img = img.astype(float) / 255.0
    # 2 - parameters
    sigma = 2.0
    thresh = 1e-6
    k = 0.06
    # 3 - compute image gradients in x and y direction
    x = 1/2*np.array([1,0,-1]).reshape(1,3)
    y = 1/2*np.array([1,0,-1]).reshape(3,1)
    Ix = signal.convolve2d(img, x, mode='same', boundary='symm')
    Iy = signal.convolve2d(img,y, mode='same',boundary='symm')
    # 4 - blur the computed gradients
    Ixx_blr = cv2.GaussianBlur(Ix ** 2, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    Iyy_blr = cv2.GaussianBlur(Iy ** 2, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    Ixy_blr = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    # 5 - compute Harris response function C
    det_M = Ixx_blr * Iyy_blr - (Ixy_blr**2)
    trace_M = Ixx_blr + Iyy_blr
    C = det_M - k * (trace_M**2)
    # 6 - detection with threshold and nqon-maximum suppression
    mask = C > thresh
    local_maxima = ndimage.maximum_filter(C, size=(3, 3), mode='constant') == C
    corners_y, corners_x = np.where(mask & local_maxima)
    corners = np.stack((corners_x, corners_y), axis=-1)
    return corners, C