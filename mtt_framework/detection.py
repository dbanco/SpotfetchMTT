# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:39:17 2025

spot_detection

@author: dpqb1
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label
from skimage.feature import hessian_matrix
from abc import ABC, abstractmethod

class Detection:
    def __init__(self, blob_label, mask, x_masked):
        self.blob_label = blob_label
        self.mask = mask
        self.x_masked = x_masked

class DetectorBase(ABC):
    """
    Abstract base class for spot detection methods.
    All custom spot detectors must inherit from this class and implement `detect`.
    """
    def __init__(self, **kwargs):
        """
        Allow flexible parameters for different detectors.
        Subclasses can define and use their own parameters.

        Parameters:
        - **kwargs: Additional parameters passed to specific detector types.
        """
        self.params = kwargs  # Store parameters (optional use in subclasses)
        
    @abstractmethod
    def detect(self, data):
        """
        Detect spots from input data.

        Parameters:
        - data: Input data

        Returns:
        - Masked data where each blob is labeled with integer >= 1
        """
        pass


class HDoGDetector(DetectorBase):
    """
    A Hessian-based Difference-of-Gaussian spot detector.
    """
    def __init__(self, sigmas=np.array([2,2,2]), dsigmas=np.array([1.5,1.5,1.5]), use_gaussian_derivatives=False):
        """
        Initialize the detector with scale parameters and derivative flag.
        """
        super().__init__(sigmas=sigmas, 
                         dsigmas=dsigmas,
                         use_gaussian_derivatives=use_gaussian_derivatives)  # Store in base class dictionary
        self.sigmas = sigmas # Store in subclass
        self.dsigmas = dsigmas
        self.use_gaussian_derivatives = use_gaussian_derivatives
        
    def detect(self, data):
        """
        Dummy detection method.

        Parameters:
        - data: Input data

        Returns:
        - Masked data where each blob is labeled with integer >= 1
        """
        return detectBlobHDoG(data,self.sigmas,self.dsigmas,self.use_gaussian_derivatives)

class ThresholdingDetector(DetectorBase):
    """
    A thresholding spot detector.
    """
    def __init__(self, threshold=5):
        """
        Initialize the detector with threhsold parameter
        """
        super().__init__(threshold=threshold)  # Store in base class dictionary
        self.threshold = threshold # Store in subclass
        
    def detect(self, data):
        """
        Dummy detection method.

        Parameters:
        - data: Input data

        Returns:
        - Masked data where each blob is labeled with integer >= 1
        """
        
        return thresholdingDetection(data,self.threshold)

####################### Functions ##############################
def thresholdingDetection(data,threshold):
    """
    Threshold data and applies connected components labeling

    Parameters:
    -----------
    data : ndarray
        Input 3D data.

    Returns:
    --------
    tuple
        - blobs : ndarray
            Labeled blob regions.
        - num_blobs : int
            Number of blobs detected.
    """   
    data[data < threshold] = 0
    blobs, num_blobs = label(data)
    return blobs, num_blobs


def detectBlobHDoG(data, sigmas, dsigmas, use_gaussian_derivatives):
    """
    Detects blobs using the Hessian-based Difference of Gaussians (DoG) method.

    Parameters:
    -----------
    data : ndarray
        Input 3D data.

    Returns:
    --------
    tuple
        - blobs : ndarray
            Labeled blob regions.
        - num_blobs : int
            Number of blobs detected.
        - hess_mat : list of ndarray
            Hessian matrices of the DoG result.
    """
    # 1. Compute normalized DoG
    dog_norm = DoG(data, sigma=sigmas, dsigma=dsigmas)

    # 2. Pre-segmentation
    hess_mat = hessian_matrix(dog_norm,use_gaussian_derivatives=use_gaussian_derivatives)
    D1 = np.zeros(hess_mat[0].shape)
    D2 = np.zeros(hess_mat[0].shape)
    D3 = np.zeros(hess_mat[0].shape)
    for i1 in range(hess_mat[0].shape[0]):
        for i2 in range(hess_mat[0].shape[1]):
            for i3 in range(hess_mat[0].shape[2]):
                h_mat = np.array([
                    [hess_mat[0][i1, i2, i3], hess_mat[1][i1, i2, i3], hess_mat[2][i1, i2, i3]],
                    [hess_mat[1][i1, i2, i3], hess_mat[3][i1, i2, i3], hess_mat[4][i1, i2, i3]],
                    [hess_mat[2][i1, i2, i3], hess_mat[4][i1, i2, i3], hess_mat[5][i1, i2, i3]]
                ])
                D1[i1,i2,i3] = h_mat[0,0]
                D2[i1,i2,i3] = np.linalg.det(h_mat[:2,:2])
                D3[i1,i2,i3] = np.linalg.det(h_mat)

    posDefIndicator = (D1 > 0) & (D2 > 0) & (D3 > 0)
    blobs, num_blobs = label(posDefIndicator)
    return blobs, num_blobs

def DoG(x, sigma, dsigma, gamma=2):
    """
    Computes the Difference of Gaussians (DoG) approximation.

    Parameters:
    -----------
    x : ndarray
        Input data.
    sigma : float
        Base Gaussian sigma.
    dsigma : float
        Increment for sigma.
    gamma : float, optional
        Scaling factor, default is 2.

    Returns:
    --------
    ndarray
        Normalized DoG result.
    """
    g1 = gaussian_filter(x, sigma=sigma)
    g2 = gaussian_filter(x, sigma=sigma + dsigma)
    return (g2 - g1) / (np.mean(sigma) * np.mean(dsigma))


