# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:39:17 2025

spot_detection

@author: dpqb1
"""
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.ndimage import label
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, blob_log
from abc import ABC, abstractmethod
from skimage import measure, feature, color

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

class HDoGDetector_2D(DetectorBase):
    """
    A Hessian-based Difference-of-Gaussian spot detector.
    """
    def __init__(self, sigmas=np.array([2,2]), dsigmas=np.array([1.5,1.5]), use_gaussian_derivatives=False):
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
        return detectBlobHDoG_2D(data,self.sigmas,self.dsigmas,self.use_gaussian_derivatives)
    
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
   
class HDoGDetector_SKImmage_2D(DetectorBase):
    """
    A Hessian-based Difference-of-Gaussian spot detector.
    """
    def __init__(self, sigmas=np.array([2,2]), dsigmas=np.array([1.5,1.5]), use_gaussian_derivatives=False):
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
        return detectBlobHDoG_Skimage_2D(data,self.sigmas,self.dsigmas)
    
class HDoGDetector_SKImage(DetectorBase):
    """
    A Hessian-based Difference-of-Gaussian spot detector using skimage library.
    """
    def __init__(self, sigmas=np.array([2,2,2]), dsigmas=np.array([1.5,1.5,1.5])):
        """
        Initialize the detector with scale parameters and derivative flag.
        """
        super().__init__(sigmas=sigmas, 
                         dsigmas=dsigmas)  # Store in base class dictionary
        
        self.sigmas = sigmas # Store in subclass
        self.dsigmas = dsigmas
        
    def detect(self, data):
        """
        Dummy detection method.

        Parameters:
        - data: Input data

        Returns:
        - Masked data where each blob is labeled with integer >= 1
        """
        return detectBlobHDoG_Skimage(data,self.sigmas,self.dsigmas)

class ThresholdingDetector(DetectorBase):
    """
    A thresholding spot detector.
    """
    def __init__(self, threshold=5, use_gaussian_filter= False, filter_size=3, sigma=1):
        """
        Initialize the detector with threhsold parameter
        """
        super().__init__(threshold=threshold)  # Store in base class dictionary
        self.threshold = threshold # Store in subclass
        self.use_gaussian_filter = use_gaussian_filter
        self.filter_size = filter_size
        self.sigma = sigma
        
    def detect(self, data):
        """
        Dummy detection method.

        Parameters:
        - data: Input data

        Returns:
        - Masked data where each blob is labeled with integer >= 1
        """
        
        # Step 1: Noise reduction via Gaussian or Median filtering
        if self.use_gaussian_filter:
            filtered_data = gaussian_filter(data, sigma=self.sigma)
        else:
            filtered_data = median_filter(data, size=self.filter_size)
        
        # Step 2: Apply thresholding and detect blobs
        return thresholdingDetection(filtered_data, self.threshold)
    
    
class ThresholdingDetector_2(DetectorBase):
    """
    A thresholding spot detector.
    """
    def __init__(self, threshold=5, use_gaussian_filter= False, filter_size=3, sigma=1):
        """
        Initialize the detector with threhsold parameter
        """
        super().__init__(threshold=threshold)  # Store in base class dictionary
        self.threshold = threshold # Store in subclass
        self.use_gaussian_filter = use_gaussian_filter
        self.filter_size = filter_size
        self.sigma = sigma
        
    def detect(self, data):
        """
        Dummy detection method.

        Parameters:
        - data: Input data

        Returns:
        - Masked data where each blob is labeled with integer >= 1
        """
        
        # Step 1: Noise reduction via Gaussian or Median filtering
        if self.use_gaussian_filter:
            filtered_data = gaussian_filter(data, sigma=self.sigma)
        else:
            filtered_data = median_filter(data, size=self.filter_size)
        
        # Step 2: Apply thresholding and detect blobs
        return thresholdingDetection_2(filtered_data, self.threshold)

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


def thresholdingDetection_2(data, threshold_value):
    """
    Threshold multidimensional data and apply connected component labeling.

    Parameters:
    -----------
    data : ndarray
        Input n-dimensional data (can be 3D, 4D, etc.).
    threshold_value : float
        The threshold value to identify regions of interest.

    Returns:
    --------
    tuple
        - blobs : ndarray
            Labeled blob regions (same dimensionality as input data).
        - num_blobs : int
            Number of blobs detected.
    """
    # Threshold the data: set values greater than threshold to True
    thresholded = data > threshold_value
    
    # Label the blobs (connected components) in the thresholded image
    labeled_image = measure.label(thresholded)
    
    # Calculate blob properties (like bounding box, area, etc.)
    blob_props = measure.regionprops(labeled_image)

    # Get the number of blobs
    num_blobs = len(blob_props)

    return labeled_image, num_blobs, blob_props


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
    hess_mat = hessian_matrix(dog_norm, sigma=sigmas)
    
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


def detectBlobHDoG_2D(data, sigmas, dsigmas, use_gaussian_derivatives):
    """
    Detects blobs using the Hessian-based Difference of Gaussians (DoG) method.

    Parameters:
    -----------
    data : ndarray
        Input 2D data.

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
    hess_mat = hessian_matrix(dog_norm, sigma=sigmas)
    D1 = np.zeros(hess_mat[0].shape)
    D2 = np.zeros(hess_mat[0].shape)
    for i1 in range(hess_mat[0].shape[0]):
        for i2 in range(hess_mat[0].shape[1]):
                h_mat = np.array([
                    [hess_mat[0][i1, i2], hess_mat[1][i1, i2]],
                    [hess_mat[1][i1, i2], hess_mat[2][i1, i2]]
                ])
                D1[i1,i2] = h_mat[0,0]
                D2[i1,i2] = np.linalg.det(h_mat[:2,:2])
                

    posDefIndicator = (D1 > 0) & (D2 > 0) 
    blobs, num_blobs = label(posDefIndicator)
    return blobs, num_blobs

def detectBlobHDoG_Skimage(data, sigmas, dsigmas):
    """
    Detects blobs using the Hessian-based Difference of Gaussians (DoG) method (skimage-based).

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
    # 1. Compute normalized DoG using skimage (Gaussian filtering)
    dog_norm = DoG(data, sigma=sigmas, dsigma=dsigmas)

    # 2. Compute the Hessian matrix using skimage
    hess_mat = hessian_matrix(dog_norm, sigma=sigmas)

    # 3. Compute eigenvalues of the Hessian matrix using skimage
    eigenvalues = hessian_matrix_eigvals(hess_mat)

    # Extract positive definite regions (blobs)
    D1 = eigenvalues[0]  # Eigenvalue associated with first principal curvature
    D2 = eigenvalues[1]  # Eigenvalue associated with second principal curvature
    D3 = eigenvalues[2]  # Eigenvalue associated with third principal curvature
    
    # Find positive definite regions (blobs)
    posDefIndicator = (D1 > 0) & (D2 > 0) & (D3 > 0)
    
    # Label the blobs
    blobs, num_blobs = label(posDefIndicator)
    return blobs, num_blobs


def detectBlobHDoG_Skimage_2D(data, sigmas, dsigmas):
    """
    Detects blobs using the Hessian-based Difference of Gaussians (DoG) method (skimage-based).

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
    # 1. Compute normalized DoG using skimage (Gaussian filtering)
    dog_norm = DoG(data, sigma=sigmas, dsigma=dsigmas)

    # 2. Compute the Hessian matrix using skimage
    hess_mat = hessian_matrix(dog_norm, sigma=sigmas)

    # 3. Compute eigenvalues of the Hessian matrix using skimage
    eigenvalues = hessian_matrix_eigvals(hess_mat)

    # Extract positive definite regions (blobs)
    D1 = eigenvalues[0]  # Eigenvalue associated with first principal curvature
    D2 = eigenvalues[1]  # Eigenvalue associated with second principal curvature
    
    # Find positive definite regions (blobs)
    posDefIndicator = (D1 > 0) & (D2 > 0)
    
    # Label the blobs
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
    return sigma[0]**(gamma-1)*(g2 - g1) / (np.mean(sigma) * np.mean(dsigma))



