# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:39:17 2025

spot_detection

@author: dpqb1
"""
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, label
from skimage.feature import hessian_matrix, hessian_matrix_det, hessian_matrix_eigvals, blob_log
from abc import ABC, abstractmethod
from skimage import measure, feature, color

# import cupy as cp
# from cupyx.scipy.ndimage import gaussian_filter, median_filter, label as gpu_gaussian_filter, gpu_median_filter, gpu_label

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
        Parameters:
        - data: Input data

        Returns:
        - Masked data where each blob is labeled with integer >= 1
        """
        return detectBlobHDoG(data,self.sigmas,self.dsigmas,self.use_gaussian_derivatives)

class MultiscaleHDoGDetector(DetectorBase):
    """
    A Hessian-based Difference-of-Gaussian spot detector.
    """
    def __init__(self, sigma_sets=np.array([[1,0.5,0],[1,1,0],[1.5,2,0],[1.5**2,2**2,0]]), 
                                          dsigmas=np.array([1.5,1.5,1.5]), 
                                          use_gaussian_derivatives=False):
        """
        Initialize the detector with scale parameters and derivative flag.
        """
        super().__init__(sigma_sets=sigma_sets, 
                         dsigmas=dsigmas,
                         use_gaussian_derivatives=use_gaussian_derivatives)  # Store in base class dictionary
        self.sigma_sets = sigma_sets # Store in subclass
        self.dsigmas = dsigmas
        self.use_gaussian_derivatives = use_gaussian_derivatives
        
    def detect(self, data):
        """
        Parameters:
        - data: Input data

        Returns:
        - Masked data where each blob is labeled with integer >= 1
        """
        return detectBlobHDoGMultiscale(data,self.sigma_sets,self.dsigmas,self.use_gaussian_derivatives)
   

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


class MeanThresholdingDetector(DetectorBase):
    """
    A thresholding spot detector.
    """
    def __init__(self, factor=1, use_gaussian_filter= False, filter_size=3, sigma=1):
        """
        Initialize the detector with threhsold parameter
        """
        super().__init__(factor=factor)  # Store in base class dictionary
        self.use_gaussian_filter = use_gaussian_filter
        self.filter_size = filter_size
        self.sigma = sigma
        self.factor = factor
        
    def detect(self, data):
        """
        Parameters:
        - data: Input data

        Returns:
        - Masked data where each blob is labeled with integer >= 1
        """
        
        # Step 1: Noise reduction via Gaussian or Median filtering
        threshold = self.factor*np.mean(data)
        if self.use_gaussian_filter:
            filtered_data = gaussian_filter(data, sigma=self.sigma)
        else:
            filtered_data = median_filter(data, size=self.filter_size)
        
        # Step 2: Apply thresholding and detect blobs
        return thresholdingDetection(filtered_data, threshold)

def detectBlobHDoGMultiscale(data,sigma_sets,dsigma_sets,use_gaussian_derivatives):
    """
    Detects blobs using multiscale Hessian-based Difference of Gaussians (DoG) method.

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
    
    # 1. Compute DoG at multiple scales and find maximum
    dog_stack = []
    hessian_masks = []
    for i, sigmas in enumerate(sigma_sets):
        # 1.1 Compute normalized DoG
        dog_norm = DoG(data, sigma=sigmas, dsigma=dsigma_sets[i])
        dog_stack.append(dog_norm)
    dog_stack = np.stack(dog_stack, axis=0)

    # NMS across scale
    dog_pad = np.pad(dog_stack, ((1,1), (0,0), (0,0), (0,0)), constant_values=-np.inf)
    nms_mask = (dog_pad[1:-1] > dog_pad[:-2]) & (dog_pad[1:-1] > dog_pad[2:])
       
    # Thresholding
    mag_thresh = np.percentile(dog_stack, 85)
    strong_response = dog_stack > mag_thresh
    nms_mask &= strong_response
    
    # Apply Hessian filter at each scale
    for i, sigmas in enumerate(sigma_sets):
        if not np.any(nms_mask[i]):
            hessian_masks.append(np.zeros_like(nms_mask[i], dtype=bool))
            continue

        hess = hessian_matrix(dog_stack[i], sigma=sigmas, order='rc')

        # Build full Hessian tensor H[..., 3, 3]
        H = np.zeros(hess[0].shape + (3, 3))
        H[..., 0, 0] = hess[0]  # Hxx
        H[..., 0, 1] = hess[1]  # Hxy
        H[..., 0, 2] = hess[2]  # Hxz
        H[..., 1, 0] = hess[1]
        H[..., 1, 1] = hess[3]  # Hyy
        H[..., 1, 2] = hess[4]  # Hyz
        H[..., 2, 0] = hess[2]
        H[..., 2, 1] = hess[4]
        H[..., 2, 2] = hess[5]  # Hzz

        # Sylvester’s criterion for positive-definite Hessian
        D1 = H[..., 0, 0]
        D2 = np.linalg.det(H[..., :2, :2])
        D3 = np.linalg.det(H)

        pos_def = (D1 > 0) & (D2 > 0) & (D3 > 0)
        hessian_masks.append(pos_def)

    hessian_masks = np.stack(hessian_masks, axis=0)
    final_mask = np.any(hessian_masks, axis=0)

    blobs, num_blobs = label(final_mask)
    return blobs, num_blobs

def detectBlobHDoGMultiscale2D(data,sigma_sets,dsigma_sets,use_gaussian_derivatives):
    """
    Detects blobs using multiscale Hessian-based Difference of Gaussians (DoG) method.

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
    
    # 1. Compute DoG at multiple scales and find maximum
    dog_stack = []
    hessian_masks = []
    for i, sigmas in enumerate(sigma_sets):
        # 1.1 Compute normalized DoG
        dog_norm = DoG(data, sigma=sigmas, dsigma=dsigma_sets[i])
        dog_stack.append(dog_norm)
    dog_stack = np.stack(dog_stack, axis=0)

    # # NMS across scale
    # dog_pad = np.pad(dog_stack, ((1,1), (0,0), (0,0)), constant_values=-np.inf)
    # nms_mask = (dog_pad[1:-1] > dog_pad[:-2]) & (dog_pad[1:-1] > dog_pad[2:])
       
    # # Thresholding
    # mag_thresh = np.percentile(dog_stack, 80)
    # strong_response = dog_stack > mag_thresh
    # nms_mask &= strong_response
    
    # Apply Hessian filter at each scale
    for i, sigmas in enumerate(sigma_sets):
        hess = hessian_matrix(dog_stack[i], sigma=sigmas, order='rc')

        # Build full Hessian tensor H[..., 3, 3]
        H = np.zeros(hess[0].shape + (2, 2))
        H[..., 0, 0] = hess[0]  # Hxx
        H[..., 0, 1] = hess[1]  # Hxy
        H[..., 1, 0] = hess[1]  # Hyx
        H[..., 1, 1] = hess[2]  # Hyy


        # Sylvester’s criterion for positive-definite Hessian
        D1 = H[..., 0,0]
        D2 = np.linalg.det(H[..., :1,:1])

        pos_def = (D1 > 0) & (D2 > 0)
        hessian_masks.append(pos_def)

    hessian_masks = np.stack(hessian_masks, axis=0)
    final_mask = np.any(hessian_masks, axis=0)

    blobs, num_blobs = label(final_mask)
    return blobs, num_blobs

def detectBlobHessianMultiscale(data,sigma_sets,dsigma_sets,use_gaussian_derivatives):
    """
    Detects blobs using multiscale Hessian-based Difference of Gaussians (DoG) method.

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
    
    # 1. Compute DoG at multiple scales and find maximum
    dog_stack = []
    hessian_masks = []
    for i, sigmas in enumerate(sigma_sets):
        # 1.1 Compute normalized DoG
        dog_norm = DoG(data, sigma=sigmas, dsigma=dsigma_sets[i])
        dog_stack.append(dog_norm)
    dog_stack = np.stack(dog_stack, axis=0)

    # NMS across scale
    dog_pad = np.pad(dog_stack, ((1,1), (0,0), (0,0), (0,0)), constant_values=-np.inf)
    nms_mask = (dog_pad[1:-1] > dog_pad[:-2]) & (dog_pad[1:-1] > dog_pad[2:])
       
    # Thresholding
    mag_thresh = np.percentile(dog_stack, 85)
    strong_response = dog_stack > mag_thresh
    nms_mask &= strong_response
    
    # Apply Hessian filter at each scale
    for i, sigmas in enumerate(sigma_sets):
        if not np.any(nms_mask[i]):
            hessian_masks.append(np.zeros_like(nms_mask[i], dtype=bool))
            continue

        hess = hessian_matrix(dog_stack[i], sigma=sigmas, order='rc')

        # Build full Hessian tensor H[..., 3, 3]
        H = np.zeros(hess[0].shape + (3, 3))
        H[..., 0, 0] = hess[0]  # Hxx
        H[..., 0, 1] = hess[1]  # Hxy
        H[..., 0, 2] = hess[2]  # Hxz
        H[..., 1, 0] = hess[1]
        H[..., 1, 1] = hess[3]  # Hyy
        H[..., 1, 2] = hess[4]  # Hyz
        H[..., 2, 0] = hess[2]
        H[..., 2, 1] = hess[4]
        H[..., 2, 2] = hess[5]  # Hzz

        # Sylvester’s criterion for positive-definite Hessian
        D1 = H[..., 0, 0]
        D2 = np.linalg.det(H[..., :2, :2])
        D3 = np.linalg.det(H)

        pos_def = (D1 > 0) & (D2 > 0) & (D3 > 0)
        hessian_masks.append(pos_def)

    hessian_masks = np.stack(hessian_masks, axis=0)
    final_mask = np.any(hessian_masks, axis=0)

    blobs, num_blobs = label(final_mask)
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
    g2 = gaussian_filter(x, sigma=np.array(sigma) + np.array(dsigma))
    return np.mean(sigma)**(gamma-1)*(g2 - g1) / (np.mean(sigma) * np.mean(dsigma))

# def hessian_matrix_cupy(img):
#     img_xx = cp.gradient(cp.gradient(img, axis=0), axis=0)
#     img_yy = cp.gradient(cp.gradient(img, axis=1), axis=1)
#     img_xy = cp.gradient(cp.gradient(img, axis=0), axis=1)
#     return img_xx, img_yy, img_xy

