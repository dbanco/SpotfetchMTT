# -*- coding: utf-8 -*-
"""
feature_extraction

Created on Tue Feb 18 16:42:59 2025
@author: dpqb1
"""
import numpy as np
from scipy.ndimage import center_of_mass

from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.
    
    A feature extractor processes a detected spot and extracts relevant features.
    """

    def __init__(self, params=None):
        """
        Initialize the feature extractor.

        Parameters:
        - params (dict, optional): Configuration parameters for the feature extraction process.
        """
        self.params = params if params is not None else {}

    @abstractmethod
    def extract_features(self, detection):
        """
        Extract features from a detected spot.

        Parameters:
        - detection (object): The detected spot containing raw measurement data.

        Returns:
        - dict: A dictionary containing extracted feature values.
        """
        pass

class BasicFeatureExtractor(FeatureExtractor):
    def extract_features(self, detection):
        """
        Extract basic features: center of mass and intensity.

        Parameters:
        - detection (object): The detected spot containing raw measurement data.

        Returns:
        - dict: Extracted feature values.
        """
        return {
            "com": detection.com,   # Center of mass
            "intensity": detection.intensity,  # Total intensity
        }



def compute_center_of_mass(x, labels=None, index=None):
    """
    Calculate the center of mass of the values of an array at labels.

    Parameters
    ----------
    input : ndarray
        Data from which to calculate center-of-mass. The masses can either
        be positive or negative.
    labels : ndarray, optional
        Labels for objects in `input`, as generated by `ndimage.label`.
        Only used with `index`. Dimensions must be the same as `input`.
    index : int or sequence of ints, optional
        Labels for which to calculate centers-of-mass. If not specified,
        the combined center of mass of all labels greater than zero
        will be calculated. Only used with `labels`.

    Returns
    -------
    center_of_mass : tuple, or list of tuples
        Coordinates of centers-of-mass.
    """
    return np.array(center_of_mass(x, labels=None, index=None))

def find_bounding_box(mask):
  """
  Finds the bounding box coordinates of non-zero pixels in a binary mask.

  Parameters:
  -----------
    mask: A 3D numpy array representing the binary mask or masked data

  Returns:
  --------
    A tuple (tth_min, tth_max, eta_min, eta_max, ome_min, ome_max) representing 
    the bounding box coordinates or None if no non-zero pixels are found.
  """
  rows = np.any(mask, axis=1)
  cols = np.any(mask, axis=0)
  tubs = np.any(mask, axis=2)
  
  if not np.any(rows) or not np.any(cols) or not np.any(tubs):
    return None  # Return None if the mask is empty

  tth_min, tth_max = np.where(rows)[0][[0, -1]]
  eta_min, eta_max = np.where(cols)[0][[0, -1]]
  ome_min, ome_max = np.where(tubs)[0][[0, -1]]
  
  return np.array([tth_min, tth_max, eta_min, eta_max, ome_min, ome_max])


def compute_intensity(x):
    """
    Calculate total intensity of the spot

    Parameters
    ----------
    x input : ndarray
        Data from which to calculate total intensity. 

    Returns
    -------
    intensity : float
    The total intensity computed as the sum of all values in x.
    """
    
    intensity = np.sum(x)
    
    return intensity

def compute_velocity(prev_com, curr_com, dt=1):
    """
    compute the velocity og the com for a blob across two consecutive time step

    Parameters
    ----------
    prev_com : tuple or array
        The Center of mass at the previous time step
    curr_com : tuple or array
        The positio of the center of mass at the current timestep
    dt : float, optional
        The default is 1.

    Returns
    -------
    velocities : tuple 
        The velocity between two time step in each tth, eta, and 
        omega dimension.

    """
    #change in position
    com_change= np.array(curr_com) - np.array(prev_com)
    velocity= com_change / dt
    
    
    return velocity


def compute_acc(prev_com, curr_com, dt=1):
    """
    compute the acceleration of the com for a blob across two consecutive time step

    Parameters
    ----------
    prev_com : tuple or array
        The Center of mass at the previous time step
    curr_com : tuple or array
        The positio of the center of mass at the current timestep
    dt : float, optional
        The default is 1.

    Returns
    -------
    acceleration : tuple 
        The acceleration between two time step in each tth, eta, and 
        omega dimension.

    """
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    