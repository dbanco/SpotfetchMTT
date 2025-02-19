# -*- coding: utf-8 -*-
"""
state_model

Description:
Module containing state models for MTT

Created on Tue Feb 18 16:37:10 2025
@author: Daniel Banco
"""

import numpy as np
from abc import ABC, abstractmethod

class StateModel(ABC):
    """
    Abstract base class for StateModel that defines common structure
    for handling measurements, tracks, and state transitions.
    """
    
    def __init__(self, initial_state, feature_extractor):
        """
        Initialize the state model with an initial state.
        
        Parameters:
        - initial_state (dict): Dictionary representing the initial state of the track.
        """
        self.state = initial_state
        self.feature_extractor = feature_extractor
        
    @abstractmethod
    def compute_features(self, data_masked):
        """
        Update the state based on a new measurement.

        Parameters:
        - data_masked (arra)

        Returns:
        - measurement (dict): a single measurement
        """
        pass
        
    @abstractmethod
    def get_measurements(self, blobs, data):
        """
        Update the state based on a new measurement.

        Parameters:
        - measurement (dict): The new measurement to update the state.

        Returns:
        - dict: Updated state.
        """
        pass
        
    @abstractmethod
    def update_state(self, measurement):
        """
        Update the state based on a new measurement.

        Parameters:
        - measurement (dict): The new measurement to update the state.

        Returns:
        - dict: Updated state.
        """
        pass
    
    @abstractmethod
    def transition(self, state, dt):
        """
        Transition the state model based on time (dt) and the current state.
        
        Parameters:
        - state (dict): Current state of the object.
        - dt (float): Time delta.
        
        Returns:
        - dict: Transitioned state.
        """
        pass


# Subclass 1: Example of a constant velocity model
class BasicModel(StateModel):
    """
    A state model that assumes constant velocity between measurements.
    """
    def compute_features(self, data_masked):
        """
        Computes features from masked data

        Parameters:
        -----------
        blobs : ndarray
            Labeled blob regions.

        Returns:
        --------
        measurements:
            List of measurement objects containing candidate spots.
        """
        
    def get_measurements(self, blobs, data):
        """
        Converts labeled blobs into a list of Measurement objects.

        Parameters:
        -----------
        blobs : ndarray
            Labeled blob regions.

        Returns:
        --------
        measurements:
            List of measurement objects containing candidate spots.
        """
        measurements = []
        unique_labels = np.unique(blobs)
        
        # Exclude background (label 0)
        unique_labels = unique_labels[unique_labels != 0]

        for blob_label in unique_labels:
            mask = (blobs == blob_label)  # Extract binary mask for the current blob
            x_masked = data * mask
            measurement = compute_featuers(x_masked)
            measurements.append(measurement)

        return measurements    
    
    def update_state(self, measurement):
        """
        Update the state based on a new measurement.
        For example, this could use a Kalman filter or simple estimation.
        
        Parameters:
        - measurement (dict): The new measurement to update the state.
        
        Returns:
        - dict: Updated state.
        """
        # Simple example: Update state with measurement directly
        self.state['position'] = measurement['position']
        self.state['velocity'] = measurement['velocity']
        return self.state
    
    def transition(self, state, dt):
        """
        Transition the state assuming constant velocity.
        
        Parameters:
        - state (dict): Current state of the object.
        - dt (float): Time delta.
        
        Returns:
        - dict: Transitioned state.
        """
        # Transition position using velocity
        new_position = state['position'] + state['velocity'] * dt
        new_state = state.copy()
        new_state['position'] = new_position
        return new_state


# Subclass 2: Example of a constant acceleration model
class ConstantAccelerationStateModel(StateModel):
    """
    A state model that assumes constant acceleration between measurements.
    """
    
    def update_state(self, measurement):
        """
        Update the state based on a new measurement.
        
        Parameters:
        - measurement (dict): The new measurement to update the state.
        
        Returns:
        - dict: Updated state.
        """
        # Simple example: Update state with measurement directly
        self.state['position'] = measurement['position']
        self.state['velocity'] = measurement['velocity']
        self.state['acceleration'] = measurement['acceleration']
        return self.state
    
    def transition(self, state, dt):
        """
        Transition the state assuming constant acceleration.
        
        Parameters:
        - state (dict): Current state of the object.
        - dt (float): Time delta.
        
        Returns:
        - dict: Transitioned state.
        """
        new_position = state['position'] + state['velocity'] * dt + 0.5 * state['acceleration'] * dt**2
        new_velocity = state['velocity'] + state['acceleration'] * dt
        new_state = state.copy()
        new_state['position'] = new_position
        new_state['velocity'] = new_velocity
        return new_state


class Measurement:
    """Represents a measurement (candidate spot) in 3D space."""
    
    def __init__(self, x):
        """
        Initialize a measured spot.
        
        Parameters:
        -----------
            - x (array): masked pixel intesity data
        
        Properties:
        - com (array): center of mass (tt,eta,ome) of the blob.
        - bound_box (array): boudning box (tth1,tth2,eta1,eta2,ome1,ome2) of blob 
        - intensity (float): Total intensity of the detected spot.
        """
        self.com = compute_center_of_mass(x)
        self.bound_box = find_bounding_box(x)
        self.intensity = np.sum(x)
        
        pass

class Track:
    """Represents a tracked spot in 3D space."""
    
    def __init__(self, measurement):
        """
        Initialize a detected spot.
        
        Parameters:
        - measurment: A measurement object
        
        Properties:
        - com (array): center of mass (tt,eta,ome) of the blob.
        - bound_box (array): boudning box (tth1,tth2,eta1,eta2,ome1,ome2) of blob 
        - intensity (float): Total intensity of the detected spot.
        - overlap
        - com_velocity
        - Maybe other things related to state transition model?
        
        """

        pass

    def update(self, measurement):
        """
        Update the track with a new detection.
        
        Parameters:
        - measurment: A measurement object
        """
        
        pass







