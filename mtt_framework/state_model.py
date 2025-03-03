# -*- coding: utf-8 -*-
"""
state_model

Description:
Module containing state models for MTT

Created on Tue Feb 18 16:37:10 2025
@author: Bahar, Daniel
"""

import numpy as np
from abc import ABC, abstractmethod
from mtt_framework.feature_extraction import (
    compute_center_of_mass, 
    find_bounding_box, 
    compute_intensity, 
    compute_velocity, 
    compute_acc
)

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
        return Measurement(data_masked)
        
    @abstractmethod
    def get_measurements(self, blobs, data):
        """
        Update the state based on a new measurement.

        Parameters:
        - measurement (dict): The new measurement to update the state.

        Returns:
        - dict: Updated state.
        """
        measurements = []
        unique_labels = np.unique(blobs)
        unique_labels = unique_labels[unique_labels != 0]
        
        for blob_label in unique_labels:
            mask = (blobs == blob_label)
            x_masked = data * mask
            measurement = self.compute_features(x_masked)
            measurements.append(measurement)
        return measurements
        
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
    def __init__(self, initial_state, feature_extractor):
        super().__init__(initial_state, feature_extractor)
        # initialize previous state
        self.prev_state = None
        
        
    def compute_features(self, data_masked):
        """
        Call the parent "compute_features" method
        """
        
        return super().compute_features(data_masked)
        
    def get_measurements(self, blobs, data):
        """
        Call the parent "get_measurements" method
        """
        return super().get_measurements(blobs, data)

    
    def update_state(self, measurement, dt):
        """
        Update the state based on a new measurement.
        For example, this could use a Kalman filter or simple estimation.
        
        Parameters:
        - measurement (dict): The new measurement to update the state.
        
        Returns:
        - dict: Updated state.
        """
        
        if self.prev_state is not None:
            # convert to numpy
            current_position = np.array(measurement.com)  # Convert to numpy array
            prev_position = np.array(self.prev_state['position'])  # Convert to numpy array

            # Calculate velocity using the difference between current and previous CoM Position
            velocity = (current_position - prev_position) / self.prev_state['dt']
            self.state['velocity'] = velocity
        else:
            # If no previous state, assume initial state
            self.state['velocity'] = np.zeros(3)
            
        # Update state with measurement directly
        self.state['position'] = measurement.com
        self.state['velocity'] = self.state['velocity']
        self.state['acceleration'] = measurement.com_acceleration
        
        self.prev_state = {
            'position': measurement.com,
            'velocity': self.state['velocity'],
            'dt': dt
        }
        
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
        # Transition position using velocity, assume consisten for now
        new_position = state['position'] + state['velocity'] * dt
        new_state = state.copy()
        new_state['position'] = new_position
        return new_state


# Subclass 2: Example of a Kalman Filter model
class KalmanModel(StateModel):
    def __init__(self, initial_state, feature_extractor, process_noise, measurement_noise, dt):
        super().__init__(initial_state, feature_extractor)
        """
        Initialize the Kalman filter model with process and measurement noise.
        
        Parameters:
        - initial_state (dict): Initial state of the object (position, velocity).
        - process_noise (float): Process noise covariance, assumes constant velocity model.
        - measurement_noise (float): Measurement noise covariance.
        - dt (float): Constant time delta between measurements.
        """
        # Initialize state (position and velocity)
        self.state = np.hstack((initial_state['position'], initial_state['velocity']))  # [tta, eta, ome, v_tta, v_eta, v_ome]
        
        # 6D state covariance matrix (position and velocity)
        self.P = np.eye(6)  # Identity matrix for simplicity
        
        # State transition matrix for constant velocity model
        self.F = np.array([
            [1, 0, 0, dt, 0, 0], 
            [0, 1, 0, 0, dt, 0], 
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (assumes we measure both position and velocity)
        self.H = np.eye(6)  # Direct measurement of state
        
        # Process noise covariance (Q), assumed constant
        self.Q = np.eye(6) * process_noise
        
        # Measurement noise covariance (R)
        self.R = np.eye(6) * measurement_noise
        
        # Time step (dt)
        self.dt = dt
        
    def compute_features(self, data_masked):
        """
        Call the parent "compute_features" method
        """
        
        return super().compute_features(data_masked)
        
    def get_measurements(self, blobs, data):
        """
        Call the parent "get_measurements" method
        """
        return super().get_measurements(blobs, data)
    
    def update_state(self, measurement, dt):
        """
        Update the state using the Kalman filter.
        
        Parameters:
        - measurement (Measurement): The new measurement (position, velocity).
        
        Returns:
        - dict: Updated state (position, velocity).
        """
        # Measurement vector
        z = np.hstack((measurement.com, measurement.com_velocity))

        # Innovation (residual)
        y = z - np.dot(self.H, self.state)

        # Innovation covariance: S_k = H * P_k|k-1 * H.T + R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Kalman Gain: K_k = P_k|k-1 * H.T * S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) 

        # State update: x_k|k = x_k|k-1 + K * y_k
        self.state = self.state + np.dot(K, y)

        # Covariance update: P_k|k = (I - K * H) * P_k|k-1
        I = np.eye(self.P.shape[0])  # Identity matrix
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)


        return {'position': self.state[:3], 'velocity': self.state[3:], 'dt':dt}  # Return position and velocity

    def transition(self, state, dt):
        """
        Predict the next state assuming constant velocity.
        
        Parameters:
        - dt (float): Time step.
        
        Returns:
        - dict: Transitioned state.
        """
        # Predict next state using the transition matrix F
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        # Return the predicted state (position and velocity)
        return {'position': self.state[:3], 'velocity': self.state[3:]}
    
    
    
# Subclass 3: Example of a constant acceleration model
class ConstantAccelerationStateModel(StateModel):
    """
    A state model that assumes constant acceleration between measurements.
    """
    
    def compute_features(self, data_masked):
        """
        Call the parent "compute_features" method
        """
        
        return super().compute_features(self, data_masked)
        
    def get_measurements(self, blobs, data):
        """
        Call the parent "get_measurements" method
        """
        return super().get_measurements(blobs, data)
    
    def update_state(self, measurement):
        """
        Update the state based on a new measurement.
        
        Parameters:
        - measurement (dict): The new measurement to update the state.
        
        Returns:
        - dict: Updated state.
        """
        # Update state with measurement directly
        self.state['position'] = measurement.com
        self.state['velocity'] = measurement.com_velocity
        self.state['acceleration'] = measurement.com_acceleration
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
    
    def __init__(self, x, prev_com= None, prev_velocity= None, dt=1):
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
        self.intensity = compute_intensity(x)
        
        if prev_com is not None and prev_velocity is not None:
            self.com_velocity = compute_velocity (self.com,prev_com)
            self.com_acceleration = compute_acc(self.com_velocity,prev_velocity)
        else:
            self.com_velocity = np.zeros_like(self.com)
            self.com_acceleration = np.zeros_like(self.com)
            
        
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

        self.com = measurement.com
        self.bound_box = measurement.bound_box
        self.intensity = measurement.intensity
        self.com_velocity = measurement.com_velocity
        self.com_acceleration = measurement.com_acceleration
        #placeholder for degree of overlap
        self.overlap = 0    

    def update(self, measurement):
        """
        Update the track with a new detection.
        
        Parameters:
        - measurment: A measurement object
        """
        
        self.com = measurement.com
        self.bound_box = measurement.bound_box
        self.intensity = measurement.intensity
        self.com_velocity = measurement.com_velocity
        self.com_acceleration = measurement.com_acceleration
        #placeholder for degree of overlap
        self.overlap = 0







