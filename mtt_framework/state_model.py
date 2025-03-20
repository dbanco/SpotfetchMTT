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
from mtt_framework.detection_spot import Detection

        

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
    def get_measurements(self, frame, blobs):
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
            x_masked = frame * mask
            # Create a Detection object
            detection = Detection(blob_label, mask, x_masked)
            measurement = self.feature_extractor.extract_features(detection)
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
        self.prediction = None
        
    def get_measurements(self, frame, blobs):
        """
        Call the parent "get_measurements" method
        """
        return super().get_measurements(frame, blobs)

    
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
            current_com = np.array(measurement['com'])  # Convert to numpy array
            prev_com = np.array(self.prev_state['com'])  # Convert to numpy array

            # Calculate velocity using the difference between current and previous CoM Position
            velocity = (current_com - prev_com) / self.prev_state['dt']
            self.state['velocity'] = velocity
        else:
            # If no previous state, assume initial state
            self.state['velocity'] = np.zeros(3)
            
        # Update state with measurement directly
        self.state['com'] = measurement["com"]
        self.state['velocity'] = self.state['velocity']
        self.state['acceleration'] = np.zeros(3)
        
        self.prev_state = {
            'com': measurement['com'],
            'velocity': self.state['velocity'],
            'dt': dt
        }
        
        return self.state
    
    def transition(self, dt):
        """
        Transition the state assuming constant velocity.
        
        Parameters:
        - state (dict): Current state of the object.
        - dt (float): Time delta.
        
        """
        # Transition com using velocity, assume consisten for now
        self.prediciton = self.state['com'] + self.state['velocity'] * dt
    
    def compute_gating_distance(self,measurement):
        diff = measurement['com'] - self.state['com']
        return np.dot(diff.T, diff)
    
    def compute_hypothesis_cost(self, tracks, measurement, event_type):
        """
        Computes the cost associated with a hypothesis based on the event type.
    
        Parameters:
        - tracks: A single track (for "persist") or a list of tracks (for "overlap").
        - measurement: The measurement being considered, containing its center of mass ('com').
        - event_type (str): The type of event ("persist" or "overlap").
    
        Returns:
        - cost (float): Scalar value based on euclidean distance between track(s) and measurement
        """
        
        # Compute cost for persistence: Distance between track's current CoM and the measurement CoM
        if event_type == 'persist':
            return euclidean_dist(tracks.state['com'], measurement['com'])
        
        # Compute cost for overlap: Average CoM of merged tracks and distance to measurement CoM
        elif event_type == 'overlap':
            # Compute the average center of mass of all nodes in the subset
            avg_com = sum(track.state['com'] for track in tracks) / len(tracks)
            return euclidean_dist(avg_com, measurement['com'])
        
        # Return a high penalty for unknown event types (optional safeguard)
        else:
            raise ValueError(f"Unknown event type: {event_type}")

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
        #self.state = np.hstack((initial_state['com'], initial_state['velocity']))  # [tta, eta, ome, v_tta, v_eta, v_ome]
        self.state = {
            'com': initial_state['com'],
            'velocity': initial_state['velocity'],
            }
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
        
        # Measurement matrix (assumes we measure both position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # Mapping position components of state to measurement
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance (Q), assumed constant
        self.Q = np.eye(6) * process_noise
        
        # Measurement noise covariance (R)
        self.R = np.eye(3) * measurement_noise
        
        # Time step (dt)
        self.dt = dt
        
    def get_measurements(self, frame, blobs):
        """
        Call the parent "get_measurements" method
        """
        return super().get_measurements(frame, blobs)
    
    def update_state(self, measurement, dt):
        """
        Update the state using the Kalman filter.
        
        Parameters:
        - measurement (Measurement): The new measurement (position, velocity).
        
        Returns:
        - dict: Updated state (position, velocity).
        """
        # Measurement vector
        z = np.array(measurement['com'])
        state_vector = np.hstack([self.state['com'], self.state['velocity']])

        # Innovation (residual)
        y = z - np.dot(self.H, state_vector)

        # Innovation covariance: S_k = H * P_k|k-1 * H.T + R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Kalman Gain: K_k = P_k|k-1 * H.T * S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) 

        # State update: x_k|k = x_k|k-1 + K * y_k
        update_state = state_vector + np.dot(K, y)
        
        self.state['com'] = update_state[:3]
        self.state['velocity']= update_state[3:]
        self.state['bbox']= measurement['bbox']
        self.state['bbox_center']= measurement['bbox_center']
        self.state['bbox_size']= measurement['bbox_size']
        self.state['intensity'] = measurement['intensity']

        # Covariance update: P_k|k = (I - K * H) * P_k|k-1
        I = np.eye(self.P.shape[0])  # Identity matrix
        self.P = np.dot(I - np.dot(K, self.H), self.P)


        return {'com': self.state['com'], 'velocity': self.state['velocity'], 'dt':dt}  # Return position and velocity

    def transition(self, dt):
        """
        Predict the next state assuming constant velocity.
        
        Parameters:
        - dt (float): Time step.
        
        Returns:
        - dict: Transitioned state.
        """
        # Predict next state using the transition matrix F
        state_vector = np.hstack([self.state['com'], self.state['velocity']])
        pred_state = np.dot(self.F, state_vector)
        
        # Update the state dictionary
        self.state['com'] = pred_state[:3]  # Position (com)
        self.state['velocity'] = pred_state[3:]  # Velocity

        
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        # Return the predicted state (position and velocity)
        return {'com': self.state['com'], 'velocity': self.state['velocity']}
    
    
    def association_likelihood(self, z_m, track_state, P_pred):
        
        # Predict the measurement for the track
        z_pred = np.dot(self.H, track_state)
        
        # Compute the innovation (difference between predicted and actual measurement)
        innovation = z_m - z_pred
        
        # Compute the Mahalanobis distance (using P_pred for covariance)
        S = np.dot(np.dot(self.H, P_pred), self.H.T) + self.R  # Innovation covariance
        inv_S = np.linalg.inv(S)
        likelihood = np.exp(-0.5 * np.dot(innovation.T, np.dot(inv_S, innovation)))
        
        return likelihood
    
    def compute_gating_distance(self, measurement):
        
        diff = measurement['com'] - self.state['com']
        return np.dot(diff.T, diff)
    
    def compute_hypothesis_cost(self, tracks, measurement, event_type):
        """
        Computes the cost associated with a hypothesis based on the event type.
    
        Parameters:
        - tracks: A single track (for "persist") or a list of tracks (for "overlap").
        - measurement: The measurement being considered, containing its center of mass ('com').
        - event_type (str): The type of event ("persist" or "overlap").
    
        Returns:
        - cost (float): Scalar value based on euclidean distance between track(s) and measurement
        """
        
        # Compute cost for persistence: Distance between track's current CoM and the measurement CoM
        if event_type == 'persist':
            return euclidean_dist(tracks.state['com'], measurement['com'])
        
        # Compute cost for overlap: Average CoM of merged tracks and distance to measurement CoM
        elif event_type == 'overlap':
            # Compute the average center of mass of all nodes in the subset
            avg_com = sum(track.state['com'] for track in tracks) / len(tracks)
            return euclidean_dist(avg_com, measurement['com'])
        
        # Return a high penalty for unknown event types (optional safeguard)
        else:
            raise ValueError(f"Unknown event type: {event_type}")
    
class MSTINModel(StateModel):
    def __init__(self, initial_state, feature_extractor, process_noise, measurement_noise, dt, skew_factor=1.0):
        super().__init__(initial_state, feature_extractor)
        """
        Initialize the MSTIN model with skewness and inflation factors.
        
        Parameters:
        - initial_state (dict): Initial state of the object (position, velocity).
        - process_noise (float): Process noise covariance, assumes constant velocity model.
        - measurement_noise (float): Measurement noise covariance.
        - dt (float): Constant time delta between measurements.
        - skew_factor (float): Skewness factor to control the skewness of the distribution.
        """
        self.state = {
            'com': initial_state['com'],
            'velocity': initial_state['velocity'],
        }
        
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
        
        # Measurement matrix (assumes we measure both position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # Mapping position components of state to measurement
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance (Q), assumed constant
        self.Q = np.eye(6) * process_noise
        
        # Measurement noise covariance (R)
        self.R = np.eye(3) * measurement_noise
        
        # Time step (dt)
        self.dt = dt
        
        # Skewness and inflation factors (control distribution skew)
        self.skew_factor = skew_factor
        self.inflation_factor = np.random.uniform(0.1, 1.0)  # Simulated skew-tail inflation factor

    def get_measurements(self, frame, blobs):
        """
        Call the parent "get_measurements" method
        """
        return super().get_measurements(frame, blobs)
    
    def update_state(self, measurement, dt):
        """
        Update the state using the MSTIN model (without Kalman Filter).
        
        Parameters:
        - measurement (Measurement): The new measurement (position, velocity).
        
        Returns:
        - dict: Updated state (position, velocity).
        """
        # Measurement vector
        z = np.array(measurement['com'])
        state_vector = np.hstack([self.state['com'], self.state['velocity']])

        # Compute the innovation directly (without Kalman filtering)
        innovation = z - state_vector[:3]  # Position innovation

        # Add skewed inflation directly into the state update
        skewed_inflation = self.skew_factor * np.random.randn(3)  # Skewness applied to position part
        inflated_innovation = innovation + skewed_inflation * self.inflation_factor

        # Update the state directly with the skewed and inflated innovation
        self.state['com'] += inflated_innovation
        self.state['velocity'] += self.skew_factor * np.random.randn(3)  # Skewed velocity update

        # Update additional tracking information
        self.state['bbox'] = measurement['bbox']
        self.state['bbox_center'] = measurement['bbox_center']
        self.state['bbox_size'] = measurement['bbox_size']
        self.state['intensity'] = measurement['intensity']
        self.state['avg_intensity'] = measurement['avg_intensity']

        # Predict next covariance based on the skewed process model (simulating the system's spread)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return {'com': self.state['com'], 'velocity': self.state['velocity'], 'dt': dt}

    def transition(self, dt):
        """
        Predict the next state assuming constant velocity and skewed inflation.
        
        Parameters:
        - dt (float): Time step.
        
        Returns:
        - dict: Transitioned state.
        """
        # Predict next state using the transition matrix F
        state_vector = np.hstack([self.state['com'], self.state['velocity']])
        pred_state = np.dot(self.F, state_vector)
        
        # Add skew and inflation to the prediction
        skewed_inflation = self.skew_factor * np.random.randn(6)  # Skewness applied
        pred_state += skewed_inflation * self.inflation_factor
        
        # Update the state dictionary
        self.state['com'] = pred_state[:3]  # Position (com)
        self.state['velocity'] = pred_state[3:]  # Velocity

        # Predict covariance with inflation (skewed state prediction)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # Return the predicted state (position and velocity)
        return {'com': self.state['com'], 'velocity': self.state['velocity']}
    
    def association_likelihood(self, z_m, track_state, P_pred):
        """
        Compute the likelihood of association between the measurement and the predicted state.
        
        Parameters:
        - z_m (np.array): Measurement vector.
        - track_state (np.array): Predicted state vector (position and velocity).
        - P_pred (np.array): Predicted state covariance.
        
        Returns:
        - likelihood (float): Association likelihood.
        """
        # Predict the measurement for the track
        z_pred = np.dot(self.H, track_state)
        
        # Compute the innovation (difference between predicted and actual measurement)
        innovation = z_m - z_pred
        
        # Compute the Mahalanobis distance (using P_pred for covariance)
        S = np.dot(np.dot(self.H, P_pred), self.H.T) + self.R  # Innovation covariance
        inv_S = np.linalg.inv(S)
        likelihood = np.exp(-0.5 * np.dot(innovation.T, np.dot(inv_S, innovation)))
        
        return likelihood
    
    def compute_gating_distance(self, measurement):
        """
        Compute the gating distance based on the center of mass.
        
        Parameters:
        - measurement (dict): Measurement with 'com' (center of mass).
        
        Returns:
        - gating_distance (float): Squared Euclidean distance.
        """
        diff = measurement['com'] - self.state['com']
        return np.dot(diff.T, diff)
    
    def compute_hypothesis_cost(self, tracks, measurement, event_type):
        """
        Computes the cost associated with a hypothesis based on the event type.
    
        Parameters:
        - tracks: A single track (for "persist") or a list of tracks (for "overlap").
        - measurement: The measurement being considered, containing its center of mass ('com').
        - event_type (str): The type of event ("persist" or "overlap").
    
        Returns:
        - cost (float): Scalar value based on Euclidean distance between track(s) and measurement.
        """
        
        # Compute cost for persistence: Distance between track's current CoM and the measurement CoM
        if event_type == 'persist':
            return np.linalg.norm(tracks.state['com'] - measurement['com'])
        
        # Compute cost for overlap: Average CoM of merged tracks and distance to measurement CoM
        elif event_type == 'overlap':
            avg_com = sum(track.state['com'] for track in tracks) / len(tracks)
            return np.linalg.norm(avg_com - measurement['com'])
        
        # Return a high penalty for unknown event types (optional safeguard)
        else:
            raise ValueError(f"Unknown event type: {event_type}")
            
            
# Subclass 3: Example of a constant acceleration model
class ConstantAccelerationStateModel(StateModel):
    """
    A state model that assumes constant acceleration between measurements.
    """
    
    def get_measurements(self, frame, blobs):
        """
        Call the parent "get_measurements" method
        """
        return super().get_measurements(frame, blobs)
    
    def update_state(self, measurement):
        """
        Update the state based on a new measurement.
        
        Parameters:
        - measurement (dict): The new measurement to update the state.
        
        Returns:
        - dict: Updated state.
        """
        # Update state with measurement directly
        self.state['position'] = measurement["com"]
        self.state['velocity'] = measurement["velocity"]
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

def persist_loglikelihood(state_model,measurement,track):
    """
    Computes loglikelihood of a measurement being associated with a track
    
    Parameters:
    -----------
    - state_model
    - measurement
    - track
    """
    
    loglikelihood = 1
    
    return loglikelihood

def death_loglikelihood(state_model,measurement,track):
    """
    Computes loglikelihood of a track dying
    
    Parameters:
    -----------
    - state_model
    - track
    """
    loglikelihood = 1
    
    return loglikelihood
    
def euclidean_dist(position1, position2, association_cost=0):
    """
    Compute the Euclidean distance between a track state and a measurement,
    incorporating an optional association cost.

    Parameters:
    - state: 3D NumPy array representing the track's current position [x, y, z].
    - measurement: 3D NumPy array representing the new measurement [x, y, z].
    - association_cost: Additional cost from the M2TA matrix (default: 0).

    Returns:
    - Computed cost (float).
    """
    distance = np.linalg.norm(position1 - position2)  # Euclidean distance
    return distance + association_cost




