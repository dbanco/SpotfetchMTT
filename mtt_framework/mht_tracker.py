# -*- coding: utf-8 -*-
"""

mht_system


Created on Tue Feb 18 16:46:03 2025
@author: dpqb1
"""
import numpy as np

class HypothesisNode:
    def __init__(self, track_id, state, scan, parent=None, cost=0):
        self.track_id = track_id
        self.state = state
        self.scan = scan
        self.parent = parent
        self.children = []
        self.cost = cost

    def add_child(self, child_node):
        self.children.append(child_node)

class HypothesisTree:
    def __init__(self):
        self.root = None
        self.nodes = {}

    def add_node(self, track_id, state, scan, parent_id=None, cost=0):
      
        if parent_id is None:
            if self.root is not None:
                raise ValueError("Root node already exists")
            new_node = HypothesisNode(track_id, state, scan, cost=cost)
            self.root = new_node
            self.nodes[track_id] = new_node
        else:
            parent_node = self.nodes.get(parent_id)
            if parent_node is None:
                raise ValueError(f"Parent node with ID {parent_id} not found")
            new_node = HypothesisNode(track_id, state, parent_node, cost=cost)
            parent_node.add_child(new_node)
            self.nodes[track_id] = new_node

    def get_node(self, track_id):
        return self.nodes.get(track_id)

    def get_best_hypothesis(self):
        
        if self.root is None:
            return None

        best_node = None
        min_cost = float('inf')

        def traverse(node, current_cost):
            nonlocal best_node, min_cost
            current_cost += node.cost
            if not node.children:  
                if current_cost < min_cost:
                    min_cost = current_cost
                    best_node = node
            else:
                for child in node.children:
                    traverse(child, current_cost)

        traverse(self.root, 0)
        
        path = []
        current = best_node
        while current:
            path.append(current)
            current = current.parent
        return path[::-1]

class MHTTracker:
    """Multiple Hypothesis Tracker for 3D spot tracking."""
    
    def __init__(self, state_model, tracks=[], gating_threshold=5.0):
        """
        Initialize the tracker.
        
        Parameters:
        - tracks (list of track hypothesis trees)
        - gating_threshold (float): Threshold for Mahalanobis distance gating.
        """
        self.tracks = tracks
        self.gating_threshold = gating_threshold
        self.state_model = state_model
        pass

    def gating(self, measurements):
        """
        Perform gating by filtering unlikely measurement-track associations.
        
        Parameters:
        - measurements (list of measurements)
        
        Returns:
        - m2ta_matrix (np.array)
        """
        self.m2ta_matrix = np.zeros(len(measurements),len(self.tracks))
        
        for k, track in enumerate(self.tracks):
            for m, measurement in enumerate(measurements):
                diff = measurement.position - track.position
                mahal_dist = np.multiply(diff.transpose(),diff)
                if mahal_dist < self.gating_threhsold:
                    self.m2ta_matrix[m,k] = 1
        
        pass

    def generate_hypotheses(self):
        """
        Generate association hypotheses between existing tracks and new detections.
        
        Parameters:
        - detections (list of Detection): List of detected spots.
        
        Returns:
        - list of tuples: Each tuple represents a hypothesis (track, detection).
        """
        
        for i, track in enumerate(self.tracks):
            update_hypothesis_tree(track, self.m2ta_matrix, measurements , euclidean_cost)
        pass

    def associate_measurements(self, detections):
        """
        Solve the data association problem using the Hungarian algorithm.
        
        Parameters:
        - detections (list of Detection): List of detected spots.
        
        Returns:
        - list of tuples: Each tuple represents an assigned (track, detection) pair.
        """
        pass

    def update_tracks(self, detections):
        """
        Update existing tracks with new detections and create new tracks.
        
        Parameters:
        - detections (list of Detection): List of detected spots.
        """
        pass

    def track_step(self, detections):
        """
        Perform one step of the MHT tracking cycle.
        
        Parameters:
        - detections (list of Detection): List of detected spots at the current time step.
        """
        pass
    
    def process_measurements(self, measurements):
        """
        Takes ikn measurements, associates them with tracks, generates
        hypothesis trees, identifies best global hypothesis, updates state
        """
        
        

def initialize_hypothesis_tree(measurements):
    """
    Initializes the hypothesis trees with multiple measurements.

    Parameters:
    - measurements: List of 3D numpy arrays representing initial measurements.

    Returns:
    - A HypothesisTree object.
    """
    tree = HypothesisTree()
    
    # Create a dummy root node (track_id=-1, no meaningful state)
    tree.add_node(track_id=-1, state=np.array([0, 0, 0]))  # Dummy root
    
    # Add each measurement as a child of the root
    for i, measurement in enumerate(measurements):
        track_id = i  # Assign unique IDs to each initial track
        tree.add_node(track_id, measurement, parent_id=-1)

    return tree

def update_hypothesis_tree(tree, m2ta_matrix, new_measurements, cost_function):
    """
    Updates the hypothesis tree given a new measurement-to-track association matrix.

    Parameters:
    - tree: HypothesisTree object representing the current tree.
    - m2ta_matrix: 2D numpy array (size M x T) where rows are new measurements
                   and columns are existing tracks, with binary or probability values.
    - new_measurements: List of 3D numpy arrays representing the new scan's measurements.
    - cost_function: Function that computes the cost between a track state and a measurement.

    Returns:
    - None (modifies the tree in place).
    """
    existing_tracks = list(tree.nodes.keys())  # Get current track IDs
    new_track_id = max(existing_tracks) + 1 if existing_tracks else 0  # Next track ID
    
    num_measurements, num_tracks = m2ta_matrix.shape
    assert num_measurements == len(new_measurements), "M2TA matrix and measurement count mismatch!"

    for m_idx in range(num_measurements):
        measurement = new_measurements[m_idx]
        
        associated_tracks = np.where(m2ta_matrix[m_idx] > 0)[0]  # Find associated tracks

        if associated_tracks.size > 0:
            # Existing track hypotheses extension
            for k in associated_tracks:
                parent_node = self.tracks[k].get_node(k)
                if parent_node:
                    cost = cost_function(parent_node.state, measurement)
                    tree.add_node(new_track_id, measurement, parent_id=track_id, cost=cost)
                    new_track_id += 1
        else:
            # Unassociated measurement: start a new track
            tree.add_node(new_track_id, measurement, parent_id=-1)  # Link to dummy root
            new_track_id += 1



import networkx as nx
import matplotlib.pyplot as plt

def visualize_hypothesis_tree(hypothesis_tree):
    """
    Visualizes the hypothesis tree using NetworkX and Matplotlib.

    Parameters:
    - hypothesis_tree (HypothesisTree): The hypothesis tree to visualize.
    """
    if hypothesis_tree.root is None:
        print("The hypothesis tree is empty.")
        return

    G = nx.DiGraph()

    def add_edges(node):
        for child in node.children:
            G.add_edge(node.track_id, child.track_id)
            add_edges(child)

    add_edges(hypothesis_tree.root)

    plt.figure(figsize=(10, 6))
    # pos = nx.drawing.nx_pydot.graphviz_layout(hypothesis_tree, prog='dot')
    pos = nx.spring_layout(G, seed=42)  # Position nodes for a visually appealing layout
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", edge_color="gray", font_size=10, font_weight="bold", arrowsize=12)
    plt.title("Hypothesis Tree Visualization")
    plt.show()
    
def euclidean_cost(state, measurement, association_cost=0):
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
    distance = np.linalg.norm(state - measurement)  # Euclidean distance
    return distance + association_cost
