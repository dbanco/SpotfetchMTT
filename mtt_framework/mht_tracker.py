# -*- coding: utf-8 -*-
"""

mht_system


Created on Tue Feb 18 16:46:03 2025
@author: dpqb1
"""
import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt
from mtt_framework.state_model import BasicModel
from itertools import chain, combinations
import copy

def euclidean_cost(position1, position2, association_cost=0):
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

class HypothesisNode:
    def __init__(self, hypoth_id, track_id, track, parents=None, event_type="persist", cost=0):
        """
        Represents a node in the hypothesis tree.

        Parameters:
        - hypoth_id (int): Unique identifier for the hypothesis node.
        - track_id (set): Set of track IDs.
        - track (BasicModel): Track state representation.
        - parents (list of HypothesisNode): List of parent nodes (None for root).
        - event_type (str): "persist", "merge", "split", or "death".
        - cost (float): Cost/log-likelihood for this hypothesis.
        """
        self.hypoth_id = hypoth_id  # Unique for each hypothesis
        self.track_id = set(track_id) if isinstance(track_id, (list, tuple, set)) else {track_id}
        self.track = track  # StateModel object
        self.parents = parents if parents else []  # Supports multiple parents
        self.children = []
        self.event_type = event_type  # Track event type
        self.cost = cost  # Log-likelihood
        self.best = False

    def add_child(self, child_node):
        """Adds a child node to this hypothesis."""
        self.children.append(child_node)

class HypothesisTree:
    def __init__(self):
        self.root = None
        self.nodes = {}
        self.next_hypoth_id = -1  # Auto-increment hypothesis ID
        self.next_track_id = -1

    def add_node(self, track, parent_ids=None, event_type="persist", cost=0):
        """
        Adds a new node to the hypothesis tree.
        - track_id is always a set.
        - Parent(s) are specified in `parent_ids`.
        - Each node is assigned an event type provided as input.
        """
        hypoth_id = self.next_hypoth_id  # Unique hypothesis ID
        self.next_hypoth_id += 1

        if parent_ids is None:
            # Root node case
            if self.root is not None:
                raise ValueError("Root node already exists")
            new_node = HypothesisNode(-1, -1, np.zeros(3), "root", cost=cost)
            self.next_track_id += 1
            self.root = new_node
        else:
            parent_nodes = [self.nodes[parent_id] for parent_id in parent_ids if parent_id in self.nodes]
            if len(parent_nodes) != len(parent_ids):
                missing_parents = set(parent_ids) - set(self.nodes.keys())
                raise ValueError(f"Parent nodes not found: {missing_parents}")

            if event_type == "birth":
                track_id = {self.next_track_id}
                self.next_track_id += 1
            else:
                # Merge track IDs from all parent nodes
                track_id = set.union(*[parent.track_id for parent in parent_nodes])

            # Create new hypothesis node
            new_node = HypothesisNode(hypoth_id, track_id, track, parent_nodes, event_type, cost=cost)

            # Attach new node to each parent
            for parent in parent_nodes:
                parent.add_child(new_node)

        # Store node in dictionary
        self.nodes[hypoth_id] = new_node
        return new_node  # Return new node for tracking

    def get_node(self, hypoth_id):
        return self.nodes.get(hypoth_id)

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
    
    def __init__(self, measurements, gating_threshold=25.0):
        """
        Initialize the tracker.
        
        Parameters:
        - tracks (list of track hypothesis trees)
        - gating_threshold (float): Threshold for Mahalanobis distance gating.
        """
        self.initial_state = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3)
        }
        self.gating_threshold = gating_threshold
        self.initialize_hypothesis_tree(measurements)
        pass
    
    def initialize_hypothesis_tree(self,measurements):
        """
        Initializes the hypothesis trees with multiple measurements.
    
        Parameters:
        - measurements: List of 3D numpy arrays representing initial measurements.
    
        Returns:
        - A HypothesisTree object.
        """
        self.tree = HypothesisTree()
        
        # Create a dummy root node (track_id=-1, no meaningful state)
        self.tree.add_node(track=np.array([0, 0, 0]))  # Dummy root
        
        # Add each measurement as a child of the root
        for measurement in measurements:
            print(measurement.com)
            state_model = BasicModel(self.initial_state, feature_extractor=None)
            state_model.update_state(measurement)
            
            self.tree.add_node(track=copy.deepcopy(state_model), event_type="birth", parent_ids=[-1])
            
        # Update leaf nodes
        self.leaf_nodes = [node for node in self.tree.nodes.values() if not node.children]

    def gating(self, measurements, cov_matrix=None):
        """
        Perform gating by filtering unlikely measurement-track associations.
        
        Parameters:
        - measurements (list of measurements)
        
        Returns:
        - m2ta_matrix (np.array)
        """
        
        self.leaf_nodes = [node for node in self.tree.nodes.values() if not node.children]
        num_leaf_nodes = len(self.leaf_nodes)
        self.m2ta_matrix = np.zeros((len(measurements),num_leaf_nodes))
        self.m2ta_to_hypoth_id = np.zeros(num_leaf_nodes)
            
        
        if cov_matrix is None:
            cov_matrix = np.eye(3)  # Assume identity if not provided
        
        for k, node in enumerate(self.leaf_nodes):
            self.m2ta_to_hypoth_id[k] = node.hypoth_id
            for m, measurement in enumerate(measurements):
                diff = measurement.com - node.track.state['position']
                print(f'diff = {diff}')
                mahal_dist = np.dot(diff.T, np.linalg.inv(cov_matrix)).dot(diff)
                print(f'distance = {mahal_dist}')
                if mahal_dist < self.gating_threshold:
                    self.m2ta_matrix[m,k] = 1
        
        pass

    def remove_unassociated_nodes(self):
        """
        Removes hypothesis nodes that do not have any association and
        recursively removes parent nodes if they no longer have children.
        """
        # Identify columns with no associations (i.e., all zeros)
        unassociated_cols = np.where(~self.m2ta_matrix.any(axis=0))[0]
        
        # Retrieve the corresponding hypothesis nodes
        to_remove = [self.nodes[self.m2ta_to_hypoth_id[k]] for k in unassociated_cols]

        # Remove nodes
        for node in to_remove:
            parent = node.parent
            del self.tree.nodes[node.hypoth_id]
            if parent:
                parent.children.remove(node)
                # Remove parents that no longer have children
                while parent and not parent.children:
                    grandparent = parent.parent
                    del self.tree.nodes[parent.hypoth_id]
                    if grandparent:
                        grandparent.children.remove(parent)
                    parent = grandparent
    
    def get_associated_leaf_nodes(self, measurement_index):
        """
        Finds leaf nodes associated with a measurement using the m2ta matrix.
        """
        associated_tracks = np.where(self.m2ta_matrix[measurement_index] > 0)[0]
        return [node for node in self.leaf_nodes if node.track_id & set(associated_tracks)]
    

    def generate_persist_hypotheses(self, associated_leaf_nodes, measurement, cost_function):
        """
        Generates persist hypotheses where each track continues with the same track_id.
        """
        return [(node.hypoth_id, cost_function(node.track.state['position'], measurement.com))
                for node in associated_leaf_nodes]
    
    def generate_overlap_hypotheses(self, associated_leaf_nodes, measurement, cost_function):
        """
        Generates overlap hypotheses where two or more distinct track_ids form a new track.
        """
        hypotheses = []
        for subset in chain.from_iterable(combinations(associated_leaf_nodes, r) for r in range(2, len(associated_leaf_nodes) + 1)):
            if all(node1.track_id.isdisjoint(node2.track_id) for node1, node2 in combinations(subset, 2)):
                avg_com = sum(node.track.state['position'] for node in subset) / len(subset)
                cost = cost_function(avg_com, measurement.com)
                parent_ids = [node.hypoth_id for node in subset]
                hypotheses.append((parent_ids, cost))
        return hypotheses
    
    def generate_split_hypotheses(self, associated_leaf_nodes, measurement, cost_function):
        """
        Generates split hypotheses where a previously merged track now associates with multiple measurements.
        """
        hypotheses = []
        for node in associated_leaf_nodes:
            if len(node.track_id) > 1:
                split_variants = self.generate_possible_splits(node.track_id)
                for split_group in split_variants:
                    predicted_coms = self.estimate_split_coms(node, split_group, measurement.com)
                    split_cost = sum(cost_function(predicted_com, measurement.com) for predicted_com in predicted_coms)
                    hypotheses.append(([node.hypoth_id], split_cost))
        return hypotheses
    
    def update_hypothesis_tree(self, new_measurements, cost_function=euclidean_cost):
        """
        Updates the hypothesis tree by adding new hypotheses to existing leaf nodes only.
        """
        self.remove_unassociated_nodes()
        
        for m_idx, measurement in enumerate(new_measurements):
            associated_leaf_nodes = self.get_associated_leaf_nodes(m_idx)
            print(associated_leaf_nodes)
    
            persist_hypotheses = self.generate_persist_hypotheses(associated_leaf_nodes, measurement, cost_function)
            overlap_hypotheses = self.generate_overlap_hypotheses(associated_leaf_nodes, measurement, cost_function)
            split_hypotheses = self.generate_split_hypotheses(associated_leaf_nodes, measurement, cost_function)
            
            state_model = BasicModel(self.initial_state, feature_extractor=None)
            state_model.update_state(measurement)
            
            for parent_id, cost in persist_hypotheses:
                self.tree.add_node(track=copy.deepcopy(state_model), parent_ids=[parent_id], event_type="persist", cost=cost)
    
            for parent_ids, cost in overlap_hypotheses:
                self.tree.add_node(track=copy.deepcopy(state_model), parent_ids=parent_ids, event_type="overlap", cost=cost)
    
            for parent_ids, cost in split_hypotheses:
                self.tree.add_node(track=copy.deepcopy(state_model), parent_ids=parent_ids, event_type="split", cost=cost)
    
            if not associated_leaf_nodes:
                self.tree.add_node(track=copy.deepcopy(state_model), parent_ids=[-1], event_type="birth", cost=0)
    
    def evaluate_hypotheses(self):
        """
        Evaluates the hypotheses to determine the best global hypothesis.
        - Calls `setup_integer_program` to solve for optimal hypotheses.
        - Marks the selected nodes and propagates the selection to their parents.
        - Resets all other nodes to best=False before marking the best path.
        """
        # Reset all nodes to best=False
        for node in self.tree.nodes.values():
            node.best = False
        
        # Get the optimal set of hypotheses
        best_hypotheses = self.setup_integer_program()
        
        # Mark selected nodes and propagate to parents
        for node in best_hypotheses:
            while isinstance(node,HypothesisNode):
                node.best = True
                node = next(iter(node.parents), None)  # Move up the tree
                
    def setup_integer_program(self):
        """
        Sets up the integer programming problem to find the best global hypothesis.
        """
        hypothesis_list = self.leaf_nodes
        num_hypotheses = len(hypothesis_list)
        track_ids = set.union(*[node.track_id for node in hypothesis_list])
        num_tracks = len(track_ids)
        
        # Cost vector
        c = np.array([node.cost for node in hypothesis_list])
        
        # Constraint matrix A and vector b
        A = np.zeros((num_tracks, num_hypotheses))
        for j, node in enumerate(hypothesis_list):
            for track_id in node.track_id:
                A[track_id, j] = 1
        b = np.ones(num_tracks)
        
        # Solve integer linear program
        result = scipy.optimize.linprog(c, A_eq=A, b_eq=b, bounds=(0, 1), method='highs')
        
        if result.success:
            selected_hypotheses = [hypothesis_list[i] for i in range(num_hypotheses) if result.x[i] > 0.5]
            return selected_hypotheses
        else:
            raise ValueError("No valid hypothesis selection found")

    def generate_possible_splits(self, track_id_set):
        """
        Generate all valid ways to split a merged track into separate track components.
        """
        return list(chain.from_iterable(combinations(track_id_set, r) for r in range(1, len(track_id_set))))

    def estimate_split_coms(self, node, split_group, measurement_com):
        """
        Estimate future center of mass (CoM) for a split hypothesis.
        
        - Travels backward in the tree to find independent track CoMs before merging.
        - Uses the first and last overlapping CoM to predict the new track locations.
        """
        previous_coms = []
        
        # Travel backward to find independent track CoMs
        for track in split_group:
            prev_com = self.find_last_independent_com(node, track)
            previous_coms.append(prev_com)
    
        first_overlap_com = self.find_first_overlap_com(node)
        last_overlap_com = node.track.state['position']
    
        # Estimate new CoM positions based on past movement
        movement_vector = last_overlap_com - first_overlap_com
        estimated_split_coms = [prev_com + movement_vector for prev_com in previous_coms]
    
        return estimated_split_coms

    def find_last_independent_com(self, node, track_id):
        """
        Find the last center of mass of an individual track before it was merged.
        """
        while any(track_id in parent.track_id for parent in node.parents):
            node = next(parent for parent in node.parents if track_id in parent.track_id)
        return node.track.state['position']
    
    def find_first_overlap_com(self, node):
        """
        Find the first center of mass when tracks initially merged.
        """
        while all(len(parent.track_id) == len(node.track_id) for parent in node.parents):
            node = next(iter(node.parents))  # Move up to the first overlap event
        return node.track.state['position']

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




def hierarchy_layout(G, root=None, level_gap=1.5, min_spacing=2.0):
    """
    Arranges nodes in a top-down hierarchical layout with no overlap.

    - Parent nodes are centered.
    - Children are evenly spread under parents.
    - No nodes overlap.

    Parameters:
    - G (networkx.DiGraph): The hypothesis tree.
    - root (int): The root node's ID.
    - level_gap (float): Vertical spacing between levels.
    - min_spacing (float): Minimum spacing between nodes to prevent overlap.

    Returns:
    - pos (dict): Dictionary mapping nodes to (x, y) positions.
    """
    if root is None:
        roots = [n for n in G.nodes if G.in_degree(n) == 0]  # Find root nodes
        if len(roots) == 1:
            root = roots[0]
        else:
            raise ValueError("Multiple root nodes found. Specify a root.")
    
    pos = {}  # Store node positions
    levels = {}  # Track depth of each node

    # Assign levels using DFS (Depth-First Search)
    def assign_levels(node, depth=0):
        if node not in levels:
            levels[node] = depth
        for child in G.successors(node):
            assign_levels(child, depth + 1)

    assign_levels(root)

    # Group nodes by depth level
    level_nodes = {}
    for node, depth in levels.items():
        if depth not in level_nodes:
            level_nodes[depth] = []
        level_nodes[depth].append(node)

    # Compute y-positions based on depth
    y_positions = {depth: -depth * level_gap for depth in level_nodes}

    # Assign x-positions dynamically to prevent overlap
    def assign_x_positions(node, x_offset=0):
        """Recursively assigns x-positions to avoid overlap."""
        children = list(G.successors(node))

        if not children:
            return x_offset  # No change for leaf nodes

        num_children = len(children)
        total_width = (num_children - 1) * min_spacing  # Spread children apart

        # Center children under the parent
        start_x = pos[node][0] - (total_width / 2)

        for i, child in enumerate(children):
            pos[child] = (start_x + i * min_spacing, y_positions[levels[child]])
            assign_x_positions(child, start_x + i * min_spacing)

    # Start layout by centering root at x = 0
    pos[root] = (0, y_positions[0])
    assign_x_positions(root)

    return pos

def visualize_hypothesis_tree(root):
    """
    Visualizes the hypothesis tree using NetworkX and Matplotlib, showing event types.
    """
    if root is None:
        print("The hypothesis tree is empty.")
        return

    G = nx.DiGraph()
    labels = {}
    node_colors = {}

    def add_edges(node):
        for child in node.children:
            G.add_edge(node.hypoth_id, child.hypoth_id)
            labels[(node.hypoth_id, child.hypoth_id)] = child.event_type  # Label edges with events
            node_colors[child.hypoth_id] = "red" if child.best else "lightblue"
            add_edges(child)

    node_colors[root.hypoth_id] = "lightblue"  # Root node color
    add_edges(root)

    try:
        pos = hierarchy_layout(G)
    except Exception as e:
        print(f"Hierarchy Layout Error: {e}")
        pos = nx.spring_layout(G)

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color=[node_colors.get(n, "lightblue") for n in G.nodes],
            edge_color="gray", font_size=10, font_weight="bold", arrowsize=12)

    # Draw event type labels on edges
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10, font_color="red")

    plt.title("Hypothesis Tree (Events: Persist, Overlap, Split, Birth, Death)")
    plt.show()


