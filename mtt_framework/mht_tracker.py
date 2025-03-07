# -*- coding: utf-8 -*-
"""

mht_tracker


Created on Tue Feb 18 16:46:03 2025
@author: dpqb1
"""
import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain, combinations
import copy
import matplotlib.colors as mcolors

class HypothesisNode:
    def __init__(self, hypoth_id, track_id, measurement_id, track, scan, parents=None, event_type="persist", cost=0, best=False):
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
        self.measurement_id = measurement_id
        self.track = track  # StateModel object
        self.parents = parents if parents else []  # Supports multiple parents
        self.children = []
        self.scan = scan
        self.event_type = event_type  # Track event type
        self.cost = cost  # Log-likelihood
        self.best = best

    def add_child(self, child_node):
        """Adds a child node to this hypothesis."""
        self.children.append(child_node)    

class HypothesisTree:
    def __init__(self):
        self.root = None
        self.nodes = {}
        self.next_hypoth_id = -1  # Auto-increment hypothesis ID
        self.next_track_id = -1
        self.fig, self.axes = None, None
        self.track_id_colors = {}

    def add_node(self, track, scan, measurement_id, parent_ids=None, event_type="persist", cost=0, best=False):
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
            new_node = HypothesisNode(-1, -1, -1, np.zeros(3), "root", cost=cost)
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
            
            # Get total cost by adding cost of parents
            parents_cost = 0
            for node in parent_nodes:
                parents_cost += node.cost
            
            # Create new hypothesis node
            new_node = HypothesisNode(hypoth_id, track_id, measurement_id, track, scan, parent_nodes, event_type, cost=cost + parents_cost, best=best)

            # Attach new node to each parent
            for parent in parent_nodes:
                parent.add_child(new_node)

        # Store node in dictionary
        self.nodes[hypoth_id] = new_node
        return new_node  # Return new node for tracking
    
    def update_leaf_nodes(self):
        """
        Updates leaf nodes with be surviving nodes
        """
        self.leaf_nodes = [node for node in self.nodes.values() if not node.children and node.event_type != 'death']
        
    def pruning(self, N):
        """
        Prunes the hypothesis tree by removing non-best nodes that are at least N scans away from the most recent best hypothesis.
        """
        # Step 1: Identify all "best" nodes at most N scans away from the latest best leaves
        best_leaves = [node for node in self.leaf_nodes if node.best]
        best_nodes = set(best_leaves)
        
        # Traverse upward to find all best nodes within N scans, ensuring we don't go beyond the root
        for _ in range(N):
            new_best_nodes = set()
            for node in best_nodes:
                for parent in node.parents:
                    if parent is self.root:
                        break
                    elif parent.best:
                        new_best_nodes.add(parent)
                if parent is self.root:
                    break
            if parent is self.root:
                break
            best_nodes = new_best_nodes
        
        # Step 2: Identify non-best children of these best nodes and mark them for removal
        to_remove = set()
        for node in best_nodes:
            for child in node.children:
                if not child.best:
                    to_remove.add(child)
        
        # Step 3: Recursively remove non-best nodes and their descendants
        def recursive_delete(node):
            for child in list(node.children):  # Use list to avoid modifying set during iteration
                recursive_delete(child)
            if node.hypoth_id in self.nodes:
                del self.nodes[node.hypoth_id]
            for parent in node.parents:
                parent.children.remove(node)  # Ensure parent references are updated
        
        for node in to_remove:
            recursive_delete(node)
        
        # Cleanup: Remove deleted nodes from their parents' children lists
        # for node in best_nodes:
        #     node.children = [child for child in node.children if child in self.nodes]

    def visualize_hypothesis_tree(self):
        """
        Visualizes the hypothesis tree using NetworkX and Matplotlib, showing event types.
        """
        if self.root is None:
            print("The hypothesis tree is empty.")
            return

        G = nx.DiGraph()
        labels = {}
        node_colors = {}

        def add_edges(node):
            if node.event_type == "death":
                node_colors[node.hypoth_id] = "purple"
            elif node.best:
                node_colors[node.hypoth_id] = "red"
            else:
                node_colors[node.hypoth_id] = "lightblue"
            
            for child in node.children:
                G.add_edge(node.hypoth_id, child.hypoth_id)
                labels[(node.hypoth_id, child.hypoth_id)] = child.event_type  # Label edges with events
                add_edges(child)

        node_colors[self.root.hypoth_id] = "lightblue"  # Root node color
        add_edges(self.root)

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
    
    def get_track_color(self, track_id):
        """Assigns a unique color to each track_id."""
        if track_id not in self.track_id_colors:
            cmap = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
            self.track_id_colors[track_id] = cmap[len(self.track_id_colors) % len(cmap)]
        return self.track_id_colors[track_id]
        
    def plot_all_tracks(self, data, time_steps, omeRange):
        """
        Plots all scans and overlays all tracks detected over time.
        """
        fig, axes = plt.subplots(len(omeRange), len(time_steps), figsize=(40, 20))
        if len(omeRange) == 1:
            axes = [axes]
        
        for t_idx, time_step in enumerate(time_steps):
            for i, ome in enumerate(omeRange):
                slice_data = data[time_step, :, :, ome]
                ax = axes[i][t_idx] if len(omeRange) > 1 else axes[t_idx]
                ax.imshow(slice_data, origin='lower', cmap='viridis')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        
        for node in self.nodes.values():
            if node.hypoth_id > -1 and node.best:
                scan_idx = node.scan
                bbox = node.track.state['bbox']
                for ome_i in np.arange(bbox[4],bbox[5]+1):
                    ax = axes[ome_i][scan_idx]
                    boxpos = node.track.state['bbox_center']
                    boxsiz = node.track.state['bbox_size']
                    colors = [self.get_track_color(tid) for tid in node.track_id]
                    for j, color in enumerate(colors):
                        spacing = 1.2*j  # Prevent overlap
                        rect = plt.Rectangle((boxpos[1] - boxsiz[1]/2 - spacing, 
                                              boxpos[0] - boxsiz[0]/2 - spacing), 
                                              boxsiz[1] + 2*spacing, 
                                              boxsiz[0] + 2*spacing, 
                                              edgecolor=color, facecolor='none', linewidth=1)
                        ax.add_patch(rect)
                        
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    
class MHTTracker:
    """Multiple Hypothesis Tracker for 3D spot tracking."""
 
    def __init__(self, track_model, gating_threshold=25.0,n_scan_pruning=4, plot_tree=False):
        """
        Initialize the tracker.
        
        Parameters:
        - tracks (list of track hypothesis trees)
        - gating_threshold (float): Threshold for Mahalanobis distance gating.
        """
        self.dt=1
        self.current_scan = 0
        self.gating_threshold = gating_threshold
        self.n_scan_pruning = n_scan_pruning
        self.track_model = track_model
        self.plot_tree= plot_tree
        pass
    
    def process_measurements(self,measurements,scan):
        # 0. Initialize tracker if it is scan 0
        self.current_scan = scan
        if self.current_scan == 0:
            self.initialize_hypothesis_tree(measurements)
            self.prediction()
        else:
            # 1. Gating
            self.gating(measurements)
            
            # 2. Generate, evaluate, prune hypotehses
            self.update_hypothesis_tree()
            self.evaluate_hypotheses()
            self.tree.pruning(self.n_scan_pruning)
            
            # 3. Prediction
            self.prediction()
            
            if self.plot_tree:
                self.tree.visualize_hypothesis_tree()
        
        
    def initialize_track(self, measurement):
        track = copy.deepcopy(self.track_model)
        track.update_state(measurement,self.dt)
        return track
    
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
        self.tree.add_node(track=self.track_model,scan=-1,measurement_id=-1)  # Dummy root
        
        # Add each measurement as a child of the root
        for i,measurement in enumerate(measurements):
            track = self.initialize_track(measurement)
            self.tree.add_node(track=track, scan=0, measurement_id=i, event_type="birth", parent_ids=[-1])
            
        # Update leaf nodes
        self.tree.update_leaf_nodes()
    
    def prediction(self):
        for node in self.tree.leaf_nodes:
            node.track.transition(self.dt)

    def gating(self, measurements, cov_matrix=None):
        """
        Perform gating by filtering unlikely measurement-track associations.
        
        Parameters:
        - measurements (list of measurements)
        
        Returns:
        - m2ta_matrix (np.array)
        """
        self.measurements = measurements
        self.tree.update_leaf_nodes()
        num_leaf_nodes = len(self.tree.leaf_nodes)
        self.m2ta_matrix = np.zeros((len(measurements),num_leaf_nodes))
        self.m2ta_to_hypoth_id = np.zeros(num_leaf_nodes)
            
        
        if cov_matrix is None:
            cov_matrix = np.eye(3)  # Assume identity if not provided
        
        for k, node in enumerate(self.tree.leaf_nodes):
            self.m2ta_to_hypoth_id[k] = node.hypoth_id
            for m, measurement in enumerate(measurements):
                gating_dist = node.track.compute_gating_distance(measurement)
                if gating_dist < self.gating_threshold:
                    self.m2ta_matrix[m,k] = 1
        
        pass


    def remove_unassociated_nodes(self):
        """
        Removes hypothesis nodes that do not have any association and
        assigns a "death" hypothesis to parent nodes that lose all children.
        Additionally, ensures that unassociated nodes labeled as 'best' 
        receive a death hypothesis instead of removal.
        """
        # Identify columns with no associations (i.e., all zeros)
        unassociated_cols = np.where(~self.m2ta_matrix.any(axis=0))[0]
        
        # Retrieve the corresponding hypothesis nodes
        unassociated_nodes = [self.tree.nodes[self.m2ta_to_hypoth_id[k]] for k in unassociated_cols]
    
        # Recursive removal function
        def recursive_remove(node):
            parents = node.parents.copy()  # Copy to avoid modification during iteration
            del self.tree.nodes[node.hypoth_id]
    
            for parent in parents:
                parent.children.remove(node)
                if not parent.children and parent is not self.tree.root:
                    recursive_remove(parent) 
        
        # Remove unassociated nodes that are not labeled as best
        for node in unassociated_nodes:
            if node.best:  
                # If the node is labeled best, assign it a death hypothesis
                self.tree.add_node(track=copy.deepcopy(node.track),
                                   scan=self.current_scan, measurement_id=None,
                                   parent_ids=[node.hypoth_id], event_type="death", 
                                   cost=0, best=True)  # Keep the 'best' label
            else:
                recursive_remove(node)
      
    def get_associated_leaf_nodes(self, measurement_index):
        """
        Finds leaf nodes associated with a measurement using the m2ta matrix.
        """
        associated_hypoths = np.where(self.m2ta_matrix[measurement_index] > 0)[0]
        return [self.tree.leaf_nodes[i] for i in associated_hypoths]

    def generate_persist_hypotheses(self, associated_leaf_nodes, measurement):
        """
        Generates persist hypotheses where each track continues with the same track_id.
        """
        return [(node.hypoth_id, self.track_model.compute_hypothesis_cost(node.track,measurement,'persist')) for node in associated_leaf_nodes]
    
    def generate_overlap_hypotheses(self, associated_leaf_nodes, measurement):
        """
        Generates overlap hypotheses where two or more distinct track_ids form a new track.
    
        Parameters:
        - associated_leaf_nodes: List of leaf nodes that have been associated with the measurement.
        - measurement: The measurement being considered for hypothesis generation.
        - cost_function: Function to compute the cost of associating a measurement with a hypothesis.
    
        Returns:
        - A list of tuples, each containing a list of parent hypothesis IDs and the associated cost.
        """
        hypotheses = []
    
        # Generate all possible subsets of associated nodes with at least two elements
        for subset_size in range(2, len(associated_leaf_nodes) + 1):
            for subset in combinations(associated_leaf_nodes, subset_size):
                # Ensure all track_ids in the subset are distinct (i.e., no overlap between track groups)
                all_distinct = all(
                    node1.track_id.isdisjoint(node2.track_id)
                    for node1, node2 in combinations(subset, 2)
                )
    
                if all_distinct:
                    # Compute cost
                    tracks = [node.track for node in subset]
                    cost = self.track_model.compute_hypothesis_cost(tracks,measurement,'overlap')
    
                    # Store the hypothesis as a tuple (parent IDs, cost)
                    parent_ids = [node.hypoth_id for node in subset]
                    hypotheses.append((parent_ids, cost))
    
        return hypotheses
    
    # def generate_split_hypotheses(self, measurements, cost_function):
    #     """
    #     Identifies all possible split hypotheses given the current tree and measurements.
    #     - A split hypothesis is considered when a node with multiple track_ids is associated with multiple measurements.
    #     """
    #     split_hypotheses = []
    #     for node in self.tree.nodes.values():
    #         if len(node.track_id) > 1:
    #             associated_measurements = [m_idx for m_idx in range(len(measurements)) if any(self.m2ta_matrix[m_idx, tid] for tid in node.track_id)]
    #             if len(associated_measurements) > 1:
    #                 split_variants = self.generate_possible_splits(node.track_id)
    #                 for split_group in split_variants:
    #                     predicted_coms = self.estimate_split_coms(node, split_group, measurements)
    #                     split_cost = sum(cost_function(predicted_com, measurements[m_idx].com) for predicted_com, m_idx in zip(predicted_coms, associated_measurements))
    #                     split_hypotheses.append((split_group, [node.hypoth_id], split_cost))
    #     return split_hypotheses
    
    def update_hypothesis_tree(self):
        """
        Updates the hypothesis tree by adding new hypotheses to existing leaf nodes only.
        """
        self.remove_unassociated_nodes()
        # split_hypotheses = self.generate_split_hypotheses(self.measurements, cost_function)
        
        for m_idx, measurement in enumerate(self.measurements):
            associated_leaf_nodes = self.get_associated_leaf_nodes(m_idx)
    
            persist_hypotheses = self.generate_persist_hypotheses(associated_leaf_nodes, measurement)
            overlap_hypotheses = self.generate_overlap_hypotheses(associated_leaf_nodes, measurement)


            # Initialize state from measurement
            track = self.initialize_track(measurement)
            
            # Create persist hypotheses
            for parent_id, cost in persist_hypotheses:
                self.tree.add_node(track=copy.deepcopy(track), scan = self.current_scan, measurement_id=m_idx, parent_ids=[parent_id], event_type="persist", cost=cost)
            
            # Create overlap hypotheses
            for parent_ids, cost in overlap_hypotheses:
                self.tree.add_node(track=copy.deepcopy(track), scan = self.current_scan, measurement_id=m_idx, parent_ids=parent_ids, event_type="overlap", cost=cost)
    
            # for parent_ids, cost in split_hypotheses:
            #     self.tree.add_node(track=copy.deepcopy(state_model), scan = self.current_scan, measurement_id=m_idx, parent_ids=parent_ids, event_type="split", cost=cost)
    
            # THE LOGIC RELATING TO BIRTHS AND DEATHS NEEDS TO BE FIXED               
    
        self.tree.update_leaf_nodes()
            
    def evaluate_hypotheses(self):
        """
        Evaluates the hypotheses to determine the best global hypothesis.
        - Calls `setup_integer_program` to solve for optimal hypotheses.
        - Marks the selected nodes and propagates the selection to their parents.
        - Resets all other nodes to best=False before marking the best path.
        """
        # Reset all nodes to best=False unless they were dead best tracks
        best_dead_hypotheses = []
        for node in self.tree.nodes.values():
            if node.best and node.event_type == 'death':
                best_dead_hypotheses.append(node)
            else:
                node.best = False
        
        # Get the optimal set of hypotheses
        best_hypotheses, unassigned_measurements = self.setup_integer_program()
        
        # Get max cost over best hypotheses to assign to newly birthed spots
        
        # Create births for unassigned measurements and make them part of best
        for m_idx, measurement in enumerate(self.measurements):
            if unassigned_measurements[m_idx]:
                track = self.initialize_track(measurement)
                self.tree.add_node(track=track, scan = self.current_scan, measurement_id=m_idx, parent_ids=[-1], event_type="birth", cost=0)
                self.tree.nodes[self.tree.next_hypoth_id-1].best = True
        
                # Make sure no other hypotheses exists involving measurement of birthed node
                for node in self.tree.leaf_nodes:
                    if node.measurement_id == m_idx:
                        if node.hypoth_id in self.tree.nodes:
                            del self.tree.nodes[node.hypoth_id]
                        for parent in node.parents:
                            parent.children.remove(node)  # Ensure parent references are updated
                
                
        # Mark selected nodes and propagate to parents
        def propagate_best(node):
            if isinstance(node, HypothesisNode):
                node.best = True
                for parent in node.parents:
                    propagate_best(parent)
        
        # Call recursively on all best hypotheses
        for node in best_hypotheses:
            propagate_best(node)
        for node in best_dead_hypotheses:
            propagate_best(node)
            
        self.tree.update_leaf_nodes()
           
    def setup_integer_program(self):
        """
        Sets up the integer programming problem to find the best global hypothesis.
        """
        hypothesis_list = self.tree.leaf_nodes
        num_hypotheses = len(hypothesis_list)
        track_ids = set.union(*[node.track_id for node in hypothesis_list])
        num_tracks = len(track_ids)
        num_measurements = self.m2ta_matrix.shape[0]
        
        # Cost vector
        c = np.array([node.cost for node in hypothesis_list])
        
        # Track constraint matrix
        track_id_to_A = {track_id: index for index, track_id in enumerate(track_ids)}
        A_t = np.zeros((num_tracks, num_hypotheses))
        for j, node in enumerate(hypothesis_list):
            for track_id in node.track_id:
                A_t[track_id_to_A[track_id], j] = 1
        b_t = np.ones(num_tracks)
        
        # Measurement constraint matrix
        A_m = np.zeros((num_measurements, num_hypotheses))
        for j, node in enumerate(hypothesis_list):
                A_m[node.measurement_id, j] = 1
        b_m = np.ones(num_measurements)
        
        # Solve integer linear program
        result = scipy.optimize.linprog(c, A_eq=A_t, b_eq=b_t,
                                           A_ub=A_m, b_ub=b_m,
                                        bounds=(0, 1), method='highs')
        
        if result.success:
            selected_hypotheses = [hypothesis_list[i] for i in range(num_hypotheses) if result.x[i] > 0.5]
            return selected_hypotheses, result['ineqlin']['residual']
        else:
            raise ValueError("No valid hypothesis selection found")

    # def generate_possible_splits(self, track_id_set):
    #     """
    #     Generate all valid ways to split a merged track into separate track components.
    #     """
    #     return list(chain.from_iterable(combinations(track_id_set, r) for r in range(1, len(track_id_set))))

    # def estimate_split_coms(self, node, split_group, measurement_com):
    #     """
    #     Estimate future center of mass (CoM) for a split hypothesis.
        
    #     - Travels backward in the tree to find independent track CoMs before merging.
    #     - Uses the first and last overlapping CoM to predict the new track locations.
    #     """
    #     previous_coms = []
        
    #     # Travel backward to find independent track CoMs
    #     for track in split_group:
    #         prev_com = self.find_last_independent_com(node, track)
    #         previous_coms.append(prev_com)
    
    #     first_overlap_com = self.find_first_overlap_com(node)
    #     last_overlap_com = node.track.state['com']
    
    #     # Estimate new CoM positions based on past movement
    #     movement_vector = last_overlap_com - first_overlap_com
    #     estimated_split_coms = [prev_com + movement_vector for prev_com in previous_coms]
    
    #     return estimated_split_coms

    # def find_last_independent_com(self, node, track_id):
    #     """
    #     Find the last center of mass of an individual track before it was merged.
    #     """
    #     while any(track_id in parent.track_id for parent in node.parents):
    #         node = next(parent for parent in node.parents if track_id in parent.track_id)
    #     return node.track.state['com']
    
    # def find_first_overlap_com(self, node):
    #     """
    #     Find the first center of mass when tracks initially merged.
    #     """
    #     while all(len(parent.track_id) == len(node.track_id) for parent in node.parents):
    #         node = next(iter(node.parents))  # Move up to the first overlap event
    #     return node.track.state['com']

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




