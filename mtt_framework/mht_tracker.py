# -*- coding: utf-8 -*-
"""

mht_tracker


Created on Tue Feb 18 16:46:03 2025
@author: Daniel, Bahar
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
        - measurement_id (int): Index of measurement
        - track (StateModel): Track state representation.
        - parents (list of HypothesisNode): List of parent nodes (None for root).
        - event_type (str): "persist", "merge", "split", or "death".
        - cost (float): Cost/log-likelihood for this hypothesis.
        - best (bool): Indicator that node is part of best global hypothesis 
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

    def add_node(self, track, scan, measurement_id, parents=None, event_type="persist", cost=0, best=False,track_id=set()):
        """
        Adds a new node to the hypothesis tree.
        - track (StateModel): Track state representation.
        - scan (int): Index of time step
        - measurement_id (int): Index of measurement
        - parents (list of HypothesisNode): List of parent nodes (None for root).
        - event_type (str): "persist", "merge", "split", or "death".
        - cost (float): Cost/log-likelihood for this hypothesis.
        - best (bool): Indicator that node is part of best global hypothesis 
        """
        hypoth_id = self.next_hypoth_id  # Unique hypothesis ID
        self.next_hypoth_id += 1

        if parents is None:
            # Root node case
            if self.root is not None:
                raise ValueError("Root node already exists")
            new_node = HypothesisNode(-1, -1, -1, np.zeros(3), -1, event_type="root", cost=cost)
            self.next_track_id += 1
            self.root = new_node
        else:
            if event_type == "birth":
                if best == True:
                    track_id = set([self.next_track_id])
                    self.next_track_id += 1
                else:    
                    track_id = set()
            elif event_type == "split":
                track_id = track_id
            else:
                # Merge track IDs from all parent nodes
                track_id = set.union(*[parent.track_id for parent in parents])
            
            # Get total cost by adding cost of parents
            parents_cost = 0
            for node in parents:
                parents_cost += node.cost
            
            # Create new hypothesis node
            new_node = HypothesisNode(hypoth_id, track_id, measurement_id, track, scan, parents, event_type, cost=cost + parents_cost, best=best)

            # Attach new node to each parent
            for parent in parents:
                parent.add_child(new_node)

        # Store node in dictionary
        self.nodes[hypoth_id] = new_node
        return new_node  # Return new node for tracking
    
    def update_leaf_nodes(self):
        """
        Updates leaf nodes with be surviving nodes
        """
        self.leaf_nodes = [node for node in self.nodes.values() if not node.children]
    
    def update_live_leaf_nodes(self):
        """
        Updates leaf nodes with be surviving nodes
        """
        self.live_leaf_nodes = [node for node in self.leaf_nodes if node.event_type != 'death']
    
    def visualize_hypothesis_tree(self):
        """
        Visualizes the hypothesis tree using NetworkX and Matplotlib, showing event types.
        """
        if self.root is None:
            print("The hypothesis tree is empty.")
            return

        G = nx.DiGraph()
        edge_labels = {}
        node_labels = {}
        node_colors = {}
        
        

        def add_edges(node):
            colors = [self.get_track_color(tid) for tid in node.track_id]
            
            if node.best:
                node_colors[node.hypoth_id] = colors[0]
            else:
                node_colors[node.hypoth_id] = "lightblue"
            
            if node.hypoth_id == -1:
                node_labels[node.hypoth_id] = 'root'
            else:
                node_labels[node.hypoth_id] = f'{node.hypoth_id}:{node.track_id}'
            
            for child in node.children:
                G.add_edge(node.hypoth_id, child.hypoth_id)
                edge_labels[(node.hypoth_id, child.hypoth_id)] = child.event_type  # Label edges with events
                add_edges(child)
            
            # Store scan number to arrange tree
            G.add_node(node.hypoth_id,scan=node.scan)
            
        node_colors[self.root.hypoth_id] = "lightblue"  # Root node color
        add_edges(self.root)

        pos = hierarchy_layout(G)

        fig = plt.figure(figsize=(8, 8))
        nx.draw_networkx(G, pos,
                         node_size=4000, 
                         node_color=[node_colors.get(n, "lightblue") for n in G.nodes],
                         labels=node_labels,
                         edge_color="gray", font_size=10, 
                         font_weight="bold", arrowsize=12)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color="red")

        plt.title("Hypothesis Tree (Events: Persist, Overlap, Split, Birth, Death)")
        plt.show()
        fig_num = plt.get_fignums()[-1]
        fig.savefig(f"fig_{fig_num}.png", dpi=300)
    
    def get_track_color(self, track_id):
        """Assigns a unique color to each track_id."""
        if track_id not in self.track_id_colors:
            cmap = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
            self.track_id_colors[track_id] = cmap[len(self.track_id_colors) % len(cmap)]
        return self.track_id_colors[track_id]
        
    def plot_all_tracks(self, data, time_steps, omeRange, vlim=None):
        """
        Plots all scans and overlays all tracks detected over time.
        """
        fig, axes = plt.subplots(len(omeRange), len(time_steps), 
                                 figsize=(2*len(time_steps), 2*len(omeRange)))
        
        if len(omeRange) == 1:
            axes = [axes]
        
        for t_idx, time_step in enumerate(time_steps):
            for i, ome in enumerate(omeRange):
                slice_data = data[time_step, :, :, ome]
                ax = axes[i][t_idx] if len(omeRange) > 1 else axes[t_idx]
                if vlim is not None:
                    vmin, vmax = vlim
                    ax.imshow(slice_data, origin='lower', cmap='viridis', vmin= vmin, vmax= vmax)
                else:
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
        
    def plot_all_tracks_2D(self, data, time_steps, omeRange, num_cols, vlim=None):
        """
        Plots all scans and overlays all tracks detected over time.
        """
        num_rows = len(time_steps)//num_cols
        fig, axes = plt.subplots(num_rows, num_cols, dpi=300,
                                 figsize=(10,0.3*len(time_steps)), constrained_layout=True)
        
        for t_idx, time_step in enumerate(time_steps):
            slice_data = data[time_step, :, :]
            col_idx = np.mod(t_idx,num_cols)
            row_idx = t_idx//num_cols
            if num_rows == 1:
                ax = axes[col_idx]
            else:
                ax = axes[row_idx,col_idx]
            if vlim is not None:
                vmin, vmax = vlim
                ax.imshow(slice_data, origin='lower', cmap='viridis', vmin= vmin, vmax= vmax)
            else:
                ax.imshow(slice_data, origin='lower', cmap='viridis')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        
        for node in self.nodes.values():
            if node.hypoth_id > -1 and node.best:
                scan_idx = node.scan
                bbox = node.track.state['bbox']
                ax_idx = np.where(time_steps == scan_idx)[0]
                
                if len(ax_idx) == 0:
                    continue
                col_idx = np.mod(ax_idx[0],num_cols)
                row_idx = ax_idx[0]//num_cols
                for ome_i in np.arange(bbox[4],bbox[5]+1):
                    if num_rows == 1:
                        ax = axes[col_idx]
                    else:
                        ax = axes[row_idx,col_idx]
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
    """Multiple Hypothesis Testing Tracker for 3D spot tracking."""
 
    def __init__(self, track_model, 
                 intensity_threshold=0,
                 size_threshold=0,
                 gating_threshold=25,
                 death_loglikelihood=-500,
                 birth_loglikelihood=-500,
                 n_scan_pruning=4,
                 birth_death_pruning=True,
                 plot_tree=False):
        """
        Initialize the tracker.
        
        Parameters:
        - tracks (list of track hypothesis trees)
        - gating_threshold (float): Threshold for Mahalanobis distance gating.
        """
        self.dt=1
        self.current_scan = 0
        self.track_model = track_model
        self.intensity_threshold = intensity_threshold
        self.size_threshold = size_threshold
        self.gating_threshold = gating_threshold
        self.death_loglikelihood = death_loglikelihood
        self.birth_loglikelihood = birth_loglikelihood
        self.n_scan_pruning = n_scan_pruning
        self.birth_death_pruning = birth_death_pruning
        self.plot_tree= plot_tree
        pass
    
    def process_measurements(self,measurements,scan):
        # 0. Initialize tracker if we do not have a tree yet and we do have measurements
        self.current_scan = scan
        if not hasattr(self,"tree") or self.tree.root == None:
            if len(measurements) > 0:
                self.initialize_hypothesis_tree(measurements)
                self.prediction()
                if self.plot_tree:
                    self.tree.visualize_hypothesis_tree()
        else:
            
            # 0.9 Filter measurements based on average intensity
            measurements = self.filter_measurements(measurements)
            measurements = self.size_filter_measurements(measurements)
            
            # 1. Gating
            self.gating(measurements)
            
            # 2. Generate, evaluate, prune hypotehses
            self.update_hypothesis_tree()
            
            if self.current_scan == 5:
                print("check here")
            
            if self.plot_tree:
                self.tree.visualize_hypothesis_tree()
                
            self.evaluate_hypotheses()
            self.pruning()
            
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
        self.tree.add_node(track=self.track_model,scan=self.current_scan-1,measurement_id=-1)  # Dummy root
        
        # Add each measurement as a child of the root
        # print(f'Measurement: {measurements}')
        for i,measurement in enumerate(measurements):
            track = self.initialize_track(measurement)
            self.tree.add_node(track=track, scan=self.current_scan, measurement_id=i, event_type="birth", parents=[self.tree.root], cost=0, best=True)
            
        # Update leaf nodes
        self.tree.update_leaf_nodes()
        self.tree.update_live_leaf_nodes()
    
    def prediction(self):
        # print(f'leaf nodes: {[node.hypoth_id for node in self.tree.live_leaf_nodes]}')
        for node in self.tree.live_leaf_nodes:
            node.track.transition(self.dt)

    def filter_measurements(self, measurements):
        """
        Parameters:
        - measurements (list of measurements)
        - intensity_threshold : float
        Returns:
        - filtered_measurements
        """
        return [m for m in measurements if m["avg_intensity"] >= self.intensity_threshold]

    def size_filter_measurements(self, measurements):
        """
        Parameters:
        - measurements (list of measurements)
        - intensity_threshold : float
        Returns:
        - filtered_measurements
        """
        return [m for m in measurements if m["bbox_size"][0]*m["bbox_size"][1]*m["bbox_size"][2] >= self.size_threshold]


    def gating(self, measurements, cov_matrix=None):
        """
        Perform gating by filtering unlikely measurement-track associations.
        
        Parameters:
        - measurements (list of measurements)
        
        Returns:
        - m2ta_matrix (np.array)
        """
        
        self.measurements = measurements
        num_leaf_nodes = len(self.tree.live_leaf_nodes)
        self.m2ta_matrix = np.zeros((len(measurements),num_leaf_nodes))
        self.m2ta_to_hypoth_id = np.zeros(num_leaf_nodes)
        self.active_track_ids = set()
        
        if cov_matrix is None:
            cov_matrix = np.eye(3)  # Assume identity if not provided
        
        for k, node in enumerate(self.tree.live_leaf_nodes):
            if node.best:
                self.active_track_ids = self.active_track_ids | node.track_id
            self.m2ta_to_hypoth_id[k] = node.hypoth_id
            for m, measurement in enumerate(measurements):
                gating_dist = node.track.compute_gating_distance(measurement)
                if gating_dist < self.gating_threshold:
                    self.m2ta_matrix[m,k] = 1       
        pass
        
    def generate_death_hypotheses(self):
        """

        """
        # Add death hypothesis to all  leaf nodes or only to leaf nodes with no association
        for node in self.tree.live_leaf_nodes:
            self.tree.add_node(track=copy.deepcopy(node.track),
                               scan=self.current_scan, 
                               measurement_id=None,
                               parents=[node], 
                               event_type="death", 
                               cost=self.death_loglikelihood + node.cost)

    def get_associated_leaf_nodes(self, measurement_index):
        """
        Finds the leaf nodes in the hypothesis tree that are associated with a given measurement.
    
        Parameters:
        - measurement_index (int): The index of the measurement in the m2ta matrix.
    
        Returns:
        - List[HypothesisNode]: A list of leaf nodes that have an association with the given measurement.
        """
        associated_hypoths = np.where(self.m2ta_matrix[measurement_index] > 0)[0]
        return [self.tree.live_leaf_nodes[i] for i in associated_hypoths]
    
    def generate_birth_hypothesis(self,track, m_idx):
        self.tree.add_node(track=copy.deepcopy(track), scan = self.current_scan, measurement_id=m_idx, 
                           parents=[self.tree.root], event_type="birth", cost=self.birth_loglikelihood)
    
    def generate_persist_hypotheses(self, track, associated_leaf_nodes, measurement, m_idx):
        """
        Generates persist hypotheses for tracks that continue with the same track ID.
    
        Parameters:
        - track (BasicModel): The track model representing the state of the track.
        - associated_leaf_nodes (List[HypothesisNode]): The leaf nodes that are associated with the measurement.
        - measurement (Measurement): The measurement being processed.
        - m_idx (int): The index of the measurement.
        """
        for node in associated_leaf_nodes:
            cost = self.track_model.compute_hypothesis_cost(node.track, measurement, 'persist')
            self.tree.add_node(track=copy.deepcopy(track), 
                               scan=self.current_scan, 
                               measurement_id=m_idx, 
                               parents=[node], 
                               event_type="persist", 
                               cost=cost)
    
    
    def generate_overlap_hypotheses(self, track, associated_leaf_nodes, measurement, m_idx):
        """
        Generates overlap hypotheses where two or more distinct track IDs merge into a new track.
    
        Parameters:
        - track (BasicModel): The track model representing the state of the track.
        - associated_leaf_nodes (List[HypothesisNode]): The leaf nodes that have been associated with the measurement.
        - measurement (Measurement): The measurement being considered for hypothesis generation.
        - m_idx (int): The index of the measurement.
        """

        # Generate all possible subsets of associated nodes with at least two elements
        for subset_size in range(2, len(associated_leaf_nodes) + 1):
            for subset in combinations(associated_leaf_nodes, subset_size):
                # Ensure all track IDs in the subset are distinct (no duplicate track groups)
                all_distinct = all(
                    node1.track_id.isdisjoint(node2.track_id)
                    for node1, node2 in combinations(subset, 2)
                )
    
                # Ensure all measurement IDs are distinct (i.e., no duplicate measurements)
                all_distinct2 = all(
                    node1.measurement_id != node2.measurement_id
                    for node1, node2 in combinations(subset, 2)
                )
    
                if all_distinct and all_distinct2:
                    # Create hypothesis node
                    parents = [node for node in subset]
                    parent_tracks = [node.track for node in subset]
                    cost = self.track_model.compute_hypothesis_cost(parent_tracks, measurement, 'overlap')
    
                    self.tree.add_node(track=copy.deepcopy(track), 
                                       scan=self.current_scan, 
                                       measurement_id=m_idx, 
                                       parents=parents, 
                                       event_type="overlap", 
                                       cost=cost)

    def find_lone_track_ancestor(self, node, tid):
        """
        Recursively search all ancestors to find the latest node where:
        - event_type is 'persist' or 'split'
        - track_id is exactly {tid}
        Returns: the first such node found (DFS), or None
        """
        if node.track_id == {tid} and node.event_type in ('persist', 'split'):
            return node
        for parent in node.parents:
            found = self.find_lone_track_ancestor(parent, tid)
            if found:
                return found
        return None

    def generate_split_hypotheses(self, track, associated_leaf_nodes, measurement, m_idx):
        """
        Identifies all possible split hypotheses given the current tree and measurements.
        - A split hypothesis is considered when a node with multiple track_ids is associated with multiple measurements.
        """
        
        for node in associated_leaf_nodes:
            if node.event_type != 'overlap' or len(node.track_id) <= 1:
                continue  # Only handle actual overlapping tracks
            
            tids = list(node.track_id)
            ancestor_dict = {}
            
            # Step 1: Find valid ancestor for each track_id using recursion
            for tid in tids:
                ancestor_dict[tid] = self.find_lone_track_ancestor(node, tid)

            # Filter out any track_ids with no valid ancestor
            ancestor_dict = {tid: anc for tid, anc in ancestor_dict.items() if anc is not None}
            valid_tids = list(ancestor_dict.keys())

            # Step 2a: Generate individual split hypotheses (one per track)
            for tid in valid_tids:
                old_track = copy.deepcopy(ancestor_dict[tid].track)
                cost = self.track_model.compute_hypothesis_cost(old_track, measurement, 'persist')
                self.tree.add_node(track=old_track,
                                   scan=self.current_scan,
                                   measurement_id=m_idx,
                                   parents=[node],
                                   event_type="split",
                                   cost=cost,
                                   track_id=tid)
    
            # Step 2b: Generate multi-track split hypotheses (combinations of 2 or more)
            for subset_size in range(2, len(valid_tids)):
                for tid_subset in combinations(valid_tids, subset_size):
                    parent_tracks = [copy.deepcopy(ancestor_dict[tid].track) for tid in tid_subset]
                    cost = self.track_model.compute_hypothesis_cost(parent_tracks, measurement, 'overlap')
                    self.tree.add_node(track=copy.deepcopy(track),
                                       scan=self.current_scan,
                                       measurement_id=m_idx,
                                       parents=[node],
                                       event_type="split",
                                       cost=cost,
                                       track_id=tid_subset)
    
    def update_hypothesis_tree(self):
        """
        Updates the hypothesis tree by adding new hypotheses to existing leaf nodes only.
        """
        self.generate_death_hypotheses()

        for m_idx, measurement in enumerate(self.measurements):
            track = self.initialize_track(measurement)
            associated_leaf_nodes = self.get_associated_leaf_nodes(m_idx)
            self.generate_persist_hypotheses(track, associated_leaf_nodes, measurement, m_idx)
            self.generate_overlap_hypotheses(track, associated_leaf_nodes, measurement, m_idx)
            self.generate_birth_hypothesis(track,m_idx)
            # self.generate_split_hypotheses(track, associated_leaf_nodes, measurement, m_idx)

        self.tree.update_leaf_nodes()
        self.tree.update_live_leaf_nodes()
            
    def evaluate_hypotheses(self):
        """
        Evaluates the hypotheses to determine the best global hypothesis.
        - Calls `solve_integer_program` to solve for optimal hypotheses.
        - Marks the selected nodes and propagates the selection to their parents.
        - Resets all other nodes to best=False before marking the best path.
        """
        # Need to only do this going back N scans. Reassigning old death hypotheses
        # should be trivial. Do I need to do this explicitly? Does this integer programming
        # deal well if certain tracks and measurements are clearly independent of one another
        # With deaths and births included, the inequality constraint is now equality
        
        # Get the optimal set of hypotheses
        best_hypotheses, unassigned_measurements = self.solve_integer_program()
                
        # Mark selected nodes and propagate to parents
        def propagate_not_best(node):
            if isinstance(node, HypothesisNode):
                node.best = False
                for parent in node.parents:
                    propagate_not_best(parent)
            
        def propagate_best(node):
            if isinstance(node, HypothesisNode):
                node.best = True
                for parent in node.parents:
                    propagate_best(parent)
        
        # Reset best labels on live tracks ignoring best hypotheses
        for node in self.tree.live_leaf_nodes:
            if node not in best_hypotheses:
                propagate_not_best(node)
        
        # Assign best labels on all best hypotheses and assign track_id to births
        for node in best_hypotheses:
            if node.event_type == 'birth':
                node.track_id = set([self.tree.next_track_id])
                self.tree.next_track_id += 1
            propagate_best(node)
           
    def solve_integer_program(self):
        """
        Sets up the integer programming problem to find the best global hypothesis.
        """
        all_hypotheses = self.tree.leaf_nodes
        num_hypotheses = len(all_hypotheses)
        num_tracks = len(self.active_track_ids)
        num_measurements = self.m2ta_matrix.shape[0]
        
        # Cost vector
        c = np.array([node.cost for node in all_hypotheses])
        
        # Constraint matrix
        A_t = np.zeros((num_tracks, num_hypotheses))
        track_id_to_A = {track_id: index for index, track_id in enumerate(self.active_track_ids)}
        A_m = np.zeros((num_measurements, num_hypotheses))
        for j, node in enumerate(all_hypotheses):
            # Track constraint matrix
            if node.event_type != 'birth':
                for track_id in node.track_id:
                    if track_id in self.active_track_ids:
                        A_t[track_id_to_A[track_id], j] = 1
            # Measurement constraint matrix
            if node.event_type != 'death':
                A_m[node.measurement_id, j] = 1

        # Form equality constraint
        A=np.concatenate((A_t,A_m))
        b = np.ones(num_tracks + num_measurements)
        
        # Solve integer linear program
        result = scipy.optimize.linprog(-c, A_eq=A, b_eq=b,
                                        bounds=(0, 1), method='highs')
        
        if result.success:
            # Assemble selected hypotheses
            selected_hypotheses = [all_hypotheses[i] for i in range(num_hypotheses) if result.x[i] > 0.5]
            return selected_hypotheses, result['ineqlin']['residual']
        else:
            raise ValueError("No valid hypothesis selection found")
    
    def pruning(self):
        """
        Prunes the hypothesis tree by removing non-best nodes that are at least 
        N scans away from the most recent best hypothesis.
        """
    
        def delete_node(node):
            """Deletes a node and updates parent-child relationships."""
            if node.hypoth_id in self.tree.nodes:
                del self.tree.nodes[node.hypoth_id]
            for parent in node.parents:
                try:
                    parent.children.remove(node)  # Use `discard()` to avoid KeyErrors
                except:
                    print('Node not present')
    
        def recursive_delete(node):
            """Recursively deletes child nodes before deleting the parent."""
            for child in list(node.children):  # Make a copy to avoid modification issues
                recursive_delete(child)
            delete_node(node)
            
        # Birth and death pruning of hypotheses that were not selected
        if self.birth_death_pruning:
            to_delete = [node for node in self.tree.leaf_nodes
                         if node.event_type in {'birth', 'death'} and not node.best]
            for node in to_delete:
                delete_node(node)
    
        # N-scan pruning
        nodes_to_delete = []
        for node in list(self.tree.nodes.values()):  # Copy values to prevent iteration issues
            if not node.best and (self.current_scan - node.scan) >= self.n_scan_pruning:
                nodes_to_delete.append(node)
    
        # Perform the deletion safely
        for node in nodes_to_delete:
            recursive_delete(node)

                    

        self.tree.update_leaf_nodes()
        self.tree.update_live_leaf_nodes()
        
        
def hierarchy_layout(G, root=None, level_gap=1.5, min_spacing=2.0):
    """
    Arranges nodes in a top-down hierarchical layout with no overlap.
    
    - Depth determined by node.scan value.
    - Parents are centered over their children.
    - Nodes are spaced dynamically to avoid overlap.

    Parameters:
    - G (networkx.DiGraph): Directed graph (tree structure).
    - root (int or str): The root node.
    - level_gap (float): Vertical spacing between levels.
    - min_spacing (float): Minimum spacing between nodes to prevent overlap.

    Returns:
    - pos (dict): Dictionary mapping nodes to (x, y) positions.
    """
    if root is None:
        # Find root nodes (nodes with no incoming edges)
        roots = [n for n in G.nodes if G.in_degree(n) == 0]
        if len(roots) == 1:
            root = roots[0]
        else:
            raise ValueError("Multiple root nodes found. Specify a root.")

    pos = {}  # Store node positions

    # Step 1: Assign levels based on `node.scan`
    levels = {}
    for node in G.nodes:
        levels[node] = G.nodes[node].get('scan', 0)  # Use node.scan value

    # Step 2: Group nodes by their scan level
    level_nodes = {}
    for node, depth in levels.items():
        if depth not in level_nodes:
            level_nodes[depth] = []
        level_nodes[depth].append(node)

    # Compute y-positions based on depth
    y_positions = {depth: -depth * level_gap for depth in sorted(level_nodes)}

    # Step 3: Compute subtree width dynamically
    def compute_subtree_width(node):
        """Recursively compute the width needed for a subtree."""
        children = list(G.successors(node))
        if not children:
            return min_spacing  # Leaf nodes require minimal space
        return sum(compute_subtree_width(child) for child in children) + (len(children) - 1) * min_spacing

    # Step 4: Assign x-positions dynamically
    def assign_x_positions(node, x_offset=0):
        """Recursively assigns x-positions with dynamic spacing."""
        children = list(G.successors(node))
        if not children:
            pos[node] = (x_offset, y_positions[levels[node]])
            return x_offset

        # Compute total width needed by all children
        subtree_width = sum(compute_subtree_width(child) for child in children)
        start_x = x_offset - subtree_width / 2

        # Place each child centered under the parent
        for child in children:
            child_width = compute_subtree_width(child)
            child_x = start_x + child_width / 2
            pos[child] = (child_x, y_positions[levels[child]])
            start_x += child_width + min_spacing  # Shift for next child

            # Recurse for the child's subtree
            assign_x_positions(child, child_x)

    # Initialize the root at x = 0
    pos[root] = (0, y_positions[levels[root]])
    assign_x_positions(root)

    return pos
        
        
        
def hierarchy_layout1(G, root=None, level_gap=1.5, min_spacing=2.0):
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



