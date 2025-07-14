import numpy as np
from cluster import Cluster

class ClusterController:
    def rebalance_clusters(self, min_size=3, max_size=8, rounds_between=5):
        """
        Periodically reassign neurons to clusters based on specialty and accuracy.
        Ensures clusters remain balanced in size.
        Call this every N rounds.
        """
        # Build a mapping of color to cluster
        color_to_cluster = {color: cid for cid, color in self.cluster_assignments.items()}
        # Track cluster sizes
        cluster_sizes = {c.cluster_id: len(c.neurons) for c in self.clusters}
        # For each neuron, check if it matches a different cluster's target specialty
        for cluster in self.clusters:
            for neuron in list(cluster.neurons):
                # Find best matching cluster for this neuron's specialty
                best_color = min(self.target_specialties, key=lambda c: abs(neuron.specialty - self.target_specialties[c]))
                best_cluster_id = color_to_cluster[best_color]
                # If neuron is not in best cluster and best cluster is under max_size
                if cluster.cluster_id != best_cluster_id and cluster_sizes[best_cluster_id] < max_size:
                    # Reassign neuron
                    cluster.neurons.remove(neuron)
                    for c in self.clusters:
                        if c.cluster_id == best_cluster_id:
                            c.neurons.append(neuron)
                            neuron.cluster = c
                            break
                    cluster_sizes[best_cluster_id] += 1
                    cluster_sizes[cluster.cluster_id] -= 1
                    print(f"[Controller] Reassigned neuron {getattr(neuron, 'position', '?')} to cluster {best_cluster_id} (specialty={neuron.specialty:.2f})")
        # Optionally, rebalance if any cluster is below min_size
        for cluster in self.clusters:
            while len(cluster.neurons) < min_size:
                # Find donor cluster with > min_size
                donor = max(self.clusters, key=lambda c: len(c.neurons))
                if donor == cluster or len(donor.neurons) <= min_size:
                    break
                # Move a random neuron from donor to cluster
                neuron = donor.neurons.pop()
                cluster.neurons.append(neuron)
                neuron.cluster = cluster
                print(f"[Controller] Rebalanced neuron {getattr(neuron, 'position', '?')} from {donor.cluster_id} to {cluster.cluster_id}")
    """
    Manages multiple clusters, delegates tasks, and optimizes routing.
    Implements complementary specialization to ensure clusters develop distinct roles.
    """
    def __init__(self, clusters, controller_id=None, lower_controllers=None):
        self.clusters = clusters  # List of Cluster objects
        self.controller_id = controller_id
        self.lower_controllers = lower_controllers if lower_controllers else []
        
        # Set controller reference in clusters
        for cluster in self.clusters:
            cluster.controller = self
        
        # Specialization targets for complementary learning
        self.target_specialties = {
            "red": 2.5,
            "green": 6.5, 
            "blue": 10.0
        }
        self.cluster_assignments = {}  # Maps cluster_id to target color
        
        # Initialize cluster assignments
        self._initialize_cluster_specialization()
    
    def _initialize_cluster_specialization(self):
        """Assign each cluster a target specialization."""
        colors = list(self.target_specialties.keys())
        for i, cluster in enumerate(self.clusters):
            if i < len(colors):
                target_color = colors[i]
                self.cluster_assignments[cluster.cluster_id] = target_color
                # Set target specialty for neurons in cluster
                target_spec = self.target_specialties[target_color]
                for neuron in cluster.neurons:
                    neuron.target_specialty = target_spec
        print(f"[Controller] Initialized cluster specializations: {self.cluster_assignments}")
    
    def on_neuron_specialty_change(self, neuron, cluster):
        """Called when a neuron's specialty changes - guide complementary specialization."""
        # Check if cluster is drifting from its assigned specialty
        target_color = self.cluster_assignments.get(cluster.cluster_id)
        if target_color:
            target_spec = self.target_specialties[target_color]
            cluster_mean = np.mean([n.specialty for n in cluster.neurons])
            
            # If cluster is drifting too far, gently guide it back
            if abs(cluster_mean - target_spec) > 2.0:
                guidance_strength = 0.1
                for n in cluster.neurons:
                    specialty_error = target_spec - n.specialty
                    n.specialty += guidance_strength * specialty_error
                print(f"[Controller] Guided cluster {cluster.cluster_id} back toward {target_color} specialty (target: {target_spec:.1f})")
    
    def ensure_complementary_specialization(self):
        """Ensure clusters maintain distinct, complementary specializations."""
        print("[Controller] Ensuring complementary specialization...")
        
        for cluster in self.clusters:
            target_color = self.cluster_assignments.get(cluster.cluster_id)
            if target_color:
                target_spec = self.target_specialties[target_color]
                cluster_neurons = cluster.neurons
                
                # Calculate how far cluster is from its target
                current_mean = np.mean([n.specialty for n in cluster_neurons])
                distance_from_target = abs(current_mean - target_spec)
                
                # If cluster has drifted significantly, apply corrective guidance
                if distance_from_target > 1.5:
                    correction_strength = min(0.2, distance_from_target * 0.1)
                    for neuron in cluster_neurons:
                        # Guide toward target with some randomness
                        error = target_spec - neuron.specialty
                        neuron.specialty += correction_strength * error + np.random.uniform(-0.1, 0.1)
                        neuron.specialty = max(1.0, min(10.0, neuron.specialty))
                    
                    print(f"[Controller] Applied corrective guidance to cluster {cluster.cluster_id} ({target_color}): {current_mean:.2f} -> {target_spec:.2f}")
    
    def equalize_specialty(self):
        """Maintain cluster specialization while allowing some variation within clusters."""
        try:
            import config
            SPECIALTY_CLAMP_MIN = getattr(config, 'SPECIALTY_CLAMP_MIN', 1.0)
            SPECIALTY_CLAMP_MAX = getattr(config, 'SPECIALTY_CLAMP_MAX', 10.0)
        except Exception:
            SPECIALTY_CLAMP_MIN = 1.0
            SPECIALTY_CLAMP_MAX = 10.0
        
        for cluster in self.clusters:
            target_color = self.cluster_assignments.get(cluster.cluster_id)
            if target_color:
                target_spec = self.target_specialties[target_color]
                # Allow some variation around target, but keep cluster focused
                for n in cluster.neurons:
                    # Gentle pull toward cluster target with variation
                    variation = np.random.uniform(-0.8, 0.8)
                    guided_specialty = target_spec + variation
                    # Blend current specialty with guided specialty
                    n.specialty = 0.8 * n.specialty + 0.2 * guided_specialty
                    n.specialty = max(SPECIALTY_CLAMP_MIN, min(SPECIALTY_CLAMP_MAX, n.specialty))
        
        cluster_means = [np.mean([n.specialty for n in c.neurons]) for c in self.clusters]
        print(f"[Controller] Specialty maintained. Cluster means: {[f'{mean:.2f}' for mean in cluster_means]}")
        print(f"[Controller] Cluster assignments: {self.cluster_assignments}")

    def route_signal(self, input_signal, goal=None, true_label=None):
        """
        Route input signal to all clusters, implement WTA/softmax competition, and provide feedback.
        """
        cluster_activations = []
        for cluster in self.clusters:
            winners = cluster.aggregate_signal(input_signal)
            # Cluster activation: sum of winner neuron activations
            activation = sum([n.active for n in cluster.neurons if n.is_winner])
            cluster_activations.append((cluster, activation))
        # Winner-take-all: select cluster(s) with highest activation
        max_activation = max([act for _, act in cluster_activations])
        winning_clusters = [c for c, act in cluster_activations if act == max_activation]
        # Log cluster competition
        print(f"[Controller] Winning cluster(s): {[c.cluster_id for c in winning_clusters]} (activation={max_activation})")
        # Feedback: if true_label is provided, check if winner(s) are correct
        feedback = {}
        if true_label is not None:
            for cluster in winning_clusters:
                # Assume winner neuron(s) output their preferred color
                winner_outputs = [n.get_preferred_color() for n in cluster.neurons if n.is_winner]
                correct = true_label in winner_outputs
                feedback[cluster.cluster_id] = {'winner_outputs': winner_outputs, 'correct': correct}
                # If incorrect, optionally weaken winner neuron weights
                if not correct:
                    for n in cluster.neurons:
                        if n.is_winner:
                            for neighbor in n.neighbors:
                                n.weight[neighbor.position] *= 0.95  # Weaken synapse
        return {
            'controller_id': self.controller_id,
            'winning_clusters': [c.cluster_id for c in winning_clusters],
            'activations': {c.cluster_id: act for c, act in cluster_activations},
            'feedback': feedback,
            'cluster_states': [c.get_state() for c in self.clusters]
        }

    def adapt_structure(self):
        # Placeholder for future meta-learning logic
        pass

    def add_cluster(self, cluster):
        self.clusters.append(cluster)

    def remove_cluster(self, cluster):
        self.clusters.remove(cluster)

    def get_state(self):
        return {
            'controller_id': self.controller_id,
            'num_clusters': len(self.clusters),
            'clusters': [c.get_state() for c in self.clusters]
        }

    def __repr__(self):
        return f"<ClusterController {self.controller_id}: clusters={len(self.clusters)} lower_controllers={len(self.lower_controllers)}>"

    # Remove equalize_confidence (no longer needed)
