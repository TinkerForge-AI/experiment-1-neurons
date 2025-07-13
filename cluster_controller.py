import numpy as np
from cluster import Cluster

class ClusterController:
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
                    # Initialize specialty closer to target if unspecialized
                    if abs(neuron.specialty - target_spec) > 3.0:
                        neuron.specialty = target_spec + np.random.uniform(-0.5, 0.5)
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
            import config_training
            SPECIALTY_CLAMP_MIN = getattr(config_training, 'SPECIALTY_CLAMP_MIN', 1.0)
            SPECIALTY_CLAMP_MAX = getattr(config_training, 'SPECIALTY_CLAMP_MAX', 10.0)
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

    def route_signal(self, input_signal, goal=None):
        # Decide which cluster(s) to activate based on goal/specialty
        # For now, pick the cluster whose specialty is closest to the input
        best_cluster = min(self.clusters, key=lambda c: abs(c.specialty - input_signal))
        result = best_cluster.aggregate_signal(input_signal)
        return {
            'controller_id': self.controller_id,
            'best_cluster': best_cluster.cluster_id,
            'result': result,
            'cluster_state': best_cluster.get_state()
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

    def equalize_confidence(self):
        """Equalize neuron confidences so total >= CONFIDENCE_TOTAL_MIN, cap at CONFIDENCE_MAX."""
        try:
            import config_training
            CONFIDENCE_TOTAL_MIN = getattr(config_training, 'CONFIDENCE_TOTAL_MIN', 9)
            CONFIDENCE_MAX = getattr(config_training, 'CONFIDENCE_MAX', 1.0)
        except Exception:
            CONFIDENCE_TOTAL_MIN = 9
            CONFIDENCE_MAX = 1.0
        # Gather all neurons
        all_neurons = []
        for cluster in self.clusters:
            all_neurons.extend(cluster.neurons)
        total_conf = sum(n.confidence for n in all_neurons)
        if total_conf < CONFIDENCE_TOTAL_MIN:
            # Distribute confidence evenly, cap at CONFIDENCE_MAX
            per_neuron = CONFIDENCE_TOTAL_MIN / len(all_neurons)
            for n in all_neurons:
                n.confidence = min(CONFIDENCE_MAX, per_neuron)
        else:
            # Cap each neuron at CONFIDENCE_MAX
            for n in all_neurons:
                n.confidence = min(CONFIDENCE_MAX, n.confidence)
        print(f"[Controller] Confidence equalized. Total: {sum(n.confidence for n in all_neurons):.2f}")
