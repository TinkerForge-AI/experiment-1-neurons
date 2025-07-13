import numpy as np
from cluster import Cluster

class ClusterController:
    def equalize_specialty(self):
        """Equalize neuron specialties within clusters, clamp to min/max."""
        try:
            import config_training
            SPECIALTY_CLAMP_MIN = getattr(config_training, 'SPECIALTY_CLAMP_MIN', 1.0)
            SPECIALTY_CLAMP_MAX = getattr(config_training, 'SPECIALTY_CLAMP_MAX', 10.0)
        except Exception:
            SPECIALTY_CLAMP_MIN = 1.0
            SPECIALTY_CLAMP_MAX = 10.0
        for cluster in self.clusters:
            # Compute mean specialty for cluster
            mean_specialty = np.mean([n.specialty for n in cluster.neurons])
            for n in cluster.neurons:
                n.specialty = max(SPECIALTY_CLAMP_MIN, min(SPECIALTY_CLAMP_MAX, mean_specialty))
        print(f"[Controller] Specialty equalized. Means: {[np.mean([n.specialty for n in c.neurons]) for c in self.clusters]}")
    """
    Manages multiple clusters, delegates tasks, and optimizes routing.
    Can be extended to recursively manage lower-level controllers.
    """
    def __init__(self, clusters, controller_id=None, lower_controllers=None):
        self.clusters = clusters  # List of Cluster objects
        self.controller_id = controller_id
        self.lower_controllers = lower_controllers if lower_controllers else []

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
