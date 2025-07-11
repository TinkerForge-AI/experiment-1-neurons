import numpy as np
from sklearn.cluster import KMeans

class NeuronEnsembleMonitor:
    """
    Monitors and analyzes neural ensemble behavior.
    
    Tracks neuron activations, clusters specialties, and maps to human-readable labels.
    """
    
    def __init__(self, neurons, iteration=0):
        """Initialize monitor with list of neurons and optional iteration."""
        self.neurons = neurons  # List of Neuron objects
        self.iterative_mapping = {}  # Maps specialty to human label
        self.logs = []
        self.iteration = iteration  # Track experiment iteration

    def get_specialties(self):
        """Get array of all neuron specialties."""
        return np.array([neuron.specialty for neuron in self.neurons])

    def get_active_neurons(self):
        """Get list of currently active neurons."""
        return [neuron for neuron in self.neurons if neuron.active]

    def find_clusters(self, n_clusters=3):
        """
        Cluster neurons by specialty using K-means.
        
        Args:
            n_clusters: Number of clusters to find
            
        Returns:
            Dictionary mapping cluster labels to neuron lists
        """
        if len(self.neurons) < n_clusters:
            n_clusters = len(self.neurons)
            
        specialties = self.get_specialties().reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(specialties)
        
        clusters = {}
        for label in set(cluster_labels):
            clusters[label] = [self.neurons[i] for i in range(len(self.neurons)) if cluster_labels[i] == label]
        return clusters

    def correlate_with_input(self, input_feature):
        """
        Correlate cluster activations with input features.
        
        Args:
            input_feature: Input feature to correlate with
            
        Returns:
            Dictionary of correlation reports by cluster
        """
        clusters = self.find_clusters()
        correlation_report = {}
        
        for label, cluster_neurons in clusters.items():
            activations = [neuron.active for neuron in cluster_neurons]
            # Calculate fraction of neurons active in this cluster
            correlation = np.mean(activations) if activations else 0.0
            correlation_report[label] = correlation
            
        return correlation_report

    def log_activity(self, input_label=None):
        """
        Log current ensemble activity state.
        
        Args:
            input_label: Label describing the current input
        """
        log_entry = {
            "input_label": input_label,
            "active_neurons": [(n.position, n.specialty) for n in self.get_active_neurons()],
            "specialties": self.get_specialties().tolist(),
            "total_active": len(self.get_active_neurons()),
            "total_neurons": len(self.neurons)
        }
        self.logs.append(log_entry)

    def update_iterative_mapping(self, specialty, human_label):
        """
        Update mapping from specialty values to human-readable labels.
        
        Args:
            specialty: Specialty value
            human_label: Human-readable label
        """
        self.iterative_mapping[specialty] = human_label

    def summarize(self):
        """
        Generate summary of current ensemble state.
        
        Returns:
            List of cluster summaries with positions, specialties, and labels
        """
        summary = []
        clusters = self.find_clusters()
        
        for label, cluster in clusters.items():
            positions = [n.position for n in cluster]
            specialties = [n.specialty for n in cluster]
            mapped_labels = [self.iterative_mapping.get(s, f"Specialty_{s:.2f}") for s in specialties]
            active_count = sum(1 for n in cluster if n.active)
            
            summary.append({
                "cluster_label": label,
                "positions": positions,
                "specialties": specialties,
                "human_labels": mapped_labels,
                "active_count": active_count,
                "total_count": len(cluster)
            })
        return summary

    def print_human_summary(self):
        """Print human-readable summary of ensemble activity."""
        print("\n=== Ensemble Activity Summary ===")
        summary = self.summarize()
        
        total_active = sum(cluster["active_count"] for cluster in summary)
        total_neurons = sum(cluster["total_count"] for cluster in summary)
        
        print(f"Overall Activity: {total_active}/{total_neurons} neurons active ({total_active/total_neurons*100:.1f}%)")
        print()
        
        for cluster in summary:
            print(f"Cluster {cluster['cluster_label']}:")
            print(f"  Active: {cluster['active_count']}/{cluster['total_count']} neurons")
            print(f"  Positions: {cluster['positions']}")
            print(f"  Specialties: {[f'{s:.2f}' for s in cluster['specialties']]}")
            print(f"  Labels: {cluster['human_labels']}")
            print()