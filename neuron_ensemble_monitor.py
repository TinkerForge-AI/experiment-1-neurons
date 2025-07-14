import numpy as np
from sklearn.cluster import KMeans

class NeuronEnsembleMonitor:
    """
    Monitors and analyzes neural ensemble behavior, now with cluster support.
    """
    
    def __init__(self, neurons, clusters=None, controllers=None, iteration=0):
        """Initialize monitor with list of neurons and optional iteration."""
        self.neurons = neurons  # List of Neuron objects
        self.clusters = clusters if clusters is not None else []
        self.controllers = controllers if controllers is not None else []
        self.iterative_mapping = {}  # Maps specialty to human label
        self.logs = []
        self.iteration = iteration  # Track experiment iteration

    def get_specialties(self):
        """Get array of all neuron specialties."""
        return np.array([neuron.specialty for neuron in self.neurons])

    def get_active_neurons(self):
        """Get list of currently active neurons."""
        return [neuron for neuron in self.neurons if neuron.active]

    def get_active_clusters(self):
        """Get list of currently active clusters."""
        return [c for c in self.clusters if any(n.active for n in c.neurons)] if self.clusters else []

    def find_clusters(self, n_clusters=None):
        """
        Cluster neurons by specialty using K-means.
        
        Args:
            n_clusters: Number of clusters to find (default from config)
            
        Returns:
            Dictionary mapping cluster labels to neuron lists
        """
        from config import NUM_CLUSTERS
        if n_clusters is None:
            n_clusters = NUM_CLUSTERS
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
            "active_clusters": [c.cluster_id for c in self.get_active_clusters()],
            "specialties": self.get_specialties().tolist(),
            "total_active": len(self.get_active_neurons()),
            "total_neurons": len(self.neurons),
            "total_clusters": len(self.clusters),
            "total_active_clusters": len(self.get_active_clusters())
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
        Generate summary of current ensemble state using actual Cluster objects.
        
        Returns:
            List of cluster summaries with positions, specialties, and labels
        """
        summary = []
        clusters = self.clusters if self.clusters else []
        
        for cluster in clusters:
            positions = [n.position for n in cluster.neurons]
            specialties = [n.specialty for n in cluster.neurons]
            mapped_labels = [self.iterative_mapping.get(s, f"Specialty_{s:.2f}") for s in specialties]
            active_count = sum(1 for n in cluster.neurons if n.active)
            
            summary.append({
                "cluster_id": cluster.cluster_id,
                "positions": positions,
                "specialties": specialties,
                "human_labels": mapped_labels,
                "active_count": active_count,
                "total_count": len(cluster.neurons)
            })
        return summary

    def print_human_summary(self):
        """Print human-readable summary of ensemble activity with controller insights."""
        print("\n=== Ensemble Activity Summary ===")
        summary = self.summarize()
        
        total_active = sum(cluster["active_count"] for cluster in summary)
        total_neurons = sum(cluster["total_count"] for cluster in summary)
        if total_neurons == 0:
            print("No neurons in clusters. Cannot compute activity percentage.")
            return
        print(f"Overall Activity: {total_active}/{total_neurons} neurons active ({(total_active/total_neurons*100):.1f}%)")
        
        # Show controller specialization assignments if available
        if self.controllers and len(self.controllers) > 0:
            controller = self.controllers[0]
            if hasattr(controller, 'cluster_assignments') and hasattr(controller, 'target_specialties'):
                print(f"Controller Assignments: {controller.cluster_assignments}")
                print(f"Target Specialties: {controller.target_specialties}")
        print()
        
        for cluster in summary:
            cluster_id = cluster['cluster_id']
            print(f"Cluster {cluster_id}:")
            print(f"  Active: {cluster['active_count']}/{cluster['total_count']} neurons ({(cluster['active_count']/cluster['total_count']*100):.1f}%)")
            print(f"  Positions: {cluster['positions']}")
            print(f"  Specialties: {[f'{s:.2f}' for s in cluster['specialties']]}")
            
            # Show target assignment if available
            if self.controllers and len(self.controllers) > 0:
                controller = self.controllers[0]
                if hasattr(controller, 'cluster_assignments'):
                    target_color = controller.cluster_assignments.get(cluster_id, "unassigned")
                    target_spec = controller.target_specialties.get(target_color, "N/A") if target_color != "unassigned" else "N/A"
                    cluster_mean = sum(cluster['specialties']) / len(cluster['specialties']) if cluster['specialties'] else 0
                    print(f"  Target: {target_color} (specialty {target_spec}) | Current Mean: {cluster_mean:.2f}")
            
            print(f"  Labels: {cluster['human_labels']}")
            print("  --- Neuron States ---")
            for neuron in [n for n in self.neurons if n.position in cluster['positions']]:
                goal_align = "✓" if getattr(neuron, 'goal_alignment', False) else "✗"
                target_spec = getattr(neuron, 'target_specialty', 'N/A')
                winner = getattr(neuron, 'is_winner', False)
                mp = getattr(neuron, 'membrane_potential', 0.0)
                print(f"    Neuron {neuron.position}: Winner={winner} | Goal:{goal_align} | Target:{target_spec} | Specialty={neuron.specialty:.2f} | MP={mp:.2f}")
            print()