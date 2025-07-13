import numpy as np
from neuron import Neuron
from cluster import Cluster
from cluster_controller import ClusterController
from neuron_ensemble_monitor import NeuronEnsembleMonitor

# Example: Create neurons and clusters
num_neurons = 12
neurons = [Neuron(position=(i,), specialty=np.random.uniform(10, 15)) for i in range(num_neurons)]

# Assign neurons to clusters (3 clusters)
clusters = [Cluster(neurons[i*4:(i+1)*4], cluster_id=f"C{i}") for i in range(3)]

# Assign cluster reference to neurons
for cluster in clusters:
    for neuron in cluster.neurons:
        neuron.cluster = cluster

# Create a higher-order controller
controller = ClusterController(clusters, controller_id="TopController")

# Create monitor
monitor = NeuronEnsembleMonitor(neurons, clusters=[*clusters], controllers=[controller])

# Simulate input and routing
input_signal = np.random.uniform(0, 1)
goal = "detect_green"
result = controller.route_signal(input_signal, goal)
print(f"Controller routed signal: {result}")

# Log and summarize activity
monitor.log_activity(input_label=goal)
monitor.print_human_summary()
