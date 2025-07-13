import os
import random
import numpy as np
from neuron import Neuron
from neuron_ensemble_monitor import NeuronEnsembleMonitor
from state_mgr.neuron_state_manager import load_neuron_state
from mappings.data_loader import load_pixel
from config import COLOR_LABELS, COLOR_RGBS, NUM_CLUSTERS, TEST_SAMPLES

def classify_pixel(neurons, monitor, pixel):
    """
    Ask the trained network: What color is this pixel?
    Returns the predicted color ("red", "green", "blue") and the activation summary.
    """
    color_labels = COLOR_LABELS
    color_rgbs = COLOR_RGBS
    activations = {}
    for idx, color in enumerate(color_labels):
        for neuron in neurons:
            neuron.clear_round()
        for neuron in neurons:
            neuron.receive_goal(f"detect_{color}")
        # Use max channel as signal, or custom mapping
        input_signal = max(pixel) / 255.0
        for neuron in neurons:
            neuron.evaluate_activation(input_signal)
        active_count = sum(1 for neuron in neurons if neuron.active)
        activations[color] = active_count
    predicted_color = max(activations, key=activations.get)
    return predicted_color, activations

def run_test(num_tests=10):
    neurons = load_neuron_state("network_state.pkl")
    if neurons is None:
        # Try loading backup if available
        if os.path.exists("network_state.pkl"):
            print("network_state.pkl exists but could not be loaded. It may be corrupted.")
        elif os.path.exists("network_state_backup.pkl"):
            print("Trying backup neuron state...")
            neurons = load_neuron_state("network_state_backup.pkl")
        if neurons is None:
            print("No saved neuron state found. Please train the network first using train_network.py.")
            return
    # Example: create clusters and controller for test output
    num_clusters = NUM_CLUSTERS
    cluster_size = max(1, len(neurons) // num_clusters)
    clusters = []
    for i in range(num_clusters):
        cluster_neurons = neurons[i*cluster_size:(i+1)*cluster_size]
        clusters.append(__import__('cluster').Cluster(cluster_neurons, cluster_id=f"C{i}"))
    from cluster_controller import ClusterController
    controller = ClusterController(clusters, controller_id="TestController")
    monitor = NeuronEnsembleMonitor(neurons, clusters=clusters, controllers=[controller])
    correct = 0
    color_labels = COLOR_LABELS
    color_rgbs = COLOR_RGBS
    for i in range(num_tests):
        idx = random.randint(0, len(color_labels) - 1)
        true_color = color_labels[idx]
        pixel = load_pixel(color_rgbs[idx])
        predicted_color, activations = classify_pixel(neurons, monitor, pixel)
        input_signal = max(pixel) / 255.0
        route_result = controller.route_signal(input_signal, true_color)
        print(f"Test {i+1}: Pixel={pixel} | True={true_color} | Predicted={predicted_color} | Activations={activations}")
        print(f"  Controller: {route_result}")
        monitor.log_activity(input_label=true_color)
        if predicted_color == true_color:
            correct += 1
    print(f"\nAccuracy: {correct}/{num_tests} correct ({correct/num_tests*100:.1f}%)")
    print("\n=== Hierarchical Activity Summary ===")
    monitor.print_human_summary()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test trained neural network on color classification.")
    parser.add_argument('-n', '--num_tests', type=int, default=TEST_SAMPLES, help='Number of test pixels to classify')
    args = parser.parse_args()
    run_test(args.num_tests)
