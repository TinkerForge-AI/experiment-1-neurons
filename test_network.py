import os
import random
import numpy as np
from neuron import Neuron
from neuron_ensemble_monitor import NeuronEnsembleMonitor
from state_mgr.neuron_state_manager import load_neuron_state
from mappings.data_loader import load_pixel

def classify_pixel(neurons, monitor, pixel):
    """
    Ask the trained network: What color is this pixel?
    Returns the predicted color ("red", "green", "blue") and the activation summary.
    """
    color_map = {"red": 0, "green": 1, "blue": 2}
    color_names = ["red", "green", "blue"]
    activations = {}
    for color in color_names:
        # Clear neuron state
        for neuron in neurons:
            neuron.clear_round()
        # Set goal
        for neuron in neurons:
            neuron.receive_goal(f"detect_{color}")
        # Calculate input signal
        input_signal = pixel[color_map[color]] / 255.0
        # Evaluate activation
        for neuron in neurons:
            neuron.evaluate_activation(input_signal)
        # Count active neurons
        active_count = sum(1 for neuron in neurons if neuron.active)
        activations[color] = active_count
    # Pick the color with the most activations
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
    monitor = NeuronEnsembleMonitor(neurons)
    correct = 0
    for i in range(num_tests):
        # Generate a random pixel that is clearly red, green, or blue
        color_options = ["red", "green", "blue"]
        true_color = random.choice(color_options)
        if true_color == "red":
            pixel = load_pixel((255, 0, 0))
        elif true_color == "green":
            pixel = load_pixel((0, 255, 0))
        else:
            pixel = load_pixel((0, 0, 255))
        predicted_color, activations = classify_pixel(neurons, monitor, pixel)
        print(f"Test {i+1}: Pixel={pixel} | True={true_color} | Predicted={predicted_color} | Activations={activations}")
        if predicted_color == true_color:
            correct += 1
    print(f"\nAccuracy: {correct}/{num_tests} correct ({correct/num_tests*100:.1f}%)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test trained neural network on color classification.")
    parser.add_argument('-n', '--num_tests', type=int, default=10, help='Number of test pixels to classify')
    args = parser.parse_args()
    run_test(args.num_tests)
