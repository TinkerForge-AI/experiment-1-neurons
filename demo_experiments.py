#!/usr/bin/env python3
"""
Demo script showing various experiments with the bio-inspired neural system.
"""

import numpy as np
from neuron import Neuron
from neuron_ensemble_monitor import NeuronEnsembleMonitor
from mappings.data_loader import load_pixel, normalize_pixel, extract_color_component

def demo_color_experiments():
    """Demonstrate experiments with different colors."""
    print("=== Bio-Inspired Neural System Demo ===\n")
    
    # Create a small network for demo
    neurons = [Neuron(position=(i,), specialty=np.random.uniform(8, 16)) for i in range(6)]
    
    # Connect neurons in a ring topology
    for i in range(len(neurons)):
        neurons[i].add_neighbor(neurons[(i+1) % len(neurons)])
        neurons[i].add_neighbor(neurons[(i-1) % len(neurons)])
    
    monitor = NeuronEnsembleMonitor(neurons)
    
    # Test different colors
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128)
    }
    
    for color_name, rgb in colors.items():
        print(f"--- Testing {color_name.upper()} pixel {rgb} ---")
        
        # Clear previous state
        for neuron in neurons:
            neuron.clear_round()
        
        # Set goal
        goal = f"detect_{color_name}"
        for neuron in neurons:
            neuron.receive_goal(goal)
        
        # Process input
        pixel = load_pixel(rgb)
        color_component = extract_color_component(pixel, color_name if color_name in ['red', 'green', 'blue'] else 'green')
        input_signal = normalize_pixel([color_component])[0]
        
        print(f"Input signal strength: {input_signal:.3f}")
        
        # Neural processing
        for neuron in neurons:
            neuron.evaluate_activation(input_signal)
        
        for neuron in neurons:
            neuron.send_message()
        
        # Monitor and report
        monitor.log_activity(input_label=color_name)
        active_neurons = monitor.get_active_neurons()
        print(f"Active neurons: {len(active_neurons)}/{len(neurons)}")
        
        if active_neurons:
            avg_specialty = np.mean([n.specialty for n in active_neurons])
            print(f"Average specialty of active neurons: {avg_specialty:.2f}")
        
        print()

def demo_network_connectivity():
    """Demonstrate different network topologies."""
    print("=== Network Topology Demo ===\n")
    
    topologies = {
        "Linear": lambda neurons: connect_linear(neurons),
        "Ring": lambda neurons: connect_ring(neurons),
        "Star": lambda neurons: connect_star(neurons)
    }
    
    for topo_name, connect_func in topologies.items():
        print(f"--- {topo_name} Topology ---")
        
        neurons = [Neuron(position=(i,), specialty=np.random.uniform(10, 15)) for i in range(5)]
        connect_func(neurons)
        
        # Count connections
        total_connections = sum(len(n.neighbors) for n in neurons)
        avg_connections = total_connections / len(neurons)
        print(f"Average connections per neuron: {avg_connections:.1f}")
        
        # Test activation spread
        for neuron in neurons:
            neuron.clear_round()
        
        # Activate first neuron manually
        neurons[0].active = True
        neurons[0].has_fired_this_round = True
        
        # Let it send messages
        neurons[0].send_message()
        
        # Count influenced neurons
        influenced = sum(1 for n in neurons if n.message_buffer)
        print(f"Neurons influenced by activation: {influenced}/{len(neurons)}")
        print()

def connect_linear(neurons):
    """Connect neurons in a linear chain."""
    for i in range(len(neurons) - 1):
        neurons[i].add_neighbor(neurons[i + 1])
        neurons[i + 1].add_neighbor(neurons[i])

def connect_ring(neurons):
    """Connect neurons in a ring."""
    for i in range(len(neurons)):
        neurons[i].add_neighbor(neurons[(i + 1) % len(neurons)])
        neurons[i].add_neighbor(neurons[(i - 1) % len(neurons)])

def connect_star(neurons):
    """Connect all neurons to the first one (star topology)."""
    center = neurons[0]
    for i in range(1, len(neurons)):
        center.add_neighbor(neurons[i])
        neurons[i].add_neighbor(center)

if __name__ == "__main__":
    demo_color_experiments()
    demo_network_connectivity()
    print("Demo completed! Try modifying the experiments or creating your own.")
