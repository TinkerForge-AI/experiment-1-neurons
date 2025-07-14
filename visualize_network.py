import pickle
import matplotlib.pyplot as plt
import networkx as nx
import os

# Load neuron and cluster state from pickle files
NEURON_STATE_PATH = 'network_state.pkl'
MONITOR_STATE_PATH = 'monitor_state.pkl'

# Helper to get color for cluster
CLUSTER_COLORS = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']

def load_state():
    if not os.path.exists(NEURON_STATE_PATH):
        print(f"State file {NEURON_STATE_PATH} not found.")
        return None, None
    with open(NEURON_STATE_PATH, 'rb') as f:
        neurons = pickle.load(f)
    clusters = None
    if os.path.exists(MONITOR_STATE_PATH):
        with open(MONITOR_STATE_PATH, 'rb') as f:
            monitor = pickle.load(f)
            clusters = getattr(monitor, 'clusters', None)
    return neurons, clusters

def visualize_network(neurons, clusters=None):
    G = nx.Graph()
    pos_dict = {}
    color_map = []
    size_map = []
    label_map = {}
    # Layout: if neurons have position as (x, y), use that; else use index
    for idx, neuron in enumerate(neurons):
        pos = neuron.position if hasattr(neuron, 'position') else (idx, 0)
        if isinstance(pos, tuple) and len(pos) == 1:
            pos = (pos[0], 0)
        pos_dict[idx] = pos
        # Color by cluster
        cluster_idx = getattr(neuron, 'cluster', None)
        if cluster_idx is not None and hasattr(cluster_idx, 'cluster_id'):
            color = CLUSTER_COLORS[int(cluster_idx.cluster_id[-1]) % len(CLUSTER_COLORS)]
        else:
            color = 'gray'
        color_map.append(color)
        # Size by specialty (biological relevance)
        size = 300 + 500 * getattr(neuron, 'specialty', 0.5) / 10.0
        size_map.append(size)
        # Label: show specialty, winner status, and membrane potential
        winner = getattr(neuron, 'is_winner', False)
        mp = getattr(neuron, 'membrane_potential', 0.0)
        label_map[idx] = f"{idx}\nS:{getattr(neuron, 'specialty', 0):.2f}\nW:{'Y' if winner else 'N'}\nMP:{mp:.2f}"
        G.add_node(idx)
    # Add neighbor edges
    for idx, neuron in enumerate(neurons):
        for neighbor in getattr(neuron, 'neighbors', []):
            try:
                neighbor_idx = neurons.index(neighbor)
                G.add_edge(idx, neighbor_idx)
            except ValueError:
                continue
    # Draw
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=pos_dict, node_color=color_map, node_size=size_map, labels=label_map, with_labels=True, font_size=8, edge_color='black')
    plt.title('Neuron Network Structure (Color: Cluster, Size: Specialty, W: Winner, MP: Membrane Potential)')
    plt.tight_layout()
    plt.show()

def main():
    neurons, clusters = load_state()
    if neurons is None:
        print("No neuron state found. Run training first.")
        return
    visualize_network(neurons, clusters)
    # Time-series plots for activations, weights, specialties
    import matplotlib.pyplot as plt
    import numpy as np
    # Activation history
    plt.figure(figsize=(10, 4))
    for idx, neuron in enumerate(neurons):
        if hasattr(neuron, 'activation_history'):
            plt.plot(neuron.activation_history, label=f'N{idx}')
    plt.title('Neuron Activation History')
    plt.xlabel('Time')
    plt.ylabel('Activation (1=fire, 0=inactive)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Weight history (mean per neuron)
    plt.figure(figsize=(10, 4))
    for idx, neuron in enumerate(neurons):
        if hasattr(neuron, 'weight_history') and neuron.weight_history:
            mean_weights = [np.mean(list(w.values())) for w in neuron.weight_history]
            plt.plot(mean_weights, label=f'N{idx}')
    plt.title('Neuron Mean Weight History')
    plt.xlabel('Time')
    plt.ylabel('Mean Synaptic Weight')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Specialty history
    plt.figure(figsize=(10, 4))
    for idx, neuron in enumerate(neurons):
        if hasattr(neuron, 'specialty_history'):
            plt.plot(neuron.specialty_history, label=f'N{idx}')
    plt.title('Neuron Specialty History')
    plt.xlabel('Time')
    plt.ylabel('Specialty')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
