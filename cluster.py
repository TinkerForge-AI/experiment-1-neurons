import numpy as np
from neuron import Neuron

class Cluster:
    """
    A group of neurons managed as a unit. Can aggregate signals and communicate with controllers.
    """
    def __init__(self, neurons, cluster_id):
        self.neurons = neurons
        self.cluster_id = cluster_id
        self.specialty = np.mean([n.specialty for n in neurons])
        self.controller = None  # Reference to higher-order controller

    def aggregate_signal(self, input_signal, true_label=None):
        # Aggregate neuron activations for this cluster
        for neuron in self.neurons:
            neuron.evaluate_activation(input_signal)
        # Guarantee at least one neuron fires per cluster
        if not any(n.active for n in self.neurons):
            import random
            chosen = random.choice(self.neurons)
            refused = False
            if hasattr(chosen, 'force_activate'):
                refused = chosen.force_activate(override_signal=True, true_label=true_label)
            else:
                chosen.active = True
            print(f"[ClusterController] Forced activation: Cluster {self.cluster_id}, Neuron {getattr(chosen, 'position', 'unknown')} (specialty={getattr(chosen, 'specialty', 'n/a'):.2f}){' [REFUSED]' if refused else ''}")
        return sum(n.active for n in self.neurons)

    def adapt_specialty(self):
        # Adapt cluster specialty based on neuron states
        self.specialty = np.mean([n.specialty for n in self.neurons])

    def receive_message(self, message):
        # Preprocess or delegate message to neurons
        for neuron in self.neurons:
            neuron.receive_message(message)

    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        self.adapt_specialty()

    def remove_neuron(self, neuron):
        self.neurons.remove(neuron)
        self.adapt_specialty()

    def get_state(self):
        return {
            'cluster_id': self.cluster_id,
            'specialty': self.specialty,
            'num_neurons': len(self.neurons)
        }

    def __repr__(self):
        return f"<Cluster {self.cluster_id}: specialty={self.specialty:.2f}, neurons={len(self.neurons)}>"
