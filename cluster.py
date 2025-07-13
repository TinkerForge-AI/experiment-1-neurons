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

    def aggregate_signal(self, input_signal, true_label=None, initial_round=False):
        # Aggregate neuron activations for this cluster
        if initial_round:
            # Activate all neurons in the initial round so they can all start learning
            print(f"[Cluster {self.cluster_id}] Initial round - activating all neurons for learning")
            for neuron in self.neurons:
                neuron.active = True
                neuron.has_fired_this_round = True
                neuron.evaluate_activation(input_signal, true_label=true_label)
        else:
            for neuron in self.neurons:
                neuron.evaluate_activation(input_signal, true_label=true_label)
        
        # Guarantee at least one neuron fires per cluster (for non-initial rounds)
        if not initial_round and not any(n.active for n in self.neurons):
            import random
            chosen = random.choice(self.neurons)
            refused = False
            if hasattr(chosen, 'force_activate'):
                refused = chosen.force_activate(override_signal=True, true_label=true_label)
            else:
                chosen.active = True
                chosen.has_fired_this_round = True
            print(f"[ClusterController] Forced activation: Cluster {self.cluster_id}, Neuron {getattr(chosen, 'position', 'unknown')} (specialty={getattr(chosen, 'specialty', 'n/a'):.2f}){' [REFUSED]' if refused else ''}")
        
        # Update has_fired_this_round for active neurons
        for neuron in self.neurons:
            if neuron.active and not neuron.has_fired_this_round:
                neuron.has_fired_this_round = True
        
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
