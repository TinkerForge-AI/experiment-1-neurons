import numpy as np
from neuron import Neuron

class Cluster:
    """
    Bio-inspired cluster: manages a group of neurons, aggregates signals, and coordinates winner-take-all (WTA) competition.
    - Each cluster aggregates neuron activations and selects the most active neuron(s) as winner(s).
    - Only winner neurons propagate output; others are suppressed for the round.
    - Supports supervised feedback via controller for error-driven learning.
    - Tracks specialty as the mean of member neurons.
    - All communication and adaptation is biologically motivated.
    """
    def __init__(self, neurons, cluster_id):
        self.neurons = neurons
        self.cluster_id = cluster_id
        self.specialty = np.mean([n.specialty for n in neurons])
        self.controller = None  # Reference to higher-order controller

    def aggregate_signal(self, input_signal, initial_round=False):
        """
        Aggregate neuron activations and implement winner-take-all (WTA) competition.
        Only the most active neuron(s) in the cluster propagate output.
        Args:
            input_signal (float): Input signal to be distributed to all neurons.
            initial_round (bool): If True, all neurons are activated for initial learning.
        Returns:
            int: Number of winner neurons for this round.
        """
        # Reset per-round evaluation guard for all neurons
        for neuron in self.neurons:
            neuron.has_evaluated_this_round = False
        # Initial round: activate all neurons for learning
        if initial_round:
            print(f"[Cluster {self.cluster_id}] Initial round - activating all neurons for learning")
            for neuron in self.neurons:
                neuron.active = True
                neuron.has_fired_this_round = True
                neuron.membrane_potential += input_signal
                neuron.evaluate_activation()
        else:
            for neuron in self.neurons:
                neuron.membrane_potential += input_signal
                neuron.evaluate_activation()
        # Sharper winner selection: neuron whose specialty is closest to cluster specialty
        from config import CLUSTER_SPECIALTIES, SPECIALTY_CLAMP_MIN, SPECIALTY_CLAMP_MAX
        # Use cluster_id to select target specialty
        # Extract cluster index from cluster_id (e.g., 'C0' -> 0)
        try:
            cluster_idx = int(str(self.cluster_id).lstrip('C'))
        except Exception:
            cluster_idx = 0
        target_specialty = CLUSTER_SPECIALTIES[cluster_idx % len(CLUSTER_SPECIALTIES)]
        specialties = [n.specialty for n in self.neurons]
        # Clamp specialties
        specialties = [max(SPECIALTY_CLAMP_MIN, min(SPECIALTY_CLAMP_MAX, s)) for s in specialties]
        # Find neuron(s) closest to target specialty
        min_dist = min([abs(s - target_specialty) for s in specialties])
        winners = [n for n, s in zip(self.neurons, specialties) if abs(s - target_specialty) == min_dist]
        # Set winner flag and allow partial/probabilistic firing for non-winners
        import random
        for neuron in self.neurons:
            neuron.is_winner = neuron in winners
            if neuron.is_winner:
                neuron.active = True
            else:
                # Non-winners: probabilistic firing and partial influence
                # Probability and influence strength can be tuned in config.py
                from config import NONWINNER_FIRE_PROB, NONWINNER_INFLUENCE_SCALE
                neuron.active = random.random() < NONWINNER_FIRE_PROB
                if neuron.active:
                    neuron.influence_scale = NONWINNER_INFLUENCE_SCALE
                else:
                    neuron.influence_scale = 0.0
        # Log cluster activity
        print(f"[Cluster {self.cluster_id}] Winner(s): {[n.position for n in winners]} (target_specialty={target_specialty}, min_dist={min_dist:.2f})")
        # Return number of winners
        return len(winners)

    def adapt_specialty(self):
        """
        Update cluster specialty as the mean of member neuron specialties.
        """
        self.specialty = np.mean([n.specialty for n in self.neurons])

    def receive_message(self, message):
        """
        Delegate incoming message to all member neurons.
        """
        for neuron in self.neurons:
            neuron.receive_message(message)

    def add_neuron(self, neuron):
        """
        Add a neuron to the cluster and update specialty.
        """
        self.neurons.append(neuron)
        self.adapt_specialty()

    def remove_neuron(self, neuron):
        """
        Remove a neuron from the cluster and update specialty.
        """
        self.neurons.remove(neuron)
        self.adapt_specialty()

    def get_state(self):
        """
        Return a summary of cluster state for logging/visualization.
        """
        return {
            'cluster_id': self.cluster_id,
            'specialty': self.specialty,
            'num_neurons': len(self.neurons)
        }

    def __repr__(self):
        """
        String representation for debugging.
        """
        return f"<Cluster {self.cluster_id}: specialty={self.specialty:.2f}, neurons={len(self.neurons)}>"
