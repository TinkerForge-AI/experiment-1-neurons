import numpy as np

class Neuron:
    """
    Bio-inspired neuron with specialty, adaptive weight, activation, and neighbor communication.
    
    Attributes:
        position: Spatial position (tuple)
        weight: Signal transformation multiplier (learnable, adapts based on prediction correctness, confidence, and noise)
        active: Current activation state (bool)
        specialty: Emergent specialty value (float)
        neighbors: List of connected neurons
        message_buffer: Incoming messages from neighbors
        goal: Current goal signal
        goal_alignment: Whether goal aligns with specialty
        has_fired_this_round: Firing state for current round
        color_success_counts: Tracks correct guesses per color for granular color association
        consecutive_correct: Number of consecutive correct guesses (for adaptive learning rate)
        stabilized: True if neuron is using reduced learning rate after stabilization
    """
    
    def __init__(self, position, specialty=None, iteration=0, cluster=None, target_specialty=None):
        """
        Initialize neuron with biologically-inspired properties:
        - membrane_potential: integrates incoming signals, decays over time (leaky integration)
        - refractory_time: time left before neuron can fire again
        - weight: synaptic strength to each neighbor (dict)
        - short_term_memory: stores recent activations for temporal integration
        - specialty: emergent property, not directly used for firing
        """
        self.position = position
        try:
            from config import SPECIALTY_INIT_MIN, SPECIALTY_INIT_MAX
        except ImportError:
            SPECIALTY_INIT_MIN, SPECIALTY_INIT_MAX = 1.0, 10.0
        self.specialty = specialty if specialty is not None else (
            target_specialty if target_specialty is not None else np.random.uniform(SPECIALTY_INIT_MIN, SPECIALTY_INIT_MAX)
        )
        self.target_specialty = target_specialty
        self.neighbors = []
        self.weight = {}  # Dict: neighbor.position -> weight
        self.membrane_potential = 0.0
        from config import ACTIVATION_THRESHOLD, LEAK_RATE, REFRACTORY_PERIOD
        self.leak_rate = LEAK_RATE  # Decay per round
        self.threshold = ACTIVATION_THRESHOLD  # Firing threshold
        self.refractory_period = REFRACTORY_PERIOD  # Rounds
        self.refractory_time = 0
        self.active = False
        self.has_fired_this_round = False
        self.iteration = iteration
        self.cluster = cluster
        self.last_sent_message = None
        self.last_received_messages = []
        self.message_history = []
        self.signal_history = []
        self.short_term_memory = []  # List of recent inputs
        self.memory_length = 5
        # Color association: track correct guesses per color (for visualization)
        try:
            from config import COLOR_LABELS
        except ImportError:
            COLOR_LABELS = ["red", "green", "blue"]
        self.color_success_counts = {color: 0 for color in COLOR_LABELS}
        # Lateral inhibition strength
        self.inhibition_strength = 0.5
        # For winner-take-all, cluster will set this
        self.is_winner = False
        # For time-series logging
        self.activation_history = []
        self.weight_history = []
        self.specialty_history = []
        # Per-round evaluation guard
        self.has_evaluated_this_round = False
        # Remove old confidence/correctness logic
        # ...existing code...

    def add_neighbor(self, neighbor):
        """Add a neighboring neuron and initialize synaptic weight."""
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            # Initialize synaptic weight
            try:
                from config import WEIGHT_RANGE
            except ImportError:
                WEIGHT_RANGE = (0.3, 1.0)
            self.weight[neighbor.position] = np.random.uniform(*WEIGHT_RANGE)

    def receive_message(self, message):
        """
        Receive a message from a neighbor and update membrane potential.
        Message format: {'from': position, 'signal': float, 'type': 'excite'|'inhibit', 'weight': float}
        """
        self.last_received_messages.append(message)
        self.message_history.append(message)
        sender_pos = message.get('from')
        signal = message.get('signal', 0.0)
        msg_type = message.get('type', 'excite')
        # Sum weighted input
        w = self.weight.get(sender_pos, 1.0)
        if msg_type == 'excite':
            self.membrane_potential += signal * w
        elif msg_type == 'inhibit':
            self.membrane_potential -= signal * w * self.inhibition_strength
        # Store in short-term memory
        self.short_term_memory.append(signal)
        if len(self.short_term_memory) > self.memory_length:
            self.short_term_memory.pop(0)

    def receive_goal(self, goal):
        """Set goal and check alignment with specialty."""
        self.goal = goal
        # For now, simple string comparison - can be enhanced
        self.goal_alignment = (str(self.specialty).startswith(str(goal)[:3]) if goal else False)

    def evaluate_activation(self):
        """
        Integrate all incoming signals, apply leaky integration, check refractory period, and fire if membrane potential > threshold.
        Implements recursive/wavefront propagation and lateral inhibition.
        """
        # Leaky integration: decay membrane potential
        self.membrane_potential *= (1.0 - self.leak_rate)
        print(f"[Neuron {self.position}] Membrane potential: {self.membrane_potential:.4f}, Threshold: {self.threshold:.4f}")
        # Add short-term memory integration
        if self.short_term_memory:
            self.membrane_potential += sum(self.short_term_memory) / len(self.short_term_memory)
        # Per-round evaluation guard to prevent infinite recursion
        if self.has_evaluated_this_round:
            return
        self.has_evaluated_this_round = True
        # Check refractory period
        if self.refractory_time > 0:
            self.active = False
            self.refractory_time -= 1
            self.activation_history.append(0)
            return
        # Fire if membrane potential exceeds threshold
        if self.membrane_potential > self.threshold:
            self.active = True
            self.has_fired_this_round = True
            self.refractory_time = self.refractory_period
            self.activation_history.append(1)
            print(f"[Neuron {self.position}] ACTIVATED! Membrane potential: {self.membrane_potential:.4f} > Threshold: {self.threshold:.4f}")
            # Lateral inhibition: send inhibitory signals to neighbors
            for neighbor in self.neighbors:
                neighbor.receive_message({'from': self.position, 'signal': 1.0, 'type': 'inhibit', 'weight': self.weight[neighbor.position]})
            # Recursive/wavefront propagation: trigger neighbors to evaluate
            for neighbor in self.neighbors:
                neighbor.evaluate_activation()
        else:
            self.active = False
            self.activation_history.append(0)
            print(f"[Neuron {self.position}] NOT ACTIVATED. Membrane potential: {self.membrane_potential:.4f} <= Threshold: {self.threshold:.4f}")
        # Reset short-term memory after evaluation
        self.short_term_memory = []
        # Log membrane potential and firing
        print(f"[Neuron] {self.position} membrane_potential={self.membrane_potential:.2f} active={self.active}")
    def get_preferred_color(self):
        """Return the color with the highest success count, or random if all zero."""
        import random
        if not self.color_success_counts:
            return None
        max_count = max(self.color_success_counts.values())
        if max_count == 0:
            # No association yet, pick random
            return random.choice(list(self.color_success_counts.keys()))
        # Return color(s) with max count
        best_colors = [c for c, v in self.color_success_counts.items() if v == max_count]
        return random.choice(best_colors)

    def send_message(self, training_round=0):
        """
        Send biologically-inspired messages to neighbors based on membrane potential, winner status, and specialty.
        """
        messages_sent = 0
        for neighbor in self.neighbors:
            specialty_diff = abs(self.specialty - neighbor.specialty)
            # Send teach-like message if this neuron is a winner and specialties are compatible
            if self.is_winner and specialty_diff <= 3.0:
                msg = {
                    'type': 'teach',
                    'specialty': self.specialty,
                    'membrane_potential': self.membrane_potential,
                    'preferred_color': self.get_preferred_color(),
                    'from': self.position,
                    'training_round': training_round
                }
                neighbor.receive_message(msg)
                self.last_sent_message = msg
                messages_sent += 1
            # Basic excite/inhibit logic
            elif specialty_diff < 2.5:
                msg_type = 'excite' if self.active else 'inhibit'
                msg = {
                    'type': msg_type,
                    'specialty': self.specialty,
                    'from': self.position,
                    'training_round': training_round
                }
                neighbor.receive_message(msg)
                self.last_sent_message = msg
                messages_sent += 1
        if messages_sent > 0:
            print(f"[Neuron] {self.position} sent {messages_sent} messages (MP={self.membrane_potential:.2f}, winner={self.is_winner})")
        return messages_sent

    def propagate_goal(self):
        """Propagate goal to neighboring neurons."""
        if not self.has_fired_this_round:
            for neighbor in self.neighbors:
                neighbor.receive_goal(self.goal)

    def adapt_weights(self):
        """
        Hebbian/STDP-like weight adaptation: strengthen weights to neighbors that were active when this neuron fired.
        """
        if self.active and self.has_fired_this_round:
            for neighbor in self.neighbors:
                # If neighbor was active, strengthen synapse
                if neighbor.active:
                    self.weight[neighbor.position] += 0.05 * (1.0 - self.weight[neighbor.position])
                else:
                    self.weight[neighbor.position] -= 0.02 * self.weight[neighbor.position]
                # Clamp weights
                self.weight[neighbor.position] = max(0.1, min(self.weight[neighbor.position], 2.0))
            # Log weight changes
            self.weight_history.append(self.weight.copy())
        # Track specialty for visualization
        self.specialty_history.append(self.specialty)

    def clear_round(self):
        """Clear round-specific state for next processing cycle."""
        self.has_fired_this_round = False
        self.active = False
        self.message_buffer = []
        self.last_received_messages = []
        # Reset membrane potential if neuron fired
        if self.refractory_time == self.refractory_period:
            self.membrane_potential = 0.0

    def clear_messages(self):
        self.message_buffer = []
        self.has_fired_this_round = False

    # Remove get_decision_log and force_activate (no longer needed)

    def force_activate(self, override_signal=True, true_label=None):
        """Force neuron activation by cluster controller, with override logic."""
        import config
        override_delta = getattr(config, 'OVERRIDE_CONFIDENCE_DELTA', 0.15)
        override_threshold = getattr(config, 'OVERRIDE_CONFIDENCE_THRESHOLD', 0.7)
        # Neuron only refuses if confidence >= threshold
        refused = False
        if self.confidence >= override_threshold:
            refused = True
            self.active = False
            print(f"[Neuron] REFUSED forced activation: Neuron {self.position}, specialty={self.specialty:.2f}, confidence={self.confidence:.2f}")
            # Penalize confidence for refusal
            self.confidence = max(0.0, self.confidence - override_delta)
            # If also incorrect, penalize again
            if true_label is not None and self.goal != true_label:
                self.confidence = max(0.0, self.confidence - override_delta)
                print(f"[Neuron] REFUSAL incorrect guess penalty applied: Neuron {self.position}, confidence={self.confidence:.2f}")
            elif true_label is not None and self.goal == true_label:
                self.confidence = min(1.0, self.confidence + override_delta)
                print(f"[Neuron] REFUSAL correct guess bonus applied: Neuron {self.position}, confidence={self.confidence:.2f}")
        else:
            self.active = True
            print(f"[Neuron] FORCED activation: Neuron {self.position}, specialty={self.specialty:.2f}, confidence={self.confidence:.2f}")
            # If correct, bonus; if incorrect, penalty
            if true_label is not None and self.goal == true_label:
                self.confidence = min(1.0, self.confidence + override_delta)
                print(f"[Neuron] FORCED correct guess bonus applied: Neuron {self.position}, confidence={self.confidence:.2f}")
            elif true_label is not None and self.goal != true_label:
                self.confidence = max(0.0, self.confidence - override_delta)
                print(f"[Neuron] FORCED incorrect guess penalty applied: Neuron {self.position}, confidence={self.confidence:.2f}")
        return refused