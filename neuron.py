import numpy as np

class Neuron:
    """
    Bio-inspired neuron with specialty, activation, and neighbor communication.
    
    Attributes:
        position: Spatial position (tuple)
        weight: Signal intensity multiplier
        active: Current activation state (bool)
        specialty: Emergent specialty value (float)
        neighbors: List of connected neurons
        message_buffer: Incoming messages from neighbors
        goal: Current goal signal
        goal_alignment: Whether goal aligns with specialty
        has_fired_this_round: Firing state for current round
    """
    
    def __init__(self, position, specialty=None, iteration=0, cluster=None, target_specialty=None):
        """Initialize neuron with position, optional specialty, cluster, and target_specialty."""
        self.position = position
        self.weight = np.random.uniform(0.3, 1.0)
        self.active = False
        # Assign specialty based on cluster or target_specialty
        self.specialty = specialty if specialty is not None else (target_specialty if target_specialty is not None else np.random.uniform(1, 10))
        self.target_specialty = target_specialty
        self.neighbors = []
        self.message_buffer = []
        self.goal = None
        self.goal_alignment = False
        self.has_fired_this_round = False
        self.iteration = iteration
        self.cluster = cluster
        self.last_sent_message = None
        self.last_received_messages = []
        self.message_history = []
        self.signal_history = []
        self.confidence = 0.5
        self.decision_log = []
        # Color association: track correct guesses per color
        try:
            from config import COLOR_LABELS
        except ImportError:
            COLOR_LABELS = ["red", "green", "blue"]
        self.color_success_counts = {color: 0 for color in COLOR_LABELS}
        self.non_activation_count = 0

    def add_neighbor(self, neighbor):
        """Add a neighboring neuron for communication."""
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)

    def receive_message(self, message):
        self.last_received_messages.append(message)
        self.message_history.append(message)
        sender_specialty = message.get('specialty', None)
        sender_confidence = message.get('confidence', None)
        sender_preferred_color = message.get('preferred_color', None)
        similarity = abs(self.specialty - sender_specialty) if sender_specialty is not None else None
        message_type = message.get('type')
        message['similarity'] = similarity
        message['from_neighbor'] = message.get('from') in [n.position for n in self.neighbors]
        # Teach message logic
        if message_type == 'teach' and sender_confidence is not None and sender_confidence > self.confidence:
            # Adjust stats toward sender at rate of abs(confidence diff)
            rate = abs(sender_confidence - self.confidence)
            old_specialty = self.specialty
            old_confidence = self.confidence
            self.specialty += rate * (sender_specialty - self.specialty)
            self.confidence += rate * (sender_confidence - self.confidence)
            # Clamp values
            self.specialty = max(1.0, min(self.specialty, 10.0))
            self.confidence = min(1.0, max(0.0, self.confidence))
            # Optionally, bias color association
            if sender_preferred_color:
                self.color_success_counts[sender_preferred_color] += 1
            # Log teach event
            print(f"[Neuron] RECEIVED TEACH: {self.position} <- {message.get('from')}, specialty {old_specialty:.2f}->{self.specialty:.2f}, confidence {old_confidence:.2f}->{self.confidence:.2f}, color {sender_preferred_color}")

    def receive_goal(self, goal):
        """Set goal and check alignment with specialty."""
        self.goal = goal
        # For now, simple string comparison - can be enhanced
        self.goal_alignment = (str(self.specialty).startswith(str(goal)[:3]) if goal else False)

    def evaluate_activation(self, input_signal, true_label=None, guess_color=None):
        self.signal_history.append(input_signal)
        # Consider signal strength, message context, specialty similarity, feedback, noise
        context = {
            'signal_strength': input_signal,
            'messages': self.last_received_messages,
            'neighbors': [n.position for n in self.neighbors],
            'specialty_similarity': [abs(self.specialty - n.specialty) for n in self.neighbors],
            'last_sent_message': self.last_sent_message,
            'confidence': self.confidence
        }
        # Decision logic: stricter threshold, use weight
        activation_value = input_signal * self.weight
        should_fire = activation_value > 0.7
        reason = f"Signal={input_signal:.2f}, Weight={self.weight:.2f}, Activation={activation_value:.2f}, Confidence={self.confidence:.2f}"
        for msg in self.last_received_messages:
            if msg.get('type') == 'inhibit':
                # Inhibition reduces activation value
                activation_value -= 0.2
                reason += " | Inhibited by neighbor"
            elif msg.get('type') == 'excite':
                activation_value += 0.2
                reason += " | Excited by neighbor"
            if msg.get('similarity') is not None and msg['similarity'] < 1.0:
                reason += f" | Similar specialty ({msg['similarity']:.2f})"
        # Re-evaluate firing after message effects
        should_fire = activation_value > 0.7
        # Add noise/randomness
        import random
        if random.random() < 0.05:
            should_fire = not should_fire
            reason += " | Randomness/noise flip"
        # --- Reach for help if confidence is low ---
        try:
            import config_training
            CONFIDENCE_REACH_THRESHOLD = getattr(config_training, 'CONFIDENCE_REACH_THRESHOLD', 0.25)
            CONFIDENCE_REACH_RATE = getattr(config_training, 'CONFIDENCE_REACH_RATE', 0.5)
        except Exception:
            CONFIDENCE_REACH_THRESHOLD = 0.25
            CONFIDENCE_REACH_RATE = 0.5
        if self.confidence < CONFIDENCE_REACH_THRESHOLD:
            candidates = self.neighbors.copy() if self.neighbors else []
            if self.cluster and hasattr(self.cluster, 'neurons'):
                candidates += [n for n in self.cluster.neurons if n is not self]
            if candidates:
                best_peer = max(candidates, key=lambda n: n.confidence)
                old_conf = self.confidence
                self.confidence += CONFIDENCE_REACH_RATE * (best_peer.confidence - self.confidence)
                self.confidence = min(1.0, max(0.0, self.confidence))
                reason += f" | REACH: {old_conf:.2f}->{self.confidence:.2f} (peer: {best_peer.position}, {best_peer.confidence:.2f})"
        self.active = should_fire
        self.decision_log.append({'iteration': self.iteration, 'decision': 'fire' if should_fire else 'inhibit', 'reason': reason, 'context': context})
        # Participation health: update non_activation_count
        if self.active:
            self.non_activation_count = 0
        else:
            self.non_activation_count += 1
        # --- Improved confidence update logic ---
        # Use config_training for granular confidence update rates
        if true_label is not None:
            try:
                import config_training
                update_correct = getattr(config_training, 'CONFIDENCE_UPDATE_CORRECT', 0.2)
                update_incorrect = getattr(config_training, 'CONFIDENCE_UPDATE_INCORRECT', 0.2)
            except Exception:
                update_correct = 0.2
                update_incorrect = 0.2
            # Use guess_color if provided, else fallback to self.goal
            guess_color = getattr(self, 'last_guess', None)
            predicted = guess_color if guess_color is not None else (self.goal if should_fire else 'none')
            if predicted == true_label:
                self.confidence += update_correct * activation_value
                # Hebbian-like: strengthen color association
                if guess_color is not None:
                    self.color_success_counts[guess_color] += 1
            else:
                self.confidence -= update_incorrect * activation_value
                # Weaken color association
                if guess_color is not None and self.color_success_counts[guess_color] > 0:
                    self.color_success_counts[guess_color] -= 1
            # Clamp confidence
            self.confidence = min(1.0, max(0.0, self.confidence))
        # Participation health: penalize if non-activation count exceeds threshold
        if self.non_activation_count > 5:
            self.confidence = max(0.0, self.confidence - 0.05)
            self.specialty = max(1.0, self.specialty - 0.05)
        self.last_received_messages = []
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

    def send_message(self):
        import config_training
        teach_threshold = getattr(config_training, 'CONFIDENCE_SPECIALIZATION_THRESHOLD', 0.85)
        for neighbor in self.neighbors:
            specialty_diff = abs(self.specialty - neighbor.specialty)
            # Send teach message if confident and specialties are close
            if self.confidence > teach_threshold and specialty_diff <= 2.0:
                msg = {
                    'type': 'teach',
                    'specialty': self.specialty,
                    'confidence': self.confidence,
                    'preferred_color': self.get_preferred_color(),
                    'from': self.position
                }
                neighbor.receive_message(msg)
                self.last_sent_message = msg
            # ...existing excite/inhibit logic...
            elif self.confidence < 0.7 or specialty_diff < 2.5:
                msg_type = 'excite' if self.active else 'inhibit'
                msg = {'type': msg_type, 'specialty': self.specialty, 'from': self.position}
                neighbor.receive_message(msg)
                self.last_sent_message = msg

    def propagate_goal(self):
        """Propagate goal to neighboring neurons."""
        if not self.has_fired_this_round:
            for neighbor in self.neighbors:
                neighbor.receive_goal(self.goal)

    def learn_specialty(self, input_data, all_neurons=None):
        """
        Adjust specialty based on successful activations, with clamping and repulsion.
        Args:
            input_data: The input that caused activation
            all_neurons: List of all neurons for repulsion
        """
        if self.active and self.has_fired_this_round:
            learning_rate = 0.5
            self.iteration += 1
            # Attraction to target specialty
            if self.target_specialty is not None:
                self.specialty += learning_rate * (self.target_specialty - self.specialty)
            elif isinstance(input_data, (int, float)):
                self.specialty += learning_rate * (input_data - self.specialty)
            # Repulsion from other neurons
            if all_neurons is not None:
                repulsion_strength = 0.2
                for other in all_neurons:
                    if other is not self:
                        diff = self.specialty - other.specialty
                        if abs(diff) < 2.0:
                            self.specialty += repulsion_strength * diff
            # Clamp specialty between 1 and 10
            self.specialty = max(1, min(self.specialty, 10))
        # Notify cluster of specialty change
        if self.cluster is not None:
            self.cluster.adapt_specialty()

    def clear_round(self):
        """Clear round-specific state for next processing cycle."""
        self.has_fired_this_round = False
        self.active = False
        self.message_buffer = []
        pass

    def clear_messages(self):
        self.message_buffer = []
        self.has_fired_this_round = False

    def get_decision_log(self):
        return self.decision_log

    def force_activate(self, override_signal=True, true_label=None):
        """Force neuron activation by cluster controller, with override logic."""
        import config_training
        override_delta = getattr(config_training, 'OVERRIDE_CONFIDENCE_DELTA', 0.15)
        override_threshold = getattr(config_training, 'OVERRIDE_CONFIDENCE_THRESHOLD', 0.7)
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