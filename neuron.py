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
        self.activation_level = 0.0  # For message-based activation adjustments
        # Color association: track correct guesses per color
        try:
            from config import COLOR_LABELS
        except ImportError:
            COLOR_LABELS = ["red", "green", "blue"]
        self.color_success_counts = {color: 0 for color in COLOR_LABELS}
        self.non_activation_count = 0
        # --- Adaptive learning rate tracking ---
        self.consecutive_correct = 0  # Track consecutive correct guesses
        self.stabilized = False  # True if learning rate is reduced

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
        
        elif message_type == 'reach' and message.get('help_needed', False):
            # Help a struggling neighbor by sharing our knowledge
            if self.confidence > 0.6:  # Only help if we're reasonably confident
                # Share a bit of our confidence and specialty knowledge
                old_confidence = self.confidence
                confidence_share = min(0.1, (self.confidence - 0.5) * 0.2)  # Share excess confidence
                self.confidence -= confidence_share  # We lose a bit
                
                # Send helpful response back
                help_msg = {
                    'type': 'help_response',
                    'confidence_boost': confidence_share,
                    'suggested_specialty': self.specialty,
                    'preferred_color': self.get_preferred_color(),
                    'from': self.position
                }
                # Find the sender and send help back
                for neighbor in self.neighbors:
                    if neighbor.position == message.get('from'):
                        neighbor.receive_message(help_msg)
                        break
                print(f"[Neuron] {self.position} helped struggling neighbor {message.get('from')} (gave {confidence_share:.2f} confidence)")
        
        elif message_type == 'help_response':
            # Receive help from a neighbor
            confidence_boost = message.get('confidence_boost', 0)
            suggested_specialty = message.get('suggested_specialty', self.specialty)
            old_confidence = self.confidence
            
            self.confidence += confidence_boost
            self.confidence = min(1.0, self.confidence)
            
            # Consider the suggested specialty (but don't completely adopt it)
            if abs(suggested_specialty - self.specialty) < 3.0:
                self.specialty += 0.1 * (suggested_specialty - self.specialty)
                self.specialty = min(10.0, max(1.0, self.specialty))
            
            print(f"[Neuron] {self.position} received help from {message.get('from')}, confidence {old_confidence:.2f}->{self.confidence:.2f}")
        
        elif message_type in ['excite', 'inhibit']:
            # Simple activation adjustments
            if message_type == 'excite':
                self.activation_level += 0.1
            else:  # inhibit
                self.activation_level -= 0.1

    def receive_goal(self, goal):
        """Set goal and check alignment with specialty."""
        self.goal = goal
        # For now, simple string comparison - can be enhanced
        self.goal_alignment = (str(self.specialty).startswith(str(goal)[:3]) if goal else False)

    def evaluate_activation(self, input_signal, true_label=None, guess_color=None):
        self.signal_history.append(input_signal)
        
        # Update goal alignment based on current specialty and goal
        if self.goal:
            # Enhanced goal alignment calculation
            goal_color = self.goal.replace("detect_", "") if "detect_" in self.goal else self.goal
            specialty_range = {
                "red": (1.5, 3.5),
                "green": (5.5, 7.5), 
                "blue": (9.0, 11.0)
            }
            if goal_color in specialty_range:
                min_spec, max_spec = specialty_range[goal_color]
                self.goal_alignment = min_spec <= self.specialty <= max_spec
            else:
                self.goal_alignment = False
        
        # Consider signal strength, message context, specialty similarity, feedback, noise
        context = {
            'signal_strength': input_signal,
            'messages': self.last_received_messages,
            'neighbors': [n.position for n in self.neighbors],
            'specialty_similarity': [abs(self.specialty - n.specialty) for n in self.neighbors],
            'last_sent_message': self.last_sent_message,
            'confidence': self.confidence,
            'goal_alignment': self.goal_alignment,
            'iteration': self.iteration
        }
        
        # Enhanced decision logic with goal alignment
        activation_value = input_signal * self.weight
        if self.goal_alignment:
            activation_value *= 1.3  # Boost for goal-aligned neurons
        
        # Apply message-based activation adjustments
        activation_value += self.activation_level
        
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
        # --- Progressive confidence update logic ---
        # Use config_training for granular confidence update rates
        if true_label is not None:
            try:
                import config_training
                update_correct = getattr(config_training, 'CONFIDENCE_UPDATE_CORRECT', 0.2)
                update_incorrect = getattr(config_training, 'CONFIDENCE_UPDATE_INCORRECT', 0.2)
                early_rounds = getattr(config_training, 'EARLY_TRAINING_ROUNDS', 50)
                
                # Get current training round from context or cluster
                training_round = 0
                if hasattr(self, 'training_round'):
                    training_round = self.training_round
                elif self.cluster and hasattr(self.cluster, 'training_round'):
                    training_round = self.cluster.training_round
                
                # Progressive scaling: be gentler in early rounds
                if training_round < early_rounds:
                    # In early rounds, reduce penalties significantly
                    progress = training_round / early_rounds
                    penalty_scale = 0.1 + 0.9 * progress  # Start at 10% penalty, scale up
                    reward_scale = 1.0  # Keep full rewards
                else:
                    penalty_scale = 1.0
                    reward_scale = 1.0
                    
            except Exception:
                update_correct = 0.2
                update_incorrect = 0.2
                penalty_scale = 1.0
                reward_scale = 1.0
            
            # Use guess_color if provided, else fallback to self.goal
            guess_color = getattr(self, 'last_guess', None)
            predicted = guess_color if guess_color is not None else (self.goal if should_fire else 'none')
            
            old_confidence = self.confidence
            # --- Adaptive learning rate logic ---
            try:
                answer_threshold = getattr(config_training, 'CONCURRENT_ANSWER_THRESHOLD', 5)
                update_correct_min = getattr(config_training, 'CONFIDENCE_UPDATE_CORRECT_MIN', 0.05)
            except Exception:
                answer_threshold = 5
                update_correct_min = 0.05
            if predicted == true_label:
                self.consecutive_correct += 1
                if self.consecutive_correct >= answer_threshold:
                    if not self.stabilized:
                        print(f"[Neuron] {self.position} stabilized: {self.consecutive_correct} correct in a row. Using reduced learning rate.")
                    self.stabilized = True
                    update_correct = update_correct_min
                self.confidence += update_correct * activation_value * reward_scale
                # Hebbian-like: strengthen color association
                if guess_color is not None:
                    self.color_success_counts[guess_color] += 1
            else:
                self.consecutive_correct = 0
                self.stabilized = False
                # Apply scaled penalty (much gentler in early rounds)
                penalty = update_incorrect * activation_value * penalty_scale
                self.confidence -= penalty
                # Weaken color association
                if guess_color is not None and self.color_success_counts[guess_color] > 0:
                    self.color_success_counts[guess_color] -= 1
                    
            # Clamp confidence
            self.confidence = min(1.0, max(0.0, self.confidence))
            
            # Log significant confidence changes
            if abs(self.confidence - old_confidence) > 0.1:
                print(f"[Neuron] {self.position} confidence: {old_confidence:.2f}->{self.confidence:.2f} (correct={predicted==true_label}, penalty_scale={penalty_scale:.2f})")
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

    def send_message(self, training_round=0):
        """Send messages to neighbors based on confidence and specialty.
        
        Args:
            training_round: Current training round for adaptive thresholds
        """
        import config_training
        
        # Progressive teach threshold: start low, increase over time
        base_threshold = getattr(config_training, 'CONFIDENCE_SPECIALIZATION_THRESHOLD', 0.85)
        early_rounds = getattr(config_training, 'EARLY_TRAINING_ROUNDS', 50)
        
        if training_round < early_rounds:
            # Start with much lower threshold, gradually increase
            progress = training_round / early_rounds
            teach_threshold = 0.3 + (base_threshold - 0.3) * progress
        else:
            teach_threshold = base_threshold
            
        messages_sent = 0
        
        for neighbor in self.neighbors:
            specialty_diff = abs(self.specialty - neighbor.specialty)
            
            # Send teach message if confident enough and specialties are compatible
            if self.confidence > teach_threshold and specialty_diff <= 3.0:
                msg = {
                    'type': 'teach',
                    'specialty': self.specialty,
                    'confidence': self.confidence,
                    'preferred_color': self.get_preferred_color(),
                    'from': self.position,
                    'training_round': training_round
                }
                neighbor.receive_message(msg)
                self.last_sent_message = msg
                messages_sent += 1
                
            # Send reach message if we need help (low confidence)
            elif self.confidence < 0.4 and specialty_diff > 1.0:
                msg = {
                    'type': 'reach',
                    'specialty': self.specialty,
                    'confidence': self.confidence,
                    'help_needed': True,
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
            print(f"[Neuron] {self.position} sent {messages_sent} messages (conf={self.confidence:.2f}, thresh={teach_threshold:.2f})")
        
        return messages_sent

    def propagate_goal(self):
        """Propagate goal to neighboring neurons."""
        if not self.has_fired_this_round:
            for neighbor in self.neighbors:
                neighbor.receive_goal(self.goal)

    def learn_specialty(self, input_data, all_neurons=None):
        """
        Adjust specialty and weight based on successful activations, prediction correctness, confidence, and repulsion.
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
            # --- Adaptive weight learning ---
            try:
                import config_training
                weight_lr = getattr(config_training, 'WEIGHT_LEARNING_RATE', 0.2)
                weight_noise_range = getattr(config_training, 'WEIGHT_NOISE_RANGE', (-0.05, 0.05))
                weight_clamp_min, weight_clamp_max = getattr(config_training, 'WEIGHT_RANGE', (0.3, 1.0))
            except Exception:
                weight_lr = 0.2
                weight_noise_range = (-0.05, 0.05)
                weight_clamp_min, weight_clamp_max = 0.3, 1.0
            import random
            # Use last decision for correctness and confidence
            last_decision = self.decision_log[-1] if self.decision_log else None
            correct = False
            confidence = self.confidence
            if last_decision and 'context' in last_decision:
                context = last_decision['context']
                true_label = context.get('true_label', None)
                guess_color = getattr(self, 'last_guess', None)
                predicted = guess_color if guess_color is not None else (self.goal if self.active else 'none')
                correct = (predicted == true_label)
            # Weight update: proportional to confidence and correctness
            noise = random.uniform(*weight_noise_range)
            old_weight = self.weight
            if correct:
                self.weight += weight_lr * confidence * (1.0 - self.weight) + noise
            else:
                self.weight -= weight_lr * (1.0 - confidence) * (self.weight - weight_clamp_min) + noise
            # Clamp weight
            self.weight = max(weight_clamp_min, min(self.weight, weight_clamp_max))
            if abs(self.weight - old_weight) > 0.01:
                print(f"[Neuron] {self.position} weight: {old_weight:.4f}->{self.weight:.4f} (correct={correct}, conf={confidence:.2f})")
        # Clamp specialty between 1 and 10
        self.specialty = max(1, min(self.specialty, 10))
        # Notify cluster of specialty change and trigger controller updates
        if self.cluster is not None:
            if hasattr(self.cluster, 'adapt_specialty'):
                self.cluster.adapt_specialty()
            # Notify controller through cluster
            if hasattr(self.cluster, 'controller') and self.cluster.controller:
                self.cluster.controller.on_neuron_specialty_change(self, self.cluster)

    def clear_round(self):
        """Clear round-specific state for next processing cycle."""
        self.has_fired_this_round = False
        self.active = False
        self.message_buffer = []
        self.last_received_messages = []
        self.activation_level = 0.0  # Reset message-based activation adjustments

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