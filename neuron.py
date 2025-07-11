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
    
    def __init__(self, position, specialty=None, iteration=0):
        """Initialize neuron with position and optional specialty."""
        self.position = position            
        self.weight = np.random.uniform(0.3, 1.0)   # Better weight range
        self.active = False                 
        self.specialty = specialty if specialty is not None else np.random.uniform(10, 15)
        self.neighbors = []                 
        self.message_buffer = []            
        self.goal = None                    
        self.goal_alignment = False         
        self.has_fired_this_round = False
        self.iteration = iteration   

    def add_neighbor(self, neighbor):
        """Add a neighboring neuron for communication."""
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)

    def receive_message(self, message):
        """Receive message from neighboring neuron."""
        self.message_buffer.append(message)

    def receive_goal(self, goal):
        """Set goal and check alignment with specialty."""
        self.goal = goal
        # For now, simple string comparison - can be enhanced
        self.goal_alignment = (str(self.specialty).startswith(str(goal)[:3]) if goal else False)

    def evaluate_activation(self, input_signal):
        """
        Evaluate whether neuron should activate based on input signal.
        
        Args:
            input_signal: Normalized input signal (0-1)
        """
        if self.has_fired_this_round:
            return
            
        # Dynamic threshold based on specialty alignment
        base_threshold = 0.5
        threshold = base_threshold * (0.8 if self.goal_alignment else 1.2)
        
        # Check for inhibition messages
        inhibition = any(msg.get('type') == 'inhibit' for msg in self.message_buffer)
        excitation = sum(1 for msg in self.message_buffer if msg.get('type') == 'excite')
        
        # Calculate activation with excitation boost
        signal_strength = input_signal * self.weight + (excitation * 0.1)
        
        if signal_strength > threshold and not inhibition:
            self.active = True
            self.has_fired_this_round = True
        else:
            self.active = False

    def send_message(self):
        """Send excitation/inhibition messages to neighbors."""
        if self.active and self.has_fired_this_round:
            for neighbor in self.neighbors:
                # Similar specialties excite, different ones inhibit
                specialty_diff = abs(self.specialty - neighbor.specialty)
                if specialty_diff < 2.0:  # Similar specialties
                    neighbor.receive_message({'type': 'excite', 'from': self.position})
                else:
                    neighbor.receive_message({'type': 'inhibit', 'from': self.position})

    def propagate_goal(self):
        """Propagate goal to neighboring neurons."""
        if not self.has_fired_this_round:
            for neighbor in self.neighbors:
                neighbor.receive_goal(self.goal)

    def learn_specialty(self, input_data):
        """
        Adjust specialty based on successful activations.
        
        Args:
            input_data: The input that caused activation
        """
        if self.active and self.has_fired_this_round:
            # Simple learning: adjust specialty toward successful input
            learning_rate = 0.5
            self.iteration += 1
            if isinstance(input_data, (int, float)):
                # Numeric specialty adjustment
                self.specialty += learning_rate * (input_data - self.specialty)
            # For more complex input types, implement appropriate learning rules

    def clear_round(self):
        """Clear round-specific state for next processing cycle."""
        self.has_fired_this_round = False
        self.active = False
        self.message_buffer = []
        pass

    def clear_messages(self):
        self.message_buffer = []
        self.has_fired_this_round = False